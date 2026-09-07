"""Model access: pluggable provider, JSON-mode calls, and an on-disk cache.

Two constraints shaped this module.

*Reviewers must be able to run the project without an API key.*  Every call is
cached to `data/cache/` keyed by a hash of (provider, model, prompt).  Those
cache files are committed, so a clean checkout reproduces the full knowledge
layer offline.  `FACTLAYER_LLM=replay` makes a missing cache entry an error
rather than a network call, which is also how the test suite stays hermetic.

*The provider must be swappable.*  Nothing above this module knows which model
answered.  Adding a provider means adding one function.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .config import load_dotenv

load_dotenv()

CACHE_DIR = Path(os.environ.get("FACTLAYER_CACHE", "data/cache"))
_LOCK = threading.Lock()


class LLMUnavailable(RuntimeError):
    """Raised when a call is needed but no provider is configured."""


class RateLimiter:
    """Token bucket shared across extraction threads.

    Provider free tiers are quoted per minute, and hitting the ceiling returns
    429s that -- without this -- get swallowed as "page failed" and silently
    shrink the corpus.  Pacing the calls is cheaper than retrying them.
    """

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] > 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.05
            time.sleep(max(0.05, wait))


_LIMITER: RateLimiter | None = None


def limiter() -> RateLimiter:
    """Global pacing budget: per-key RPM multiplied by the number of keys."""
    global _LIMITER
    if _LIMITER is None:
        per_key = int(os.environ.get("FACTLAYER_RPM", "8"))
        n = max(1, len(key_pool())) if provider_name() == "gemini" else 1
        _LIMITER = RateLimiter(per_key * n)
    return _LIMITER


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def _is_transient(exc: Exception) -> bool:
    """Retryable server-side conditions, as opposed to a bad request.

    A 503 "model is experiencing high demand" is not a programming error and
    not a quota problem -- it is weather.  Treating it as fatal killed a full
    corpus run on its first document, so transient server errors get the same
    backoff as rate limits.
    """
    text = str(exc)
    if _is_rate_limit(exc):
        return True
    return any(
        marker in text
        for marker in ("500", "502", "503", "504", "UNAVAILABLE", "INTERNAL",
                       "DEADLINE_EXCEEDED", "Timeout", "timed out", "Connection")
    )


def _retry_delay_from(exc: Exception, attempt: int) -> float:
    """Honour a server-supplied retry delay when present, else back off."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if m:
        return float(m.group(1)) + random.uniform(0.2, 1.0)
    return min(60.0, 2.0 ** attempt) + random.uniform(0.2, 1.2)


@dataclass
class LLMResult:
    data: dict | list
    raw: str
    cached: bool
    model: str


def _cache_key(provider: str, model: str, prompt: str) -> str:
    return hashlib.sha1(f"{provider}\x00{model}\x00{prompt}".encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _read_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _write_cache(key: str, payload: dict) -> None:
    p = _cache_path(key)
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1))


def _extract_json(text: str) -> dict | list:
    """Pull a JSON value out of a model response, tolerating code fences."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost bracketed region.
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]!r}")


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class KeyPool:
    """Round-robin over several API keys, parking any that reports exhaustion.

    Free tiers meter per key, per day.  A 500-page corpus sits close enough to
    the daily ceiling that a single key can strand a run half-finished, so the
    pool spreads calls across whatever keys are configured and skips one that
    has started returning 429s until its cooldown expires.  With one key
    configured this degrades to exactly the single-key behaviour.
    """

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._i = 0
        self._cooldown: dict[str, float] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.keys)

    def next_key(self) -> str:
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self.keys)):
                k = self.keys[self._i % len(self.keys)]
                self._i += 1
                if self._cooldown.get(k, 0.0) <= now:
                    return k
            # Every key is cooling down; use the one that frees up soonest.
            return min(self.keys, key=lambda k: self._cooldown.get(k, 0.0))

    def penalise(self, key: str, seconds: float = 90.0) -> None:
        with self._lock:
            self._cooldown[key] = time.monotonic() + seconds


_POOL: KeyPool | None = None


def key_pool() -> KeyPool:
    """Keys from GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY."""
    global _POOL
    if _POOL is None:
        raw = os.environ.get("GEMINI_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            keys = [single] if single else []
        _POOL = KeyPool(keys)
    return _POOL


def _call_gemini(model: str, prompt: str, temperature: float) -> str:
    from google import genai
    from google.genai import types

    pool = key_pool()
    if not len(pool):
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    api_key = pool.next_key()
    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                max_output_tokens=32768,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit(exc):
            pool.penalise(api_key)
        raise
    return resp.text or ""


def _call_openai(model: str, prompt: str, temperature: float) -> str:
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_anthropic(model: str, prompt: str, temperature: float) -> str:
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json={
            "model": model,
            "max_tokens": 16000,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=180,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


_PROVIDERS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}

_DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-5",
}


def provider_name() -> str:
    return os.environ.get("FACTLAYER_LLM", "gemini").lower()


def model_name() -> str:
    p = provider_name()
    return os.environ.get("FACTLAYER_MODEL", _DEFAULT_MODELS.get(p, "gemini-2.5-flash"))


def complete_json(prompt: str, *, temperature: float = 0.0, tag: str = "") -> LLMResult:
    """Run a JSON-mode completion, using the cache when possible.

    `tag` is recorded in the cache entry for provenance only; it does not
    affect the key, so re-tagging a prompt does not invalidate the cache.
    """
    provider = provider_name()
    model = model_name()
    key = _cache_key(provider, model, prompt)

    hit = _read_cache(key)
    if hit is not None:
        return LLMResult(data=hit["data"], raw=hit.get("raw", ""), cached=True, model=hit.get("model", model))

    if provider == "replay":
        raise LLMUnavailable(
            "FACTLAYER_LLM=replay but this prompt is not in the cache. "
            "Set a provider and API key to extract new documents."
        )

    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise LLMUnavailable(f"unknown provider {provider!r}")

    attempts = int(os.environ.get("FACTLAYER_RETRIES", "5"))
    last: Exception | None = None
    raw = ""
    for attempt in range(attempts):
        limiter().acquire()
        try:
            raw = fn(model, prompt, temperature)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            time.sleep(_retry_delay_from(exc, attempt))
    else:
        raise last if last else RuntimeError("no response")

    data = _extract_json(raw)
    _write_cache(key, {"data": data, "raw": raw[:20000], "model": model, "provider": provider, "tag": tag})
    return LLMResult(data=data, raw=raw, cached=False, model=model)


def cache_stats() -> dict:
    files = list(CACHE_DIR.rglob("*.json")) if CACHE_DIR.exists() else []
    return {
        "entries": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "dir": str(CACHE_DIR),
    }
