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


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def limiter(model: str = "") -> RateLimiter:
    """Pacing budget for one model.

    Buckets are per model, not global.  Providers meter each model separately,
    so a single shared budget both under-uses the models that are free and --
    worse -- spends its allowance throttling retries against a model that has
    already hit its ceiling, which is how a run ends up stalled rather than
    falling through to the next model.
    """
    per_model = int(os.environ.get("FACTLAYER_RPM", "10")) * max(1, len(key_pool()))
    with _LIMITERS_LOCK:
        if model not in _LIMITERS:
            _LIMITERS[model] = RateLimiter(per_model)
        return _LIMITERS[model]


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
    """Key on the prompt alone.

    Provider and model are recorded inside each entry for provenance, but both
    are deliberately excluded from the key. A cached answer to an identical
    prompt is a cached answer whoever produced it, and keying on either one
    breaks the two things the cache exists for: replaying a committed corpus
    after the default model changes, and replaying it at all under
    FACTLAYER_LLM=replay, whose provider name would otherwise never match the
    "gemini" that wrote the entry.
    """
    return hashlib.sha1(prompt.encode()).hexdigest()


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

    def next_key_for(self, model: str) -> str:
        """Round-robin over the keys that are not cooling down on this model."""
        with self._lock:
            n = len(self.keys)
            for _ in range(n):
                k = self.keys[self._i % n]
                self._i += 1
                if _cooling(model, k) <= 0:
                    return k
            # All cooling: take the one that frees up soonest.
            return min(self.keys, key=lambda k: _cooling(model, k))


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
    api_key = pool.next_key_for(model)
    timeout_ms = int(float(os.environ.get("FACTLAYER_TIMEOUT_S", "150")) * 1000)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
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
            cool_down(model, _retry_delay_from(exc, 0), api_key)
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


# Free tiers meter each model separately and generously to none of them, so a
# 500-page corpus can exhaust one model's daily allowance mid-run.  Rather than
# stopping, fall through to the next model in the chain.  Order is by preference;
# every entry is a capable extraction model.
_FALLBACK_CHAINS = {
    # Ordered by measured latency on a trivial prompt, best first. Slow models
    # are excluded rather than demoted: one that takes 19s on a trivial prompt
    # takes minutes on a 20k-character extraction, and blocks a worker that
    # could have used a free model instead.
    "gemini": [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ],
}

# A rate-limited model is not exhausted for the day -- it is busy for a while.
# Cooldowns are timestamps, so a model returns to the chain on its own.
#
# They are keyed by (model, key), not by model alone. Quotas are metered per
# project, so when several keys from different projects are configured, one key
# hitting its ceiling on a model says nothing about the others. Parking the
# model globally would throw away the capacity the extra keys were added for.
_cooldown: dict[tuple[str, str], float] = {}
_exhausted_lock = threading.Lock()


def _cooling(model: str, api_key: str = "") -> float:
    """Seconds until this (model, key) pair is worth trying again."""
    with _exhausted_lock:
        return max(0.0, _cooldown.get((model, api_key), 0.0) - time.monotonic())


def cool_down(model: str, seconds: float, api_key: str = "") -> None:
    with _exhausted_lock:
        cur = _cooldown.get((model, api_key), 0.0)
        _cooldown[(model, api_key)] = max(cur, time.monotonic() + seconds)


def model_cooling(model: str) -> float:
    """Seconds until *some* configured key could try this model again.

    Zero as soon as any one key is free, which is what the fallback loop needs.
    """
    keys = key_pool().keys or [""]
    return min(_cooling(model, k) for k in keys)


def model_chain() -> list[str]:
    """Models to try, in order. An explicit FACTLAYER_MODEL pins to one."""
    p = provider_name()
    pinned = os.environ.get("FACTLAYER_MODEL")
    if pinned:
        return [pinned]
    override = os.environ.get("FACTLAYER_MODEL_CHAIN")
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return _FALLBACK_CHAINS.get(p, [_DEFAULT_MODELS.get(p, "gemini-3.8-flash")])


def model_name() -> str:
    """The model a call would use right now, skipping any that are cooling."""
    chain = model_chain()
    for m in chain:
        if model_cooling(m) <= 0:
            return m
    return chain[0]


def exhausted_models() -> list[tuple[str, int]]:
    """Models with no free key right now, and seconds until one frees up."""
    return sorted(
        (m, round(model_cooling(m))) for m in model_chain() if model_cooling(m) > 0
    )


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

    attempts = int(os.environ.get("FACTLAYER_RETRIES", "2"))
    chain = model_chain()
    last: Exception | None = None
    raw = ""
    used = model

    # Two passes. The first tries every model that is not cooling down; a
    # rate-limited model is skipped immediately rather than slept on, because
    # the whole point of the chain is that another model is probably free. Only
    # if every model is cooling do we wait, and then for the shortest cooldown.
    for round_no in range(2):
        for candidate in chain:
            if model_cooling(candidate) > 0:
                continue
            used = candidate
            for attempt in range(attempts):
                limiter(candidate).acquire()
                try:
                    raw = fn(candidate, prompt, temperature)
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if _is_rate_limit(exc):
                        # The provider already parked the (model, key) pair that
                        # failed. If another key is still free for this model,
                        # retry it; otherwise fall through to the next model.
                        if model_cooling(candidate) > 0:
                            break
                        continue
                    if not _is_transient(exc):
                        raise
                    if attempt == attempts - 1:
                        cool_down(candidate, 5.0)
                        break

                    time.sleep(min(4.0, 1.5 ** attempt) + random.uniform(0.1, 0.5))
            if raw:
                break
        if raw:
            break
        if round_no == 0:
            waits = [w for w in (model_cooling(m) for m in chain) if w > 0]
            if not waits:
                break
            time.sleep(min(min(waits) + 0.5, 65.0))

    if not raw:
        raise last if last else RuntimeError("no response from any model in the chain")

    data = _extract_json(raw)
    _write_cache(key, {"data": data, "raw": raw[:20000], "model": used,
                       "provider": provider, "tag": tag})
    return LLMResult(data=data, raw=raw, cached=False, model=used)


def cache_stats() -> dict:
    files = list(CACHE_DIR.rglob("*.json")) if CACHE_DIR.exists() else []
    return {
        "entries": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "dir": str(CACHE_DIR),
    }
