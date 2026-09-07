"""Embeddings for metric names, with an on-disk cache.

Lexical similarity is not enough to decide which metrics are worth comparing.
"PTL freight tonnage" and "Part-truckload tonnage" are the same measurement and
score 63 on token-set ratio -- below any threshold that is not also flooded with
false pairs.  An abbreviation is invisible to string distance.

So candidate generation runs on meaning as well as spelling: unique metric names
are embedded once, and pairs are proposed when they are close *either* lexically
or semantically.  Recall comes from the union; precision is restored downstream
by the model adjudicating each proposed pair.

Embeddings are optional.  With no provider configured the resolver degrades to
lexical-only candidates and says so, rather than failing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

CACHE_PATH = Path(os.environ.get("FACTLAYER_EMBED_CACHE", "data/embeddings.json"))
_LOCK = threading.Lock()

EMBED_MODELS = {
    "gemini": "models/gemini-embedding-001",
    "openai": "text-embedding-3-small",
}


class EmbeddingsUnavailable(RuntimeError):
    pass


def _load_cache() -> dict[str, list[float]]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    with _LOCK:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache))


def _key(provider: str, model: str, text: str) -> str:
    return hashlib.sha1(f"{provider}\x00{model}\x00{text}".encode()).hexdigest()[:20]


def _embed_gemini(model: str, texts: list[str]) -> list[list[float]]:
    from google import genai

    # Use the same key pool as completions. Reading GEMINI_API_KEY directly
    # meant a run configured through the plural GEMINI_API_KEYS silently had no
    # embeddings at all -- and because the caller swallowed the error, the whole
    # semantic tier disappeared without a word.
    from .llm import key_pool

    pool = key_pool()
    api_key = pool.keys[0] if len(pool) else (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise EmbeddingsUnavailable("no Gemini API key configured")
    client = genai.Client(api_key=api_key)
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = client.models.embed_content(model=model, contents=batch)
        out.extend(list(e.values) for e in resp.embeddings)
    return out


def _embed_openai(model: str, texts: list[str]) -> list[list[float]]:
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY")  # noqa: SIM910
    if not api_key:
        raise EmbeddingsUnavailable("OPENAI_API_KEY is not set")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    out: list[list[float]] = []
    for i in range(0, len(texts), 256):
        r = httpx.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": texts[i : i + 256]},
            timeout=120,
        )
        r.raise_for_status()
        out.extend(d["embedding"] for d in r.json()["data"])
    return out


def embed(texts: list[str], provider: str | None = None) -> np.ndarray:
    """Embed strings, reusing cached vectors. Returns an L2-normalised matrix."""
    from .llm import provider_name

    provider = provider or provider_name()
    if provider == "replay":
        # Replay may still serve fully-cached vectors; only fail if some miss.
        provider_for_model = "gemini"
    else:
        provider_for_model = provider
    model = os.environ.get("FACTLAYER_EMBED_MODEL", EMBED_MODELS.get(provider_for_model, ""))
    if not model:
        raise EmbeddingsUnavailable(f"no embedding model for provider {provider!r}")

    cache = _load_cache()
    missing = [t for t in texts if _key(provider_for_model, model, t) not in cache]

    if missing:
        if provider == "replay":
            raise EmbeddingsUnavailable(
                f"{len(missing)} metric names are not in the embedding cache"
            )
        fn = {"gemini": _embed_gemini, "openai": _embed_openai}.get(provider_for_model)
        if fn is None:
            raise EmbeddingsUnavailable(f"no embedding backend for {provider!r}")
        vectors = fn(model, missing)
        for t, v in zip(missing, vectors):
            cache[_key(provider_for_model, model, t)] = v
        _save_cache(cache)

    mat = np.array([cache[_key(provider_for_model, model, t)] for t in texts], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# Why the last semantic_pairs() call produced nothing, if it produced nothing.
LAST_ERROR: str | None = None


def semantic_pairs(
    keys: list[str], threshold: float = 0.80, top_k: int = 10
) -> list[tuple[str, str, float]]:
    """Pairs of metric names that are close in meaning.

    Degrades to an empty list when embeddings are unavailable, so the caller can
    fall back to lexical candidates -- but records *why* in LAST_ERROR. Silently
    returning nothing once hid a dead semantic tier behind a plausible-looking
    run, which is the failure mode this module exists to avoid.
    """
    global LAST_ERROR
    LAST_ERROR = None
    uniq = sorted(set(k for k in keys if k))
    if len(uniq) < 2:
        return []
    try:
        mat = embed(uniq)
    except EmbeddingsUnavailable as exc:
        LAST_ERROR = f"unavailable: {exc}"
        return []
    except Exception as exc:  # noqa: BLE001
        LAST_ERROR = f"{type(exc).__name__}: {str(exc)[:160]}"
        return []

    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)

    pairs: dict[tuple[str, str], float] = {}
    k = min(top_k, len(uniq) - 1)
    for i, a in enumerate(uniq):
        idx = np.argpartition(sims[i], -k)[-k:]
        for j in idx:
            score = float(sims[i, j])
            if score < threshold:
                continue
            b = uniq[j]
            key = (a, b) if a < b else (b, a)
            pairs[key] = max(pairs.get(key, 0.0), score)
    return [(a, b, s) for (a, b), s in sorted(pairs.items(), key=lambda kv: -kv[1])]


def cache_size() -> int:
    return len(_load_cache())
