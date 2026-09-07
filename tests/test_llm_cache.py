"""The cache contract, which is what lets a reviewer run this without a key.

The README promises that a clean checkout with no API key reproduces the
knowledge layer from committed cache files.  That promise broke once already:
the cache key included the provider name, so `FACTLAYER_LLM=replay` could never
match an entry written by `gemini`, and every replay was a miss.  These tests
pin the properties that make the promise true.
"""

from __future__ import annotations

import json

import pytest

from factlayer import llm


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    return tmp_path


def _seed(cache_dir, prompt: str, data, model="gemini-3.5-flash", provider="gemini"):
    key = llm._cache_key(provider, model, prompt)
    path = llm._cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": data, "raw": "", "model": model, "provider": provider}))
    return key


def test_cache_key_ignores_provider_and_model(cache_dir):
    """An answer to an identical prompt is reusable whoever produced it."""
    prompt = "what is the revenue"
    assert (
        llm._cache_key("gemini", "gemini-3.8-flash", prompt)
        == llm._cache_key("replay", "gemini-2.5-flash", prompt)
        == llm._cache_key("anthropic", "claude-sonnet-4-5", prompt)
    )


def test_cache_key_distinguishes_prompts(cache_dir):
    assert llm._cache_key("gemini", "m", "prompt A") != llm._cache_key("gemini", "m", "prompt B")


def test_replay_serves_an_entry_written_under_another_provider(cache_dir, monkeypatch):
    """The exact regression: gemini writes it, replay must read it."""
    _seed(cache_dir, "extract this page", {"facts": [{"metric": "revenue"}]}, provider="gemini")
    monkeypatch.setenv("FACTLAYER_LLM", "replay")

    res = llm.complete_json("extract this page")
    assert res.cached is True
    assert res.data["facts"][0]["metric"] == "revenue"
    assert res.model == "gemini-3.5-flash"  # provenance is preserved


def test_replay_refuses_a_miss_rather_than_calling_out(cache_dir, monkeypatch):
    """A cache miss in replay mode must be an error, never a silent network call."""
    monkeypatch.setenv("FACTLAYER_LLM", "replay")
    with pytest.raises(llm.LLMUnavailable):
        llm.complete_json("a prompt that was never issued")


def test_json_is_recovered_from_code_fences():
    assert llm._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._extract_json('here you go: [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        llm._extract_json("no json here at all")


def test_transient_errors_are_retryable_and_bad_requests_are_not():
    assert llm._is_transient(Exception("503 UNAVAILABLE high demand"))
    assert llm._is_transient(Exception("429 RESOURCE_EXHAUSTED"))
    assert llm._is_rate_limit(Exception("429 RESOURCE_EXHAUSTED"))
    assert not llm._is_transient(Exception("400 INVALID_ARGUMENT"))
    assert not llm._is_rate_limit(Exception("503 UNAVAILABLE"))


def test_server_supplied_retry_delay_is_honoured():
    delay = llm._retry_delay_from(Exception("{'retryDelay': '17s'}"), 0)
    assert 17.0 <= delay <= 18.5


def _one_key(monkeypatch, *keys):
    monkeypatch.setattr(llm, "_POOL", llm.KeyPool(list(keys)))
    monkeypatch.setattr(llm, "_cooldown", {})


def test_cooldown_moves_the_chain_to_the_next_model(monkeypatch):
    """A rate-limited model must be skipped, not slept on."""
    monkeypatch.setenv("FACTLAYER_MODEL_CHAIN", "model-a,model-b,model-c")
    _one_key(monkeypatch, "k1")
    assert llm.model_name() == "model-a"
    llm.cool_down("model-a", 60, "k1")
    assert llm.model_name() == "model-b"
    llm.cool_down("model-b", 60, "k1")
    assert llm.model_name() == "model-c"


def test_one_key_exhausting_does_not_park_the_model_for_the_others(monkeypatch):
    """Quotas are per project, so keys must be parked independently.

    Parking the model globally on the first 429 would discard exactly the
    capacity a second key was added to provide.
    """
    monkeypatch.setenv("FACTLAYER_MODEL_CHAIN", "model-a,model-b")
    _one_key(monkeypatch, "k1", "k2")

    llm.cool_down("model-a", 60, "k1")
    assert llm._cooling("model-a", "k1") > 0
    assert llm._cooling("model-a", "k2") == 0
    # Some key is still free, so the model stays in play...
    assert llm.model_cooling("model-a") == 0
    assert llm.model_name() == "model-a"
    # ...and the pool hands out the key that is not cooling.
    assert llm.key_pool().next_key_for("model-a") == "k2"

    # Only when every key is spent does the chain fall through.
    llm.cool_down("model-a", 60, "k2")
    assert llm.model_cooling("model-a") > 0
    assert llm.model_name() == "model-b"


def test_rate_budget_scales_with_key_count(monkeypatch):
    monkeypatch.setenv("FACTLAYER_RPM", "10")
    monkeypatch.setattr(llm, "_LIMITERS", {})
    monkeypatch.setattr(llm, "_POOL", llm.KeyPool(["k1", "k2", "k3"]))
    assert llm.limiter("model-a").rpm == 30


def test_rate_limiter_admits_up_to_its_budget_without_blocking():
    rl = llm.RateLimiter(rpm=50)
    import time

    t0 = time.monotonic()
    for _ in range(50):
        rl.acquire()
    assert time.monotonic() - t0 < 1.0
