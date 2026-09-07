"""Runtime configuration, loaded from the environment or a local .env file.

Credentials never enter the repository.  `.env` is gitignored; the committed
`.env.example` documents the variables without carrying values.  When no key is
present the system still runs, serving the committed extraction cache, so a
reviewer can evaluate the whole pipeline without an account of their own.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(".env")


def load_dotenv(path: Path = ENV_PATH) -> list[str]:
    """Load KEY=VALUE lines into os.environ without overriding what is set."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def has_llm_credentials() -> bool:
    return any(
        os.environ.get(k)
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )


def describe() -> dict:
    from .llm import cache_stats, model_name, provider_name

    return {
        "provider": provider_name(),
        "model": model_name(),
        "credentials_present": has_llm_credentials(),
        "cache": cache_stats(),
    }


# Load once at import so every entry point (CLI, API, tests) sees the same env.
load_dotenv()
