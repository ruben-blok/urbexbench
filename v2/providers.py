"""OpenRouter provider resolution for UrbexBench v2.

Selects a deterministic provider per model so runs are reproducible:
the model author's endpoint when available, otherwise the cheapest active
endpoint.
"""

import json
import urllib.request
from typing import Optional

MODELS_API_URL = "https://openrouter.ai/api/v1/models"

# Canonical author tags per model author, in preference order (non-flex,
# non-priority first). Falls back to the author slug itself when unmapped.
AUTHOR_TAGS = {
    "google": ["google-ai-studio", "google-vertex/global"],
    "openai": ["openai"],
    "deepseek": ["deepseek"],
    "qwen": ["alibaba"],
    "nvidia": ["nvidia"],
    "xiaomi": ["xiaomi/fp8"],
}


def fetch_endpoints(model_id: str) -> list:
    """Return the raw endpoint list for a model from OpenRouter."""
    req = urllib.request.Request(
        f"{MODELS_API_URL}/{model_id}/endpoints",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response).get("data", {}).get("endpoints", [])


def _active(endpoints: list) -> list:
    """Keep only endpoints OpenRouter reports as active."""
    return [e for e in endpoints if e.get("status") == 0]


def price_sum(endpoint: dict) -> float:
    """Prompt + completion price per token, used as the cheapest metric."""
    pricing = endpoint.get("pricing") or {}
    try:
        return float(pricing.get("prompt") or 0) + float(pricing.get("completion") or 0)
    except (TypeError, ValueError):
        return float("inf")


def resolve_provider(model_id: str, override: Optional[str] = None) -> Optional[dict]:
    """Resolve the provider tag to pin for a model.

    Returns {"tag", "source", "provider_name"} or None when no active
    endpoint exists at all. ``source`` is "override", "author" or
    "fallback-cheapest".
    """
    if override:
        return {"tag": override, "source": "override", "provider_name": override}

    endpoints = _active(fetch_endpoints(model_id))
    if not endpoints:
        return None

    by_tag = {e.get("tag"): e for e in endpoints}
    author = model_id.split("/", 1)[0]
    for candidate in AUTHOR_TAGS.get(author, [author]):
        if candidate in by_tag:
            return {
                "tag": candidate,
                "source": "author",
                "provider_name": by_tag[candidate].get("provider_name"),
            }

    cheapest = min(endpoints, key=price_sum)
    return {
        "tag": cheapest.get("tag"),
        "source": "fallback-cheapest",
        "provider_name": cheapest.get("provider_name"),
    }


def provider_preferences(tag: str) -> dict:
    """OpenRouter request body fragment that pins a single provider."""
    return {"only": [tag], "allow_fallbacks": False}
