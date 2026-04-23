"""OpenAI API pricing per 1M tokens.

Single source of truth for model pricing across Architect modules.
Prices are in USD per 1M tokens (input and output).
Website: https://developers.openai.com/api/docs/pricing/
"""

from typing import Any, Dict

# OpenAI API pricing per 1M tokens (as of February 2026)
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "o3": {"input": 2.00, "output": 8.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
}

DEFAULT_PRICING = {"input": 2.00, "output": 10.00}  # fallback for unknown models


def get_pricing(model: str) -> Dict[str, float]:
    """Resolve pricing for model (handles partial names like o3).

    Args:
        model: Model name (e.g., "o3", "gpt-4o", "o3-mini-2025-01-31")

    Returns:
        Dict with "input" and "output" keys (USD per 1M tokens)
    """
    if model in PRICING:
        return PRICING[model]
    for key in PRICING:
        if model.lower().startswith(key.lower()):
            return PRICING[key]
    return DEFAULT_PRICING.copy()
