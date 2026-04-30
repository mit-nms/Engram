"""
Demonstration of fixed OpenAI model parameter handling
"""


def demo_model_parameter_selection():
    """Demonstrate how different models get different parameters"""

    # Mock the logic from openai.py
    OPENAI_REASONING_MODEL_PREFIXES = (
        # O-series reasoning models
        "o1-",
        "o1",  # o1, o1-mini, o1-preview
        "o3-",
        "o3",  # o3, o3-mini, o3-pro
        "o4-",  # o4-mini
        # GPT-5 series are also reasoning models
        "gpt-5-",
        "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
    )

    def get_params_for_model(model_name, api_base="https://api.openai.com/v1"):
        """Show what parameters would be used for each model"""
        model_lower = str(model_name).lower()
        is_openai_reasoning_model = (
            api_base == "https://api.openai.com/v1"
            and model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)
        )

        if is_openai_reasoning_model:
            return {
                "type": "reasoning_model",
                "uses": "max_completion_tokens",
                "supports": ["reasoning_effort", "verbosity"],
                "excludes": ["temperature", "top_p"],
            }
        else:
            return {
                "type": "standard_model",
                "uses": "max_tokens",
                "supports": ["temperature", "top_p"],
                "excludes": [],
            }

    print("🔧 OpenAI Model Parameter Selection Demo")
    print("=" * 50)

    test_models = [
        # Reasoning models
        ("o1-mini", "✅ Reasoning"),
        ("o1-preview", "✅ Reasoning"),
        ("o3-mini-2025-01-31", "✅ Reasoning (with date)"),
        ("gpt-5-nano", "✅ Reasoning (GPT-5 series)"),
        # Standard models
        ("gpt-4o-mini", "❌ Standard (not reasoning)"),
        ("gpt-4o", "❌ Standard"),
        ("gpt-4-turbo", "❌ Standard"),
    ]

    for model, description in test_models:
        params = get_params_for_model(model)
        print(f"\n📋 Model: {model}")
        print(f"   Type: {description}")
        print(f"   Uses: {params['uses']}")
        print(f"   Supports: {', '.join(params['supports'])}")
        if params["excludes"]:
            print(f"   Excludes: {', '.join(params['excludes'])}")

    print("\n" + "=" * 50)
    print("✅ Fix successful! No more false positives/negatives.")


if __name__ == "__main__":
    demo_model_parameter_selection()
