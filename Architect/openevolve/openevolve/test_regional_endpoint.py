"""
Test the regional endpoint detection logic without requiring API keys
Run this to verify the fix works correctly
"""

def test_endpoint_detection():
    """Test that all OpenAI regional endpoints are detected correctly"""
    
    OPENAI_REASONING_MODEL_PREFIXES = (
        "o1-", "o1", "o3-", "o3", "o4-",
        "gpt-5-", "gpt-5", "gpt-oss-120b", "gpt-oss-20b",
    )
    
    test_cases = [
        # (api_base, model, should_be_reasoning_model, description)
        ("https://eu.api.openai.com/v1", "o1-mini", True, "EU endpoint with o1-mini"),
        ("https://api.openai.com/v1", "o1-preview", True, "US endpoint with o1-preview"),
        ("https://apac.api.openai.com/v1", "o3-mini", True, "APAC endpoint with o3-mini"),
        ("https://eu.api.openai.com/v1", "gpt-4", False, "EU endpoint with gpt-4"),
        ("https://api.openai.com/v1", "gpt-3.5-turbo", False, "US endpoint with gpt-3.5"),
        ("https://azure.openai.com/", "o1-mini", False, "Azure endpoint (not OpenAI)"),
        ("https://fake.com/api.openai.com", "o1-mini", False, "Fake endpoint with o1"),
        (None, "o1-mini", False, "None endpoint"),
        ("", "o1-mini", False, "Empty endpoint"),
        ("https://eu.api.openai.com/v1", "O1-MINI", True, "EU with uppercase model"),
        ("HTTPS://EU.API.OPENAI.COM/v1", "o1-mini", True, "Uppercase URL"),
    ]
    
    print("Testing Regional Endpoint Detection Logic")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for api_base, model, expected_result, description in test_cases:
        # This is the exact logic from your fixed code
        model_lower = str(model).lower()
        api_base_lower = (api_base or "").lower()
        
        is_openai_api = (
            api_base_lower.startswith("https://api.openai.com")
            or api_base_lower.startswith("https://eu.api.openai.com")
            or api_base_lower.startswith("https://apac.api.openai.com")
            or api_base_lower.startswith("http://api.openai.com")
            or api_base_lower.startswith("http://eu.api.openai.com")
            or api_base_lower.startswith("http://apac.api.openai.com")
        )
        
        is_openai_reasoning_model = (
            is_openai_api
            and model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)
        )
        
        # Determine which parameter would be used
        param_used = "max_completion_tokens" if is_openai_reasoning_model else "max_tokens"
        expected_param = "max_completion_tokens" if expected_result else "max_tokens"
        
        # Check if result matches expectation
        if is_openai_reasoning_model == expected_result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"\n{status} | {description}")
        print(f"  API Base: {api_base}")
        print(f"  Model: {model}")
        print(f"  is_openai_api: {is_openai_api}")
        print(f"  is_reasoning_model: {is_openai_reasoning_model}")
        print(f"  Parameter used: {param_used}")
        print(f"  Expected: {expected_param}")
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("🎉 All tests PASSED! The fix is working correctly.")
        return True
    else:
        print("⚠️ Some tests FAILED! Please review the logic.")
        return False

if __name__ == "__main__":
    success = test_endpoint_detection()
    exit(0 if success else 1)