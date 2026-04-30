"""Function schemas for OpenAI function calling."""

FUNCTION_SCHEMAS = {
    "implement_algorithm": {
        "name": "implement_algorithm",
        "description": "Implement an algorithm based on the given description",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code implementation of the algorithm"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the implementation approach and key decisions"
                }
            },
            "required": ["code"]
        }
    }
}
