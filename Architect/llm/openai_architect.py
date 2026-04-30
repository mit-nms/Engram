import os
import sys
from openai import OpenAI
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import re

# Add the parent directory to the Python path so that SystemBench can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .func_schema import FUNCTION_SCHEMAS
from Architect.task import Task
from Architect.pricing_table import get_pricing


class OpenAIArchitect:
    """An architect that uses GPT-4o to design and test algorithms"""

    def __init__(
        self,
        model: str,
        task: Task,
        api_key: Optional[str] = None,
    ):
        """Initialize the architect with OpenAI API key, the environment evaluator, and the task type

        Args:
            model: The OpenAI model to use
            task: The Task object
            api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        """
        self.model = model
        self.task = task

        # Initialize cost tracking
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        # Validate model and get pricing
        self.price_info = get_pricing(model)

        # Set up API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Load system prompt
        system_prompt_path = Path(__file__).parent / "default_prompts" / "system_prompt.txt"
        self._system_prompt = system_prompt_path.read_text()

        # Load fix prompt
        fix_prompt_path = Path(__file__).parent / "default_prompts" / "fix_prompt.txt"
        self._fix_prompt = fix_prompt_path.read_text()

        # Initialize prompt
        self._implement_prompt = task.task_prompt

        # Initialize conversation history with system prompt followed by task prompt
        self._conversation_history = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._implement_prompt},
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_llm_response(self, prompt: str) -> str:
        """Get a response from the LLM."""
        try:
            # Ensure prompt is a string
            if isinstance(prompt, (dict, list)):
                prompt = json.dumps(prompt, indent=2)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            self._update_usage_stats(response)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            self._conversation_history.append({"role": "user", "content": f"API Error: {error_msg}"})
            raise

    def override_implement_prompt(self, new_prompt: str) -> None:
        """Override the implementation prompt for the next generation."""
        self._implement_prompt = new_prompt

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._conversation_history

    def _create_chat_completion(
        self, messages: List[Dict[str, str]], functions: List[Dict] = None, function_call: str = None
    ) -> Any:
        """Helper method to create chat completion with the new API"""
        try:
            # Convert functions to tools format
            tools = None
            tool_choice = "auto"
            if functions:
                tools = [{"type": "function", "function": func} for func in functions]
                if function_call:
                    tool_choice = {"type": "function", "function": {"name": function_call}}

            # Ensure all message contents are strings
            processed_messages = []
            for msg in messages:
                content = msg['content']
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, indent=2)
                processed_messages.append({"role": msg["role"], "content": content})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                tools=tools,
                tool_choice=tool_choice,
            )

            self._update_usage_stats(response)

            if response.choices[0].message.tool_calls:
                return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            self._conversation_history.append({"role": "system", "content": f"API Error: {error_msg}"})
            raise

    def evaluate_implementation(self, algorithm_code: str) -> Dict[str, Any]:
        """Evaluate an implementation using the evaluator."""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Set the implementation in the evaluator
                self.task.evaluator.set_code(self.task.evaluator.target_name, algorithm_code)

                # Get the objective value
                score, sim_dirs, results = self.task.evaluate()

                # Add successful result to conversation history
                self._conversation_history.append(
                    {"role": "system", "content": f"Evaluation completed successfully with score: {score}"}
                )

                return {
                    "success": True,
                    "score": score,
                    "code": algorithm_code,
                    "sim_dirs": sim_dirs,
                    "results": results
                }
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__

                # Add error to conversation history
                self._conversation_history.append(
                    {
                        "role": "system",
                        "content": f"Evaluation failed (Attempt {retry_count + 1}/{max_retries}): {error_type} - {error_msg}",
                    }
                )

                retry_count += 1
                if retry_count < max_retries:
                    # Try to fix the implementation
                    self._conversation_history.append({"role": "user", "content": f"{error_msg}\n\n{self._fix_prompt}"})

                    result = self._create_chat_completion(
                        messages=self._conversation_history,
                        functions=[FUNCTION_SCHEMAS["implement_algorithm"]],
                        function_call="implement_algorithm",
                    )

                    algorithm_code = result["code"]
                    continue

                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "code": algorithm_code,
                    "sim_dirs": sim_dirs,
                    "results": results
                }

    def implement_algorithm(self, debug: bool = False) -> Tuple[str, str]:
        """Ask LLM to implement an algorithm based on a description"""
        # Use the implementation prompt from the task prompt path
        prompt = self._implement_prompt
        print(f"Implement prompt: {prompt}")
        # Add the prompt to conversation history
        self._conversation_history.append({"role": "user", "content": prompt})

        # Get implementation using function calling
        try:
            result = self._create_chat_completion(
                messages=self._conversation_history,
                functions=[FUNCTION_SCHEMAS["implement_algorithm"]],
                function_call="implement_algorithm",
            )

            if debug:
                self.print_conversation_history()

            # Add the result to conversation history
            self._conversation_history.append({"role": "assistant", "content": json.dumps(result)})

            return result["code"], result.get("reasoning", "Initial implementation")
        except Exception as e:
            error_msg = str(e)
            if debug:
                print(f"Error in implement_algorithm: {error_msg}")
            raise

    def get_implementation(self) -> str:
        """Get the implementation from the architect."""
        implementation, _ = self.implement_algorithm()
        return implementation

    def print_conversation_history(self) -> None:
        """Print the conversation history in a readable format"""
        print("\n=== Conversation History ===")
        for i, message in enumerate(self._conversation_history, 1):
            role = message["role"].upper()
            content = message["content"]
            print(f"\n{i}. [{role}]")
            print(f"{'='*50}")
            print(content)
            print(f"{'='*50}")
        print("\n=== End of Conversation ===\n", flush=True)

    def _update_usage_stats(self, response) -> None:
        """Update token usage and cost statistics from an API response."""
        usage = response.usage

        # Extract basic token counts
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Update token counts
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        # Calculate and update costs using per-million token pricing
        prompt_cost = (prompt_tokens / 1000000) * self.price_info["input"]
        completion_cost = (completion_tokens / 1000000) * self.price_info["output"]
        self.total_cost += prompt_cost + completion_cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_cost": round(self.total_cost, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "pricing_info": self.price_info
        }

def create_and_test_algorithm(
    architect: OpenAIArchitect, debug: bool = False, scenarios: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create and test an algorithm using the architect."""
    try:
        if debug:
            print("\n=== Starting Algorithm Design Process ===")
            print(f"Task Type: {architect.task.evaluator.target_name}\n")
            print("\n=== Implementation ===")
            print("Prompt: Implement the algorithm")

        # Get implementation from architect
        implementation, reasoning = architect.implement_algorithm(debug=debug)
        if debug:
            print("\nGenerated implementation:")
            print(implementation)

        # Run evaluation
        results = architect.evaluate_implementation(implementation)

        # Add metadata to results
        results["reasoning"] = reasoning
        results["prompt"] = architect._implement_prompt
        results["usage_stats"] = architect.get_usage_stats()

        return results

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        if debug:
            print(f"\nError ({error_type}): {error_msg}")
            import traceback
            traceback.print_exc()

        return {
            "code": "",
            "reasoning": "",
            "success": False,
            "score": 0.0,
            "error": error_msg,
            "error_type": error_type,
            "usage_stats": architect.get_usage_stats()
        }
