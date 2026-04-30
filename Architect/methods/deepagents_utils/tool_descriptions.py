import os
from pathlib import Path
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AnyMessage

# Get the directory where this file is located
_DIR = Path(__file__).parent


def _load_prompt_file(filename: str) -> str:
    """Load a prompt from a text file in the same directory."""
    filepath = _DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    return filepath.read_text(encoding='utf-8').strip()


class CustomSummarizationMiddleware(SummarizationMiddleware):
    name = "CustomSummarizationMiddleware"

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Return messages unchanged without trimming; the default implementation will return no summary if the messages are too long."""
        return messages


RUN_SIMULATION_TOOL_DESCRIPTION = """This tool runs a simulation with your code implementation and write the results to a CSV file and save it to the agent's filesystem.

IMPORTANT: You should use this tool to test and validate your code changes. After modifying code, always run a simulation to verify the changes work correctly and measure performance improvements.

Usage:
- The file_path parameter must be an absolute path starting with `/` (e.g., `/my_implementation.py`)
- The file must exist in the agent's filesystem (use write_file or edit_file to create/modify files first)
- The file must contain valid Python code that implements the required functionality
- The code file has to ONLY contain the implementation asked by the user, the imports will be handled by the simulator.
- If the simulation succeeds, a CSV file with detailed metrics is automatically saved to the agent filesystem.
- The result files will be too large, so use an offset to read them using the read_file tool.

Workflow: Make code changes → Use run_simulation to test → Analyze results → Iterate."""

CONTINUE_MESSAGE = _load_prompt_file("continue_message.txt")
SUMMARY_PROMPT = _load_prompt_file("summay_prompt.txt")

HANDOFF_RUN_SIMULATION_DESCRIPTION = """Run a simulation with your code implementation.

This tool:
1. Archives your code to experiments/exp_XXX/snapshot.py
2. Runs the simulation and records the score
3. Saves results to experiments/exp_XXX/results/
4. Returns the score and experiment ID

IMPORTANT:
- file_path must be an absolute path starting with `/` (e.g., `/my_implementation.py`)
- The file must exist in your filesystem
- After running, update the research journal with your findings

Workflow: Write code -> run_simulation -> analyze results -> update journal -> iterate"""

HANDOFF_SUMMARY_REQUEST_PROMPT = """Provide your summary report now. Use EXACTLY this format:

## Agent {{AGENT_NUMBER}} Summary

### Best Result
- Score: [your best score]
- Approach: [brief description of what achieved this score]
- Experiment: [which exp_XXX folder contains this code]

### What I Tried
1. [Approach 1]: Score [X.XXX] - [outcome/insight]
2. [Approach 2]: Score [X.XXX] - [outcome/insight]
...

### Key Insights
- [Important finding 1]
- [Important finding 2]

### Recommended Next Steps
- [Specific suggestion for next agent]

### Approaches That Didn't Work (and Why)
- [Approach]: [why it failed for me — future agents may revisit with different implementation]

Do NOT use any tools. Do NOT make any more tool calls. Just respond with a message.
"""