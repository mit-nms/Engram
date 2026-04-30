"""Utility for building FrontierCS research task prompts using gen_env."""
import sys
from pathlib import Path

_FCS_REPO = Path(__file__).parent / "frontier_cs_repo"
_SCRIPTS = _FCS_REPO / "research" / "scripts"
_SRC = _FCS_REPO / "src"


def _ensure_imports():
    for p in [str(_SRC), str(_SCRIPTS)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def build_research_task_prompt(problem_id: str) -> str:
    """
    Build the combined task prompt for a research problem.
    Mirrors exactly what generate_solutions.py sends to the LLM:
      system_prompt = get_system_prompt_for_problem(name, path)
      task_prompt = f"Problem:\\n\\n{readme}\\n\\nGenerate solution code:"
      combined = f"{system_prompt}\\n\\n{task_prompt}"
    """
    _ensure_imports()
    from gen_env import get_system_prompt_for_problem
    from gen_io import read_readme

    problem_path = _FCS_REPO / "research" / "problems" / problem_id
    if not problem_path.is_dir():
        raise FileNotFoundError(f"Research problem not found: {problem_path}")

    # problem_name is the slash-separated ID (e.g. "symbolic_regression/mccormick")
    system_prompt = get_system_prompt_for_problem(problem_id, problem_path)
    readme = read_readme(problem_path)
    task_prompt = f"Problem:\n\n{readme}\n\nGenerate solution code:"
    return f"{system_prompt}\n\n{task_prompt}"
