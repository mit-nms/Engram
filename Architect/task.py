"""
Task specification for Glia optimization.

A Task is the minimal interface for specifying an optimization problem:
- task_prompt: Text description of the task
- evaluator: An ADRSEvaluator that can evaluate candidate solutions

Usage:
    task = load_task_from_paths(
        task_prompt_path="/path/to/task_prompt.txt",
        evaluator_path="/path/to/evaluator.py",  # or directory containing evaluator.py
        task_name="my_task",  # optional, defaults to evaluator directory name
    )
"""

import os
from typing import Optional, Tuple, List, Dict, Any


class Task:
    """
    Minimal task specification for optimization.
    - task_prompt: natural-language description of the problem
    - evaluator: evaluation wrapper (ADRSEvaluator or compatible)
    """

    def __init__(
        self,
        name: str,
        task_prompt: str,
        evaluator: Any,
    ):
        self.name = name
        self.task_prompt = task_prompt
        self.evaluator = evaluator

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation (for logging/resuming)."""
        return {
            "name": self.name,
            "task_prompt": self.task_prompt[:500] + "..." if len(self.task_prompt) > 500 else self.task_prompt,
        }

    def evaluate(self) -> Tuple[float, List[str], Dict[str, Any]]:
        """Evaluate the code currently loaded in the evaluator.

        Returns:
            (score, sim_dirs, results) — score is the objective value.
        """
        from Architect.types import Scenario, ProblemSpec

        design_config = self.evaluator.get_current_design_config()
        default_scenario = self.evaluator.get_default_scenario()

        def objective_fn(results: Dict[str, Any]) -> float:
            scores = [
                r["score"]
                for r in results.values()
                if r.get("success") and r.get("score") is not None
            ]
            return sum(scores) / len(scores) if scores else float("-inf")

        spec = ProblemSpec(
            design_config=design_config,
            scenarios=[default_scenario],
            objective_fn=objective_fn,
        )
        return self.evaluator.get_objective_value(spec)


def load_task_from_paths(
    task_prompt_path: str,
    evaluator_path: str,
    task_name: Optional[str] = None,
) -> Task:
    """Load a Task from a task prompt file and evaluator file/directory.

    Args:
        task_prompt_path: Path to the task prompt text file.
        evaluator_path: Path to the evaluator .py file, or the directory that
            contains evaluator.py / evaluate.py.
        task_name: Human-readable task name (defaults to evaluator directory basename).

    Returns:
        A Task instance ready for optimization.
    """
    import importlib.util
    import sys
    from SystemBench.ADRS.adrs_evaluator import ADRSEvaluator

    # Read task prompt
    task_prompt_path = os.path.abspath(task_prompt_path)
    with open(task_prompt_path, "r") as f:
        task_prompt = f.read()

    # Resolve evaluator directory and optional explicit filename
    evaluator_path = os.path.abspath(evaluator_path)
    if os.path.isfile(evaluator_path):
        evaluator_dir = os.path.dirname(evaluator_path)
        evaluator_file = os.path.basename(evaluator_path)
    else:
        evaluator_dir = evaluator_path
        evaluator_file = None

    # Derive task name from evaluator directory if not given
    if task_name is None:
        task_name = os.path.basename(evaluator_dir)

    # If the evaluator module defines (or imports) a native Evaluator subclass,
    # instantiate it directly — it manages its own setup (e.g. VidurEvaluator).
    # Otherwise fall back to the generic ADRSEvaluator wrapper.
    import inspect
    from SystemBench.evaluator import Evaluator as _BaseEvaluator

    eval_file = os.path.join(evaluator_dir, evaluator_file or "evaluator.py")
    if not os.path.isfile(eval_file):
        eval_file = os.path.join(evaluator_dir, "evaluate.py")

    evaluator_class = None
    if os.path.isfile(eval_file):
        module_name = os.path.splitext(os.path.basename(eval_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, eval_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, mod)
        if evaluator_dir not in sys.path:
            sys.path.insert(0, evaluator_dir)
        spec.loader.exec_module(mod)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, _BaseEvaluator) and obj is not _BaseEvaluator:
                evaluator_class = obj
                break

    if evaluator_class is not None:
        evaluator = evaluator_class()
    else:
        evaluator = ADRSEvaluator(
            adrs_env_path=evaluator_dir,
            evaluator_file=evaluator_file,
            name=task_name,
        )

    return Task(
        name=task_name,
        task_prompt=task_prompt,
        evaluator=evaluator,
    )
