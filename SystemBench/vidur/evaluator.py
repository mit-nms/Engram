"""
ADRS-compatible evaluator for Vidur (LLM Inference Load Balancing).

Used by load_task_from_paths() and ADRSEvaluator.
Returns combined_score for optimization.
"""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Architect.types import Scenario
from SystemBench.vidur.env_evaluator import VidurEvaluator

TARGET_NAME = "CustomGlobalScheduler"

_DEFAULT_SCENARIO = Scenario(
    name="sarathi_qps7.5",
    config={"replica_scheduler": "sarathi", "qps": 7.5},
)

_evaluator = None


def _get_evaluator() -> VidurEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = VidurEvaluator()
    return _evaluator


def evaluate(program_path: str) -> dict:
    """Evaluate a program implementing CustomGlobalScheduler.

    Args:
        program_path: Path to a Python file defining CustomGlobalScheduler.

    Returns:
        dict with at least 'combined_score' (float) and 'runs_successfully' (float).
    """
    with open(program_path, "r") as f:
        code = f.read()

    evaluator = _get_evaluator()
    result = evaluator.run_simulation_with_algorithm_code(code, _DEFAULT_SCENARIO)

    score = result.get("score", float("-inf"))
    success = result.get("success", False)
    metrics = result.get("metrics", {})

    return {
        "combined_score": score if success else float("-inf"),
        "runs_successfully": 1.0 if success else 0.0,
        "success": success,
        "error": result.get("error", ""),
        **metrics,
    }
