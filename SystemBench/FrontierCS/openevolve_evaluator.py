"""
OpenEvolve evaluator bridge for Frontier-CS problems (algorithmic and research).

Wraps the existing FrontierCSEvaluator to conform to OpenEvolve's
evaluate(program_path) interface. Reads FCS_TRACK and FCS_PROBLEM_ID
from environment variables.

Usage:
    FCS_TRACK=algorithmic FCS_PROBLEM_ID=0 python -c "
    from SystemBench.FrontierCS.openevolve_evaluator import evaluate
    result = evaluate('path/to/solution.cpp')
    print(result)
    "

    FCS_TRACK=research FCS_PROBLEM_ID=flash_attn python -c "
    from SystemBench.FrontierCS.openevolve_evaluator import evaluate
    result = evaluate('path/to/solution.py')
    print(result)
    "
"""

import os
import sys

# Ensure Glia root is on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GLIA_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _GLIA_ROOT not in sys.path:
    sys.path.insert(0, _GLIA_ROOT)

# Ensure openevolve is importable
_OPENEVOLVE_DIR = os.path.join(_GLIA_ROOT, "Architect", "openevolve")
if _OPENEVOLVE_DIR not in sys.path:
    sys.path.insert(0, _OPENEVOLVE_DIR)

from openevolve.evaluation_result import EvaluationResult
from SystemBench.FrontierCS.frontier_cs_evaluator import FrontierCSEvaluator
from Architect.types import Scenario

# Module-level cache to avoid re-creating evaluator per call
_evaluator = None
_track = None
_problem_id = None
_target_name = None


def _get_evaluator():
    """Lazily create and cache the FrontierCSEvaluator."""
    global _evaluator, _track, _problem_id, _target_name

    track = os.environ.get("FCS_TRACK")
    problem_id = os.environ.get("FCS_PROBLEM_ID")
    if not track or not problem_id:
        raise RuntimeError(
            "Environment variables FCS_TRACK and FCS_PROBLEM_ID must be set. "
            "Example: FCS_TRACK=algorithmic FCS_PROBLEM_ID=0"
        )

    # Convert algorithmic problem IDs to int
    if track == "algorithmic" and problem_id.isdigit():
        problem_id = int(problem_id)

    # Reuse cached evaluator if same problem
    if _evaluator is not None and _track == track and _problem_id == problem_id:
        return _evaluator, _track, _problem_id, _target_name

    judge_url = os.environ.get("FCS_JUDGE_URL", "http://localhost:8081")
    timeout = os.environ.get("FCS_TIMEOUT")
    timeout = int(timeout) if timeout else None

    target_name = "Solution" if track == "research" else "solution"

    evaluator = FrontierCSEvaluator(
        track=track,
        problem_id=problem_id,
        target_name=target_name,
        backend="docker",
        timeout=timeout,
        judge_url=judge_url,
    )

    _evaluator = evaluator
    _track = track
    _problem_id = problem_id
    _target_name = target_name

    return evaluator, track, problem_id, target_name


def evaluate(program_path):
    """
    Evaluate a program file for a Frontier-CS problem.

    Args:
        program_path: Path to the program file (C++ for algorithmic, Python for research)

    Returns:
        EvaluationResult with metrics and artifacts
    """
    try:
        # Read code from file
        with open(program_path, "r") as f:
            code = f.read()

        if not code.strip():
            return EvaluationResult(
                metrics={"combined_score": 0.0, "score": 0.0},
                artifacts={"error": "Empty program file", "error_type": "empty_file"},
            )

        evaluator, track, problem_id, target_name = _get_evaluator()

        # Set the code on the evaluator and run simulation
        evaluator.set_code(target_name, code)
        design_config = evaluator.get_current_design_config()
        scenario = Scenario(
            name="default_evaluation",
            config={"track": track, "problem_id": problem_id},
        )
        result = evaluator.run_simulation(design_config, scenario)

        # Extract metrics and artifacts
        score = result.get("score", 0.0)
        success = result.get("success", False)
        metrics = result.get("metrics", {})

        return EvaluationResult(
            metrics={
                "combined_score": metrics.get("combined_score", score),
                "score": score,
                "runs_successfully": 1.0 if success else 0.0,
            },
            artifacts={
                "error": result.get("error", "") if not success else None,
                "error_type": result.get("error_type", "") if not success else None,
                "stdout": result.get("stdout", ""),
                "track": track,
                "problem_id": str(problem_id),
            },
        )

    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "score": 0.0, "runs_successfully": 0.0},
            artifacts={"error": str(e), "error_type": type(e).__name__},
        )


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: FCS_TRACK=<track> FCS_PROBLEM_ID=<id> python openevolve_evaluator.py <program_path>")
        sys.exit(1)

    result = evaluate(sys.argv[1])
    print("Metrics:", json.dumps(result.metrics, indent=2, default=str))
    print("Artifacts:", json.dumps(result.artifacts, indent=2, default=str))
