"""
Evaluator for Vidur Task
"""

import importlib.util
import numpy as np
import os
import sys

# Add necessary directories to Python path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
glia_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
openevolve_dir = os.path.join(glia_root, 'Architect', 'openevolve')
# Add Glia root for SystemBench imports
if glia_root not in sys.path:
    sys.path.insert(0, glia_root)
# Add openevolve directory for openevolve imports
if openevolve_dir not in sys.path:
    sys.path.insert(0, openevolve_dir)

from Architect.types import Scenario
from SystemBench.vidur.env_evaluator import VidurEvaluator
from openevolve.evaluation_result import EvaluationResult

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


def evaluate(program_path):
    """
    Evaluate the evolved program

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    try:
        with open(program_path, "r") as file:
            program_code = file.read()

        result = _get_evaluator().run_simulation_with_algorithm_code(program_code, _DEFAULT_SCENARIO)
        metrics = {
            'score': result.get('score', float('-inf')),
            'combined_score': result.get('score', float('-inf')),
        }
        artifacts = {
            'error': None if not result.get('error') else result['error'],
            'error_type': None if not result.get('error_type') else result['error_type'],
        }
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return EvaluationResult(metrics={'score': float("-inf")}, artifacts={'error': str(e), 'error_type': None})


def main():
    """
    Main function to test the evaluator
    """
    # Example program path - you'll need to replace this with an actual program file
    test_program_path = "initial_program.py"
    
    # Test the evaluator
    print("Testing the evaluator...")
    print(f"Program path: {test_program_path}")
    
    try:
        result = evaluate(test_program_path)
        print("Evaluation result:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
        print("Artifacts:")
        for key, value in result.artifacts.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    

if __name__ == "__main__":
    main()