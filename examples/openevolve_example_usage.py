#!/usr/bin/env python3
"""
Example usage of the OpenEvolve Optimizer.

This script demonstrates how to run the OpenEvolve optimizer for evolutionary
code optimization. OpenEvolve uses LLM-based evolution to improve code iteratively.
"""

import os
import signal
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add the root directory to the path so we can import modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env file if it exists
_env_file = os.path.join(root_dir, ".env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip().strip("'\""))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from _common import get_results_base_dir, normalize_initial_program_path, auto_increment_run_dir, print_initial_program, resolve_problem_config


def resolve_openevolve_paths(base_dir: Path, problem_name: str):
    """Resolve OpenEvolve-specific paths for a given problem.

    Common fields (task_prompt_path, evaluator_path, initial_program_path) come from
    resolve_problem_config. OpenEvolve-specific files (config.yaml, evaluator.py) are
    derived from evaluator_path, with a special override for fcs_alg_* problems whose
    configs live in a dedicated openevolve_configs/ directory.
    """
    task_prompt_path, evaluator_path, initial_program_path = resolve_problem_config(base_dir, problem_name)

    if problem_name.startswith("fcs_alg_"):
        problem_id = problem_name[len("fcs_alg_"):]
        initial_program_path = config_dir / "initial_program.cpp"
        openevolve_evaluator_path = evaluator_path / "openevolve_evaluator.py"
    else:
        openevolve_evaluator_path = evaluator_path / "evaluator.py"

    return task_prompt_path, evaluator_path, initial_program_path, openevolve_evaluator_path


def run_openevolve_optimization(problem_name: str, run_id: int, model: str = "o3", config_path: str = None):
    """Run OpenEvolve optimization for a given problem."""
    base_dir = Path(__file__).parent.parent
    results_base = get_results_base_dir(base_dir)
    task_prompt_path, evaluator_path, initial_program_path, openevolve_evaluator_path = \
        resolve_openevolve_paths(base_dir, problem_name)

    if problem_name == "vidur":
        results_dir = results_base / "openevolve_results" / f"vidur_openevolve_results_{run_id}"
    elif problem_name.startswith("fcs_alg_"):
        results_dir = results_base / f"{problem_name}_openevolve_results_{run_id}"
    else:
        results_dir = results_base / f"{problem_name}_openevolve_results" / f"{problem_name}_openevolve_results_{run_id}"

    if results_dir.exists():
        print(f"Results directory {results_dir} already exists. Skipping run {run_id}.")
        return run_id, 0

    initial_program_path = normalize_initial_program_path(initial_program_path)

    cmd = [
        "python",
        "-m",
        "Architect.main",
        "--method", "openevolve",
        "--model", model,
        "--task_prompt_path", str(task_prompt_path),
        "--evaluator_path", str(evaluator_path),
        "--openevolve_config_path", str(config_path),
        "--initial_program_path", str(initial_program_path) if isinstance(initial_program_path, str) else (initial_program_path[0] if initial_program_path else ""),
        "--results_dir", str(results_dir),
        "--debug",
    ]

    print(f"[Run {run_id}] Running OpenEvolve optimization:")
    print(f"[Run {run_id}] " + " ".join(cmd))
    print(f"[Run {run_id}]   Task Prompt: {task_prompt_path}")
    print(f"[Run {run_id}]   Evaluator Path: {evaluator_path}")
    print(f"[Run {run_id}]   Config Path: {config_path}")
    print_initial_program(initial_program_path)
    print(f"[Run {run_id}]   Results Directory: {results_dir}")
    print()

    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return_code = proc.wait()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return run_id, return_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenEvolve optimization")
    parser.add_argument("--problem_name", type=str, default="cloudcast",
                        help="Problem to optimize (vidur, cloudcast, eplb, fcs_alg_<id>, or any ADRS problem name)")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of parallel runs to perform")
    parser.add_argument("--model", type=str, default="o3",
                        help="Model to use")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to OpenEvolve config YAML (required)")
    args = parser.parse_args()

    problem_name = args.problem_name
    num_runs = args.num_runs
    model = args.model
    config_path = args.config_path

    print(f"Starting {num_runs} parallel OpenEvolve optimization runs for '{problem_name}'...")
    print()

    with ProcessPoolExecutor(max_workers=num_runs) as executor:
        futures = {
            executor.submit(run_openevolve_optimization, problem_name, run_id, model, config_path): run_id
            for run_id in range(num_runs)
        }

        results = []
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result_run_id, return_code = future.result()
                results.append((result_run_id, return_code))
                if return_code == 0:
                    print(f"[Run {result_run_id}] Completed successfully")
                else:
                    print(f"[Run {result_run_id}] Completed with exit code {return_code}")
            except Exception as e:
                print(f"[Run {run_id}] Failed with exception: {e}")
                results.append((run_id, -1))

    print()
    print(f"All {num_runs} runs completed.")
    successful = sum(1 for _, code in results if code == 0)
    print(f"Successful runs: {successful}/{num_runs}")
