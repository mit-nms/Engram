#!/usr/bin/env python3
"""
Aggregate Linear Agent results into a single JSON file.

For each agent_N directory, this script picks the JSON log with the highest
iteration count from <agent_N>/<task>/logs/, then concatenates all
all_iterations in agent order.

Usage:
    python aggregate_linear_logs.py <linear_run_dir> [output_file]

Example:
    python aggregate_linear_logs.py \
      /data2/projects/pantea-work/Glia/results/cloudcast_linear_agent_results/linear_agent_cloudcast_model_gpt-5.2-xhigh_num_agents_20_give_files_False_run0
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Linear-agent logs are already in "score" space where higher is better.
def _is_better_score(new_score: float, current_best: float) -> bool:
    return new_score > current_best


def find_agent_dirs(linear_run_dir: Path) -> List[Tuple[int, Path]]:
    """Find all agent_N subdirectories and return them sorted by N."""
    out: List[Tuple[int, Path]] = []
    for entry in linear_run_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.match(r"agent_(\d+)$", entry.name)
        if not m:
            continue
        out.append((int(m.group(1)), entry))
    return sorted(out, key=lambda x: x[0])


def find_best_json_in_agent(agent_dir: Path) -> Optional[Path]:
    """
    Find highest-iteration JSON for one agent.

    Expected structure: agent_N/<task>/logs/*single_agent*iterations.json
    """
    candidates: List[Tuple[int, Path]] = []
    for task_dir in agent_dir.iterdir():
        if not task_dir.is_dir():
            continue
        logs_dir = task_dir / "logs"
        if not logs_dir.exists():
            continue
        for f in logs_dir.iterdir():
            if (
                f.suffix == ".json"
                and "iterations" in f.name
                and "usage_stats" not in f.name
                and ("single_agent" in f.name.lower() or "linear_agent" in f.name.lower())
            ):
                m = re.search(r"(\d+)iterations", f.name)
                if m:
                    candidates.append((int(m.group(1)), f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}", file=sys.stderr)
        return None


def aggregate_linear_results(linear_run_dir: Path) -> Dict[str, Any]:
    """Aggregate all final per-agent logs into one run-level JSON."""
    agent_dirs = find_agent_dirs(linear_run_dir)
    print(f"Found {len(agent_dirs)} agent directories")

    all_iterations: List[Dict[str, Any]] = []
    iteration_counter = 0
    best_solution: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    baselines: Dict[str, Any] = {}
    problem_name = ""

    # Preload optional summary metadata when available.
    summary_path = linear_run_dir / "linear_agent_summary.json"
    summary_data = read_json_file(summary_path) if summary_path.exists() else None
    if summary_data and isinstance(summary_data, dict):
        problem_name = str(summary_data.get("problem_name", "") or "")

    for agent_idx, agent_dir in agent_dirs:
        json_path = find_best_json_in_agent(agent_dir)
        if not json_path:
            print(f"Warning: No final iteration JSON found in {agent_dir}", file=sys.stderr)
            continue

        print(f"  Agent {agent_idx}: {json_path.name}")
        run_data = read_json_file(json_path)
        if not run_data:
            continue

        if not problem_name:
            problem_name = str(run_data.get("problem_name", "") or "")
        if not baselines and isinstance(run_data.get("baselines"), dict):
            baselines = run_data.get("baselines", {})

        # IMPORTANT: keep per-agent file order exactly as recorded.
        # Global order is: all agent_01 iterations, then all agent_02 iterations, ...
        iterations = [it for it in run_data.get("all_iterations", []) if isinstance(it, dict)]
        print(f"    -> {len(iterations)} iterations")

        for item in iterations:
            iteration_counter += 1
            source_sim_num = item.get("simulation_number")
            mapped_item = dict(item)
            # Keep raw item content, but enforce linear-stage label and global index.
            mapped_item["agent_number"] = agent_idx
            mapped_item["iteration"] = iteration_counter
            mapped_item["simulation_number"] = iteration_counter
            mapped_item["linear_metadata"] = {
                "agent_index": agent_idx,
                "source_json": str(json_path),
                "source_simulation_number": source_sim_num,
                "source_experiment_id": item.get("experiment_id"),
            }
            all_iterations.append(mapped_item)

            score = item.get("score")
            if score is None or score == float("inf") or score == float("-inf"):
                continue
            score = float(score)
            if best_score is None:
                best_score = score
                best_solution = {
                    "code": item.get("code", ""),
                    "score": score,
                    "agent_number": agent_idx,
                    "iteration": iteration_counter,
                }
            elif _is_better_score(score, best_score):
                best_score = score
                best_solution = {
                    "code": item.get("code", ""),
                    "score": score,
                    "agent_number": agent_idx,
                    "iteration": iteration_counter,
                }

    if best_solution is None:
        best_solution = {"code": "", "score": 0.0}

    convergence_reason = "completed"
    if summary_data and isinstance(summary_data, dict):
        convergence_reason = str(summary_data.get("final_convergence_reason", convergence_reason))

    result = {
        "best_solution": best_solution,
        "all_iterations": all_iterations,
        "final_code": best_solution.get("code", ""),
        "total_iterations": len(all_iterations),
        "total_agents": len(agent_dirs),
        "convergence_reason": convergence_reason,
        "baselines": baselines,
        "linear_info": {
            "problem_name": problem_name,
            "run_dir": str(linear_run_dir),
            "summary_file": str(summary_path) if summary_path.exists() else None,
        },
    }

    # Keep important fields from linear_agent_summary.json so this output can
    # directly replace/read as that file in downstream tooling.
    if summary_data and isinstance(summary_data, dict):
        for key in [
            "problem_name",
            "model",
            "num_agents_requested",
            "num_agents_completed",
            "results_dir",
            "agent_results",
            "total_simulations_across_agents",
            "final_convergence_reason",
        ]:
            if key in summary_data and key not in result:
                result[key] = summary_data[key]
        result["final_best_solution"] = summary_data.get("final_best_solution", best_solution)

    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python aggregate_linear_logs.py <linear_run_dir> [output_file]")
        sys.exit(1)

    linear_run_dir = Path(sys.argv[1])
    if not linear_run_dir.exists():
        print(f"Error: Directory does not exist: {linear_run_dir}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = linear_run_dir / "aggregated_results.json"

    print(f"Scanning linear run directory: {linear_run_dir}")
    result = aggregate_linear_results(linear_run_dir)

    if not result.get("all_iterations"):
        print("Error: No iterations found", file=sys.stderr)
        sys.exit(1)

    print(f"\nWriting results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\nDone!")
    print(f"  Best score: {result['best_solution']['score']}")
    print(f"  Total iterations: {result['total_iterations']}")
    print(f"  Total agents: {result['total_agents']}")


if __name__ == "__main__":
    main()
