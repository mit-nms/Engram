#!/usr/bin/env python3
"""
Aggregate DeepAgents Tree results into a single JSON file.

This script collects all iterations from each round's AgenticDeepAgents run
and concatenates them in order, similar to aggregate_openevolve_logs.py.

Usage:
    python aggregate_tree_logs.py <tree_results_dir> [output_file]

Example:
    python aggregate_tree_logs.py results_tree/cloudcast
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def find_round_dirs(logs_dir: Path) -> List[Tuple[int, Path]]:
    """Find all round_N directories and return sorted by round number."""
    round_dirs = []
    for entry in logs_dir.iterdir():
        if entry.is_dir():
            match = re.match(r"round_(\d+)", entry.name)
            if match:
                round_num = int(match.group(1))
                round_dirs.append((round_num, entry))
    return sorted(round_dirs, key=lambda x: x[0])


def find_best_json_in_round(round_dir: Path) -> Optional[Path]:
    """
    Find the AgenticDeepAgents JSON file with the most iterations in a round directory.
    
    The structure is: round_N/<task_name>/logs/*.json
    We pick the JSON with the highest iteration count (e.g., 8iterations > 5iterations).
    """
    # Find the task subdirectory (e.g., 'cloudcast')
    task_dirs = [d for d in round_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for task_dir in task_dirs:
        logs_dir = task_dir / "logs"
        if not logs_dir.exists():
            continue
        
        # Find JSON files matching the pattern *agentic_deepagents_*iterations.json
        # Exclude *_usage_stats.json files
        json_files = []
        for f in logs_dir.iterdir():
            if (f.suffix == ".json" 
                and "agentic_deepagents" in f.name 
                and "iterations" in f.name
                and "usage_stats" not in f.name):  # Exclude usage_stats files
                # Extract iteration count from filename
                match = re.search(r"(\d+)iterations", f.name)
                if match:
                    iter_count = int(match.group(1))
                    json_files.append((iter_count, f))
        
        if json_files:
            # Return the one with the most iterations
            json_files.sort(key=lambda x: x[0], reverse=True)
            return json_files[0][1]
    
    return None


def read_json_file(json_path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file and return its contents."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}", file=sys.stderr)
        return None


def aggregate_tree_results(tree_dir: Path) -> Dict[str, Any]:
    """
    Aggregate all AgenticDeepAgents iterations from a tree results directory.
    
    Returns a Vidur-compatible format with all_iterations containing every
    simulation from every round.
    """
    logs_dir = tree_dir / "logs"
    if not logs_dir.exists():
        print(f"Error: logs directory not found at {logs_dir}", file=sys.stderr)
        return {}
    
    # Also try to load the main tree JSON for metadata
    tree_json_files = list(logs_dir.glob("*-deepagents_tree_*rounds.json"))
    main_tree_data = None
    if tree_json_files:
        # Get the one with the most rounds
        tree_json_files.sort(key=lambda x: int(re.search(r"(\d+)rounds", x.name).group(1)) if re.search(r"(\d+)rounds", x.name) else 0, reverse=True)
        main_tree_data = read_json_file(tree_json_files[0])
    
    # Find all round directories
    round_dirs = find_round_dirs(logs_dir)
    print(f"Found {len(round_dirs)} round directories")
    
    all_iterations = []
    iteration_counter = 0
    best_score = float('-inf')
    best_solution = None
    
    # Process each round in order
    for round_num, round_dir in round_dirs:
        json_path = find_best_json_in_round(round_dir)
        if not json_path:
            print(f"Warning: No AgenticDeepAgents JSON found in {round_dir}", file=sys.stderr)
            continue
        
        print(f"  Round {round_num}: {json_path.name}")
        
        round_data = read_json_file(json_path)
        if not round_data:
            continue
        
        round_iterations = round_data.get("all_iterations", [])
        print(f"    -> {len(round_iterations)} iterations")
        
        for iter_data in round_iterations:
            # Renumber iterations sequentially across all rounds
            iteration_counter += 1
            
            # Map to Vidur-compatible format
            mapped_iter = {
                "iteration": iteration_counter,
                "code": iter_data.get("code", ""),
                "score": iter_data.get("score", float("-inf")),
                "success": iter_data.get("success", False),
                "sim_dirs": iter_data.get("sim_dirs", []),
                "error": iter_data.get("error", ""),
                # Tree-specific metadata
                "tree_metadata": {
                    "round": round_num,
                    "simulation_number": iter_data.get("simulation_number"),
                    "file_path": iter_data.get("file_path"),
                    "source_json": str(json_path),
                }
            }
            
            all_iterations.append(mapped_iter)
            
            # Track best solution
            score = iter_data.get("score", float("-inf"))
            if score != float("-inf") and score > best_score:
                best_score = score
                best_solution = {
                    "code": iter_data.get("code", ""),
                    "score": score,
                    "round": round_num,
                    "iteration": iteration_counter,
                }
    
    # If no best solution found, try to get from main tree data
    if not best_solution and main_tree_data:
        main_best = main_tree_data.get("best_solution", {})
        if main_best:
            best_solution = {
                "code": main_best.get("code", ""),
                "score": main_best.get("score", 0.0),
            }
            best_score = main_best.get("score", 0.0)
    
    # Ensure we have a best solution
    if not best_solution:
        best_solution = {"code": "", "score": 0.0}
    
    # Build the result
    result = {
        "best_solution": best_solution,
        "all_iterations": all_iterations,
        "final_code": best_solution.get("code", ""),
        "total_iterations": len(all_iterations),
        "total_rounds": len(round_dirs),
        "convergence_reason": "completed",
        "baselines": main_tree_data.get("baselines", {}) if main_tree_data else {},
        # Include tree-specific info
        "tree_info": {
            "nodes": main_tree_data.get("nodes", {}) if main_tree_data else {},
            "rounds": main_tree_data.get("rounds", []) if main_tree_data else [],
            "tree_stats": main_tree_data.get("tree_stats", {}) if main_tree_data else {},
            "config": main_tree_data.get("config", {}) if main_tree_data else {},
        }
    }
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_tree_logs.py <tree_results_dir> [output_file]")
        print("\nExample:")
        print("  python aggregate_tree_logs.py results_tree/cloudcast")
        sys.exit(1)
    
    tree_dir = Path(sys.argv[1])
    if not tree_dir.exists():
        print(f"Error: Directory {tree_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = tree_dir / "aggregated_results.json"
    
    print(f"Scanning tree results directory: {tree_dir}")
    result = aggregate_tree_results(tree_dir)
    
    if not result.get("all_iterations"):
        print("Error: No iterations found", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nWriting results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDone!")
    print(f"  Best score: {result['best_solution']['score']}")
    print(f"  Total iterations: {result['total_iterations']}")
    print(f"  Total rounds: {result['total_rounds']}")


if __name__ == "__main__":
    main()

