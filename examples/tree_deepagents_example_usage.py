#!/usr/bin/env python3
"""
Example usage of the AgenticDeepAgents optimizer.

This script demonstrates how to run the AgenticDeepAgents optimizer using
deep agents to optimize code through iterative simulation and refinement.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the root directory to the path so we can import modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Architect.task import load_task_from_paths
from Architect.methods.deepagents_tree import DeepAgentsTreeOptimizer
from _common import get_results_base_dir, normalize_initial_program_path, auto_increment_run_dir, print_initial_program, resolve_problem_config


def run_deepagents_tree_optimization(results_base_dir=None, problem_name="cloudcast",
                                     model="o3", tree_rounds=10, max_review_iterations=40,
                                     early_stop_patience=10, capture_simulation_output=False,
                                     give_files=False, enable_continue_message=False): # Whether to add the continue message to the agent's messages
    """Run DeepAgentsTreeOptimizer optimization."""
    
    # Get the base directory (root of the project)
    base_dir = Path(__file__).parent.parent
    system_prompt_path = base_dir / "Architect" / "methods" / "deepagents_utils" / "best_score_system_prompt.txt"
    results_base = get_results_base_dir(base_dir) if results_base_dir is None else Path(results_base_dir)
    results_dir = results_base / f"deepagents_tree_{problem_name}_model_{model}_give_files_{give_files}_capture_simulation_output_{capture_simulation_output}_run0"
    results_dir = auto_increment_run_dir(results_dir)
    debug = True

    task_prompt_path, evaluator_path, initial_program_path = resolve_problem_config(base_dir, problem_name, give_files)

    # Convert to strings
    evaluator_path = str(evaluator_path)
    results_dir = str(results_dir)
    task_prompt_path = str(task_prompt_path)
    system_prompt_path = str(system_prompt_path)
    initial_program_path = normalize_initial_program_path(initial_program_path)

    # Verify files exist
    if not os.path.exists(evaluator_path):
        raise FileNotFoundError(f"Evaluator path not found: {evaluator_path}")
    if not os.path.exists(task_prompt_path):
        raise FileNotFoundError(f"User prompt file not found: {task_prompt_path}")
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")

    print("=" * 60)
    print("DeepAgentsTree Optimization")
    print("=" * 60)
    print(f"User Prompt: {task_prompt_path}")
    print(f"Evaluator Path: {evaluator_path}")
    print(f"Model: {model}")
    print(f"Results Directory: {results_dir}")
    print(f"User Prompt: {task_prompt_path}")
    print(f"System Prompt: {system_prompt_path}")
    print_initial_program(initial_program_path)
    print("=" * 60)
    print()

    # Load the task
    print("Loading task...")
    task = load_task_from_paths(task_prompt_path, evaluator_path)
    print(f"Loaded task: {task.name}")
    print()
    
    # Create the optimizer
    print("Initializing AgenticDeepAgents optimizer...")
    optimizer = DeepAgentsTreeOptimizer(
        task=task,
        model=model,
        results_dir=results_dir,
        debug=debug,
        tree_rounds=tree_rounds,
        initial_program_path=initial_program_path,
        task_prompt_path=task_prompt_path,
        system_prompt_path=system_prompt_path,
        max_review_iterations=max_review_iterations,
        early_stop_patience=early_stop_patience,
        capture_simulation_output=capture_simulation_output,
        enable_continue_message=enable_continue_message, # Whether to add the continue message to the agent's messages
    )
    print("Optimizer initialized successfully!")
    print()

    # optimizer._display_available_tools()
    # return None
    
    # Run optimization
    print("Starting optimization...")
    print("-" * 60)
    results = optimizer.optimize()
    print("-" * 60)
    print()
    
    # Print results
    print("Optimization completed!")
    print(f"Success: {results.get('success', 'Unknown')}")
    if 'score' in results:
        print(f"Score: {results['score']:.6f}")
    if 'code' in results:
        print("\nGenerated Code:")
        print(results['code'])
    if 'analysis' in results:
        print(f"\nAnalysis: {results['analysis']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgenticDeepAgents optimization")
    parser.add_argument(
        "--results_base_dir",
        type=str,
        default=None,
        help="Name of the results directory (e.g., 'deepagents_run2'). Will be created under results/ directory."
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default="cloudcast",
        help="Name of the problem (e.g., 'vidur', 'cloudcast', 'eplb', 'llm_sql')."
    )
    parser.add_argument(
        "--tree_rounds",
        type=int,
        default=10,
        help="Number of tree expansion rounds (default: 10)."
    )
    parser.add_argument(
        "--max_review_iterations",
        type=int,
        default=40,
        help="Max iterations per AgenticDeepAgents run (default: 40)."
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop AgenticDeepAgents if no improvement for this many iterations (default: 10)."
    )
    parser.add_argument(
        "--capture_simulation_output",
        action="store_true",
        help="Capture simulation output (default: False)."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs to perform (default: 1)."
    )
    parser.add_argument(
        "--give_files",
        action="store_true",
        help="Give files to the agent (default: False)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o3",
        help="Model to use (default: o3)."
    )
    parser.add_argument(
        "--disable_continue_message",
        action="store_true",
        help="Disable continue message (default: False)."
    )
    args = parser.parse_args()
    
    try:
        for i in range(args.num_runs):
            results = run_deepagents_tree_optimization(
                results_base_dir=args.results_base_dir, 
                problem_name=args.problem_name,
                model=args.model,
                tree_rounds=args.tree_rounds,
                max_review_iterations=args.max_review_iterations,
                early_stop_patience=args.early_stop_patience,
                capture_simulation_output=args.capture_simulation_output,
                give_files=args.give_files,
                enable_continue_message=not args.disable_continue_message, # Whether to add the continue message to the agent's messages
            )
    except Exception as e:
        print(f"Error running optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

