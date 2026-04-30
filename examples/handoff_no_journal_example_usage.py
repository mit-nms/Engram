#!/usr/bin/env python3
"""
Example usage of the AgenticHandoffNoJournal optimizer (ablation: KB only, no research journal).

Sequential agent handoffs with knowledge base archive but no summaries passed between agents.
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the root directory to the path so we can import modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Architect.task import load_task_from_paths
from Architect.methods.agentic_handoff_no_journal import AgenticHandoffNoJournal
from _common import normalize_initial_program_path, auto_increment_run_dir, print_initial_program, get_results_base_dir, resolve_problem_config

RUN_IN_PARALLEL = True


def run_handoff_no_journal_optimization(
    results_dir_name: str = None,
    problem_name: str = "cloudcast",
    model: str = "o3",
    max_agents: int = 3,
    agent_timeout_minutes: float = 30.0,
    debug: bool = False,
    give_files: bool = False,
    early_stop_patience: int = 10,
    enable_continue_message: bool = False,
):
    """Run AgenticHandoffNoJournal optimization (KB only, no research journal)."""

    base_dir = Path(__file__).parent.parent
    prompts_dir = base_dir / "Architect" / "methods" / "deepagents_utils"
    system_prompt_path = prompts_dir / "handoff_system_prompt_no_journal.txt"

    results_base = get_results_base_dir(base_dir)
    handoff_suffix = "handoff_no_journal"
    results_dir = results_base / (results_dir_name or f"{handoff_suffix}_{problem_name}_model_{model}_give_files_{give_files}_run0")
    results_dir = auto_increment_run_dir(results_dir)
    debug = True

    task_prompt_path, evaluator_path, initial_program_path = resolve_problem_config(base_dir, problem_name, give_files)
    initial_program_path = normalize_initial_program_path(initial_program_path)

    print("=" * 60)
    print("AgenticHandoffNoJournal Optimization (no research journal, KB only)")
    print("=" * 60)
    print(f"User Prompt: {task_prompt_path}")
    print(f"Evaluator Path: {evaluator_path}")
    print(f"Model: {model}")
    print(f"Results Directory: {results_dir}")
    print(f"System Prompt: {system_prompt_path}")
    print(f"Max Agents: {max_agents}")
    print(f"Agent Timeout: {agent_timeout_minutes} minutes")
    print(f"Debug: {debug}")
    print_initial_program(initial_program_path)
    print("=" * 60)
    print()

    print("Loading task...")
    task = load_task_from_paths(task_prompt_path, evaluator_path)
    print(f"Loaded task: {task.name}")
    print()

    print("Initializing AgenticHandoffNoJournal optimizer...")
    optimizer = AgenticHandoffNoJournal(
        task=task,
        model=model,
        results_dir=results_dir,
        debug=debug,
        task_prompt_path=task_prompt_path,
        system_prompt_path=system_prompt_path,
        initial_program_path=initial_program_path,
        max_agents=max_agents,
        agent_timeout_minutes=agent_timeout_minutes,
        enable_continue_message=enable_continue_message,
        early_stop_patience=early_stop_patience,
        run_baselines=False,
    )
    print("Optimizer initialized!")
    print()

    print("Starting optimization...")
    print("-" * 60)
    results = optimizer.optimize()
    print("-" * 60)
    print()

    print("Optimization completed!")
    print(f"Best score: {results.get('best_solution', {}).get('score', 'N/A')}")
    print(f"Total simulations: {results.get('total_simulations', 0)}")
    print(f"Total agents: {results.get('total_agents', 0)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgenticHandoffNoJournal optimization (KB only, no journal)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory name (created under results/)")
    parser.add_argument("--problem_name", type=str, default="cloudcast", help="Problem to optimize")
    parser.add_argument("--max_agents", type=int, default=30, help="Maximum number of sequential agents to run")
    parser.add_argument("--agent_timeout", type=float, default=60.0, help="Wall-clock timeout per agent in minutes")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--give_files", action="store_true", help="Give files to the agents (optimal)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--model", type=str, default="o3", help="Model to use")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Stop if no improvement for this many iterations")
    parser.add_argument("--enable_continue_message", action="store_true", help="Enable continue message (default: False)")
    parser.add_argument("--no_parallel", action="store_true", help="Disable parallel runs (default: run num_runs in parallel)")
    args = parser.parse_args()

    try:
        if RUN_IN_PARALLEL and not args.no_parallel:
            with ProcessPoolExecutor(max_workers=args.num_runs) as executor:
                futures = {
                    executor.submit(
                        run_handoff_no_journal_optimization,
                        f"run{i}" if args.results_dir is None else f"{args.results_dir}_run{i}",
                        args.problem_name,
                        args.model,
                        args.max_agents,
                        args.agent_timeout,
                        args.debug,
                        args.give_files,
                        args.early_stop_patience,
                        args.enable_continue_message,
                    ): i
                    for i in range(args.num_runs)
                }
                for future in as_completed(futures):
                    run_idx = futures[future]
                    try:
                        future.result()
                        print(f"Run {run_idx} completed successfully.")
                    except Exception as e:
                        print(f"Run {run_idx} failed: {e}")
                        raise
        else:
            for i in range(args.num_runs):
                run_handoff_no_journal_optimization(
                    args.results_dir,
                    args.problem_name,
                    args.model,
                    args.max_agents,
                    args.agent_timeout,
                    args.debug,
                    args.give_files,
                    args.early_stop_patience,
                    args.enable_continue_message,
                )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
