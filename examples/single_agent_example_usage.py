#!/usr/bin/env python3
"""
Example usage of the SingleAgent optimizer (ablation: no handoff, no research journal, no knowledge base).
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
from Architect.methods.single_agent import SingleAgent
from _common import normalize_initial_program_path, auto_increment_run_dir, print_initial_program, get_results_base_dir, resolve_problem_config

RUN_IN_PARALLEL = True


def run_single_agent_optimization(
    results_dir_name: str = None,
    problem_name: str = "cloudcast",
    model: str = "o3",
    agent_timeout_minutes: float = 30.0,
    debug: bool = False,
    give_files: bool = False,
    early_stop_patience: int = 10,
    use_summarization_middleware: bool = False,
    enable_continue_message: bool = False,
):
    """Run SingleAgent optimization (one agent until timeout or early stop).

    Args:
        results_dir_name: Name for results directory (created under results/)
        problem_name: Problem to optimize (vidur, cloudcast, eplb, llm_sql, etc.)
        model: Model to use
        agent_timeout_minutes: Wall-clock timeout in minutes
        debug: Whether to print debug output
        give_files: Whether to give files to the agent (problem-specific)
        early_stop_patience: Stop if no improvement for this many iterations (when enable_continue_message=True)
        use_summarization_middleware: Enable summarization middleware for long conversations
        enable_continue_message: Enable continue message to keep agent running until timeout/early stop
    """

    base_dir = Path(__file__).parent.parent
    prompts_dir = base_dir / "Architect" / "methods" / "deepagents_utils"

    # Single-agent system prompt (no handoff, no summary, no journal/kb)
    system_prompt_path = prompts_dir / "system_prompt_single_agent.txt"


    results_base = get_results_base_dir(base_dir)
    results_dir = results_base / (results_dir_name or f"single_agent_{problem_name}_model_{model}_give_files_{give_files}_summarization_{use_summarization_middleware}_continue_message_{enable_continue_message}_run0")
    results_dir = auto_increment_run_dir(results_dir)
    debug = True
    # Problem-specific configuration
    task_prompt_path, evaluator_path, initial_program_path = resolve_problem_config(base_dir, problem_name, give_files)

    initial_program_path = normalize_initial_program_path(initial_program_path)

    for name, path in [
        ("User prompt", task_prompt_path),
        ("System prompt", system_prompt_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    print("=" * 60)
    print("SingleAgent Optimization (ablation: no handoff, no journal, no kb)")
    print("=" * 60)
    print(f"User Prompt: {task_prompt_path}")
    print(f"Evaluator Path: {evaluator_path}")
    print(f"Model: {model}")
    print(f"Results Directory: {results_dir}")
    print(f"System Prompt: {system_prompt_path}")
    print(f"Agent Timeout: {agent_timeout_minutes} minutes")
    print(f"Debug: {debug}")
    print(f"Enable continue message: {enable_continue_message}")
    print(f"Use summarization middleware: {use_summarization_middleware}")
    print(f"Early stop patience: {early_stop_patience}")
    print_initial_program(initial_program_path)
    print("=" * 60)
    print()

    print("Loading task...")
    task = load_task_from_paths(task_prompt_path, evaluator_path)
    print(f"Loaded task: {task.name}")
    print()

    print("Initializing SingleAgent optimizer...")
    optimizer = SingleAgent(
        task=task,
        model=model,
        results_dir=results_dir,
        debug=debug,
        task_prompt_path=task_prompt_path,
        system_prompt_path=system_prompt_path,
        initial_program_path=initial_program_path,
        agent_timeout_minutes=agent_timeout_minutes,
        use_summarization_middleware=use_summarization_middleware,
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
    print(f"Convergence reason: {results.get('convergence_reason', 'N/A')}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SingleAgent optimization (ablation)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory name (created under results/)")
    parser.add_argument("--problem_name", type=str, default="cloudcast", help="Problem to optimize")
    parser.add_argument("--agent_timeout", type=float, default=60.0, help="Wall-clock timeout in minutes")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--give_files", action="store_true", help="Give files to the agent (problem-specific)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--model", type=str, default="o3", help="Model to use")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop if no improvement for this many iterations when --enable_continue_message is set (default: 10).",
    )
    parser.add_argument(
        "--use_summarization_middleware",
        action="store_true",
        help="Enable summarization middleware to compact long message history.",
    )
    parser.add_argument(
        "--enable_continue_message",
        action="store_true",
        help="Enable continue message so the agent runs until timeout or early stop (recommended for ablation).",
    )
    args = parser.parse_args()

    try:
        if RUN_IN_PARALLEL:
            with ProcessPoolExecutor(max_workers=args.num_runs) as executor:
                base_name = args.results_dir or f"single_agent_{args.problem_name}_model_{args.model}_give_files_{args.give_files}_summarization_{args.use_summarization_middleware}_continue_message_{args.enable_continue_message}"
                futures = {
                    executor.submit(run_single_agent_optimization, f"{base_name}_run{i}", args.problem_name, args.model, args.agent_timeout, args.debug, args.give_files, args.early_stop_patience, args.use_summarization_middleware, args.enable_continue_message): i
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
                run_single_agent_optimization(
                    args.results_dir,
                    args.problem_name,
                    args.model,
                    args.agent_timeout,
                    args.debug,
                    args.give_files,
                    args.early_stop_patience,
                    args.use_summarization_middleware,
                    args.enable_continue_message,
                )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
