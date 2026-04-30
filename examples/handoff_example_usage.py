#!/usr/bin/env python3
"""
Example usage of the AgenticHandoff optimizer with sequential agent handoffs.
"""

import os
import sys
import argparse
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

from Architect.task import load_task_from_paths
from Architect.methods.agentic_handoff import AgenticHandoff
from Architect.methods.agentic_handoff_no_kb import AgenticHandoffNoKB
from _common import get_results_base_dir, normalize_initial_program_path, auto_increment_run_dir, print_initial_program, resolve_problem_config


def run_handoff_optimization(
    results_dir_name: str = None,
    problem_name: str = "cloudcast",
    model: str = "o3",
    max_agents: int = 3,
    agent_timeout_minutes: float = 30.0,
    debug: bool = False,
    give_files: bool = False,
    early_stop_patience: int = 10,
    enable_continue_message: bool = False,
    remove_kb: bool = False,
):
    """Run AgenticHandoff optimization with sequential agent handoffs.

    Args:
        results_dir_name: Name for results directory (created under results/)
        problem_name: Problem to optimize (vidur, cloudcast, eplb, llm_sql)
        max_agents: Maximum number of sequential agents to run
        agent_timeout_minutes: Wall-clock timeout per agent in minutes
        debug: Whether to print debug output
        give_files: Whether to give files to the agents
        early_stop_patience: Stop AgenticHandoff if no improvement for this many iterations
        enable_continue_message: Enable continue message
        remove_kb: If True, use AgenticHandoffNoKB (no knowledge base; experiments wiped on handoff)
    """

    # Get the base directory (root of the project)
    base_dir = Path(__file__).parent.parent
    prompts_dir = base_dir / "Architect" / "methods" / "deepagents_utils"

    # Prompt paths (no-KB ablation uses different system prompt)
    system_prompt_path = prompts_dir / "handoff_system_prompt_no_kb.txt" if remove_kb else prompts_dir / "handoff_system_prompt.txt"
    if problem_name.startswith("fcs_alg_"):
        system_prompt_path = prompts_dir / "handoff_system_prompt_algorithms.txt"
    elif problem_name.startswith("fcs_res_"):
        system_prompt_path = prompts_dir / "handoff_system_prompt_research.txt"

    results_base = get_results_base_dir(base_dir)
    handoff_suffix = "no_kb" if remove_kb else "handoff"
    results_dir = results_base / (results_dir_name or f"{handoff_suffix}_{problem_name}_model_{model}_give_files_{give_files}_run0")

    results_dir = auto_increment_run_dir(results_dir)
    debug = True

    # Problem-specific configuration.
    task_prompt_path, evaluator_path, initial_program_path = resolve_problem_config(base_dir, problem_name, give_files)

    # Convert to strings
    evaluator_path = str(evaluator_path)
    results_dir = str(results_dir)
    task_prompt_path = str(task_prompt_path)
    system_prompt_path = str(system_prompt_path)

    initial_program_path = normalize_initial_program_path(initial_program_path)

    # Verify files exist
    for name, path in [("Task prompt", task_prompt_path), ("User prompt", task_prompt_path),
                       ("System prompt", system_prompt_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    print("=" * 60)
    print("AgenticHandoff Optimization" + (" (no-KB ablation)" if remove_kb else ""))
    print("=" * 60)
    print(f"Task Prompt: {task_prompt_path}")
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

    # Load the task
    print("Loading task...")
    task = load_task_from_paths(task_prompt_path, evaluator_path)
    print(f"Loaded task: {task.name}")
    print()

    # Create the optimizer
    HandoffClass = AgenticHandoffNoKB if remove_kb else AgenticHandoff
    print(f"Initializing {HandoffClass.__name__} optimizer...")
    optimizer = HandoffClass(
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
        run_baselines=not problem_name.startswith("fcs_"),
    )
    print("Optimizer initialized!")
    print()

    # Run optimization
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
    parser = argparse.ArgumentParser(description="Run AgenticHandoff optimization")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory name (created under results/)")
    parser.add_argument("--problem_name", type=str, default="cloudcast",
                        help="Problem to optimize. Built-in: vidur, cloudcast, eplb, llm_sql, llm_sql_ggr_adrs, llm_sql_ggr_ours. "
                             "Frontier-CS: fcs_alg_<id> (e.g., fcs_alg_228) or fcs_res_<id> (e.g., fcs_res_flash_attn).")
    parser.add_argument("--max_agents", type=int, default=30,
                        help="Maximum number of sequential agents to run")
    parser.add_argument("--agent_timeout", type=float, default=60.0,
                        help="Wall-clock timeout per agent in minutes")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of runs to perform")
    parser.add_argument("--give_files", action="store_true",
                        help="Give files to the agents (optimal)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--model", type=str, default="o3",
                        help="Model to use")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop AgenticHandoff if no improvement for this many iterations (default: 10)."
    )
    parser.add_argument(
        "--enable_continue_message",
        action="store_true",
        help="Enable continue message (default: False)."
    )
    parser.add_argument(
        "--remove_kb",
        action="store_true",
        help="Use no-KB ablation: AgenticHandoffNoKB (no knowledge base; experiments wiped on handoff)."
    )
    args = parser.parse_args()

    try:
        for i in range(args.num_runs):
            run_handoff_optimization(
                args.results_dir,
                args.problem_name,
                args.model,
                args.max_agents,
                args.agent_timeout,
                args.debug,
                args.give_files,
                args.early_stop_patience,
                args.enable_continue_message,
                args.remove_kb,
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
