#!/usr/bin/env python3
"""
Example usage of a strict linear multi-agent loop:
- each step is a fresh SingleAgent run
- continue message is disabled
- summarization middleware is disabled
- no context is shared between agents except code written by the previous agent
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the root directory to the path so we can import modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Architect.task import load_task_from_paths
from Architect.methods.single_agent import SingleAgent
from _common import normalize_initial_program_path, auto_increment_run_dir, resolve_problem_config

RUN_IN_PARALLEL = True


def run_linear_agent_optimization(
    results_dir_name: str | None = None,
    problem_name: str = "cloudcast",
    model: str = "o3",
    num_agents: int = 3,
    agent_timeout_minutes: float = 30.0,
    debug: bool = False,
    give_files: bool = False,
) -> dict[str, Any]:
    """Run a strict linear chain of fresh SingleAgent instances."""
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")

    base_dir = Path(__file__).parent.parent
    prompts_dir = base_dir / "Architect" / "methods" / "deepagents_utils"
    system_prompt_path = prompts_dir / "system_prompt_single_agent.txt"

    run_name = (
        results_dir_name
        or f"linear_agent_{problem_name}_model_{model}_num_agents_{num_agents}_give_files_{give_files}_run0"
    )
    results_base_dir = os.environ.get("GLIA_RESULTS_BASE_DIR") or str(base_dir / "results")
    results_dir = (
        Path(results_base_dir)
        / Path(f"{problem_name}_linear_agent_results")
        / Path(run_name)
    )

    results_dir = auto_increment_run_dir(results_dir)

    task_prompt_path, evaluator_path, initial_program_path = resolve_problem_config(base_dir, problem_name, give_files)
    initial_program_path = normalize_initial_program_path(initial_program_path)

    for name, path in [
        ("User prompt", str(task_prompt_path)),
        ("System prompt", str(system_prompt_path)),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    print("=" * 70)
    print("LinearAgent Optimization (fresh agents, code-only handoff)")
    print("=" * 70)
    print(f"Problem: {problem_name}")
    print(f"Model: {model}")
    print(f"Results Directory: {results_dir}")
    print(f"Num Agents: {num_agents}")
    print(f"Agent Timeout: {agent_timeout_minutes} minutes")
    print(f"Debug: {debug}")
    print("Enable continue message: False")
    print("Use summarization middleware: False")
    print("=" * 70)
    print()

    print("Loading task...")
    task = load_task_from_paths(str(task_prompt_path), str(evaluator_path))
    print(f"Loaded task: {task.name}")
    print()

    handoff_dir = Path(results_dir) / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    next_initial_program_path: str | list[str] | None = initial_program_path
    agent_summaries: list[dict[str, Any]] = []
    final_results: dict[str, Any] = {}
    total_simulations = 0

    for agent_idx in range(1, num_agents + 1):
        agent_results_dir = Path(results_dir) / f"agent_{agent_idx:02d}"
        print("-" * 70)
        print(f"Starting Agent {agent_idx}/{num_agents}")
        print(f"Input passed to Agent {agent_idx}: {next_initial_program_path}")
        print("-" * 70)

        optimizer = SingleAgent(
            task=task,
            model=model,
            results_dir=str(agent_results_dir),
            debug=debug,
            task_prompt_path=str(task_prompt_path),
            system_prompt_path=str(system_prompt_path),
            initial_program_path=next_initial_program_path,
            agent_timeout_minutes=agent_timeout_minutes,
            use_summarization_middleware=False,
            enable_continue_message=False,
            early_stop_patience=10,
            run_baselines=False,
        )

        agent_results = optimizer.optimize()
        final_results = agent_results

        best_solution = agent_results.get("best_solution", {})
        agent_score = best_solution.get("score", None)
        best_code = best_solution.get("code", None)
        agent_total_sims = int(agent_results.get("total_simulations", 0))
        total_simulations += agent_total_sims

        handoff_file: str | None = None
        if isinstance(best_code, str) and best_code.strip():
            handoff_path = handoff_dir / f"agent_{agent_idx:02d}_best.py"
            handoff_path.write_text(best_code, encoding="utf-8")
            handoff_file = str(handoff_path)
            next_initial_program_path = handoff_file
            print(f"Agent {agent_idx} handoff code saved to: {handoff_file}")
        else:
            print(
                f"Warning: Agent {agent_idx} produced no valid best_solution.code; "
                "reusing previous handoff input for the next agent."
            )

        agent_summary = {
            "agent_index": agent_idx,
            "score": agent_score,
            "total_simulations": agent_total_sims,
            "handoff_file": handoff_file,
            "convergence_reason": agent_results.get("convergence_reason", "unknown"),
            "results_dir": str(agent_results_dir),
        }
        agent_summaries.append(agent_summary)
        print(f"Agent {agent_idx} score: {agent_score}")

    aggregate = {
        "problem_name": problem_name,
        "model": model,
        "num_agents_requested": num_agents,
        "num_agents_completed": len(agent_summaries),
        "results_dir": str(results_dir),
        "agent_results": agent_summaries,
        "total_simulations_across_agents": total_simulations,
        "final_best_solution": final_results.get("best_solution", {}),
        "final_convergence_reason": final_results.get("convergence_reason", "unknown"),
    }

    summary_path = Path(results_dir) / "linear_agent_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 70)
    print("Linear chain completed!")
    print(f"Summary saved to: {summary_path}")
    print(
        "Final best score: "
        f"{aggregate.get('final_best_solution', {}).get('score', 'N/A')}"
    )
    print(f"Total simulations across agents: {total_simulations}")
    print("=" * 70)

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strict linear multi-agent optimization (code-only handoff)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory name (created under results/)")
    parser.add_argument("--problem_name", type=str, default="cloudcast", help="Problem to optimize")
    parser.add_argument("--num_agents", type=int, default=20, help="Number of sequential fresh agents")
    parser.add_argument("--agent_timeout", type=float, default=60.0, help="Wall-clock timeout per agent in minutes")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repeated runs")
    parser.add_argument("--give_files", action="store_true", help="Give files to the first agent (problem-specific)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--model", type=str, default="o3", help="Model to use")
    args = parser.parse_args()

    try:
        base_name = (
            args.results_dir
            or f"linear_agent_{args.problem_name}_model_{args.model}_num_agents_{args.num_agents}_give_files_{args.give_files}"
        )
        if RUN_IN_PARALLEL:
            with ProcessPoolExecutor(max_workers=args.num_runs) as executor:
                futures = {
                    executor.submit(
                        run_linear_agent_optimization,
                        f"{base_name}_run{i}",
                        args.problem_name,
                        args.model,
                        args.num_agents,
                        args.agent_timeout,
                        args.debug,
                        args.give_files,
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
            for run_idx in range(args.num_runs):
                run_linear_agent_optimization(
                    results_dir_name=f"{base_name}_run{run_idx}",
                    problem_name=args.problem_name,
                    model=args.model,
                    num_agents=args.num_agents,
                    agent_timeout_minutes=args.agent_timeout,
                    debug=args.debug,
                    give_files=args.give_files,
                )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
