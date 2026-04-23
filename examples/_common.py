"""Shared utilities for example scripts."""

import os
from pathlib import Path


def get_results_base_dir(base_dir: Path) -> Path:
    """Return results base directory, respecting GLIA_RESULTS_BASE_DIR env var."""
    return Path(os.environ.get("GLIA_RESULTS_BASE_DIR") or str(base_dir / "results"))


def normalize_initial_program_path(initial_program_path):
    """Filter non-existing paths and normalize to strings."""
    if not initial_program_path:
        return None
    if isinstance(initial_program_path, list):
        result = [str(p) for p in initial_program_path if Path(p).exists()]
        return result if result else None
    return str(initial_program_path) if Path(initial_program_path).exists() else None


def auto_increment_run_dir(results_dir: Path) -> Path:
    """Bump the trailing runN suffix until the directory does not exist."""
    i = 0
    while results_dir.exists():
        results_dir = Path(str(results_dir).replace(f"run{i}", f"run{i + 1}"))
        i += 1
    return results_dir


def print_initial_program(initial_program_path) -> None:
    """Print initial program path(s) in a standardized format."""
    if not initial_program_path:
        return
    if isinstance(initial_program_path, list):
        print(f"Initial Programs: {', '.join(initial_program_path)}")
    else:
        print(f"Initial Program: {initial_program_path}")


def resolve_problem_config(base_dir: Path, problem_name: str, give_files: bool = False) -> tuple[Path, Path, str | list[Path]]:
    # Problem-specific configuration.
    if problem_name == "vidur":
        task_prompt_path = base_dir / "SystemBench" / "vidur" / "deepagents_files" / "task_prompt.txt"
        evaluator_path = base_dir / "SystemBench" / "vidur"
        initial_program_path = base_dir / "SystemBench" / "vidur" / "deepagents_files" / "llq_scheduler.py"

    elif problem_name == "cloudcast":
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "cloudcast"
        if give_files:
            task_prompt_path = base_dir / "SystemBench" / "ADRS" / "cloudcast" / "deepagents_files" / "task_prompt_direction_with_optimal.txt"
            initial_program_path = [base_dir / "SystemBench" / "ADRS" / "cloudcast" / "initial_program.py",
                                    base_dir / "SystemBench" / "ADRS" / "cloudcast" / "optimal.py"]
        else:
            print("Using task prompt no cheat no file")
            task_prompt_path = base_dir / "SystemBench" / "ADRS" / "cloudcast" / "deepagents_files" / "task_prompt_direction.txt"
            initial_program_path = base_dir / "SystemBench" / "ADRS" / "cloudcast" / "initial_program.py"

    elif problem_name == "eplb":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "eplb" / "deepagents_files" / "adrs.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "eplb"
        initial_program_path = base_dir / "SystemBench" / "ADRS" / "eplb" / "initial_program.py"

    elif problem_name == "llm_sql":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "normal" / "task_prompt.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "llm_sql"
        initial_program_path = [
            base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "normal" / "initial_program.py",
            base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "sample_data",
        ]
    elif problem_name == "llm_sql_ggr_ours":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "GGR_ours" / "task_prompt.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "llm_sql"
        initial_program_path = [
            base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "GGR_ours" / "initial_program.py",
            base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "sample_data",
        ]
    elif problem_name == "llm_sql_ggr_adrs":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "GGR_ADRS" / "task_prompt.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "llm_sql"
        initial_program_path = [
            base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "GGR_ADRS" / "initial_program.py",
            base_dir / "SystemBench" / "ADRS" / "llm_sql" / "deepagents_files" / "sample_data",
        ]
    elif problem_name == "cant-be-late":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "cant-be-late" / "deepagents_files" / "adrs.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "cant-be-late"
        initial_program_path = base_dir / "SystemBench" / "ADRS" / "cant-be-late" / "initial_program.py"
    elif problem_name == "cant-be-late-multi":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "cant-be-late-multi" / "deepagents_files" /"adrs.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "cant-be-late-multi"
        initial_program_path = base_dir / "SystemBench" / "ADRS" / "cant-be-late-multi" / "initial_program.py"
    elif problem_name == "txn_scheduling":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "txn_scheduling" / "deepagents_files" / "adrs.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "txn_scheduling"
        initial_program_path = base_dir / "SystemBench" / "ADRS" / "txn_scheduling" / "initial_program.py"
    elif problem_name == "prism":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "prism" / "deepagents_files" / "adrs.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "prism"
        initial_program_path = base_dir / "SystemBench" / "ADRS" / "prism" / "initial_program.py"
    elif problem_name == "telemetry_repair":
        task_prompt_path = base_dir / "SystemBench" / "ADRS" / "telemetry_repair" / "deepagents_files" / "adrs.txt"
        evaluator_path = base_dir / "SystemBench" / "ADRS" / "telemetry_repair"
        initial_program_path = base_dir / "SystemBench" / "ADRS" / "telemetry_repair" / "initial_program.py"
    elif problem_name.startswith("fcs_alg_"):
        problem_id = problem_name.replace("fcs_alg_", "")
        fcs_base = base_dir / "SystemBench" / "FrontierCS"
        os.environ["FCS_TRACK"] = "algorithmic"
        os.environ["FCS_PROBLEM_ID"] = problem_id
        task_prompt_path = fcs_base / "frontier_cs_repo" / "algorithmic" / "problems" / problem_id / "statement.txt"
        initial_program_path = fcs_base / "algorithmic" / "initial_program.cpp"
        evaluator_path = fcs_base

    elif problem_name.startswith("fcs_res_"):
        problem_id = problem_name.replace("fcs_res_", "")
        fcs_base = base_dir / "SystemBench" / "FrontierCS"
        os.environ["FCS_TRACK"] = "research"
        os.environ["FCS_PROBLEM_ID"] = problem_id
        initial_program_path = fcs_base / "research" / "initial_program.py"
        fcs_repo = fcs_base / "frontier_cs_repo"
        task_prompt_path = fcs_repo / "research" / "problems" / problem_id / "readme"
        evaluator_path = fcs_base
    else:
        raise ValueError(f"Invalid problem name: {problem_name}.")
    return task_prompt_path, evaluator_path, initial_program_path