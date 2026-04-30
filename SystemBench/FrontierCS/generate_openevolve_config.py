#!/usr/bin/env python3
"""
Generate OpenEvolve config.yaml and initial_program for a Frontier-CS problem.

Reads problem statement and environment from the frontier_cs_repo submodule,
uses the same prompts as the repo's generate_solutions scripts.

Usage:
    python SystemBench/FrontierCS/generate_openevolve_config.py \
        --track algorithmic --problem_id 0 \
        --output SystemBench/FrontierCS/openevolve_configs/alg_0/

    python SystemBench/FrontierCS/generate_openevolve_config.py \
        --track research --problem_id flash_attn \
        --output SystemBench/FrontierCS/openevolve_configs/res_flash_attn/
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure Glia root and frontier_cs src are importable
_THIS_DIR = Path(__file__).resolve().parent
_GLIA_ROOT = _THIS_DIR.parent.parent
_FCS_REPO = _THIS_DIR / "frontier_cs_repo"
_FCS_SRC = _FCS_REPO / "src"

if str(_GLIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_GLIA_ROOT))
if str(_FCS_SRC) not in sys.path:
    sys.path.insert(0, str(_FCS_SRC))


# ---------------------------------------------------------------------------
# Prompt helpers — reuse the exact same prompts from the frontier_cs repo
# ---------------------------------------------------------------------------

# Algorithmic prompt (from frontier_cs_repo/algorithmic/scripts/generate_solutions.py)
CPP_SYSTEM_PROMPT = (
    "You are a competitive programmer. You will be given a problem statement, "
    "please implement a solution in C++. The execution time and memory limit are "
    "also stated in the statement so be aware of the complexity of the program. "
    "Your response should ONLY contain the C++ code, with no additional "
    "explanation or text."
)


def _get_research_system_prompt(problem_path: Path) -> str:
    """Build research system prompt using gen_env (same as generate_solutions.py)."""
    scripts_path = _FCS_REPO / "research" / "scripts"
    for p in [str(_FCS_SRC), str(scripts_path)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from gen_env import get_system_prompt_for_problem

    # problem_name relative to research/problems/ (e.g. "symbolic_regression/mccormick")
    try:
        rel = problem_path.relative_to(_FCS_REPO / "research" / "problems")
        problem_name = str(rel)
    except ValueError:
        problem_name = problem_path.name
    return get_system_prompt_for_problem(problem_name, problem_path)


# ---------------------------------------------------------------------------
# Problem statement reading
# ---------------------------------------------------------------------------

def _read_problem_statement(track: str, problem_id, fcs_repo: Path) -> str:
    """Read the problem statement from the frontier_cs_repo."""
    if track == "algorithmic":
        stmt_path = fcs_repo / "algorithmic" / "problems" / str(problem_id) / "statement.txt"
        if stmt_path.exists():
            return stmt_path.read_text()
        return f"Algorithmic problem {problem_id} (statement not found locally)."
    else:
        problem_dir = fcs_repo / "research" / "problems" / str(problem_id)
        for name in ("readme", "readme.md", "README.md", "README"):
            p = problem_dir / name
            if p.exists():
                return p.read_text()
        return f"Research problem {problem_id} (readme not found locally)."


# ---------------------------------------------------------------------------
# Initial program
# ---------------------------------------------------------------------------

def _get_initial_program(track: str, problem_id, fcs_repo: Path, problem_path: Path = None) -> str:
    """Get the initial program for the problem."""
    if track == "algorithmic":
        return (
            "// EVOLVE-BLOCK-START\n"
            "#include <bits/stdc++.h>\n"
            "using namespace std;\n"
            "int main() {\n"
            "    // Your C++17 solution here\n"
            "    return 0;\n"
            "}\n"
            "// EVOLVE-BLOCK-END\n"
        )
    else:
        # research: use the shared initial_program.py (same as handoff)
        initial_prog = _THIS_DIR / "research" / "initial_program.py"
        code = initial_prog.read_text(encoding="utf-8")
        return f"# EVOLVE-BLOCK-START\n{code}\n# EVOLVE-BLOCK-END\n"


# ---------------------------------------------------------------------------
# Timeout discovery
# ---------------------------------------------------------------------------

def _discover_timeout(track: str, problem_id, fcs_repo: Path) -> int:
    """Discover evaluation timeout from problem config."""
    if track == "algorithmic":
        config_path = fcs_repo / "algorithmic" / "problems" / str(problem_id) / "config.yaml"
    else:
        config_path = fcs_repo / "research" / "problems" / str(problem_id) / "config.yaml"

    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            if track == "algorithmic":
                time_str = config.get("time", "2s")
                seconds = int(time_str.replace("s", "").replace("m", ""))
                return max(30, seconds * 10)
            else:
                # Research problems may have a timeout key
                t = config.get("timeout") or config.get("time_limit")
                if t:
                    return int(t)
        except Exception:
            pass

    return 30 if track == "algorithmic" else 600


# ---------------------------------------------------------------------------
# Config YAML generation
# ---------------------------------------------------------------------------

def generate_config(track: str, problem_id, fcs_repo: Path, model: str = "o3") -> str:
    """Generate the OpenEvolve config.yaml content."""
    if track == "research":
        problem_path = fcs_repo / "research" / "problems" / str(problem_id)
        system_prompt = _get_research_system_prompt(problem_path)
        readme = _read_problem_statement("research", problem_id, fcs_repo)
        timeout = _discover_timeout("research", problem_id, fcs_repo)

        # system_message = env-aware prompt + readme (mirrors generate_solutions.py exactly)
        system_message = f"{system_prompt}\n\nProblem:\n\n{readme}\n\nGenerate solution code:"
        indented_message = "\n".join("    " + line for line in system_message.splitlines())

        config = f"""# OpenEvolve config for Frontier-CS research/{problem_id}
max_iterations: 100
checkpoint_interval: 10
log_level: "DEBUG"

llm:
  models:
    - name: "{model}"
      weight: 1.0
      api_base: "https://api.openai.com/v1"
  temperature: 0.7
  top_p: 0.95
  max_tokens: 32000
  timeout: 600

prompt:
  system_message: |
{indented_message}
  num_top_programs: 3
  use_template_stochasticity: true

database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7

evaluator:
  timeout: {timeout}
  cascade_evaluation: false
  parallel_evaluations: 2
  use_llm_feedback: false

diff_based_evolution: true
allow_full_rewrites: true
max_code_length: 60000
"""
        return config

    # Algorithmic track
    statement = _read_problem_statement(track, problem_id, fcs_repo)
    timeout = _discover_timeout(track, problem_id, fcs_repo)
    preamble = CPP_SYSTEM_PROMPT

    # Build system_message: preamble + problem statement
    system_message = f"{preamble}\n\nProblem:\n\n{statement}"

    # Escape for YAML block scalar
    # Use literal block style (|) — just need to indent content
    indented_message = "\n".join("    " + line for line in system_message.splitlines())

    diff_based = "true" if track == "research" else "false"
    allow_rewrites = "true"
    parallel_evals = 4

    config = f"""# OpenEvolve config for Frontier-CS {track}/{problem_id}

max_iterations: 100
checkpoint_interval: 10
log_level: "DEBUG"

# LLM configuration
llm:
  models:
    - name: "{model}"
      weight: 1.0
      api_base: "https://api.openai.com/v1"
  temperature: 0.7
  top_p: 0.95
  max_tokens: 32000
  timeout: 600

# Prompt configuration
prompt:
  system_message: |
{indented_message}
  num_top_programs: 3
  use_template_stochasticity: true

# Database configuration
database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7

# Evaluator configuration
evaluator:
  timeout: {timeout}
  cascade_evaluation: false
  cascade_thresholds: [0.5, 0.75]
  parallel_evaluations: {parallel_evals}
  use_llm_feedback: false

# Evolution settings
diff_based_evolution: {diff_based}
allow_full_rewrites: {allow_rewrites}
max_code_length: 60000
"""
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate OpenEvolve config for a Frontier-CS problem")
    parser.add_argument("--track", required=True, choices=["algorithmic", "research"])
    parser.add_argument("--problem_id", required=True)
    parser.add_argument("--output", required=True, help="Output directory for config.yaml and initial_program")
    parser.add_argument("--model", default="o3", help="LLM model name (default: o3)")
    parser.add_argument(
        "--fcs_repo",
        default=str(_FCS_REPO),
        help="Path to frontier_cs_repo (default: auto-detected)",
    )
    args = parser.parse_args()

    fcs_repo = Path(args.fcs_repo)
    if not fcs_repo.is_dir():
        print(f"Error: frontier_cs_repo not found at {fcs_repo}")
        sys.exit(1)

    # Normalize problem_id
    problem_id = args.problem_id
    if args.track == "algorithmic" and problem_id.isdigit():
        problem_id = int(problem_id)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate config
    config_content = generate_config(args.track, problem_id, fcs_repo, args.model)
    config_path = output_dir / "config.yaml"
    config_path.write_text(config_content)
    print(f"Written: {config_path}")

    # Generate initial program
    problem_path = fcs_repo / args.track / "problems" / str(problem_id)
    initial_code = _get_initial_program(args.track, problem_id, fcs_repo, problem_path)
    ext = ".cpp" if args.track == "algorithmic" else ".py"
    initial_path = output_dir / f"initial_program{ext}"
    initial_path.write_text(initial_code)
    print(f"Written: {initial_path}")


if __name__ == "__main__":
    main()
