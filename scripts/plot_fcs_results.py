#!/usr/bin/env python3
"""
Plot OpenEvolve and Handoff results for Frontier-CS algorithmic problems.

For each problem, builds an all_methods JSON and invokes plot_methods_paper.py
to produce a separate set of plots.

Usage:
    source .venv/bin/activate
    python scripts/plot_fcs_results.py
    python scripts/plot_fcs_results.py --problems 0 1 5
    python scripts/plot_fcs_results.py --max-num-sims 50 --num-runs 5
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
AGGREGATE_SCRIPT = ROOT_DIR / "scripts" / "aggregate_openevolve_logs.py"
PLOT_SCRIPT = ROOT_DIR / "scripts" / "plot_methods_paper.py"

# Problem ID -> human-readable name
PROBLEM_NAMES = {
    0: "Pack the Polyominoes",
    1: "Treasure Packing",
    5: "Hamiltonian Path Challenge",
    6: "Worldmap",
    7: "Build a Computer",
}

# Directory prefixes
HANDOFF_PREFIX = "fcs_alg"           # fcs_alg_{pid}_run{N}
OPENEVOLVE_PREFIX = "fcs_openevolve_alg"  # fcs_openevolve_alg_{pid}_run{N}

# Sub-directory inside the results dir that contains logs
FCS_SUBDIR_TEMPLATE = "fcs_algorithmic_{pid}"


def find_latest_handoff_json(run_dir: Path, pid: int) -> str | None:
    """Find the handoff JSON with the highest iteration count."""
    logs_dir = run_dir / FCS_SUBDIR_TEMPLATE.format(pid=pid) / "logs"
    if not logs_dir.is_dir():
        return None
    candidates = list(logs_dir.glob("*handoff*iterations.json"))
    # Filter out usage_stats files
    candidates = [c for c in candidates if "usage_stats" not in c.name]
    if not candidates:
        return None

    def extract_iters(p: Path) -> int:
        m = re.search(r"(\d+)iterations\.json$", p.name)
        return int(m.group(1)) if m else 0

    best = max(candidates, key=extract_iters)
    return str(best)


def aggregate_openevolve(run_dir: Path) -> str | None:
    """Run aggregate_openevolve_logs.py and return the output JSON path."""
    gen_dir = run_dir / "generated_programs"
    if not gen_dir.is_dir():
        return None
    output_json = run_dir / "aggregated_results.json"
    if not output_json.is_file():
        try:
            result = subprocess.run(
                [sys.executable, str(AGGREGATE_SCRIPT), str(run_dir), str(output_json)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  Warning: aggregation failed for {run_dir.name}: {result.stderr.strip()}")
        except Exception as e:
            print(f"  Warning: aggregation error for {run_dir.name}: {e}")
    if output_json.is_file():
        return str(output_json)
    return None


def collect_method_paths(pid: int, num_runs: int) -> dict[str, list[str]]:
    """Collect all JSON paths for both methods for a given problem."""
    methods: dict[str, list[str]] = {}

    # Handoff runs
    handoff_paths = []
    for i in range(1, num_runs + 1):
        run_dir = RESULTS_DIR / f"{HANDOFF_PREFIX}_{pid}_run{i}"
        if not run_dir.is_dir():
            continue
        p = find_latest_handoff_json(run_dir, pid)
        if p:
            handoff_paths.append(p)
    if handoff_paths:
        methods[f"Glia Handoff (o3)"] = handoff_paths

    # OpenEvolve runs
    oe_paths = []
    for i in range(1, num_runs + 1):
        run_dir = RESULTS_DIR / f"{OPENEVOLVE_PREFIX}_{pid}_run{i}"
        if not run_dir.is_dir():
            continue
        p = aggregate_openevolve(run_dir)
        if p:
            oe_paths.append(p)
    if oe_paths:
        methods[f"OpenEvolve (o3)"] = oe_paths

    return methods


def main():
    parser = argparse.ArgumentParser(description="Plot Frontier-CS algorithmic results per problem")
    parser.add_argument("--problems", type=int, nargs="+", default=list(PROBLEM_NAMES.keys()),
                        help="Problem IDs to plot (default: all)")
    parser.add_argument("--max-num-sims", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=5, help="Max number of runs to look for per method")
    parser.add_argument("--output-dir", type=str, default="plots_fcs",
                        help="Base output directory (per-problem subdirs created inside)")
    args = parser.parse_args()

    print(f"Results directory: {RESULTS_DIR}")
    print(f"Looking for up to {args.num_runs} runs per method\n")

    for pid in args.problems:
        name = PROBLEM_NAMES.get(pid, f"problem_{pid}")
        print(f"{'='*60}")
        print(f"Problem {pid}: {name}")
        print(f"{'='*60}")

        methods = collect_method_paths(pid, args.num_runs)

        for method_key, paths in methods.items():
            print(f"  {method_key}: {len(paths)} run(s)")

        if not methods:
            print(f"  No results found -- skipping\n")
            continue

        # Clean output directory to avoid stale cached JSON from previous runs
        output_path = ROOT_DIR / args.output_dir / f"problem_{pid}"
        if output_path.exists():
            shutil.rmtree(output_path)

        # Write the all_methods JSON for this problem
        json_path = ROOT_DIR / f"all_methods_fcs_{pid}.json"
        with open(json_path, "w") as f:
            json.dump(methods, f, indent=2)
        print(f"  Wrote {json_path}")

        # Baseline cache path for this problem
        baseline_cache = (ROOT_DIR / "SystemBench" / "FrontierCS" / "frontier_cs_repo"
                          / "algorithmic" / "solutions" / str(pid) / "baseline_cache.json")

        # Output directory for this problem's plots
        output_dir = str(ROOT_DIR / args.output_dir / f"problem_{pid}")

        # Invoke plot_methods_paper.py
        cmd = [
            sys.executable, str(PLOT_SCRIPT),
            "--all-methods-json", str(json_path),
            "-o", output_dir,
            "--max-num-sims", str(args.max_num_sims),
            "--problem-name", "fcs_algorithmic",
        ]
        if baseline_cache.is_file():
            cmd.extend(["--baseline-cache", str(baseline_cache)])
        print(f"  Running: {' '.join(cmd)}\n")
        subprocess.run(cmd)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
