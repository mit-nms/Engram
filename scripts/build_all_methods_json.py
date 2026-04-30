#!/usr/bin/env python3
"""
Build all_methods JSON file for plot_methods_paper.py from base result directories.

Scans base directories for OpenEvolve, Glia Tree, Glia Handoff, and Single Agent results.
- OpenEvolve: runs aggregate_openevolve_logs.py if aggregated_results.json doesn't exist
- Glia Tree: runs aggregate_tree_logs.py if aggregated_results.json doesn't exist
- Glia Handoff: finds the last (highest-iteration) JSON log per run
- Single Agent: finds the last (highest-iteration) single-agent JSON log per run

Usage:
    python build_all_methods_json.py --output output_name.json dir1 dir2 [dir3 ...]

Example:
    python build_all_methods_json.py \\
        --output all_methods_cant_be_late.json \\
        /data2/projects/pantea-work/Glia/results/glia_results_cant_be_late \\
        /data2/projects/pantea-work/Glia/results/OpenEvolve_Results_new
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Script directory for calling aggregate scripts
SCRIPTS_DIR = Path(__file__).resolve().parent


def find_openevolve_dirs(base_dir: Path) -> List[Path]:
    """Find OpenEvolve result directories (contain generated_programs)."""
    dirs = []
    if not base_dir.exists():
        return dirs
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and (subdir / "generated_programs").exists():
            dirs.append(subdir)
    return dirs


def find_handoff_dirs(base_dir: Path) -> List[Path]:
    """
    Find Glia Handoff run directories.
    Matches directories that either:
    1. Have "handoff" in their name, OR
    2. Contain a subdirectory with a "logs" folder that has handoff JSON files.
    """
    dirs = []
    if not base_dir.exists():
        return dirs
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Pattern 1: name contains "handoff"
        if "handoff" in subdir.name.lower():
            dirs.append(subdir)
            continue
        # Pattern 2: check if any subdirectory contains logs with handoff JSON files
        for nested_subdir in subdir.iterdir():
            if not nested_subdir.is_dir():
                continue
            logs_dir = nested_subdir / "logs"
            if not logs_dir.exists():
                continue
            # Check if logs directory contains handoff JSON files
            for f in logs_dir.iterdir():
                if (
                    f.suffix == ".json"
                    and "handoff" in f.name.lower()
                    and "iterations" in f.name
                    and "usage_stats" not in f.name
                ):
                    dirs.append(subdir)
                    break
            if subdir in dirs:
                break
    return dirs


def find_tree_dirs(base_dir: Path) -> List[Path]:
    """Find Glia Tree run directories (deepagents_tree_* pattern)."""
    dirs = []
    if not base_dir.exists():
        return dirs
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and "deepagents_tree" in subdir.name.lower():
            dirs.append(subdir)
    return dirs


def find_single_agent_dirs(base_dir: Path) -> List[Path]:
    """
    Find Single Agent run directories.
    Matches directories that either:
    1. Have "single_agent" in their name, OR
    2. Contain a subdirectory with a "logs" folder that has single-agent JSON files.
    """
    dirs = []
    if not base_dir.exists():
        return dirs
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Pattern 1: name contains "single_agent"
        if "single_agent" in subdir.name.lower():
            dirs.append(subdir)
            continue
        # Pattern 2: check if any subdirectory contains logs with single-agent JSON files
        for nested_subdir in subdir.iterdir():
            if not nested_subdir.is_dir():
                continue
            logs_dir = nested_subdir / "logs"
            if not logs_dir.exists():
                continue
            for f in logs_dir.iterdir():
                if (
                    f.suffix == ".json"
                    and "single_agent" in f.name.lower()
                    and "iterations" in f.name
                    and "usage_stats" not in f.name
                ):
                    dirs.append(subdir)
                    break
            if subdir in dirs:
                break
    return dirs


def ensure_openevolve_aggregated(openevolve_dir: Path) -> Optional[Path]:
    """Run aggregate_openevolve_logs.py if aggregated_results.json doesn't exist."""
    agg_path = openevolve_dir / "aggregated_results.json"
    if agg_path.exists():
        return agg_path
    script = SCRIPTS_DIR / "aggregate_openevolve_logs.py"
    if not script.exists():
        print(f"Warning: {script} not found, skipping {openevolve_dir.name}", file=sys.stderr)
        return None
    print(f"Running aggregate_openevolve_logs.py for {openevolve_dir.name}...")
    result = subprocess.run(
        [sys.executable, str(script), str(openevolve_dir), str(agg_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: aggregation failed for {openevolve_dir.name}: {result.stderr}", file=sys.stderr)
        return None
    return agg_path if agg_path.exists() else None


def find_last_handoff_json(run_dir: Path) -> Optional[Path]:
    """
    Find the last (highest-iteration) handoff JSON log in a run directory.
    Looks for *handoff*iterations.json or *agentic_handoff*iterations.json in ADRS/logs/.
    Automatically discovers ADRS subdirectories by scanning for directories with a "logs" subdirectory.
    """
    # Automatically discover ADRS subdirectories
    if not run_dir.exists():
        return None

    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue
        logs_dir = subdir / "logs"
        if not logs_dir.exists():
            continue

        candidates = []
        for f in logs_dir.iterdir():
            if (
                f.suffix == ".json"
                and "handoff" in f.name.lower()
                and "iterations" in f.name
                and "usage_stats" not in f.name
            ):
                m = re.search(r"(\d+)iterations", f.name)
                if m:
                    candidates.append((int(m.group(1)), f))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    return None


def ensure_tree_aggregated(tree_run_dir: Path) -> Optional[Path]:
    """
    For a deepagents_tree_* run dir, find ADRS subdir and run aggregate_tree_logs.py
    if aggregated_results.json doesn't exist.
    Automatically discovers ADRS subdirectories by scanning for directories with a "logs" subdirectory.
    """
    if not tree_run_dir.exists():
        return None

    # Automatically discover ADRS subdirectories
    for subdir in tree_run_dir.iterdir():
        if not subdir.is_dir():
            continue
        adrs_dir = subdir
        if not (adrs_dir / "logs").exists():
            continue

        agg_path = adrs_dir / "aggregated_results.json"
        if agg_path.exists():
            return agg_path
        script = SCRIPTS_DIR / "aggregate_tree_logs.py"
        if not script.exists():
            print(f"Warning: {script} not found, skipping {tree_run_dir.name}", file=sys.stderr)
            return None
        print(f"Running aggregate_tree_logs.py for {tree_run_dir.name}/{adrs_dir.name}...")
        result = subprocess.run(
            [sys.executable, str(script), str(adrs_dir), str(agg_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: tree aggregation failed for {tree_run_dir.name}: {result.stderr}", file=sys.stderr)
            return None
        return agg_path if agg_path.exists() else None
    return None


def find_last_single_agent_json(run_dir: Path) -> Optional[Path]:
    """
    Find the last (highest-iteration) single-agent JSON log in a run directory.
    Looks for *single_agent*iterations.json in ADRS/logs/.
    Automatically discovers ADRS subdirectories by scanning for directories with a "logs" subdirectory.
    """
    if not run_dir.exists():
        return None

    for subdir in run_dir.iterdir():
        if not subdir.is_dir():
            continue
        logs_dir = subdir / "logs"
        if not logs_dir.exists():
            continue

        candidates = []
        for f in logs_dir.iterdir():
            if (
                f.suffix == ".json"
                and "single_agent" in f.name.lower()
                and "iterations" in f.name
                and "usage_stats" not in f.name
            ):
                m = re.search(r"(\d+)iterations", f.name)
                if m:
                    candidates.append((int(m.group(1)), f))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    return None


def collect_from_base_dir(
    base_dir: Path,
    label_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    Scan a base directory and collect paths for OpenEvolve, Tree, Handoff, and Single Agent.
    Returns dict: method_label -> list of absolute paths.

    label_overrides: optional dict mapping "pattern" -> custom_label. When base_dir
        path contains "pattern", the custom label is used (method type inferred from
        label: Handoff/Tree/OpenEvolve). E.g. {"many_agents": "Glia Handoff (o3) many agents"}.
    """
    out: Dict[str, List[str]] = {}
    base_str = str(base_dir)
    overrides = label_overrides or {}

    def pick_label(default: str, keywords: Tuple[str, ...]) -> str:
        """Use override if base contains pattern and override label matches keywords."""
        for pattern, custom in overrides.items():
            if pattern in base_str:
                if any(kw in custom for kw in keywords):
                    return custom
        return default

    # OpenEvolve
    for d in find_openevolve_dirs(base_dir):
        p = ensure_openevolve_aggregated(d)
        if p:
            key = pick_label("OpenEvolve (o3)", ("OpenEvolve", "openevolve"))
            out.setdefault(key, []).append(str(p.resolve()))

    # Glia Tree
    for d in find_tree_dirs(base_dir):
        p = ensure_tree_aggregated(d)
        if p:
            key = pick_label("Glia Tree (o3)", ("Tree", "tree"))
            out.setdefault(key, []).append(str(p.resolve()))

    # Glia Handoff
    for d in find_handoff_dirs(base_dir):
        p = find_last_handoff_json(d)
        if p:
            key = pick_label("Glia Handoff (o3)", ("Handoff", "handoff"))
            out.setdefault(key, []).append(str(p.resolve()))

    # Glia Single Agent
    for d in find_single_agent_dirs(base_dir):
        p = find_last_single_agent_json(d)
        if p:
            key = pick_label("Glia Single Agent (o3)", ("Single Agent", "single_agent", "single"))
            out.setdefault(key, []).append(str(p.resolve()))

    return out


def merge_methods(accum: Dict[str, List[str]], new: Dict[str, List[str]]) -> None:
    """Merge new method paths into accum (extend lists)."""
    for k, v in new.items():
        accum.setdefault(k, []).extend(v)


def main():
    parser = argparse.ArgumentParser(
        description="Build all_methods JSON for plot_methods_paper.py from base result directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="Base result directories to scan (e.g. glia_results_cant_be_late, OpenEvolve_Results_new)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output JSON file path (e.g. all_methods_cant_be_late.json)",
    )
    parser.add_argument("--label-override", action="append", metavar="PATTERN:LABEL",
                        help="When base dir path contains PATTERN, use LABEL for that method type")
    args = parser.parse_args()

    label_overrides: Dict[str, str] = {}
    for s in (args.label_override or []):
        if ":" in s:
            pat, lbl = s.split(":", 1)
            label_overrides[pat.strip()] = lbl.strip()

    merged: Dict[str, List[str]] = {}
    for base in args.dirs:
        if not base.exists():
            print(f"Warning: directory does not exist: {base}", file=sys.stderr)
            continue
        print(f"Scanning {base}...")
        collected = collect_from_base_dir(base, label_overrides=label_overrides)
        merge_methods(merged, collected)

    if not merged:
        print("No results found in any directory.", file=sys.stderr)
        sys.exit(1)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort paths within each method for reproducibility
    for k in merged:
        merged[k] = sorted(merged[k])
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Wrote {out_path}")
    for k, paths in merged.items():
        print(f"  {k}: {len(paths)} runs")


if __name__ == "__main__":
    main()