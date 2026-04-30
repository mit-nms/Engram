#!/usr/bin/env python3
"""
Collect best scheduler programs per Vidur paper run, rerun Vidur (sarathi @ 7.5 QPS),
summarize request_e2e_time from request_metrics.csv, and plot paper-style bar charts.

Usage:
  python vidur_e2e_rerun.py collect [--paper-json PATH] [--out-dir DIR] [--approach NAME ...]
  python vidur_e2e_rerun.py run [--manifest PATH] [--results-csv PATH] [--resume] [--approach NAME ...]
  python vidur_e2e_rerun.py baselines [--output-json PATH]
  python vidur_e2e_rerun.py plot [--results-csv PATH] [--out-prefix PATH]
    [--paper-json PATH] [--no-seq-par4] [--baseline-json PATH] [--no-baselines]

``baselines`` runs Vidur (sarathi @ 7.5 QPS) for round_robin, llq, lor, and mo
(MoGlobalScheduler, Expert) and writes JSON used as horizontal reference lines on bar plots.

Do not edit the plan file; this script implements the Vidur E2E bar workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from plot_methods_paper import METHOD_COLORS
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Paths & method keys (aligned with plot_methods_paper.py)
# ---------------------------------------------------------------------------

APPROACH_DIRS = {
    "Best Shot (o3)": "best_shot",
    "Evolution (o3)": "evolution",
    "OpenEvolve (o3)": "openevolve",
    "Glia (o3)": "glia",
}

# Shorthand for --approach (case-insensitive)
_APPROACH_ARG_ALIASES = {"eoh": "Evolution (o3)"}


def _parse_approach_selection(names: Optional[List[str]]) -> Optional[frozenset]:
    """None or empty list → all approaches; otherwise validated set of manifest method keys."""
    if not names:
        return None
    out: List[str] = []
    for raw in names:
        s = raw.strip()
        key = _APPROACH_ARG_ALIASES.get(s.lower(), s)
        if key not in APPROACH_DIRS:
            valid = ", ".join(sorted(APPROACH_DIRS)) + ", eoh"
            raise SystemExit(f"Unknown --approach {raw!r}. Use one of: {valid}")
        out.append(key)
    return frozenset(out)

PAPER_JSON_KEYS = {
    "Best Shot (o3)": "Best Shot (o3)",
    "Evolution (o3)": "Evolution (o3)",
}

METHOD_ORDER_FOR_PLOT = [
    "Glia Best of 4 (o3)",
    "Sequential Glia (o3)",
    "OpenEvolve (o3)",
    "Best Shot (o3)",
    "Evolution (o3)",
]

# METHOD_COLORS = {
#     "Best Shot (o3)": "#9467bd",
#     "Evolution (o3)": "#b22222",
#     "Glia (o3)": "#006400",
#     "OpenEvolve (o3)": "#ff8c00",
#     "Glia Best of 4 (o3)": "magenta", #"#006400",
#     "Sequential Glia (o3)": "#b22222",
# }

METHOD_NAMES = {
    "Best Shot (o3)": "FunSearch",
    "Evolution (o3)": "EoH",
    "Glia (o3)": "Single-Context Glia",
    "OpenEvolve (o3)": "OpenEvolve",
    "Glia Best of 4 (o3)": "MCG-Par4",
    "Sequential Glia (o3)": "MCG-Seq",
}

Y_LABEL_SIZE = 18
BETTER_FONT_SIZE = 14
# Match plot_bars() baselines in plot_methods_paper.py
BASELINE_LABEL_FONT_SIZE = 10
BASELINE_LINE_COLOR = "black"
BASELINE_LINE_ALPHA = 0.5
BASELINE_LINE_WIDTH = 1.5
BASELINE_TEXT_COLOR = "black"
BASELINE_TEXT_ALPHA = 0.8

# Built-in Vidur global schedulers (see vidur/global_scheduler_registry.py). ``mo`` is MoGlobalScheduler (Expert).
VIDUR_E2E_BASELINES: List[Tuple[str, str]] = [
    # ("round_robin", "Round Robin"),
    ("llq", "LLQ"),
    # ("lor", "LOR"),
    ("mo", "Expert"),
]


def _is_valid_numeric(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return False
    return True


def _iteration_ok(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    score = item.get("score")
    cost = item.get("usage_stats", {}).get("total_cost", 0)
    code = item.get("code")
    return (
        _is_valid_numeric(score)
        and _is_valid_numeric(cost)
        and code is not None
        and isinstance(code, str)
        and len(code.strip()) > 0
    )


def best_code_from_all_iterations(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Highest score wins (Vidur combined_score)."""
    if not isinstance(data, dict) or "all_iterations" not in data:
        return None, None
    best_score = float("-inf")
    best_code: Optional[str] = None
    for item in data["all_iterations"]:
        if not _iteration_ok(item):
            continue
        s = float(item["score"])
        if s > best_score:
            best_score = s
            best_code = str(item["code"])
    return best_code, best_score if best_code else (None, None)


def best_code_from_openevolve_summary(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    if isinstance(data, dict) and "best_solution" in data:
        b = data["best_solution"]
        if isinstance(b, dict) and b.get("code"):
            return str(b["code"]), float(b.get("score", float("nan")))
    return best_code_from_all_iterations(data)


def load_paper_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _paper_paths_for_key(paper: Dict[str, Any], key: str) -> List[Any]:
    """Resolve a path list from paper JSON. Keys may be ``key`` or ``#key`` (plot skip prefix)."""
    for k in (key, f"#{key}"):
        v = paper.get(k)
        if isinstance(v, list):
            return v
    return []


def cmd_collect(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    paper = load_paper_json(Path(args.paper_json).resolve())
    manifest: Dict[str, Any] = {"approaches": {}, "paper_json": str(args.paper_json)}
    only = _parse_approach_selection(getattr(args, "approaches", None))

    for method_key, sub in APPROACH_DIRS.items():
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        manifest["approaches"][method_key] = []

    # Best Shot & Evolution from logs
    for method_key, json_key in PAPER_JSON_KEYS.items():
        if only is not None and method_key not in only:
            continue
        paths = _paper_paths_for_key(paper, json_key)
        sub = APPROACH_DIRS[method_key]
        for i, log_path in enumerate(paths, start=1):
            lp = Path(log_path)
            if not lp.is_file():
                raise FileNotFoundError(f"Missing log for {method_key} run {i}: {lp}")
            with open(lp, "r") as f:
                data = json.load(f)
            code, score = best_code_from_openevolve_summary(data)
            if code is None:
                raise ValueError(f"No valid code in {lp}")
            out_py = out_dir / sub / f"run_{i:02d}.py"
            out_py.write_text(code, encoding="utf-8")
            manifest["approaches"][method_key].append(
                {
                    "run": i,
                    "source_log": str(lp),
                    "best_score": score,
                    "code_file": str(out_py.relative_to(out_dir)),
                }
            )

    # OpenEvolve — existing best .py files
    mk = "OpenEvolve (o3)"
    if only is None or mk in only:
        oe_src = (REPO_ROOT / "scripts/glia_paper_plots/best_codes_openevolve").resolve()
        sub = APPROACH_DIRS[mk]
        for i in range(1, 11):
            src = oe_src / f"best_code_run_{i}.py"
            if not src.is_file():
                raise FileNotFoundError(f"Missing OpenEvolve file: {src}")
            dst = out_dir / sub / f"run_{i:02d}.py"
            shutil.copy2(src, dst)
            manifest["approaches"][mk].append(
                {
                    "run": i,
                    "source_file": str(src),
                    "code_file": str(dst.relative_to(out_dir)),
                }
            )

    # Glia — existing best .py files
    gk = "Glia (o3)"
    if only is None or gk in only:
        glia_src = (REPO_ROOT / "scripts/glia_paper_plots/best_codes_glia").resolve()
        gsub = APPROACH_DIRS[gk]
        for i in range(1, 11):
            src = glia_src / f"ai_global_scheduler_vidur_{i}_best.py"
            if not src.is_file():
                raise FileNotFoundError(f"Missing Glia file: {src}")
            dst = out_dir / gsub / f"run_{i:02d}.py"
            shutil.copy2(src, dst)
            manifest["approaches"][gk].append(
                {
                    "run": i,
                    "source_file": str(src),
                    "code_file": str(dst.relative_to(out_dir)),
                }
            )

    man_path = out_dir / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path} and scheduler copies under {out_dir}")


def _scenario():
    from Architect.types import Scenario

    return Scenario(
        name="sarathi_qps7.5",
        config={"replica_scheduler": "sarathi", "qps": 7.5},
    )


def _sanitize_scheduler_code(code: str) -> str:
    """Adjust saved programs for Vidur's typical Python 3.9 eval environment."""
    import re

    if "class AIGlobalScheduler" in code:
        code = code.replace("class AIGlobalScheduler", "class CustomGlobalScheduler", 1)
    if re.search(r"\bint\s*\|\s*float\b", code):
        if "Union" not in code:

            def _add_union(m: re.Match) -> str:
                names = m.group(1).strip()
                return f"from typing import Union, {names}"

            if re.search(r"^from typing import ", code, re.MULTILINE):
                code = re.sub(
                    r"^from typing import (.+)$",
                    _add_union,
                    code,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                code = "from typing import Union\n" + code
        code = re.sub(r"\bint\s*\|\s*float\b", "Union[int, float]", code)
    return code


def _e2e_stats_from_csv(csv_path: Path) -> Tuple[float, float, float, float, float]:
    df = pd.read_csv(csv_path)
    if "request_e2e_time" not in df.columns:
        raise KeyError(f"No request_e2e_time in {csv_path}")
    s = df["request_e2e_time"].astype(float)
    return (
        float(s.min()),
        float(s.mean()),
        float(s.median()),
        float(s.quantile(0.99)),
        float(s.max()),
    )


def _glia_valid_iteration_counts_per_paper_json(paper_json: Path) -> List[int]:
    """Count valid score/cost iterations per Glia (o3) log (same filter as extract_scores_and_costs)."""
    with open(paper_json, "r") as f:
        paper = json.load(f)
    paths = _paper_paths_for_key(paper, "Glia (o3)")
    counts: List[int] = []
    for path_str in paths:
        p = Path(path_str)
        if not p.is_file():
            counts.append(0)
            continue
        with open(p, "r") as jf:
            data = json.load(jf)
        n = 0
        for item in data.get("all_iterations", []):
            if _iteration_ok(item):
                n += 1
        counts.append(n)
    return counts


def _par4_e2e_bar_from_values(vals: List[float]) -> Optional[Tuple[float, float, float]]:
    """Mean of min(S) over all 4-subsets S; bootstrap CI over those subset minima."""
    if len(vals) < 4:
        return None
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from stat_utils import bootstrap_ci_mean

    subset_mins = [min(c) for c in combinations(vals, 4)]
    arr = np.asarray(subset_mins, dtype=float)
    return bootstrap_ci_mean(arr, random_state=42)


def _sequential_e2e_bar_time_weighted(
    M: List[float], L: List[int]
) -> Optional[Tuple[float, float, float]]:
    """Time-weighted running best E2E over all permutations (budget L[i] per Glia run), matching serial multi-context intuition."""
    if not M or not L or len(M) != len(L):
        return None
    tot = float(sum(L))
    if tot <= 0:
        return None
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from stat_utils import bootstrap_ci_mean

    seq_vals: List[float] = []
    for perm in permutations(range(len(M))):
        best = float("inf")
        acc = 0.0
        for j in perm:
            best = min(best, M[j])
            acc += L[j] * best
        seq_vals.append(acc / tot)
    arr = np.asarray(seq_vals, dtype=float)
    # Permutation count can be huge; cap bootstrap subsample size inside bootstrap_ci_mean.
    return bootstrap_ci_mean(arr, random_state=43, max_num_sims=min(5000, len(arr)))


def _backfill_e2e_min_column(df: pd.DataFrame) -> pd.DataFrame:
    if "e2e_min" in df.columns and df["e2e_min"].notna().all():
        return df
    out = df.copy()
    mins: List[float] = []
    for _, r in out.iterrows():
        if (
            str(r.get("success", "")).lower() in ("true", "1", "yes")
            and r.get("request_metrics_csv")
            and Path(str(r["request_metrics_csv"])).is_file()
        ):
            try:
                mins.append(float(pd.read_csv(r["request_metrics_csv"])["request_e2e_time"].min()))
            except Exception:
                mins.append(float("nan"))
        else:
            mins.append(float("nan"))
    out["e2e_min"] = mins
    return out


def cmd_run(args: argparse.Namespace) -> None:
    from SystemBench.vidur.env_evaluator import VidurEvaluator

    manifest_path = Path(args.manifest).resolve()
    out_dir = manifest_path.parent
    results_path = Path(args.results_csv).resolve()
    limit = getattr(args, "limit", None)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    only = _parse_approach_selection(getattr(args, "approaches", None))

    scenario = _scenario()
    by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if (
        args.resume
        and results_path.is_file()
        and results_path.stat().st_size > 0
    ):
        prev = pd.read_csv(results_path)
        for _, r in prev.iterrows():
            k = (str(r["method_key"]), int(r["run_id"]))
            by_key[k] = r.to_dict()

    new_success_count = 0

    def already_done(method_key: str, run_id: int) -> bool:
        r = by_key.get((method_key, run_id))
        if not r:
            return False
        if str(r.get("success", "")).lower() not in ("true", "1", "yes"):
            return False
        csvp = r.get("request_metrics_csv")
        return bool(csvp) and Path(str(csvp)).is_file()

    stop_all = False
    for method_key, entries in manifest["approaches"].items():
        if stop_all:
            break
        if only is not None and method_key not in only:
            continue
        for entry in entries:
            run_id = int(entry["run"])
            if limit is not None and new_success_count >= limit:
                stop_all = True
                break
            if args.resume and already_done(method_key, run_id):
                print(f"skip (done) {method_key} run_{run_id:02d}")
                continue
            rel = entry["code_file"]
            code_path = (out_dir / rel).resolve()
            code = _sanitize_scheduler_code(code_path.read_text(encoding="utf-8"))
            print(f"Running {method_key} run_{run_id:02d} ...")
            ev = VidurEvaluator()
            result = ev.run_simulation_with_algorithm_code(code, scenario)
            success = bool(result.get("success"))
            err = result.get("error", "")
            csv_path = ""
            stats = (float("nan"),) * 5
            if success:
                info_metrics = result.get("info", {}).get("metrics", {})
                op = info_metrics.get("output_path")
                if op:
                    csv_path = str(Path(op).resolve())
                    stats = _e2e_stats_from_csv(Path(csv_path))
            by_key[(method_key, run_id)] = {
                "method_key": method_key,
                "run_id": run_id,
                "code_path": str(code_path),
                "success": success,
                "error": err,
                "request_metrics_csv": csv_path,
                "e2e_min": stats[0],
                "e2e_mean": stats[1],
                "e2e_median": stats[2],
                "e2e_p99": stats[3],
                "e2e_max": stats[4],
            }
            if success:
                new_success_count += 1
            if limit is not None and new_success_count >= limit:
                stop_all = True
            rows_sorted = sorted(by_key.values(), key=lambda x: (x["method_key"], int(x["run_id"])))
            pd.DataFrame(rows_sorted).to_csv(results_path, index=False)
            if stop_all:
                break
        if stop_all:
            break

    rows_sorted = sorted(by_key.values(), key=lambda x: (x["method_key"], int(x["run_id"])))
    pd.DataFrame(rows_sorted).to_csv(results_path, index=False)
    print(f"Wrote {results_path}")


def setup_plot_style():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "font.family": "sans-serif",
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "grid.alpha": 0.3,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    plt.rcParams["axes.labelweight"] = "bold"


def _order_method_keys(keys: List[str]) -> List[str]:
    ordered = [k for k in METHOD_ORDER_FOR_PLOT if k in keys]
    ordered.extend(k for k in keys if k not in ordered)
    return ordered


def _load_vidur_baseline_e2e_json(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        blob = json.load(f)
    return blob.get("baselines", {})


def cmd_baselines(args: argparse.Namespace) -> None:
    """Run sarathi@7.5 QPS once per built-in global scheduler; write E2E stats JSON."""
    from SystemBench.vidur.env_evaluator import VidurEvaluator

    scenario = _scenario()
    out: Dict[str, Any] = {
        "scenario": {"name": scenario.name, "config": scenario.config},
        "baselines": {},
    }
    for cli_name, label in VIDUR_E2E_BASELINES:
        print(f"Baseline: {label} ({cli_name}) …")
        ev = VidurEvaluator()
        result = ev.run_simulation_builtin_global_scheduler(cli_name, scenario)
        if not result.get("success"):
            out["baselines"][cli_name] = {
                "label": label,
                "success": False,
                "error": result.get("error", ""),
            }
            print(f"  failed: {result.get('error', '')}")
            continue
        info_metrics = result.get("info", {}).get("metrics", {})
        op = info_metrics.get("output_path")
        if not op:
            out["baselines"][cli_name] = {
                "label": label,
                "success": False,
                "error": "no output_path in metrics",
            }
            continue
        stats = _e2e_stats_from_csv(Path(op))
        out["baselines"][cli_name] = {
            "label": label,
            "success": True,
            "request_metrics_csv": str(Path(op).resolve()),
            "e2e_min": stats[0],
            "e2e_mean": stats[1],
            "e2e_median": stats[2],
            "e2e_p99": stats[3],
            "e2e_max": stats[4],
        }
        print(f"  e2e_mean={stats[1]:.4f}s")

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


def cmd_plot(args: argparse.Namespace) -> None:
    from matplotlib import pyplot as plt

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from stat_utils import bootstrap_ci_mean

    results_path = Path(args.results_csv).resolve()
    df = pd.read_csv(results_path)
    df = _backfill_e2e_min_column(df)

    def _success_mask(series: pd.Series) -> pd.Series:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])

    ok = df[_success_mask(df["success"])]
    if ok.empty:
        raise RuntimeError("No successful runs in results CSV")

    metric_defs = [
        ("e2e_min", "Min RT (s)"),
        ("e2e_mean", "Mean RT (s)"),
        ("e2e_median", "Median RT (s)"),
        ("e2e_p99", "P99 RT (s)"),
        ("e2e_max", "Max RT (s)"),
    ]

    prefix = Path(args.out_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    _SKIP_METHODS = {"Glia (o3)"}
    methods_present = sorted(
        (m for m in ok["method_key"].unique() if m not in _SKIP_METHODS),
        key=lambda x: METHOD_ORDER_FOR_PLOT.index(x) if x in METHOD_ORDER_FOR_PLOT else 99,
    )

    paper_json = Path(getattr(args, "paper_json", REPO_ROOT / "scripts/all_methods_paper.json")).resolve()
    include_seq_par = not getattr(args, "no_seq_par4", False)
    bj_arg = getattr(args, "baseline_json", None)
    baseline_json = Path(bj_arg).resolve() if bj_arg is not None else (results_path.parent / "vidur_baseline_e2e.json")
    show_baselines = not getattr(args, "no_baselines", False)
    baseline_data = _load_vidur_baseline_e2e_json(baseline_json) if show_baselines else {}
    if show_baselines and not baseline_data:
        print(
            f"Note: no baseline file at {baseline_json}; run:\n"
            f"  python3 scripts/vidur_e2e_rerun.py baselines --output-json {baseline_json}"
        )

    for col, ylabel in metric_defs:
        if col not in ok.columns:
            print(f"Warning: skip metric {col}: column missing from results")
            continue
        bar_values_and_errors: Dict[str, Tuple[float, float, float]] = {}
        for mk in methods_present:
            vals = ok[ok["method_key"] == mk][col].astype(float).values
            if len(vals) == 0:
                continue
            if len(vals) > 1:
                m, lo, hi = bootstrap_ci_mean(vals, random_state=42)
            else:
                m = float(vals[0])
                lo = hi = m
            bar_values_and_errors[mk] = (float(m), float(lo), float(hi))

        if include_seq_par and col in ok.columns:
            glia = ok[ok["method_key"] == "Glia (o3)"].sort_values("run_id")
            M = [float(x) for x in glia[col].values if pd.notna(x)]
            L = _glia_valid_iteration_counts_per_paper_json(paper_json)
            n = min(len(M), len(L))
            if n >= 1:
                M_use, L_use = M[:n], L[:n]
                if len(M_use) >= 4:
                    p4 = _par4_e2e_bar_from_values(M_use)
                    if p4 is not None:
                        bar_values_and_errors["Glia Best of 4 (o3)"] = (
                            float(p4[0]),
                            float(p4[1]),
                            float(p4[2]),
                        )
                sq = _sequential_e2e_bar_time_weighted(M_use, L_use)
                if sq is not None:
                    bar_values_and_errors["Sequential Glia (o3)"] = (
                        float(sq[0]),
                        float(sq[1]),
                        float(sq[2]),
                    )

        ordered = _order_method_keys(list(bar_values_and_errors.keys()))
        method_names = [METHOD_NAMES[m] for m in ordered]
        method_colors = [METHOD_COLORS[m] for m in ordered]
        bar_values = [bar_values_and_errors[m][0] for m in ordered]
        bar_low = np.array([bar_values_and_errors[m][0] - bar_values_and_errors[m][1] for m in ordered])
        bar_up = np.array([bar_values_and_errors[m][2] - bar_values_and_errors[m][0] for m in ordered])
        yerr = np.array([bar_low, bar_up])

        fig, ax = plt.subplots(figsize=(9, 6))
        x_pos = range(len(method_names))
        ax.bar(
            x_pos,
            bar_values,
            yerr=yerr,
            color=method_colors,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
            capsize=5,
            error_kw={"linewidth": 2, "capthick": 2},
        )

        values_txt = prefix.parent / f"{prefix.name}_{col}_bar_values.txt"
        with open(values_txt, "w") as f:
            for mn, v, el, eu in zip(method_names, bar_values, bar_low, bar_up):
                f.write(f"{mn}: {v} - {el} + {eu}\n")
            if baseline_data:
                f.write("\n# Baselines (horizontal lines)\n")
                for cli_name, display_name in VIDUR_E2E_BASELINES:
                    row = baseline_data.get(cli_name)
                    if row and row.get("success") and col in row and pd.notna(row.get(col)):
                        f.write(f"{display_name}: {float(row[col])}\n")

        ax.set_ylabel(ylabel, fontsize=Y_LABEL_SIZE, fontweight="bold")
        ax.set_xticks(list(x_pos))

        def fit_text(text: str) -> str:
            if " " not in text:
                return text
            if text == "Single-Context Glia":
                return "SCG"
            if text in ("MCG-Par4", "MCG-Seq"):
                return text
            if text == "OpenEvolve":
                return text
            return text.replace(" ", "\n", 1)

        ax.set_xticklabels(
            [fit_text(m) for m in method_names],
            rotation=25,
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
            label.set_multialignment("center")

        y_min, y_max = ax.get_ylim()
        y_offset = 0.015 * (y_max - y_min)
        whisker_tops = [v + u for v, u in zip(bar_values, bar_up)]
        for i, (v, u) in enumerate(zip(bar_values, bar_up)):
            ax.text(i, v + u + y_offset, f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        xmax = ax.get_xlim()[1]
        baseline_ys: List[float] = []
        if baseline_data:
            for cli_name, display_name in VIDUR_E2E_BASELINES:
                row = baseline_data.get(cli_name)
                if not row or not row.get("success"):
                    continue
                if col not in row or row[col] is None or (isinstance(row[col], float) and math.isnan(float(row[col]))):
                    continue
                yv = float(row[col])
                baseline_ys.append(yv)
                ax.axhline(
                    y=yv,
                    color=BASELINE_LINE_COLOR,
                    linestyle="--",
                    linewidth=BASELINE_LINE_WIDTH,
                    alpha=BASELINE_LINE_ALPHA,
                    label=None,
                    zorder=4,
                )
                ax.text(
                    xmax,
                    yv,
                    f" {display_name}",
                    ha="left",
                    va="center",
                    color=BASELINE_TEXT_COLOR,
                    alpha=BASELINE_TEXT_ALPHA,
                    fontsize=BASELINE_LABEL_FONT_SIZE,
                    fontweight="bold",
                    clip_on=False,
                )

        top_candidates = whisker_tops + baseline_ys + [y_max]
        hi = max(top_candidates) if top_candidates else y_max
        lo = min([y_min] + baseline_ys) if baseline_ys else y_min
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + 2 * y_offset + pad)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        arrow_x = xlim[0] + 0.1 * (xlim[1] - xlim[0])
        arrow_y_start = ylim[0] + 0.75 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.55 * (ylim[1] - ylim[0])
        ax.annotate(
            "",
            xy=(arrow_x, arrow_y_end),
            xytext=(arrow_x, arrow_y_start),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        )
        ax.text(
            arrow_x - 0.02 * (xlim[1] - xlim[0]),
            (arrow_y_start + arrow_y_end) / 2,
            "Better",
            rotation=90,
            va="center",
            ha="right",
            fontsize=BETTER_FONT_SIZE,
            fontweight="bold",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out_pdf = prefix.parent / f"{prefix.name}_{col}_bar.pdf"
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_pdf}")

    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Vidur request_e2e_time rerun & bar plots")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="Extract/copy best codes into best_codes_vidur_e2e_rerun/")
    c.add_argument(
        "--paper-json",
        type=Path,
        default=REPO_ROOT / "scripts/all_methods_paper.json",
    )
    c.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "scripts/glia_paper_plots/best_codes_vidur_e2e_rerun",
    )
    c.add_argument(
        "--approach",
        action="append",
        dest="approaches",
        metavar="NAME",
        help="Only this method (repeatable). Keys: Best Shot (o3), Evolution (o3), OpenEvolve (o3), Glia (o3); alias eoh=Evolution. Default: all.",
    )
    c.set_defaults(func=cmd_collect, approaches=None)

    r = sub.add_parser("run", help="Run VidurEvaluator for each manifest entry")
    r.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "scripts/glia_paper_plots/best_codes_vidur_e2e_rerun/manifest.json",
    )
    r.add_argument(
        "--results-csv",
        type=Path,
        default=REPO_ROOT / "scripts/glia_paper_plots/best_codes_vidur_e2e_rerun/simulation_results.csv",
    )
    r.add_argument("--resume", action="store_true")
    r.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many successful new runs (for smoke tests)",
    )
    r.add_argument(
        "--approach",
        action="append",
        dest="approaches",
        metavar="NAME",
        help="Only run these methods (repeatable); same names as collect --approach.",
    )
    r.set_defaults(func=cmd_run, approaches=None)

    pl = sub.add_parser("plot", help="Bar charts from simulation_results.csv")
    pl.add_argument(
        "--results-csv",
        type=Path,
        default=REPO_ROOT / "scripts/glia_paper_plots/best_codes_vidur_e2e_rerun/simulation_results.csv",
    )
    pl.add_argument(
        "--out-prefix",
        type=Path,
        default=REPO_ROOT / "scripts/glia_paper_plots/vidur_request_e2e_methods",
        help="Output base name (suffix _e2e_mean_bar.pdf etc. added)",
    )
    pl.add_argument(
        "--paper-json",
        type=Path,
        default=REPO_ROOT / "scripts/all_methods_paper.json",
        help="Glia (o3) log paths (for iteration counts used in MCG-Seq bars)",
    )
    pl.add_argument(
        "--no-seq-par4",
        action="store_true",
        help="Omit MCG-Par4 / MCG-Seq bars (subset-based Glia combinations only)",
    )
    pl.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="E2E stats for built-in schedulers (default: <results-dir>/vidur_baseline_e2e.json)",
    )
    pl.add_argument(
        "--no-baselines",
        action="store_true",
        help="Do not draw Round Robin / LLQ / LOR / Expert (Mo) horizontal lines",
    )
    pl.set_defaults(func=cmd_plot)

    bl = sub.add_parser(
        "baselines",
        help="Run Vidur once per built-in scheduler (RR, LLQ, LOR, Mo) and write E2E JSON for plots",
    )
    bl.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "scripts/glia_paper_plots/best_codes_vidur_e2e_rerun/vidur_baseline_e2e.json",
    )
    bl.set_defaults(func=cmd_baselines)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
