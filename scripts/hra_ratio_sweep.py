#!/usr/bin/env python3
"""Sweep DECODE_TO_PREFILL_RATIO in HRA.py on Vidur (sarathi @ 7.5 QPS) and plot mean E2E.

Usage:
  python scripts/hra_ratio_sweep.py --run   [--ratios 0.3,0.4,...] [--resume]
  python scripts/hra_ratio_sweep.py --plot  [--results-csv PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

HRA_PATH = REPO_ROOT / "scripts/glia_paper_plots/HRA.py"
OUT_DIR = REPO_ROOT / "scripts/glia_paper_plots/hra_ratio_sweep"
RESULTS_CSV = OUT_DIR / "ratio_sweep_results.csv"
PLOT_PDF = OUT_DIR / "ratio_sweep_mean_e2e.pdf"

DEFAULT_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]


def _substitute_ratio(code: str, ratio: float) -> str:
    pattern = re.compile(r"^(DECODE_TO_PREFILL_RATIO\s*:\s*float\s*=\s*)[-+0-9.eE]+", re.MULTILINE)
    new_code, n = pattern.subn(rf"\g<1>{ratio}", code, count=1)
    if n == 0:
        raise RuntimeError("Could not find DECODE_TO_PREFILL_RATIO assignment in HRA.py")
    return new_code


def _e2e_stats(csv_path: Path):
    df = pd.read_csv(csv_path)
    s = df["request_e2e_time"].astype(float)
    return (
        float(s.min()),
        float(s.mean()),
        float(s.median()),
        float(s.quantile(0.99)),
        float(s.max()),
    )


def cmd_run(ratios: List[float], resume: bool, results_csv: Path) -> None:
    from SystemBench.vidur.env_evaluator import VidurEvaluator
    from Architect.types import Scenario

    from vidur_e2e_rerun import _sanitize_scheduler_code

    scenario = Scenario(name="sarathi_qps7.5", config={"replica_scheduler": "sarathi", "qps": 7.5})
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    base_code = HRA_PATH.read_text(encoding="utf-8")

    rows_by_ratio: dict = {}
    if resume and results_csv.is_file() and results_csv.stat().st_size > 0:
        prev = pd.read_csv(results_csv)
        for _, r in prev.iterrows():
            rows_by_ratio[float(r["ratio"])] = r.to_dict()

    for ratio in ratios:
        if resume and ratio in rows_by_ratio:
            row = rows_by_ratio[ratio]
            if str(row.get("success", "")).lower() in ("true", "1", "yes"):
                print(f"skip (done) ratio={ratio}")
                continue

        code = _sanitize_scheduler_code(_substitute_ratio(base_code, ratio))
        print(f"Running ratio={ratio} ...")
        ev = VidurEvaluator()
        result = ev.run_simulation_with_algorithm_code(code, scenario)
        success = bool(result.get("success"))
        err = result.get("error", "")
        csv_path = ""
        stats = (float("nan"),) * 5
        if success:
            op = result.get("info", {}).get("metrics", {}).get("output_path")
            if op:
                csv_path = str(Path(op).resolve())
                stats = _e2e_stats(Path(csv_path))
        rows_by_ratio[ratio] = {
            "ratio": ratio,
            "success": success,
            "error": err,
            "request_metrics_csv": csv_path,
            "e2e_min": stats[0],
            "e2e_mean": stats[1],
            "e2e_median": stats[2],
            "e2e_p99": stats[3],
            "e2e_max": stats[4],
        }
        df_out = pd.DataFrame(sorted(rows_by_ratio.values(), key=lambda x: float(x["ratio"])))
        df_out.to_csv(results_csv, index=False)
        if success:
            print(f"  ratio={ratio}  e2e_mean={stats[1]:.4f}s")
        else:
            print(f"  ratio={ratio}  FAILED: {err}")

    print(f"Wrote {results_csv}")


def _setup_plot_style() -> None:
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plt.rcParams["axes.labelweight"] = "bold"


def cmd_plot(results_csv: Path, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    ok = df[df["success"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if ok.empty:
        raise RuntimeError("No successful runs in results CSV")
    ok = ok.sort_values("ratio")

    COLOR_LINE    = "#2E86AB"   # steel blue — main sweep line
    COLOR_VLINE   = "#555555"   # neutral charcoal — default marker
    COLOR_MIN     = "#1B998B"   # teal — best (lowest) point
    COLOR_MAX     = "#E84855"   # coral-red — worst (highest) point
    COLOR_ARROW   = "#333333"   # near-black arrow & "Better" label

    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ok["ratio"], ok["e2e_mean"], marker="o", linewidth=2.5, color=COLOR_LINE,
            markerfacecolor="white", markeredgewidth=2)
    ax.set_ylim(bottom=0, top=26)
    ax.axvline(0.6, linestyle="--", color=COLOR_VLINE, alpha=0.5, linewidth=1.4)
    ax.text(0.6, ax.get_ylim()[1], " default=0.6", va="top", ha="left", fontsize=12,
            alpha=0.65, color=COLOR_VLINE)

    min_idx = ok["e2e_mean"].idxmin()
    max_idx = ok["e2e_mean"].idxmax()
    min_x, min_y = float(ok.loc[min_idx, "ratio"]), float(ok.loc[min_idx, "e2e_mean"])
    max_x, max_y = float(ok.loc[max_idx, "ratio"]), float(ok.loc[max_idx, "e2e_mean"])
    ax.plot([min_x], [min_y], marker="*", color=COLOR_MIN, markersize=16, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8,
            label=f"min: ratio={min_x}, mean={min_y:.3f}s")
    ax.plot([max_x], [max_y], marker="X", color=COLOR_MAX, markersize=13, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8,
            label=f"max: ratio={max_x}, mean={max_y:.3f}s")

    ax.set_xlabel("DECODE_TO_PREFILL_RATIO", fontsize=18, fontweight="bold")
    ax.set_ylabel("Avg RT (s)", fontsize=18, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=14)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    arrow_x = xlim[0] + 0.92 * (xlim[1] - xlim[0])
    arrow_y_start = ylim[0] + 0.88 * (ylim[1] - ylim[0])
    arrow_y_end = ylim[0] + 0.73 * (ylim[1] - ylim[0])
    ax.annotate(
        "",
        xy=(arrow_x, arrow_y_end),
        xytext=(arrow_x, arrow_y_start),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=COLOR_ARROW),
    )
    ax.text(
        arrow_x - 0.01 * (xlim[1] - xlim[0]),
        (arrow_y_start + arrow_y_end) / 2,
        "Better",
        rotation=90,
        va="center",
        ha="right",
        fontsize=14,
        fontweight="bold",
        color=COLOR_ARROW,
    )

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def _parse_ratios(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep HRA DECODE_TO_PREFILL_RATIO around 0.6")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run Vidur for each ratio")
    mode.add_argument("--plot", action="store_true", help="Plot mean E2E vs ratio")
    p.add_argument("--ratios", type=_parse_ratios, default=DEFAULT_RATIOS,
                   help="Comma-separated ratios (default: spread around 0.6)")
    p.add_argument("--resume", action="store_true", help="Skip ratios already successful in results CSV")
    p.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    p.add_argument("--out", type=Path, default=PLOT_PDF, help="Output PDF for --plot mode")
    args = p.parse_args()

    if args.run:
        cmd_run(args.ratios, args.resume, args.results_csv)
    else:
        cmd_plot(args.results_csv, args.out)


if __name__ == "__main__":
    main()
