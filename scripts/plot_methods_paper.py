"""
Plotting script for comparing multiple methods across different runs.

USAGE:
    Method 1: Specify individual methods as arguments
        python plot_methods_paper.py \\
            path/to/method1.json:"Method Name 1" \\
            path/to/method2.json:"Method Name 2" \\
            -o output_directory \\
            [--show-raw-scores] \\
            [--max-num-sims 100]

    Method 2: Use an all_methods.json file
        python plot_methods_paper.py \\
            --all-methods-json path/to/all_methods.json \\
            -o output_directory \\
            [--show-raw-scores] \\
            [--max-num-sims 100]

OPTIONS:
    -o, --output-dir PATH       (Required) Directory to save plots
    --all-methods-json PATH     JSON file containing all methods and their runs
    --show-raw-scores           If set, add raw scores to the plots
    --max-num-sims INTEGER      Maximum number of simulations to plot (default: 100)

EXAMPLES:
    # Plot two methods with custom names
    python plot_methods_paper.py \\
        results/best_shot.json:"Best Shot (o3)" \\
        results/evolution.json:"Evolution (o3)" \\
        -o plots/cloudcast_comparison \\
        --max-num-sims 50

    # Plot using all_methods.json format
    python plot_methods_paper.py \\
        --all-methods-json all_methods.json \\
        -o plots/cloudcast_comparison \\
        --show-raw-scores \\
        --max-num-sims 100

all_methods.json format:
    {
        "Method Name 1": [
            "path/to/run1.json",
            "path/to/run2.json"
        ],
        "Method Name 2": [
            "path/to/run1.json"
        ]
    }

OUTPUT:
    The script generates multiple PDF plots in the specified output directory:
    - *_distribution.pdf: Score distribution plot
    - *_stairs.pdf: Best score progression over simulations
    - *_aggregated_simulations.pdf: Aggregated comparison across runs
    - *_aggregated_simulations_with_round_robin.pdf: Comparison with round-robin methods
    - *_aggregated_simulations_with_round_robin_bar.pdf: Bar chart comparison
    - Additional plots for sequential envelopes and best-of-n analysis
"""

import click
import json
import os
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import gaussian_kde
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import colorsys
import re
from itertools import combinations, permutations
import math
import pickle
from stat_utils import bootstrap_ci_mean, bootstrap_ci_best_of_n
import time
import matplotlib
from matplotlib import colors as mcolors
CUT_LENGTH = 100

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

console = Console()
# Global problem name; set from --problem-name or auto-detected from output_dir in main().
problem_name = None

# Plot toggles for optional Glia aggregate variants.
DISABLE_PLOT_PAR = False
DISABLE_PLOT_SEQ = True
# When True, skip plotting lines/bars for the base method 'Glia (o3)' while
# still allowing its data to be processed for derived quantities (e.g., Multi-Context Glia).
SKIP_GLIA_O3_PLOTS = True

# FONT SIZE
BASELINE_FONT_SIZE = 10
BETTER_FONT_SIZE = 14
Y_LABEL_FONT_SIZE = 18
X_LABEL_FONT_SIZE = Y_LABEL_FONT_SIZE
LEGEND_FONT_SIZE = 16

# Score direction: True = higher is better, False = lower is better.
# Add new problems here when adding benchmarks.
HIGHER_IS_BETTER = {
    "cant-be-late": True,   # score = -avg_cost
    "cant-be-late-multi": True,   # score = -avg_cost
    "llm_sql": True,        # accuracy / score
    "cloudcast": False,     # cost
    "eplb": True,          # score (inverted in data)
    "vidur": False,         # RT (response time)
    "prism": True,         # higher score is better
    "telemetry_repair": True,         # higher score is better
    "txn_scheduling": True,         # higher score is better
    "fcs_algorithmic": True,  # Frontier-CS algorithmic problems
    # Frontier-CS (FCS) problems
    "cross_entropy": True,
    "grammar_fuzzing__fuzzer__sql": True,
    "nbody_simulation__random_10k": True,
}
def is_maximize_for_problem(name: str) -> bool:
    return HIGHER_IS_BETTER.get(name, True)

def get_metric_name(prob_name: str) -> str:
    """Get metric name for a given problem."""
    if prob_name == "vidur":
        return "Avg RT (s)"
    elif prob_name == "cloudcast":
        return "Cost ($)"
    elif prob_name == "eplb":
        return "Score"
    elif prob_name == "llm_sql":
        return "Score"
    else:
        return "Score"

def get_baseline_names(prob_name: str) -> Dict[str, str]:
    """Get baseline names for a given problem."""
    if prob_name == "vidur":
        return {
            "baseline_v0": "Random",
            "baseline_v1": "Round Robin",
            "baseline_v2": "LLQ",
            "baseline_v3": "Expert"
        }
    elif prob_name == "cloudcast":
        return {
            # "baseline_v0": "Dijkstra",
            # "baseline_v1": "Direct",
            # "baseline_v2": "MinST",
            "baseline_v3": "Human SOTA",
            # "baseline_v4": "Initial Program",
            # "baseline_v5": "ADRS Expert"
        }
    elif prob_name == "llm_sql":
        return {
            "baseline_v0": "GGR"
        }
    else:
        return {}

def get_baseline_scores(prob_name: str) -> Dict[str, float]:
    """Get baseline scores for a given problem."""
    if prob_name == "vidur":
        return {
            "Random": 20 / 0.3053086850567041,
            "Round Robin": 20 / 0.31575420142147625,
            "LLQ": 20 / 0.36604211306794665,
            "LOR": 59.5712,
            "Expert": 20 / 0.7809078797222101
        }
    elif prob_name == "cloudcast":
        return {
            # "Dijkstra": 1 / 0.0009552371980292827,
            # "Direct": 1 / 0.0008332159929137445,
            # "MinST": 1 / 0.0009552371980292827,
            "Human SOTA": 1 / 0.0015939194701005796,
            # "Initial Program": 1 / 0.0009552371980292827,
            # "ADRS Expert": 629.24
        }
    elif prob_name == "llm_sql":
        return {
            "GGR": 0.677
        }
    else:
        return {}

def extract_top_baselines_from_cache(cache_path: str, top_n: int = 10) -> Dict[str, float]:
    """Extract top N baselines from a baseline_cache.json file.

    Returns dict of {model_name: score} for the top N unique scores.
    """
    with open(cache_path) as f:
        cache = json.load(f)

    entries = []
    for key, val in cache.items():
        score = val.get("score")
        if score is None or score == float("-inf") or score <= 0:
            continue
        # Use the reasoning field as the model name, fall back to key
        label = val.get("reasoning", key)
        label = label.replace(" baseline", "").strip()
        if not label:
            label = key
        entries.append((score, label))

    # Sort descending by score, deduplicate by score
    entries.sort(key=lambda x: x[0], reverse=True)
    seen_scores = set()
    unique = []
    for score, label in entries:
        if score not in seen_scores:
            seen_scores.add(score)
            unique.append((score, label))

    top = unique[:top_n]
    return {label: score for score, label in top}


# Initialize with default problem_name
METRIC_NAME = get_metric_name(problem_name)
BASELINE_NAMES = get_baseline_names(problem_name)
BASELINE_SCORES = get_baseline_scores(problem_name)

# Convert RGB to HSV and filter out hard-to-read hues (red/yellow)
def is_red(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    hue_deg = h * 360
    return hue_deg < 30 or hue_deg > 330  # red tones


def is_yellow(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    hue_deg = h * 360
    return 35 <= hue_deg <= 85  # yellow/yellow-green tones are hard to see on white backgrounds


def get_colors(n_colors=10):
    colors = []
    count = 0
    while len(colors) < n_colors:
        count += 1
        husl_colors = sns.color_palette("husl", count * n_colors)
        for husl_color in husl_colors:
            if not is_red(husl_color) and not is_yellow(husl_color):
                colors.append(husl_color)
                if len(colors) == n_colors:
                    break

    return colors

METHOD_COLORS = {
    "Best Shot (o3)": get_colors(5)[2],
    "Evolution (o3)": "firebrick", #get_colors(5)[1],
    "Glia (o3)": "darkgreen", #get_colors(8)[7],
    "OpenEvolve (o3)": "darkorange", #get_colors(5)[3], #get_colors(5)[1] # get_colors(8)[7],
    "Glia Best of 4 (o3)": "darkgreen", #"magenta",
    "Sequential Glia (o3)": "firebrick",
    "One-shot (o3)": "blue",
    "Tree (o3)": "darkgreen",
    "Tree (gpt-5.2)": "teal",
    "Handoff (o3)": "navy",
    "OpenEvolve GGR Normal (o3)": "cyan",
    "OpenEvolve GGR Ours (o3)": "lightgreen",
    "OpenEvolve GGR ADRS (o3)": "lightblue",
    "Handoff Normal (o3)": "purple",
    "Handoff GGR Ours (o3)": "darkgreen",
    "Handoff GGR ADRS (o3)": "darkorange",
    "o3": "blue",
    "gpt-4o": "maroon",
    "gpt-5.2": "darkgreen",
    "gpt-5.2-xhigh": "darkorange",
    "Single Agent": "black",
    "Engram w/ Direction (o3)": "navy",
    "OE w/ Direction (o3)": "darkorange",
    "Engram Minimal (o3)": "navy",
    "OE Minimal (o3)": "darkorange",
    "No Journal": "firebrick",
    "Sequential": "green",
    # llm_sql short-name keys
    "Engram": "navy",
    "OpenEvolve": "darkorange",
    "FunSearch": get_colors(5)[2],
    "Engram GGR": "navy",
    "OpenEvolve GGR": "darkorange",
    "Engram GGR ADRS": "navy",
    "OpenEvolve GGR ADRS": "darkorange",
    "Engram (gpt-5.2-xhigh)": "navy",
    "OpenEvolve (gpt-5.2-xhigh)": "darkorange",
}

METHOD_COLORS["With Supervisor"] = METHOD_COLORS["Glia (o3)"]
METHOD_COLORS["Original"] = METHOD_COLORS["Handoff (o3)"]

METHOD_NAMES = {
    "Best Shot (o3)": "FunSearch",
    "Evolution (o3)": "EoH",
    "Glia (o3)": "Single-Context Glia",
    "OpenEvolve (o3)": "OpenEvolve",
    "Glia Best of 4 (o3)": "Glia",
    "Sequential Glia (o3)": "Glia",
    "One-shot (o3)": "One-shot (o3)",
    "Tree (o3)": "Tree (o3)",
    "Tree (gpt-5.2)": "Tree (gpt-5.2)",
    "Tree (o3) with hints": "Tree (o3) with hints",
    "Tree (o3) no continue message": "Tree (o3) no continue message",
    "Handoff (o3)": "Engram",
    "OpenEvolve GGR Normal (o3)": "OpenEvolve GGR Normal",
    "OpenEvolve GGR Ours (o3)": "OpenEvolve GGR Ours",
    "OpenEvolve GGR ADRS (o3)": "OpenEvolve GGR ADRS",
    "Handoff Normal (o3)": "Handoff Normal",
    "Handoff GGR Ours (o3)": "Handoff GGR Ours",
    "Handoff GGR ADRS (o3)": "Handoff GGR ADRS",
    "o3": "o3",
    "gpt-4o": "gpt-4o",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-xhigh": "gpt-5.2-xhigh",
    # llm_sql short-name keys
    "Engram": "Engram",
    "OpenEvolve": "OpenEvolve",
    "Evolution": "EoH",
    "FunSearch": "FunSearch",
    "Engram GGR": "Engram GGR",
    "OpenEvolve GGR": "OpenEvolve GGR",
    "Engram GGR ADRS": "Engram GGR ADRS",
    "OpenEvolve GGR ADRS": "OpenEvolve GGR ADRS",
    "Engram (gpt-5.2-xhigh)": "Engram (gpt-5.2-xhigh)",
    "OpenEvolve (gpt-5.2-xhigh)": "OpenEvolve (gpt-5.2-xhigh)",
}

METHOD_LINES = {
    "Evolution (o3)": (0, (5, 2, 1, 2, 1, 2)),
    "Best Shot (o3)": "dashdot", 
    "Glia (o3)": "dashed", #(0, (5, 2)),
    "OpenEvolve (o3)": (0, (1, 1)),
    "Glia Best of 4 (o3)": "dotted",
    "Sequential Glia (o3)": (0, (2, 1, 1, 1)), #"solid",
    "One-shot (o3)": "solid",
    "Tree (o3)": "solid",
    "Tree (gpt-5.2)": "dashed",
    "Tree (o3) with hints": "dotted",
    "Handoff (o3)": "solid",
    "Tree (o3) no continue message": "dashed",
    "OpenEvolve GGR Normal (o3)": (0, (1, 1)),
    "OpenEvolve GGR Ours (o3)": "solid",
    "OpenEvolve GGR ADRS (o3)": "dashed",
    "Handoff Normal (o3)": "dotted",
    "Handoff GGR Ours (o3)": "solid",
    "Handoff GGR ADRS (o3)": "dashed",
    "o3": "solid",
    "gpt-4o": "dashed",
    "gpt-5.2": "dotted",
    "gpt-5.2-xhigh": "dashdot",
    "No Journal": (0, (1, 1)),
    # llm_sql short-name keys
    "Engram": "solid",
    "OpenEvolve": (0, (1, 1)),
    "Evolution": (0, (5, 2, 1, 2, 1, 2)),
    "FunSearch": "dashdot",
    "Engram GGR": "solid",
    "OpenEvolve GGR": (0, (1, 1)),
    "Engram GGR ADRS": "solid",
    "OpenEvolve GGR ADRS": (0, (1, 1)),
    "Engram (gpt-5.2-xhigh)": "solid",
    "OpenEvolve (gpt-5.2-xhigh)": (0, (1, 1)),
    }

METHOD_LINES["Original"] = METHOD_LINES["Handoff (o3)"]
METHOD_ORDER = ["Handoff Normal (o3)",
                "Handoff GGR Ours (o3)",
                "Handoff GGR ADRS (o3)",
                "OpenEvolve GGR Normal (o3)",
                "OpenEvolve GGR Ours (o3)",
                "OpenEvolve GGR ADRS (o3)",
                "Handoff (o3)",
                "Tree (o3)",
                "Tree (gpt-5.2)",
                "Glia Best of 4 (o3)",
                "Sequential Glia (o3)",
                "Glia (o3)",
                "OpenEvolve (o3)",
                "Best Shot (o3)",
                "Evolution (o3)",
                "One-shot (o3)"]

# Pool of line styles for auto-assigning to new methods (must be hashable for set membership)
_LINE_STYLE_POOL = [
    "solid", "dashed", "dashdot", "dotted",
    (0, (1, 1)), (0, (5, 2)), (0, (2, 1, 1, 1)), (0, (5, 2, 1, 2, 1, 2)),
    (0, (3, 1, 1, 1)), (0, (1, 2)), (0, (4, 2, 1, 2)),
]


def _color_to_rgb(c):
    """Convert color to (r, g, b) tuple in [0,1] for distance computation."""
    if isinstance(c, str):
        return tuple(mcolors.to_rgb(c))
    if hasattr(c, "__iter__") and len(c) >= 3:
        return tuple(c)[:3]
    return (0.5, 0.5, 0.5)


def _color_distance_rgb(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance in RGB space (0-1). Larger = more distinct."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _pick_most_distinct_color(used_rgbs: List[Tuple[float, float, float]]) -> Union[Tuple[float, float, float], str]:
    """Pick a color that maximizes minimum distance to all used colors. Excludes red/yellow hues."""
    # Candidate pool: evenly spaced hues (non-red, non-yellow), high saturation, medium-high value for visibility
    n_hues = 36
    candidates = []
    for i in range(n_hues):
        hue_deg = (30 + (i / n_hues) * 300) % 360  # skip red: start at 30°, span 300°
        if 35 <= hue_deg <= 85:
            continue
        hue = hue_deg / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        candidates.append((r, g, b))
    if not candidates:
        return "gray"
    if not used_rgbs:
        return candidates[0]
    used = [(_color_to_rgb(c) if not isinstance(c, tuple) else c) for c in used_rgbs]
    best_c = None
    best_min_dist = -1.0
    for c in candidates:
        min_d = min(_color_distance_rgb(c, u) for u in used)
        if min_d > best_min_dist:
            best_min_dist = min_d
            best_c = c
    return best_c if best_c is not None else candidates[0]


def add_better_direction_arrow(ax, is_maximize: bool) -> None:
    """Add a high-contrast vertical direction arrow with label."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    arrow_x = xlim[1] - 0.08 * x_span

    if is_maximize:
        # Higher is better: arrow points upward
        arrow_y_start = ylim[0] + 0.35 * y_span
        arrow_y_end = ylim[0] + 0.58 * y_span
    else:
        # Lower is better: arrow points downward
        arrow_y_start = ylim[0] + 0.58 * y_span
        arrow_y_end = ylim[0] + 0.35 * y_span

    ax.annotate(
        "",
        xy=(arrow_x, arrow_y_end),
        xytext=(arrow_x, arrow_y_start),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.4,
            color="black",
            mutation_scale=18,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
        ),
        zorder=6,
    )
    ax.text(
        arrow_x - 0.015 * x_span,
        (arrow_y_start + arrow_y_end) / 2,
        "Better",
        rotation=90,
        va="center",
        ha="right",
        fontsize=BETTER_FONT_SIZE,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        zorder=7,
    )


def _ensure_method_style(method_key: str, existing_keys_for_plot: List[str] = None) -> None:
    """
    If method_key is missing from METHOD_* dicts/lists, add a non-existing color, name, line style, and order.
    
    existing_keys_for_plot:
        Optional list of method keys that are actually being plotted together in the current figure.
        When provided, we pick a new color that is maximally distinct from the colors of just these
        methods (plus any already-assigned new methods), instead of all global METHOD_COLORS.
    """
    if method_key not in METHOD_COLORS:
        if existing_keys_for_plot:
            used_rgbs = [
                _color_to_rgb(METHOD_COLORS[k])
                for k in existing_keys_for_plot
                if k in METHOD_COLORS
            ]
        else:
            used_rgbs = [_color_to_rgb(v) for v in METHOD_COLORS.values()]
        new_color = _pick_most_distinct_color(used_rgbs)
        METHOD_COLORS[method_key] = new_color

    if method_key not in METHOD_NAMES:
        METHOD_NAMES[method_key] = method_key

    if method_key not in METHOD_LINES:
        used_lines = set(METHOD_LINES.values())
        for ls in _LINE_STYLE_POOL:
            if ls not in used_lines:
                METHOD_LINES[method_key] = ls
                break
        else:
            METHOD_LINES[method_key] = "solid"

    if method_key not in METHOD_ORDER:
        METHOD_ORDER.append(method_key)


def order_methods(grouped: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Order methods: first those in METHOD_ORDER (in METHOD_ORDER's order), 
    then others in their original JSON/data order.
    
    Parameters
    ----------
    grouped : Dict[str, List[Dict[str, Any]]]
        Dictionary of method names to their run data lists
        
    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Ordered dictionary with methods in METHOD_ORDER first, then others in original order
    """
    grouped = {
        method: runs
        for method, runs in grouped.items()
        if not _should_skip_method(method)
    }

    # Preserve original order from grouped dict (which comes from JSON order)
    original_order = list(grouped.keys())
    
    # Build ordered list: METHOD_ORDER items first, then others in original order
    ordered_methods = []
    methods_in_order = set()
    
    # Add methods in METHOD_ORDER first (in METHOD_ORDER's order)
    for method in METHOD_ORDER:
        if method in grouped:
            ordered_methods.append(method)
            methods_in_order.add(method)
    
    # Add remaining methods in their original order
    for method in original_order:
        if method not in methods_in_order:
            ordered_methods.append(method)
    
    # Create new ordered dictionary
    return {method: grouped[method] for method in ordered_methods}


def _should_skip_method(method: str) -> bool:
    """Whether a method should be excluded from plotting based on toggles."""
    if method == "Glia Best of 4 (o3)":
        return DISABLE_PLOT_PAR
    if method == "Sequential Glia (o3)":
        return DISABLE_PLOT_SEQ
    return False

def ensure_all_method_styles(method_keys) -> None:
    """
    Ensure every method key has entries in METHOD_COLORS, METHOD_NAMES, METHOD_LINES, and METHOD_ORDER.
    
    Colors for any newly encountered methods are chosen to be maximally distinct from the colors of
    the methods that actually appear together in the current plot (method_keys), instead of from all
    methods globally. This helps avoid collisions like a new method ending up too close to Glia SCG.
    """
    method_keys = list(method_keys)
    # Start with any methods that already have colors assigned among this set
    existing_keys_for_plot: List[str] = [k for k in method_keys if k in METHOD_COLORS]
    for key in method_keys:
        _ensure_method_style(key, existing_keys_for_plot=existing_keys_for_plot)
        if key not in existing_keys_for_plot:
            existing_keys_for_plot.append(key)

def is_eoh_method(method_key: str) -> bool:
    """Return True when a method is named EOH/EoH."""
    display_name = METHOD_NAMES.get(method_key, method_key)
    normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return normalize(method_key) == "eoh" or normalize(display_name) == "eoh"


def is_linear_method(method_key: str) -> bool:
    """Return True for linear-agent style method names."""
    display_name = METHOD_NAMES.get(method_key, method_key)
    normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return "linear" in normalize(method_key) or "linear" in normalize(display_name)


def is_no_journal_method(method_key: str) -> bool:
    """Return True for no-journal style method names (e.g. Handoff no journal)."""
    display_name = METHOD_NAMES.get(method_key, method_key)
    normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    nkey, ndisplay = normalize(method_key), normalize(display_name)
    return "nojournal" in nkey or "nojournal" in ndisplay

def is_SCG_method(method_key: str) -> bool:
    if method_key == "Glia (o3)":
        return True
    return False

def should_expand_method_to_max_sims(method: str, method_max_length: float, max_num_sims: int) -> bool:
    """True when this method should be extended to max_num_sims (e.g. EOH, linear, no-journal)."""
    return (
        (is_eoh_method(method) or is_linear_method(method) or is_no_journal_method(method))
        and method_max_length < max_num_sims and not is_SCG_method(method)
    )


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "text.usetex": False,  # Disable LaTeX rendering
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "font.family": "sans-serif",
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "figure.dpi": 100,  # Higher base DPI
            "savefig.dpi": 600,  # Much higher save DPI
            "lines.linewidth": 2.5,  # Thicker lines
            "lines.markersize": 8,  # Larger markers
            "grid.alpha": 0.3,  # More visible grid
            "axes.linewidth": 1.2,  # Thicker axes
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plt.rcParams['axes.labelweight'] = 'bold'

def extract_scores(data: Union[Dict, List]) -> List[float]:
    """Extract scores from different possible data structures."""
    scores = []

    if isinstance(data, dict):
        # Try different possible structures
        if "all_iterations" in data:
            scores = [item.get("score", 0) for item in data["all_iterations"] if isinstance(item, dict)]

        elif "baselines" in data:
            # Skip baselines for now as they're handled separately
            pass

        else:
            # Try to extract scores from any nested dictionaries
            for key, value in data.items():
                if isinstance(value, dict) and "score" in value:
                    scores.append(value["score"])
                elif isinstance(value, (int, float)):
                    scores.append(value)

            if not scores:
                # If no scores found, try to find them in nested structures
                for key, value in data.items():
                    if isinstance(value, dict):
                        try:
                            nested_scores = extract_scores(value)
                            scores.extend(nested_scores)
                        except ValueError:
                            continue

    elif isinstance(data, list):
        # If it's a list, try different possible structures
        for item in data:
            if isinstance(item, dict) and "score" in item:
                scores.append(item["score"])
            elif isinstance(item, (int, float)):
                scores.append(item)

    # Convert all scores to float and filter out None values
    scores = [float(score) for score in scores if score is not None]
    # remove inf
    scores = [score for score in scores if score != float("inf") and score != float("-inf")]
    if not scores:
        print("WARNING: No scores found in the data structure")
        raise ValueError("Could not find scores in the data structure")

    return scores


def extract_baseline_scores(data: Dict) -> Dict[str, float]:
    """Extract baseline scores from the data."""
    baseline_scores = {}
    if isinstance(data, dict) and "baselines" in data:
        for key, value in data["baselines"].items():
            if key not in BASELINE_NAMES:
                continue
            if isinstance(value, dict) and "score" in value:
                score = BASELINE_SCORES[BASELINE_NAMES[key]] #value["score"]
                if score is not None and score != float("inf") and score != float("-inf"):
                    baseline_scores[BASELINE_NAMES[key]] = float(score)
    for key in BASELINE_SCORES:
        if key not in baseline_scores:
            baseline_scores[key] = BASELINE_SCORES[key]
    if 'Random' in baseline_scores:
        # remove Random
        del baseline_scores['Random']
    return baseline_scores


def extract_costs(data: Union[Dict, List]) -> List[float]:
    """Extract costs from different possible data structures."""
    costs = []
    print("\nDEBUG: Extracting costs from data structure:")

    if isinstance(data, dict):
        if "all_iterations" in data:
            costs = []
            for item in data["all_iterations"]:
                if isinstance(item, dict) and "usage_stats" in item:
                    cost = item["usage_stats"].get("total_cost", 0)
                    costs.append(cost)
        else:
            print("No 'all_iterations' found")
            # Try to find usage_stats directly
            if "usage_stats" in data and "total_cost" in data["usage_stats"]:
                costs.append(data["usage_stats"]["total_cost"])
                print(f"Found cost in root usage_stats: {data['usage_stats']['total_cost']}")

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "usage_stats" in item and "total_cost" in item["usage_stats"]:
                    costs.append(item["usage_stats"]["total_cost"])
                    print(f"Found cost in list item: {item['usage_stats']['total_cost']}")

    # Convert all costs to float and filter out None values
    costs = [float(cost) for cost in costs if cost is not None]
    # remove inf
    costs = [cost for cost in costs if cost != float("inf") and cost != float("-inf")]
    if not costs:
        print("WARNING: No costs found in the data structure")
        raise ValueError("Could not find costs in the data structure")

    return costs


def extract_scores_and_costs(data: Union[Dict, List], method_label: str = "") -> Tuple[List[float], List[float], int]:
    """Extract scores and costs together to ensure they stay aligned."""
    scores = []
    costs = []
    initial_num_simulations = 0

    if isinstance(data, dict):
        if "all_iterations" in data:
            initial_num_simulations = len(data["all_iterations"])
            skipped_invalid_cost = 0
            skipped_invalid_score = 0
            for item in data["all_iterations"]:
                if isinstance(item, dict):
                    score = item.get("score")
                    cost = item.get("usage_stats", {}).get("total_cost", 0)
                    score_valid = (
                        score is not None
                        and score != float("inf")
                        and score != float("-inf")
                        and score != float("nan")
                    )
                    cost_valid = (
                        cost is not None
                        and cost != float("inf")
                        and cost != float("-inf")
                        and cost != float("nan")
                    )

                    # Only append if both score and cost are valid
                    if score_valid and cost_valid:
                        scores.append(float(score))
                        costs.append(float(cost))
                    else:
                        if score_valid and not cost_valid:
                            skipped_invalid_cost += 1
                        elif not score_valid:
                            skipped_invalid_score += 1

    if not scores or not costs:
        raise ValueError("Could not find valid score-cost pairs in the data structure")

    return scores, costs, initial_num_simulations


def print_best_solution(data: Dict[str, Any], method_name: str, save_path: str, is_maximize: bool = False) -> None:
    """Print the best solution code for a method in a nice box."""
    if not isinstance(data, dict) or "all_iterations" not in data:
        return

    # Find the best score and its corresponding solution
    best_score = float("-inf") if is_maximize else float("inf")
    best_code = None

    for item in data["all_iterations"]:
        if isinstance(item, dict):
            score = item.get("score")
            code = item.get("code")
            if score is not None and score != float("inf") and code is not None:
                if score > best_score if is_maximize else score < best_score:
                    best_score = score
                    best_code = code

    best_code += "\n\n"
    best_code += "# score = " + str(best_score)
    best_code += "\n\n"
    best_code += "# method_name = " + str(method_name)

    if best_code is not None:
        console.print(f"\n[bold cyan]Best Solution for {method_name}[/bold cyan] (Score: {best_score:.4f})")
        with open(f"{save_path}_{method_name}_best_solution.py", "w") as f:
            f.write(best_code)
        syntax = Syntax(best_code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, expand=False, border_style="cyan"))
    return best_code, best_score


def _is_valid_numeric_value(v: Any) -> bool:
    return v is not None and v != float("inf") and v != float("-inf") and v != float("nan")


def save_best_code_per_base_method(
    data: Dict[str, Any],
    save_path: str,
    is_maximize: bool,
    max_num_sims: int = None,
) -> None:
    """
    For each base method (e.g., 'Handoff (o3)', 'OpenEvolve (o3)'), find the single
    best code snippet across all its runs and save it as a standalone .py file.
    """
    # Group runs by base method name (strip trailing 'Run N')
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for run_name, run_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", run_name)
        grouped.setdefault(base_name, []).append((run_name, run_data))

    for base_name, runs in grouped.items():
        best_score = float("-inf") if is_maximize else float("inf")
        best_code = None
        best_run_name = None
        best_item_cost = None
        best_item_cost_valid = False

        for run_name, run_data in runs:
            if not isinstance(run_data, dict) or "all_iterations" not in run_data:
                continue
            valid_items = []
            for item in run_data["all_iterations"]:
                if not isinstance(item, dict):
                    continue
                score = item.get("score")
                code = item.get("code")
                # Keep semantics aligned with extract_scores_and_costs(): missing cost defaults to 0.
                cost = item.get("usage_stats", {}).get("total_cost", 0)
                score_valid = _is_valid_numeric_value(score)
                cost_valid = _is_valid_numeric_value(cost)
                if not score_valid or not cost_valid or code is None:
                    continue
                valid_items.append((float(score), code, cost, cost_valid))
            if max_num_sims is not None:
                valid_items = valid_items[:max_num_sims]
            for score, code, cost, cost_valid in valid_items:
                if (score > best_score) if is_maximize else (score < best_score):
                    best_score = score
                    best_code = code
                    best_run_name = run_name
                    best_item_cost = cost
                    best_item_cost_valid = cost_valid

        if best_code is None:
            continue

        annotated_code = best_code
        annotated_code += "\n\n"
        annotated_code += f"# score = {best_score}\n"
        annotated_code += f"# base_method_name = {base_name}\n"
        if best_run_name is not None and best_run_name != base_name:
            annotated_code += f"# run_name = {best_run_name}\n"

        # Sanitize base_name for filesystem safety (keep simple readable characters)
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name)
        file_path = f"{save_path}_{safe_base}_best_solution.py"
        with open(file_path, "w") as f:
            f.write(annotated_code)
        console.print(
            f"[bold cyan]Saved best solution for {base_name}[/bold cyan] "
            f"to {file_path} (Score: {best_score:.4f})"
        )

def analyze_expert_crossover(data: Dict[str, Any], max_num_sims: int = None, percent_expert: float = 0.95) -> None:
    """Analyze when each run first surpasses the Expert baseline score."""
    # First get the Expert baseline score from any of the runs
    expert_score = None
    for method_name, method_data in data.items():
        baselines = extract_baseline_scores(method_data)
        if "Expert" in baselines:
            expert_score = baselines["Expert"] * percent_expert
            break

    if expert_score is None:
        print("No Expert baseline score found in any run")
        return

    print(f"\nAnalyzing when runs surpass *{percent_expert}* Expert baseline score ({expert_score:.4f}):")
    print("=" * 80)

    for method_name, method_data in data.items():
        try:
            scores, _, _ = extract_scores_and_costs(method_data)

            # Limit number of simulations if specified
            if max_num_sims is not None:
                scores = scores[:max_num_sims]

            # Calculate best scores so far at each point
            best_so_far = [scores[0]]
            for score in scores[1:]:
                best_so_far.append(max(best_so_far[-1], score))

            # Find first point that surpasses expert
            for i, score in enumerate(best_so_far):
                if score > expert_score:
                    print(f"{method_name}: First surpassed {percent_expert} * Expert baseline at simulation {i+1} with score {score:.4f}")
                    break
            else:
                print(f"{method_name}: Never surpassed {percent_expert} * Expert baseline within first {len(scores)} simulations")

        except ValueError as e:
            print(f"Warning: Could not analyze {method_name}: {e}")
    print("=" * 80)


def get_envelope_curve(scores: List[float], is_maximize: bool = False) -> List[float]:
    """Get the envelope curve of the scores."""
    if is_maximize:
        envelop = [scores[0]]
        for score in scores[1:]:
            envelop.append(max(envelop[-1], score))
        return envelop
    else:
        envelop = [scores[0]]
        for score in scores[1:]:
            envelop.append(min(envelop[-1], score))
        return envelop


def compute_fastness_metrics_for_envelope(
    envelope: List[float],
    is_maximize: bool,
    alphas: List[float] = None,
) -> Dict[str, float]:
    """
    Compute "fastness" metrics from a best-so-far (envelope) curve.

    Metrics:
      - T_alpha: first simulation index (1-based) where progress >= alpha, where progress is measured
        relative to the run's start and end (final) envelope value.
      - norm_auc: area under *progress* curve / horizon (in [0,1] when there is nonzero improvement).

    This definition works for both maximize and minimize problems by using a normalized progress
    fraction from start -> final.
    """
    if alphas is None:
        alphas = [0.95, 0.99]
    if not envelope:
        return {f"T_{int(a*100)}": float("nan") for a in alphas} | {"norm_auc": float("nan")}

    start = float(envelope[0])
    final = float(envelope[-1])
    denom = (final - start) if is_maximize else (start - final)

    # If no improvement (or degenerate), define T_alpha as 1 and norm_auc as 0.
    if denom == 0:
        out = {f"T_{int(a*100)}": 1.0 for a in alphas}
        out["norm_auc"] = 0.0
        return out

    # Progress fraction p(t) in [..] (clipped to [0,1] for stability).
    if is_maximize:
        progress = [(v - start) / denom for v in envelope]
    else:
        progress = [(start - v) / denom for v in envelope]
    progress = [min(1.0, max(0.0, float(p))) for p in progress]

    # Hitting times (1-based simulation count)
    out: Dict[str, float] = {}
    for a in alphas:
        t = None
        for i, p in enumerate(progress):
            if p >= a:
                t = i + 1
                break
        out[f"T_{int(a*100)}"] = float(t if t is not None else len(progress))

    # Normalized AUC of progress curve (higher => reaches final faster)
    # Use trapezoid rule over indices 1..T with unit spacing, then normalize by horizon length.
    if len(progress) == 1:
        out["norm_auc"] = progress[0]
    else:
        # trapezoid is the modern name (NumPy 2.0+); trapz existed in NumPy 1.x
        trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = trapezoid_fn(progress, dx=1.0)
        out["norm_auc"] = float(auc / (len(progress) - 1))
    return out


def compute_and_save_fastness_table(
    data: Dict[str, Any],
    save_path: str,
    is_maximize: bool,
    alphas: List[float] = None,
    max_num_sims: int = 1000,
) -> None:
    """
    Compute per-run fastness metrics for each base method (strip 'Run N'),
    aggregate with median (and IQR), and save a CSV table.
    """
    if alphas is None:
        alphas = [0.95, 0.99]

    # Group runs by base method
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run_name, run_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", run_name)
        grouped.setdefault(base_name, []).append(run_data)

    rows = []
    for method, runs in grouped.items():
        per_run = []
        finals = []
        for run in runs:
            try:
                scores, _, _ = extract_scores_and_costs(run)
            except ValueError:
                continue
            env = get_envelope_curve(scores, is_maximize)
            clipped_env = env[:min(len(env), max_num_sims)]
            finals.append(float(clipped_env[-1]))
            per_run.append(compute_fastness_metrics_for_envelope(clipped_env, is_maximize, alphas=alphas))

        if not per_run:
            continue

        # Aggregate with median + IQR for robustness
        def med_iqr(vals: List[float]) -> Tuple[float, float, float, float]:
            v = np.array([x for x in vals if not (np.isnan(x) or np.isinf(x))], dtype=float)
            if v.size == 0:
                return float("nan"), float("nan"), float("nan"), float("nan")
            q1 = float(np.percentile(v, 25))
            med = float(np.percentile(v, 50))
            mean = float(np.mean(v))
            q3 = float(np.percentile(v, 75))
            return med, mean, q1, q3

        row = {"method": method}
        med_final, mean_final, q1_final, q3_final = med_iqr(finals)
        row["final_best_median"] = med_final
        row["final_best_mean"] = mean_final
        row["final_best_q1"] = q1_final
        row["final_best_q3"] = q3_final

        for a in alphas:
            key = f"T_{int(a*100)}"
            vals = [r[key] for r in per_run]
            med, mean, q1, q3 = med_iqr(vals)
            row[f"{key}_median"] = med
            row[f"{key}_mean"] = mean
            row[f"{key}_q1"] = q1
            row[f"{key}_q3"] = q3

        auc_vals = [r["norm_auc"] for r in per_run]
        med, mean, q1, q3 = med_iqr(auc_vals)
        row["norm_auc_median"] = med
        row["norm_auc_mean"] = mean
        row["norm_auc_q1"] = q1
        row["norm_auc_q3"] = q3

        rows.append(row)

    # Sort by "fastness": higher norm_auc first (ties broken by smaller T_95)
    def sort_key(r):
        t95 = r.get("T_95_median", float("inf"))
        auc = r.get("norm_auc_median", float("-inf"))
        mean = r.get("norm_auc_mean", float("-inf"))
        return (-auc, mean, t95)
    rows.sort(key=sort_key)

    out_csv = f"{save_path}_fastness_metrics.csv"
    with open(out_csv, "w") as f:
        if not rows:
            f.write("method,final_best_median,final_best_mean,final_best_q1,final_best_q3,T_95_median,T_95_mean,T_95_q1,T_95_q3,T_99_median,T_99_mean,T_99_q1,T_99_q3,norm_auc_median,norm_auc_mean,norm_auc_q1,norm_auc_q3\n")
            return
        # Header from keys (stable ordering)
        keys = [
            "method",
            "final_best_median","final_best_mean","final_best_q1","final_best_q3",
            "T_95_median","T_95_mean","T_95_q1","T_95_q3",
            "T_99_median","T_99_mean","T_99_q1","T_99_q3",
            "T_99_median","T_99_q1","T_99_q3",
            "norm_auc_median","norm_auc_mean","norm_auc_q1","norm_auc_q3",
        ]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"[fastness] Wrote fastness table to: {out_csv}")


def append_multi_context_glia_fastness_row(
    out_csv: str,
    mcg_envelope: List[float],
    is_maximize: bool,
    alphas: List[float],
) -> None:
    """
    Append a fastness row for Multi-Context Glia (MCG) based on a single aggregated
    best-of-n envelope (e.g., best-of-4 Glia curve).
    """
    if not mcg_envelope:
        return

    env = [float(v) for v in mcg_envelope]
    metrics = compute_fastness_metrics_for_envelope(env, is_maximize, alphas=alphas)

    final = float(env[-1])
    # With a single envelope, median/mean/Q1/Q3 all coincide.
    med_final = mean_final = q1_final = q3_final = final

    T95 = metrics.get("T_95", float("nan"))
    T99 = metrics.get("T_99", float("nan"))
    auc = metrics.get("norm_auc", float("nan"))

    T95_med = T95_mean = T95_q1 = T95_q3 = T95
    T99_med = T99_mean = T99_q1 = T99_q3 = T99
    auc_med = auc_mean = auc_q1 = auc_q3 = auc

    row = {
        "method": "Multi-Context Glia",
        "final_best_median": med_final,
        "final_best_mean": mean_final,
        "final_best_q1": q1_final,
        "final_best_q3": q3_final,
        "T_95_median": T95_med,
        "T_95_mean": T95_mean,
        "T_95_q1": T95_q1,
        "T_95_q3": T95_q3,
        "T_99_median": T99_med,
        "T_99_mean": T99_mean,
        "T_99_q1": T99_q1,
        "T_99_q3": T99_q3,
        "norm_auc_median": auc_med,
        "norm_auc_mean": auc_mean,
        "norm_auc_q1": auc_q1,
        "norm_auc_q3": auc_q3,
    }

    # Match the same column order (including the duplicated T_99 fields) used in
    # compute_and_save_fastness_table.
    keys = [
        "method",
        "final_best_median", "final_best_mean", "final_best_q1", "final_best_q3",
        "T_95_median", "T_95_mean", "T_95_q1", "T_95_q3",
        "T_99_median", "T_99_mean", "T_99_q1", "T_99_q3",
        "T_99_median", "T_99_q1", "T_99_q3",
        "norm_auc_median", "norm_auc_mean", "norm_auc_q1", "norm_auc_q3",
    ]

    with open(out_csv, "a") as f:
        f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def plot_comparison(data: Dict[str, Any], save_path: str, show_raw_scores: bool = True, max_num_sims: int = None, is_maximize: bool = False):
    """Plot comparison between different methods."""
    setup_plot_style()

    # # Print best solutions for each method
    # console.print("\n[bold]Best Solutions[/bold]")
    # console.print("=" * 80)
    # for method_name, method_data in data.items():
    #     _, _ = print_best_solution(method_data, method_name, save_path, is_maximize=is_maximize)
    # console.print("=" * 80 + "\n")

    # Analyze when runs surpass Expert baseline
    # analyze_expert_crossover(data, max_num_sims)

    # Create three subplots: distribution, time-based stairs, and cost-based stairs
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot settings for all axes
    for ax in [ax1, ax2, ax3]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, alpha=0.2)

    # Get all scores, costs, and baselines for range calculation
    all_scores = []
    all_costs = []
    all_baselines = {}
    method_data_pairs = {}  # Store aligned score-cost pairs for each method

    for method_name, method_data in data.items():
        base_for_skip = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        if SKIP_GLIA_O3_PLOTS and base_for_skip == "Glia (o3)":
            continue
        try:
            scores, costs, initial_num_simulations = extract_scores_and_costs(method_data, method_label=method_name)

            # Limit number of simulations if specified
            if max_num_sims is not None:
                scores = scores[:max_num_sims]
                costs = costs[:max_num_sims]
                initial_num_simulations = min(initial_num_simulations, max_num_sims)

            print()
            print(f"Scores for {method_name}: min {min(scores)}, max {max(scores)}")
            print(f"Costs for {method_name}: min {min(costs)}, max {max(costs)}")
            print(f"total cost for {method_name}: {sum(costs)}")
            print(f"average score for {method_name}: {sum(scores) / len(scores)}")
            print(f"Total number of simulations for {method_name}: {initial_num_simulations}")
            all_scores.extend(scores)
            all_costs.extend(costs)
            method_data_pairs[method_name] = (scores, costs)

            # Extract baselines from each method's data
            baselines = extract_baseline_scores(method_data)
            if baselines:
                all_baselines.update(baselines)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    if not all_scores:
        raise ValueError("No valid scores found in any method")

    if all_baselines:
        print("\nBaseline scores:")
        for name, score in all_baselines.items():
            print(f"{name}: {score}")
    print()

    # Include baselines in range calculation
    all_values = all_scores + list(all_baselines.values())
    range_data = (min(all_values), max(all_values))
    # Plot distributions (left subplot)
    for method_name, (scores, _) in method_data_pairs.items():
        # Strip "Run X" suffix for color lookup
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        color = METHOD_COLORS[base_name]
        try:
            # Compute and plot KDE
            kde = gaussian_kde(scores, bw_method="scott")
            kde.set_bandwidth(kde.factor * 0.5)
            x_kde = np.linspace(range_data[0], range_data[1], 1000)
            y_kde = kde(x_kde)

            # Plot the distribution
            ax1.plot(x_kde, y_kde, label=method_name, color=color, linewidth=2)
            ax1.fill_between(x_kde, y_kde, color=color, alpha=0.2)
        except ValueError as e:
            print(f"Warning: Could not plot distribution for {method_name}: {e}")
            continue

    # Add baseline vertical lines to distribution plot
    if all_baselines:
        ytop = ax1.get_ylim()[1]
        for name, score in all_baselines.items():
            ax1.axvline(x=score, color="black", linestyle="--", alpha=0.5, label=None)
            ax1.text(score, ytop, name, rotation=90, ha="right", va="top", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

    LineStyles = ["solid", "dashed", "dotted", "dashdot", (0, (1, 1)), (0, (2, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 2, 1, 2, 1, 2)), "dotted", "dotted", "dotted"]
    count = 0
    # Plot time-based stairs (middle subplot)
    for method_name, (scores, _) in method_data_pairs.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        color = METHOD_COLORS[base_name]
        try:
            # Calculate score progression (best scores so far)
            score_generations = get_envelope_curve(scores, is_maximize)
            score_path = f"{save_path}_score_generations.txt"
            with open(score_path, "a") as f:
                last_score = score_generations[-1]
                f.write(f"{method_name} {last_score}\n")
            # Plot best scores
            ax2.stairs(
                score_generations,
                np.arange(len(score_generations) + 1),
                edgecolor=color,
                linewidth=2,
                baseline=None,
                label=f"{method_name} (best)",
                linestyle=LineStyles[count % len(LineStyles)],
            )

            # Plot raw scores
            if show_raw_scores:
                ax2.stairs(
                    scores,
                    np.arange(len(scores) + 1),
                    edgecolor=color,
                    linewidth=1,
                    baseline=None,
                    label=f"{method_name} (raw)",
                    linestyle=LineStyles[count % len(LineStyles)],
                    alpha=0.5,
                )
            count += 1
        except ValueError as e:
            print(f"Warning: Could not plot stairs for {method_name}: {e}")
            continue

    # Plot cost-based stairs (right subplot)
    count = 0
    for method_name, (scores, costs) in method_data_pairs.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        color = METHOD_COLORS[base_name]
        try:
            # Calculate score progression (best scores so far)
            score_generations = get_envelope_curve(scores, is_maximize)
            cumulative_costs = np.cumsum(costs)

            # Plot best scores vs cumulative cost
            ax3.stairs(
                score_generations,
                np.concatenate(([0], cumulative_costs)),
                edgecolor=color,
                linewidth=2,
                baseline=None,
                label=f"{method_name}",
                linestyle=LineStyles[count % len(LineStyles)],
            )

            # Plot raw scores vs cumulative cost
            if show_raw_scores:
                ax3.stairs(
                    scores,
                    np.concatenate(([0], cumulative_costs)),
                    edgecolor=color,
                    linewidth=1,
                    linestyle="--",
                    baseline=None,
                    label=f"{method_name} (raw)",
                    alpha=0.5,
                )
            count += 1
        except ValueError as e:
            print(f"Warning: Could not plot cost-based stairs for {method_name}: {e}")
            continue

    # Add baseline horizontal lines to stairs plots
    if all_baselines:
        for ax in [ax2, ax3]:
            xmax = ax.get_xlim()[1]
            for name, score in all_baselines.items():
                ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, label=None)
                ax.text(xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

    # Customize plots
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Density")

    ax2.set_xlabel("Num Simulations")
    ax2.set_ylabel("Score")

    ax3.set_xlabel("Cumulative Cost")
    ax3.set_ylabel("Score")

    # Create a single legend for all plots
    handles1 = [line for line in ax1.get_lines() if line.get_label() and not line.get_label().startswith("_")]
    labels1 = [line.get_label() for line in handles1]

    handles2 = [line for line in ax2.get_lines() if line.get_label() and not line.get_label().startswith("_")]
    labels2 = [line.get_label() for line in handles2]

    # Place the combined legend below all subplots
    fig.legend(handles1 + handles2, labels1 + labels2, ncol=len(data), frameon=False, loc="lower center")

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make space for the legend

    # Save plots
    # plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Also save individual plots
    # Distribution plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.2)

    for method_name, (scores, _) in method_data_pairs.items():
        scores = [score for score in scores if score > 0]
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        color = METHOD_COLORS[base_name]
        # Use pretty name for legend (without "Run X")
        legend_label = METHOD_NAMES.get(base_name, base_name)
        try:
            kde = gaussian_kde(scores, bw_method="scott")
            kde.set_bandwidth(kde.factor * 0.5)
            range_data = (min(scores), max(scores))
            print(f"Range data: {range_data}")
            x_kde = np.linspace(range_data[0], range_data[1], 1000)
            y_kde = kde(x_kde)
            ax.plot(x_kde, y_kde, label=legend_label, color=color, linewidth=2, linestyle=METHOD_LINES[base_name])
            ax.fill_between(x_kde, y_kde, color=color, alpha=0.1, linestyle=METHOD_LINES[base_name])
        except ValueError as e:
            print(f"Warning: Could not plot distribution for {method_name}: {e}")
            continue

    if all_baselines:
        ytop = ax.get_ylim()[1]
        for name, score in all_baselines.items():
            ax.axvline(x=score, color="black", linestyle="--", alpha=0.5, label=None)
            ax.text(score, ytop, name, rotation=90, ha="right", va="top", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

    # ax.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65])
    ax.set_xlabel(f"{METRIC_NAME}")
    ax.set_ylabel("Density")
    # Add arrow indicating better direction (left to right when lower is better, right to left when higher is better)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    arrow_y = ylim[0] + 0.5 * (ylim[1] - ylim[0])
    # Determine is_maximize from problem_name
    is_maximize = is_maximize_for_problem(problem_name)
    print(f"is_maximize: {is_maximize}")
    if is_maximize:
        # Higher is better: arrow points right to left (higher values on right are better)
        arrow_x_start = xlim[0] + 0.65 * (xlim[1] - xlim[0])
        arrow_x_end = xlim[0] + 0.85 * (xlim[1] - xlim[0])
    else:
        # Lower is better: arrow points left to right (lower values on left are better)
        arrow_x_start = xlim[0] + 0.85 * (xlim[1] - xlim[0])
        arrow_x_end = xlim[0] + 0.65 * (xlim[1] - xlim[0])

    # Draw the arrow
    ax.annotate('', xy=(arrow_x_end, arrow_y), xytext=(arrow_x_start, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Add "Better" text next to the arrow
    ax.text((arrow_x_start + arrow_x_end) / 2 , 
            arrow_y + 0.02 * (ylim[1] - ylim[0]),
            'Better', rotation=0, va='center', ha='left', 
            fontsize=BETTER_FONT_SIZE, fontweight='bold')

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),  # above the axes
        ncol=4,
    )

    plt.tight_layout()
    plt.savefig(f"{save_path}_distribution.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Boxplot showing min, max, P25, P75, median for each method
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.8)

    # Collect data for boxplot
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []

    for method_name, (scores, _) in method_data_pairs.items():
        scores = [score for score in scores if score > 0]
        if not scores:
            continue
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        legend_label = METHOD_NAMES.get(base_name, base_name)
        color = METHOD_COLORS[base_name]
        boxplot_data.append(scores)
        boxplot_labels.append(legend_label)
        boxplot_colors.append(color)

    if boxplot_data:
        # Create boxplot with custom whiskers at min/max
        positions = np.arange(1, len(boxplot_data) + 1)
        box_width = 0.6
        bp = ax.boxplot(
            boxplot_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            whis=[0, 100],  # Whiskers extend to min and max
            showfliers=False,  # Don't show outliers since whiskers cover full range
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black",
                          markeredgewidth=1.0, markersize=7, zorder=5),
            medianprops=dict(color="black", linewidth=1.5, solid_capstyle="butt"),
            whiskerprops=dict(linewidth=2, linestyle="-"),
            capprops=dict(linewidth=2.5, solid_capstyle="round"),
            boxprops=dict(linewidth=2),
        )

        # Clip median lines to stay within box borders
        for i, median_line in enumerate(bp["medians"]):
            pos = positions[i]
            # Inset the median line slightly so it doesn't touch the box edges
            inset = 0.03
            median_line.set_xdata([pos - box_width/2 + inset, pos + box_width/2 - inset])

        # Color each box according to method color with gradient-like effect
        for i, (patch, color) in enumerate(zip(bp["boxes"], boxplot_colors)):
            # Create a lighter version of the color for gradient effect
            rgb = mcolors.to_rgb(color)
            lighter_rgb = tuple(min(1.0, c + 0.15) for c in rgb)
            darker_rgb = tuple(max(0.0, c - 0.2) for c in rgb)

            patch.set_facecolor(lighter_rgb)
            patch.set_alpha(0.85)
            patch.set_edgecolor(darker_rgb)
            patch.set_linewidth(2)

        # Color whiskers and caps to match boxes (darker shade)
        for i, color in enumerate(boxplot_colors):
            rgb = mcolors.to_rgb(color)
            darker_rgb = tuple(max(0.0, c - 0.15) for c in rgb)
            bp["whiskers"][2*i].set_color(darker_rgb)
            bp["whiskers"][2*i + 1].set_color(darker_rgb)
            bp["caps"][2*i].set_color(darker_rgb)
            bp["caps"][2*i + 1].set_color(darker_rgb)

        # Add baseline horizontal lines
        if all_baselines:
            for name, score in all_baselines.items():
                ax.axhline(y=score, color="black", linestyle="--", alpha=0.6, linewidth=1.5, label=None)
                ax.text(len(boxplot_data) + 0.6, score, f" {name}", ha="left", va="center",
                        color="black", alpha=0.8, fontsize=BASELINE_FONT_SIZE, fontweight="medium")

        ax.set_ylabel(f"{METRIC_NAME}", fontsize=Y_LABEL_FONT_SIZE, fontweight="bold")
        # ax.set_xlabel("Method", fontsize=16, fontweight="bold")
        if problem_name == "cloudcast":
            ax.set_ylim(0, 2100)
        # Set x-tick labels with better formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(boxplot_labels, rotation=0, ha="center", fontsize=X_LABEL_FONT_SIZE, fontweight="bold")

        # Add some padding to y-axis
        ymin, ymax = ax.get_ylim()
        y_padding = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - y_padding, ymax + y_padding)

    plt.tight_layout()
    plt.savefig(f"{save_path}_boxplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Time-based stairs plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.2)
    count = 0
    for method_name, (scores, _) in method_data_pairs.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        color = METHOD_COLORS[base_name]
        # Use pretty name for legend (without "Run X")
        legend_label = METHOD_NAMES.get(base_name, base_name)
        try:
            score_generations = get_envelope_curve(scores, is_maximize)

            ax.stairs(
                score_generations,
                np.arange(len(score_generations) + 1),
                edgecolor=color,
                linewidth=2,
                baseline=None,
                label=legend_label,
                linestyle=LineStyles[count % len(LineStyles)],
            )

            if show_raw_scores:
                ax.stairs(
                    scores,
                    np.arange(len(scores) + 1),
                    edgecolor=color,
                    linewidth=1,
                    linestyle="--",
                    baseline=None,
                    label=None,  # Don't show raw scores in legend
                    alpha=0.5,
                )
            count += 1
        except ValueError as e:
            print(f"Warning: Could not plot stairs for {method_name}: {e}")
            continue

    ax.set_xlim(0, max_num_sims)
    if all_baselines:
        xmax = ax.get_xlim()[1]
        for name, score in all_baselines.items():
            ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, label=None)
            ax.text(xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

    ax.set_xlabel("Num Simulations")
    ax.set_ylabel(f"Best {METRIC_NAME} So Far")
    # ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.savefig(f"{save_path}_stairs.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_stairs.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Cost-based stairs plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.2)

    for method_name, (scores, costs) in method_data_pairs.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        color = METHOD_COLORS[base_name]
        # Use pretty name for legend (without "Run X")
        legend_label = METHOD_NAMES.get(base_name, base_name)
        try:
            score_generations = get_envelope_curve(scores, is_maximize)
            cumulative_costs = np.cumsum(costs)

            ax.stairs(
                score_generations,
                np.concatenate(([0], cumulative_costs)),
                edgecolor=color,
                linewidth=2,
                baseline=None,
                label=legend_label,
            )

            if show_raw_scores:
                ax.stairs(
                    scores,
                    np.concatenate(([0], cumulative_costs)),
                    edgecolor=color,
                    linewidth=1,
                    linestyle="--",
                    baseline=None,
                    label=None,  # Don't show raw scores in legend
                    alpha=0.5,
                )
        except ValueError as e:
            print(f"Warning: Could not plot cost-based stairs for {method_name}: {e}")
            continue

    if all_baselines:
        xmax = ax.get_xlim()[1]
        for name, score in all_baselines.items():
            ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, label=None)
            ax.text(xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

    ax.set_xlabel("Cumulative Cost")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="lower center")
    plt.tight_layout()
    # plt.savefig(f"{save_path}_cost_stairs.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_cost_stairs.png", dpi=300, bbox_inches="tight")
    plt.close()

def convert_list_of_list_to_list_of_dicts(list_of_lists):
    output_list = []
    for lst in list_of_lists:
        output_list.append({i + 1: lst[i] for i in range(len(lst))})
    return output_list

def plot_aggregated_methods(
    data: Dict[str, Any],
    save_path: str,
    band: str = "std",  # or "std"
    pct_low: float = 5,  # used only if band == "percentile"
    pct_high: float = 95,
    max_num_sims: int = 1000,
    is_maximize: bool = False,
):
    """
    Aggregate multiple runs for each method into ONE line with a variability band.

    Parameters
    ----------
    data       : Dict[str, Any]
        Your existing flat dict: {"Best Shot (o4-mini) Run 1": {...}, ...}
    save_path  : str
        File path prefix (no extension) where the figure will be saved.
    band       : {"std", "percentile"}
        "std"  – shades ±1 standard deviation
        "percentile" – shades the [pct_low, pct_high] percentile band
    pct_low / pct_high : float
        Percentile limits when band == "percentile".
    max_num_sims : int
        Maximum number of simulations to plot.
    is_maximize : bool
        Whether to maximize the score.
    """
    multi_context_glia_x = None
    multi_context_glia_y = None
    multi_context_glia_ci_lows = None
    multi_context_glia_ci_highs = None
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run_name, run_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", run_name)  # strip trailing "Run N"
        grouped.setdefault(base_name, []).append(run_data)

    if not grouped:
        raise ValueError("No runs found to aggregate.")

    setup_plot_style()

    # Extract baselines from the first run of each method
    all_baselines = {}
    for method, runs in grouped.items():
        if runs:  # If there are any runs for this method
            first_run = runs[0]
            baselines = extract_baseline_scores(first_run)
            if baselines:
                all_baselines.update(baselines)

    # Create two subplots side by side for the main aggregated plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for ax in [ax1, ax2]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, alpha=0.2)

    # Create a separate figure for scatter plots
    fig_scatter, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5))
    for ax in [ax3, ax4]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, alpha=0.2)

    # order the grouped methods: METHOD_ORDER first, then others in JSON order
    grouped = order_methods(grouped)

    # Plot method curves for both plots
    for method, runs in grouped.items():
        if SKIP_GLIA_O3_PLOTS and method == "Glia (o3)":
            continue
        color = METHOD_COLORS[method]
        # First find minimum length for this method
        method_max_length = float("-inf")
        for run in runs:
            try:
                scores, costs, _ = extract_scores_and_costs(run)
                method_max_length = min(max(method_max_length, len(scores)), max_num_sims)
            except ValueError:
                continue

        if method_max_length == float("-inf"):
            print(f"Warning: No valid runs found for method {method}")
            continue
        if should_expand_method_to_max_sims(method, method_max_length, max_num_sims):
            method_max_length = max_num_sims
        if method == "Glia (o3)" and CUT_LENGTH is not None:
            method_max_length = min(method_max_length, CUT_LENGTH)

        print(f"Method {method}: extending runs to length {method_max_length}")

        best_curves = []
        cost_curves = []
        min_scores = []  # For scatter plots
        min_score_indices = []  # For scatter plots
        min_score_costs = []  # For scatter plots

        for run in runs:
            try:
                scores, costs, _ = extract_scores_and_costs(run)
                # Truncate to this method's minimum length
                if len(scores) < method_max_length:
                    scores = scores + [scores[-1]] * (method_max_length - len(scores))
                    costs = costs + [costs[-1]] * (method_max_length - len(costs))
                else:
                    scores = scores[:method_max_length]
                    costs = costs[:method_max_length]
            except ValueError:
                continue

            # best-so-far trajectory
            current_best = get_envelope_curve(scores, is_maximize)
            bs = current_best
            best_curves.append(bs)

            # Store max score and its index for scatter plots
            min_score = min(bs)
            min_score_idx = bs.index(min_score)
            min_scores.append(min_score)
            min_score_indices.append(min_score_idx)
            min_score_costs.append(np.sum(costs[:min_score_idx + 1]))

            # cumulative costs
            cumulative_costs = np.cumsum(costs)
            cost_curves.append((bs, cumulative_costs))

        if not best_curves:
            continue

        # Plot 1: Best score vs num simulations (line plot)
        # padded = np.array(best_curves)  # No padding needed since all curves have same length
        x, mean, ci_lows, ci_highs, _ = get_average_of_Dicts(convert_list_of_list_to_list_of_dicts(best_curves))

        ax1.plot(x, mean, color=color, linewidth=2, label=method)

        if band == "std":
            ax1.fill_between(x, ci_lows, ci_highs, color=color, alpha=0.2)
        elif band == "percentile":
            low = np.percentile(padded, pct_low, axis=0)
            high = np.percentile(padded, pct_high, axis=0)
            ax1.fill_between(x, low, high, color=color, alpha=0.2)
        elif band is None:
            pass
        else:
            raise ValueError("band must be 'std' or 'percentile'")

        # Plot 2: Best score vs cumulative cost (line plot)
        # First, interpolate all curves to a common cost grid
        max_cost = max(costs[-1] for _, costs in cost_curves)
        cost_grid = np.linspace(0, max_cost, method_max_length)  # Use method_max_length points
        interpolated_scores = []

        for scores, costs in cost_curves:
            # Interpolate this curve's scores onto the common cost grid
            interpolated = np.interp(cost_grid, costs, scores)
            interpolated_scores.append(interpolated)

        interpolated_scores = np.array(interpolated_scores)
        mean_scores = interpolated_scores.mean(axis=0)

        ax2.plot(cost_grid, mean_scores, color=color, linewidth=2)  # Remove label here

        if band == "std":
            std_scores = interpolated_scores.std(axis=0)
            ax2.fill_between(cost_grid, mean_scores - std_scores, mean_scores + std_scores, color=color, alpha=0.2)
        elif band == "percentile":
            low_scores = np.percentile(interpolated_scores, pct_low, axis=0)
            high_scores = np.percentile(interpolated_scores, pct_high, axis=0)
            ax2.fill_between(cost_grid, low_scores, high_scores, color=color, alpha=0.2)

        # Plot scatter plots in separate figure
        ax3.scatter(min_score_indices, min_scores, color=color, label=method, alpha=0.7, s=100)
        ax4.scatter(min_score_costs, min_scores, color=color, label=method, alpha=0.7, s=100)

    # Add baseline horizontal lines to main plots
    if all_baselines:
        for ax in [ax1, ax2]:
            xmax = ax.get_xlim()[1]
            for name, score in all_baselines.items():
                ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, label=None)
                ax.text(xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

        # Add baseline lines to scatter plots
        for ax in [ax3, ax4]:
            xmax = ax.get_xlim()[1]
            for name, score in all_baselines.items():
                ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, label=None)
                ax.text(xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.7, fontsize=BASELINE_FONT_SIZE)

    # Set labels for main plots
    ax1.set_xlabel("Num Simulations")
    ax1.set_ylabel("Best Score So Far")
    ax2.set_xlabel("Cumulative Cost")
    ax2.set_ylabel("Best Score So Far")

    # Set labels for scatter plots
    ax3.set_xlabel("Simulation Number of Max Score")
    ax3.set_ylabel("Maximum Score")
    ax4.set_xlabel("Cumulative Cost at Max Score")
    ax4.set_ylabel("Maximum Score")

    # Add legends
    # For main plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

    # For scatter plots
    handles, labels = ax3.get_legend_handles_labels()
    fig_scatter.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

    # Save main plots
    plt.figure(fig.number)
    plt.tight_layout()
    # plt.savefig(f"{save_path}_aggregated.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_aggregated.png", dpi=300, bbox_inches="tight")

    # Save scatter plots
    plt.figure(fig_scatter.number)
    plt.tight_layout()
    # plt.savefig(f"{save_path}_max_scores_scatter.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_max_scores_scatter.png", dpi=300, bbox_inches="tight")

    # Create standalone aggregated simulation plot
    fig_sim, ax_sim = plt.subplots(figsize=(9, 6))
    ax_sim.set_xlim(0, max_num_sims)
    ax_sim.spines["right"].set_visible(False)
    ax_sim.spines["top"].set_visible(False)
    ax_sim.grid(True, alpha=0.2)
    # Let matplotlib choose y-ticks based on the actual metric scale (e.g., cloudcast uses small values).
    ax_sim.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_sim.tick_params(axis="y", which="both", left=True, labelleft=True)
    # Plot method curves for simulation plot
    for method, runs in grouped.items():
        color = METHOD_COLORS[method]
        # First find minimum length for this method
        method_max_length = float("-inf")
        for run in runs:
            try:
                scores, costs, _ = extract_scores_and_costs(run)
                method_max_length = min(max(method_max_length, len(scores)), max_num_sims)
            except ValueError:
                continue

        if method_max_length == float("-inf"):
            print(f"Warning: No valid runs found for method {method}")
            continue
        if should_expand_method_to_max_sims(method, method_max_length, max_num_sims):
            method_max_length = max_num_sims
        if method == "Glia (o3)" and CUT_LENGTH is not None:
            method_max_length = min(method_max_length, CUT_LENGTH)

        best_curves = []

        for run in runs:
            try:
                scores, costs, _ = extract_scores_and_costs(run)
                if method == "Evolution + LLM Analysis (o3)":
                    method_max_length = 50
                    # Truncate to this method's minimum length
                if len(scores) < method_max_length:
                    scores = scores + [scores[-1]] * (method_max_length - len(scores))
                    costs = costs + [costs[-1]] * (method_max_length - len(costs))
                if len(scores) > method_max_length:
                    scores = scores[:method_max_length]
                    costs = costs[:method_max_length]

            except ValueError:
                continue

            # best-so-far trajectory
            bs = get_envelope_curve(scores, is_maximize)
            best_curves.append(bs)

        if not best_curves:
            continue

        # Compute per-run final best scores (for consistency with fastness table)
        per_run_finals = [float(curve[-1]) for curve in best_curves if curve]

        # Plot: Best score vs num simulations (line plot)
        # padded = np.array(best_curves)  # No padding needed since all curves have same length
        x, mean, ci_lows, ci_highs, _ = get_average_of_Dicts(convert_list_of_list_to_list_of_dicts(best_curves))
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method)
        if METHOD_NAMES[base_name] in ["Glia (Evolutionary)", "Seed Idea + EoH"]:
            # shift x by 8 to right, account for seed generation
            new_x = x + 14
            new_x[0] = x[0]
            x = new_x

        if band == None:
            ax_sim.plot(x, mean, color=color, linewidth=3, label=METHOD_NAMES[method], linestyle=METHOD_LINES[method])
        else:
            ax_sim.plot(x, mean, color=color, linewidth=2, label=METHOD_NAMES[method], linestyle=METHOD_LINES[method])

        if band == "std":
            ax_sim.fill_between(x, np.array(ci_lows), np.array(ci_highs), color=color, alpha=0.1)
            if METHOD_NAMES[base_name] == "Glia (Evolutionary)":
                multi_context_glia_x = x
                multi_context_glia_y = mean
                multi_context_glia_ci_lows = ci_lows
                multi_context_glia_ci_highs = ci_highs
        elif band == "percentile":
            low = np.percentile(padded, pct_low, axis=0)
            high = np.percentile(padded, pct_high, axis=0)
            ax_sim.fill_between(x, low, high, color=color, alpha=0.2)

    # Add baseline horizontal lines
    add_baseline_lines(ax_sim, BASELINE_SCORES)
    # Set labels
    ax_sim.set_xlabel("Num Simulations")
    ax_sim.set_ylabel(f"Best {METRIC_NAME} So Far")
    ncols = 3
    if len(grouped) == 4:
        ncols = 2
    ax_sim.legend(frameon=False, ncol=ncols, loc="upper center", bbox_to_anchor=(0.5, 1.2))

    # add arrow indicating better direction (upward when higher is better, downward when lower is better)
    add_better_direction_arrow(ax_sim, is_maximize)

    # Save standalone simulation plot
    plt.figure(fig_sim.number)
    plt.tight_layout()
    plt.savefig(f"{save_path}_aggregated_simulations.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_aggregated_simulations.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    plt.close(fig_scatter)
    plt.close(fig_sim)
    return multi_context_glia_x, multi_context_glia_y, multi_context_glia_ci_lows, multi_context_glia_ci_highs

def plot_aggregated_methods_with_round_robin(
    data: Dict[str, Any],
    save_path: str,
    band: str = "std",  # or "std"
    pct_low: float = 5,  # used only if band == "percentile"
    pct_high: float = 95,
    max_num_sims: int = 1000,
    is_maximize: bool = False,
    best_of_4_x: List[float] = None,
    best_of_4_y: List[float] = None,
    best_of_4_ci_lows: List[float] = None,
    best_of_4_ci_highs: List[float] = None,
    sequential_envelopes_x: List[float] = None,
    sequential_envelopes_y: List[float] = None,
    sequential_envelopes_ci_lows: List[float] = None,
    sequential_envelopes_ci_highs: List[float] = None,
):
    bar_values_and_errors = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run_name, run_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", run_name)  # strip trailing "Run N"
        grouped.setdefault(base_name, []).append(run_data)

    if not grouped:
        raise ValueError("No runs found to aggregate.")

    setup_plot_style()

    # Extract baselines from the first run of each method
    all_baselines = {}
    # order the grouped methods: METHOD_ORDER first, then others in JSON order
    grouped = order_methods(grouped)
    for method, runs in grouped.items():
        if SKIP_GLIA_O3_PLOTS and method == "Glia (o3)":
            continue
        if runs:  # If there are any runs for this method
            first_run = runs[0]
            baselines = extract_baseline_scores(first_run)
            if baselines:
                all_baselines.update(baselines)

    # Create standalone aggregated simulation plot
    fig_sim, ax_sim = plt.subplots(figsize=(9, 6))
    ax_sim.set_xlim(0, max_num_sims)
    ax_sim.spines["right"].set_visible(False)
    ax_sim.spines["top"].set_visible(False)
    ax_sim.grid(True, alpha=0.2)

    if (
        not DISABLE_PLOT_PAR
        and best_of_4_x is not None
        and best_of_4_y is not None
        and best_of_4_ci_lows is not None
        and best_of_4_ci_highs is not None
    ):
        ax_sim.plot(best_of_4_x, best_of_4_y, color=METHOD_COLORS["Glia Best of 4 (o3)"], linewidth=2, label=METHOD_NAMES["Glia Best of 4 (o3)"], linestyle=METHOD_LINES["Glia Best of 4 (o3)"])
        ax_sim.fill_between(best_of_4_x, np.array(best_of_4_ci_lows), np.array(best_of_4_ci_highs), color=METHOD_COLORS["Glia Best of 4 (o3)"], alpha=0.1)
        bar_values_and_errors["Glia Best of 4 (o3)"] = (best_of_4_y[-1], best_of_4_ci_lows[-1], best_of_4_ci_highs[-1])

    # if sequential_envelopes_x is not None and sequential_envelopes_y is not None and sequential_envelopes_ci_lows is not None and sequential_envelopes_ci_highs is not None:
    #     ax_sim.plot(sequential_envelopes_x, sequential_envelopes_y, color=METHOD_COLORS["Sequential Glia (o3)"], linewidth=2, label=METHOD_NAMES['Sequential Glia (o3)'], linestyle=METHOD_LINES["Sequential Glia (o3)"])
    #     ax_sim.fill_between(sequential_envelopes_x, np.array(sequential_envelopes_ci_lows), np.array(sequential_envelopes_ci_highs), color=METHOD_COLORS["Sequential Glia (o3)"], alpha=0.1)
    #     bar_values_and_errors["Sequential Glia (o3)"] = (sequential_envelopes_y[-1], sequential_envelopes_ci_lows[-1], sequential_envelopes_ci_highs[-1])
    # Plot method curves for simulation plot
    for method, runs in grouped.items():
        if SKIP_GLIA_O3_PLOTS and method == "Glia (o3)":
            continue
        color = METHOD_COLORS[method]
        # First find minimum length for this method
        method_max_length = float("-inf")
        for run in runs:
            try:
                scores, costs, _ = extract_scores_and_costs(run)
                method_max_length = min(max(method_max_length, len(scores)), max_num_sims)
            except ValueError:
                continue

        if method_max_length == float("-inf"):
            print(f"Warning: No valid runs found for method {method}")
            continue
        if should_expand_method_to_max_sims(method, method_max_length, max_num_sims):
            method_max_length = max_num_sims
        if method == "Glia (o3)" and CUT_LENGTH is not None:
            method_max_length = min(method_max_length, CUT_LENGTH)

        best_curves = []
        per_run_finals = []

        for run in runs:
            try:
                scores, costs, _ = extract_scores_and_costs(run)

                # For bar statistics, use the same "final best" definition as the fastness table:
                # the final value of the unmodified best-so-far envelope.
                full_env = get_envelope_curve(scores, is_maximize)
                if full_env:
                    per_run_finals.append(float(full_env[-1]))

                if method == "Evolution + LLM Analysis (o3)":
                    method_max_length = 50
                    # Truncate/pad to this method's minimum length for plotting
                if len(scores) < method_max_length:
                    scores = scores + [scores[-1]] * (method_max_length - len(scores))
                    costs = costs + [costs[-1]] * (method_max_length - len(costs))
                if len(scores) > method_max_length:
                    scores = scores[:method_max_length]
                    costs = costs[:method_max_length]

            except ValueError:
                continue

            # best-so-far trajectory for plotting (clipped/padded to method_max_length <= max_num_sims)
            bs = get_envelope_curve(scores, is_maximize)
            best_curves.append(bs)

            # Bar statistics use the same clipped endpoint as the line plot and AUC,
            # so all three metrics are evaluated at the same simulation budget.
            if bs:
                per_run_finals.append(float(bs[-1]))

        if not best_curves or not per_run_finals:
            continue

        # Plot: Best score vs num simulations (line plot)
        # padded = np.array(best_curves)  # No padding needed since all curves have same length
        x, mean, ci_lows, ci_highs, _ = get_average_of_Dicts(convert_list_of_list_to_list_of_dicts(best_curves))
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method)
        if METHOD_NAMES[base_name] in ["Glia (Evolutionary)", "Seed Idea + EoH"]:
            # shift x by 8 to right, account for seed generation
            new_x = x + 14
            new_x[0] = x[0]
            x = new_x

        if method == "Glia (o3)":
            x = x[:CUT_LENGTH]
            mean = mean[:CUT_LENGTH]
            ci_lows = ci_lows[:CUT_LENGTH]
            ci_highs = ci_highs[:CUT_LENGTH]
            if (
                not DISABLE_PLOT_SEQ
                and sequential_envelopes_x is not None
                and sequential_envelopes_y is not None
                and sequential_envelopes_ci_lows is not None
                and sequential_envelopes_ci_highs is not None
            ):
                # For indices < CUT_LENGTH, force MCG-Seq to match Single-Context Glia exactly
                # (mean and CI). The sequential mean and SCG mean are mathematically equal in
                # this region, but cached values can drift; this guarantees the curves overlap.
                seq_x_arr = np.array(sequential_envelopes_x, dtype=float)
                seq_y_arr = np.array(sequential_envelopes_y, dtype=float)
                seq_lo_arr = np.array(sequential_envelopes_ci_lows, dtype=float)
                seq_hi_arr = np.array(sequential_envelopes_ci_highs, dtype=float)
                overlap = min(CUT_LENGTH, len(seq_y_arr), len(mean), len(ci_lows), len(ci_highs))
                seq_y_arr[:overlap] = np.array(mean)[:overlap]
                seq_lo_arr[:overlap] = np.array(ci_lows)[:overlap]
                seq_hi_arr[:overlap] = np.array(ci_highs)[:overlap]
                sequential_envelopes_y = seq_y_arr.tolist()
                sequential_envelopes_ci_lows = seq_lo_arr.tolist()
                sequential_envelopes_ci_highs = seq_hi_arr.tolist()
                ax_sim.plot(sequential_envelopes_x, sequential_envelopes_y, color=METHOD_COLORS["Sequential Glia (o3)"], linewidth=2, label=METHOD_NAMES['Sequential Glia (o3)'], linestyle=METHOD_LINES["Sequential Glia (o3)"])
                ax_sim.fill_between(sequential_envelopes_x, np.array(sequential_envelopes_ci_lows), np.array(sequential_envelopes_ci_highs), color=METHOD_COLORS["Sequential Glia (o3)"], alpha=0.1)
                bar_values_and_errors["Sequential Glia (o3)"] = (sequential_envelopes_y[-1], sequential_envelopes_ci_lows[-1], sequential_envelopes_ci_highs[-1])
        ax_sim.plot(x, mean, color=color, linewidth=2, label=METHOD_NAMES[method], linestyle=METHOD_LINES[method])

        if band == "std":
            ax_sim.fill_between(x, np.array(ci_lows), np.array(ci_highs), color=color, alpha=0.2)
        elif band == "percentile":
            low = np.percentile(padded, pct_low, axis=0)
            high = np.percentile(padded, pct_high, axis=0)
            ax_sim.fill_between(x, low, high, color=color, alpha=0.2)

        # Use the same statistic as the fastness table: mean of per-run final best scores.
        # Also compute CIs directly on these finals, so that ci_low <= mean <= ci_high and
        # the error bar lengths are always non-negative (as required by matplotlib).
        if per_run_finals:
            if len(per_run_finals) > 1:
                bar_mean, ci_low, ci_high = bootstrap_ci_mean(per_run_finals)
            else:
                bar_mean = per_run_finals[0]
                ci_low = ci_high = bar_mean
            bar_values_and_errors[method] = (float(bar_mean), float(ci_low), float(ci_high))
        else:
            bar_values_and_errors[method] = (mean[-1], ci_lows[-1], ci_highs[-1])

    # Add baseline horizontal lines
    add_baseline_lines(ax_sim, BASELINE_SCORES)

    # Set labels
    ax_sim.set_xlabel("Num Simulations")
    ax_sim.set_ylabel(f"Best {METRIC_NAME} So Far")
    if problem_name == "llm_sql":
        ax_sim.set_ylim(0.5, None)
    add_better_direction_arrow(ax_sim, is_maximize)
    ncols = 3
    has_par = (not DISABLE_PLOT_PAR) and (best_of_4_y is not None)
    has_seq = (not DISABLE_PLOT_SEQ) and (sequential_envelopes_y is not None)
    if len(grouped) + int(has_par) + int(has_seq) == 4:
        ncols = 2
    ax_sim.legend(frameon=False, ncol=ncols, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=LEGEND_FONT_SIZE)

    # Save standalone simulation plot
    plt.figure(fig_sim.number)
    plt.tight_layout()
    plt.savefig(f"{save_path}_aggregated_simulations_with_round_robin.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_aggregated_simulations_with_round_robin.png", dpi=300, bbox_inches="tight")

    plt.close(fig_sim)
    plot_bars(bar_values_and_errors, save_path)


def plot_bars(bar_values_and_errors: Dict[str, Tuple[float, float, float]], save_path: str):
# Create bar plot
    ensure_all_method_styles(bar_values_and_errors.keys())
    ordered_methods = list(order_methods(bar_values_and_errors).keys())
    fig, ax = plt.subplots(figsize=(9, 6))
    # (mean, ci_low, ci_high) per method -> error bar half-lengths: below = mean - ci_low, above = ci_high - mean
    bar_errors_lower = np.array([bar_values_and_errors[method][0] - bar_values_and_errors[method][1] for method in ordered_methods])
    bar_errors_upper = np.array([bar_values_and_errors[method][2] - bar_values_and_errors[method][0] for method in ordered_methods])
    yerr = np.array([bar_errors_lower, bar_errors_upper])
    method_names = [METHOD_NAMES[method] for method in ordered_methods]
    method_colors = [METHOD_COLORS[method] for method in ordered_methods]
    x_pos = range(len(method_names))
    bar_values = [bar_values_and_errors[method][0] for method in ordered_methods]
    with open(f"{save_path}_bar_values.txt", "w") as f:
        for method, value, bar_error_lower, bar_error_upper in zip(method_names, bar_values, bar_errors_lower, bar_errors_upper):
            f.write(f"{method}: {value} - {bar_error_lower} + {bar_error_upper}\n")
    bars = ax.bar(
        x_pos,
        bar_values,
        yerr=yerr,
        color=method_colors,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
        capsize=5,
        error_kw={'linewidth': 2, 'capthick': 2}
    )

    # Customize the plot
    # ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(f"Best {METRIC_NAME}", fontsize=Y_LABEL_FONT_SIZE, fontweight="bold")
    # ax.set_title('Average Request Slowdown by Method', fontsize=14, fontweight='bold')

    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)

    def fit_text(text):
        if " " not in text:
            return text
        if "4" in text:
            return "4-way Parallel \nGlia"
        if text == "Single Agent":
            return text
        if text == "Single Agent (Summarization)":
            return "Summarization"
        if text == "Multi-Context Glia":
            return "Glia"
        if text == "Single-Context Glia":
            return "SCG"
        if text == "Coding Agent w/ Summarization":
            return "Coding Agent \n w/ Summarization"
        # Replace the first space with a newline; centering is handled via tick label settings below
        return text.replace(" ", "\n", 1)

    ax.set_xticklabels(
        [fit_text(method_name) for method_name in method_names],
        rotation=25,
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    # Ensure multi-line labels are centered on each line
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")
        label.set_multialignment("center")

    # Add average value labels above each bar's upper whisker.
    y_min, y_max = ax.get_ylim()
    y_offset = 0.015 * (y_max - y_min)
    whisker_tops = []
    for bar, value, err_up in zip(bars, bar_values, bar_errors_upper):
        whisker_top = value + err_up
        whisker_tops.append(whisker_top)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            whisker_top + y_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    # Ensure labels are not clipped when they are near the top limit.
    max_label_y = max(whisker_tops) + 2 * y_offset if whisker_tops else y_max
    if max_label_y > y_max:
        ax.set_ylim(y_min, max_label_y)

    xmax = ax.get_xlim()[1]
    for name, score in BASELINE_SCORES.items():
        if name == "Random":
            continue
        ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, linewidth=1.5, label=None)
        ax.text(
            xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.8, fontsize=BASELINE_FONT_SIZE, fontweight="bold"
        )

    # add arrow indicating better direction (upward when higher is better, downward when lower is better)

    xlim = ax.get_xlim()
    # get the max y value from the bars
    current_ylim = ax.get_ylim()
    if problem_name == "cloudcast":
        ax.set_ylim(0, max(1400, current_ylim[1]))
    elif problem_name == "llm_sql":
        ax.set_ylim(0.5, None)

    ylim = ax.get_ylim()
    
    arrow_x = xlim[0] + 0.1 * (xlim[1] - xlim[0])
    # Determine is_maximize from problem_name
    is_maximize = is_maximize_for_problem(problem_name)
    if is_maximize:
        # Higher is better: arrow points upward
        arrow_y_start = ylim[0] + 0.75 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.95 * (ylim[1] - ylim[0])
    else:       
        # Lower is better: arrow points downward
        arrow_y_start = ylim[0] + 0.95 * (ylim[1] - ylim[0])
        arrow_y_end = ylim[0] + 0.75 * (ylim[1] - ylim[0])

    # Draw the arrow
    ax.annotate('', xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Add rotated "Better" text next to the arrow
    ax.text(arrow_x - 0.02 * (xlim[1] - xlim[0]), 
            (arrow_y_start + arrow_y_end) / 2,
            'Better', rotation=90, va='center', ha='right', 
            fontsize=BETTER_FONT_SIZE, fontweight='bold')

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f"{save_path}_aggregated_simulations_with_round_robin_bar.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_aggregated_simulations_with_round_robin_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_round_robin_envelope(data: Dict[str, Any], save_path: str, max_num_sims: int = 1000, is_maximize: bool = False):
    """Plot the round-robin envelope of the methods."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.2)
    envelopes = []
    all_baselines = []
    for method_name, method_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        if base_name != "Glia (o3)":
            continue
        color = METHOD_COLORS[base_name]
        try:
            scores, costs, _ = extract_scores_and_costs(method_data)
            score_generations = get_envelope_curve(scores, is_maximize)
            envelopes.append(score_generations)
            all_baselines.append(extract_baseline_scores(method_data))
        except ValueError as e:
            print(f"Warning: Could not plot envelope for {method_name}: {e}")
            continue

    if not envelopes:
        print(
            "Warning: plot_round_robin_envelope() found 0 runs for base_name == 'Glia (o3)'. "
            "Skipping round-robin envelope."
        )
        plt.close(fig)
        return None, None

    x = []
    y = []
    num_simulations = 0
    min_score = float("inf")
    max_length = max(len(envelope) for envelope in envelopes)
    for i in range(max_length):
        existing_envelopes = [envelope[i] for envelope in envelopes if i < len(envelope)]
        num_simulations += len(existing_envelopes)
        if len(existing_envelopes) > 0:
            min_score = min(min_score, min(existing_envelopes))
            x.append(num_simulations)
            y.append(min_score)

    ax.plot(x, y, color=METHOD_COLORS["Parallel Glia (o3)"], linewidth=2, label=METHOD_NAMES["Parallel Glia (o3)"])

    add_baseline_lines(ax, BASELINE_SCORES)

    ax.set_xlabel("Num Simulations")
    ax.set_ylabel(f"Best {METRIC_NAME} So Far")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.savefig(f"{save_path}_round_robin_envelope.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_round_robin_envelope.png", dpi=300, bbox_inches="tight")
    plt.close()
    return x, y

def add_baseline_lines(ax: plt.Axes, all_baselines: Dict[str, float]):
    xmax = ax.get_xlim()[1]
    for name, score in all_baselines.items():
        if name == "Random":
            continue
        ax.axhline(y=score, color="black", linestyle="--", alpha=0.5, linewidth=1.5, label=None)
        ax.text(xmax, score, f" {name}", ha="left", va="center", color="black", alpha=0.8, fontsize=BASELINE_FONT_SIZE, fontweight="bold")

def plot_sequential_envelopes(data: Dict[str, Any], save_path: str, max_num_sims: int = 1000, is_maximize: bool = False, multi_context_glia_x: List[float] = None, multi_context_glia_y: List[float] = None, multi_context_glia_ci_lows: List[float] = None, multi_context_glia_ci_highs: List[float] = None, best_of_4_x: List[float] = None, best_of_4_y: List[float] = None, best_of_4_ci_lows: List[float] = None, best_of_4_ci_highs: List[float] = None):
    """Plot the round-robin envelope of the methods."""
    if DISABLE_PLOT_SEQ:
        print("Skipping sequential envelopes plot because DISABLE_PLOT_SEQ=True")
        return None, None, None, None

    setup_plot_style()
    sequential_envelopes_x = None
    sequential_envelopes_y = None
    sequential_envelopes_ci_lows = None
    sequential_envelopes_ci_highs = None
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, max_num_sims)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.2)
    all_envelopes = []
    all_baselines = []
    for method_name, method_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        if base_name != "Glia (o3)":
            continue
        color = METHOD_COLORS[base_name]
        try:
            scores, costs, _ = extract_scores_and_costs(method_data)
            score_generations = get_envelope_curve(scores, is_maximize)
            all_envelopes.append(score_generations)
            all_baselines.append(extract_baseline_scores(method_data))
        except ValueError as e:
            print(f"Warning: Could not plot envelope for {method_name}: {e}")
            continue

    if not all_envelopes:
        print(
            "Warning: plot_sequential_envelopes() found 0 runs for base_name == 'Glia (o3)'. "
            "Skipping sequential envelopes."
        )
        plt.close(fig)
        return None, None, None, None

    def put_envelopes_sequentially(ordered_envelopes: List[List[float]], max_num_sims: int) -> Dict[float, float]:
        """Put the envelopes sequentially."""
        all_envelopes_sequential = {}
        num_simulations = 0
        min_score = float("inf")
        for envelope in ordered_envelopes:
            if num_simulations >= max_num_sims:
                break
            for i in range(len(envelope)):
                if num_simulations >= max_num_sims:
                    break
                min_score = min(min_score, envelope[i])
                num_simulations += 1
                all_envelopes_sequential[num_simulations] = min_score
        return all_envelopes_sequential

    def get_all_sequential_envelopes(all_envelopes: List[List[float]], max_num_sims: int) -> List[List[float]]:
        """Get all the sequential envelopes."""
        all_curves = []
        # add a tiny random noise to the envelopes so we can distinguish between them
        import random
        random.seed(42)
        all_envelopes = [[envelope_val + random.random() * 0.000001 for envelope_val in envelope] for envelope in all_envelopes]

        # for the len(all_envelopes) envelopes, we have len(all_envelopes)! combinations we can get put_envelopes_sequentially on
        all_permutations = permutations(range(len(all_envelopes)), len(all_envelopes))
        # assert len(all_permutations) == math.factorial(len(all_envelopes))
        count = 0
        for permutation in all_permutations:
            if count % 10000 == 0:
                print(f"Processed {count} permutations")
            ordered_envelopes = [all_envelopes[i] for i in permutation]
            all_curves.append(put_envelopes_sequentially(ordered_envelopes, max_num_sims))
            count += 1
        return all_curves

    print(f"Getting all sequential envelopes...")
    if os.path.exists(f"{save_path}_all_sequential_envelopes.pkl"):
        with open(f"{save_path}_all_sequential_envelopes.pkl", "rb") as f:
            all_sequential_envelopes = pickle.load(f)
    else:
        all_sequential_envelopes = get_all_sequential_envelopes(all_envelopes, max_num_sims)
        with open(f"{save_path}_all_sequential_envelopes.pkl", "wb") as f:
            pickle.dump(all_sequential_envelopes, f)

    print("Computing the CI for the sequential envelopes...")
    if os.path.exists(f"{save_path}_sequential_envelopes_x.json"):
        with open(f"{save_path}_sequential_envelopes_x.json", "r") as f:
            sequential_envelopes_x = json.load(f)
        with open(f"{save_path}_sequential_envelopes_y.json", "r") as f:
            sequential_envelopes_y = json.load(f)
        with open(f"{save_path}_sequential_envelopes_ci_lows.json", "r") as f:
            sequential_envelopes_ci_lows = json.load(f)
        with open(f"{save_path}_sequential_envelopes_ci_highs.json", "r") as f:
            sequential_envelopes_ci_highs = json.load(f)
    else:
        start_time = time.time()
        sequential_envelopes_x, sequential_envelopes_y, _, _, sequential_envelopes_extended_dicts = get_average_of_Dicts(all_sequential_envelopes, skip_bootstrap=True)
        sequential_envelopes_ci_lows = []
        sequential_envelopes_ci_highs = []
        for index, i in enumerate(sequential_envelopes_x):
            print(f"Computing CI for sim_num={index}...")
            inputs = [sequential_envelopes_extended_dicts[j][i] for j in range(len(sequential_envelopes_extended_dicts))]
            unique_inputs = list(set(inputs))
            if len(unique_inputs) > 1:
                se = np.std(unique_inputs, ddof=1) / np.sqrt(len(unique_inputs))
            else:
                se = 0.0
            band = 1.96 * se
            sequential_envelopes_ci_lows.append(sequential_envelopes_y[index] - band)
            sequential_envelopes_ci_highs.append(sequential_envelopes_y[index] + band)
        end_time = time.time()
        print(f"Time taken to average the stats for sequential envelopes: {end_time - start_time} seconds")
        with open(f"{save_path}_sequential_envelopes_x.json", "w") as f:
            json.dump(np.array(sequential_envelopes_x).tolist(), f)
        with open(f"{save_path}_sequential_envelopes_y.json", "w") as f:
            json.dump(np.array(sequential_envelopes_y).tolist(), f)
        with open(f"{save_path}_sequential_envelopes_ci_lows.json", "w") as f:
            json.dump(np.array(sequential_envelopes_ci_lows).tolist(), f)
        with open(f"{save_path}_sequential_envelopes_ci_highs.json", "w") as f:
            json.dump(np.array(sequential_envelopes_ci_highs).tolist(), f)

    # Compute fastness / AUC-style metrics for the Multi-Context Glia (sequential) envelope.
    # This reuses the generic fastness helper so that MCG has comparable T_95 / T_99 and norm_auc.
    try:
        seq_env = list(sequential_envelopes_y)
        if max_num_sims is not None:
            seq_env = seq_env[: min(len(seq_env), max_num_sims)]
        fastness_metrics = compute_fastness_metrics_for_envelope(seq_env, is_maximize=is_maximize)
        final_val = float(seq_env[-1]) if seq_env else float("nan")
        mcg_fastness = {
            "method": "Sequential Glia (o3)",
            "final_best": final_val,
            **fastness_metrics,
        }
        with open(f"{save_path}_fastness.json", "w") as f:
            json.dump(mcg_fastness, f)
        print(f"[fastness] Wrote Multi-Context Glia (sequential) fastness metrics to: {save_path}_fastness.json")
    except Exception as e:
        print(f"Warning: failed to compute fastness metrics for sequential envelopes: {e}")

    ax.plot(sequential_envelopes_x, sequential_envelopes_y, color=METHOD_COLORS["Sequential Glia (o3)"], linewidth=2, label=METHOD_NAMES["Sequential Glia (o3)"], linestyle=METHOD_LINES["Sequential Glia (o3)"])
    ax.fill_between(sequential_envelopes_x, np.array(sequential_envelopes_ci_lows), np.array(sequential_envelopes_ci_highs), color=METHOD_COLORS["Sequential Glia (o3)"], alpha=0.1)

    if (
        not DISABLE_PLOT_PAR
        and best_of_4_x is not None
        and best_of_4_y is not None
        and best_of_4_ci_lows is not None
        and best_of_4_ci_highs is not None
    ):
        ax.plot(best_of_4_x, best_of_4_y, color=METHOD_COLORS["Glia Best of 4 (o3)"], linewidth=2, label=METHOD_NAMES["Glia Best of 4 (o3)"], linestyle=METHOD_LINES["Glia Best of 4 (o3)"])
        ax.fill_between(best_of_4_x, np.array(best_of_4_ci_lows), np.array(best_of_4_ci_highs), color=METHOD_COLORS["Glia Best of 4 (o3)"], alpha=0.1)

    if multi_context_glia_x is not None and multi_context_glia_y is not None and multi_context_glia_ci_lows is not None and multi_context_glia_ci_highs is not None:
        ax.plot(multi_context_glia_x, multi_context_glia_y, color=METHOD_COLORS["Seed + Evolution + LLM Analysis (o3)"], linewidth=2, linestyle=METHOD_LINES["Seed + Evolution + LLM Analysis (o3)"], label=METHOD_NAMES["Seed + Evolution + LLM Analysis (o3)"])
        ax.fill_between(multi_context_glia_x, np.array(multi_context_glia_ci_lows), np.array(multi_context_glia_ci_highs), color=METHOD_COLORS["Seed + Evolution + LLM Analysis (o3)"], linestyle=METHOD_LINES["Seed + Evolution + LLM Analysis (o3)"], alpha=0.1)

    add_baseline_lines(ax, BASELINE_SCORES)
    ax.set_xlabel("Num Simulations")
    ax.set_ylabel(f"Best {METRIC_NAME} So Far")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.savefig(f"{save_path}_sequential_envelopes.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_sequential_envelopes.png", dpi=300, bbox_inches="tight")
    plt.close()
    return sequential_envelopes_x, sequential_envelopes_y, sequential_envelopes_ci_lows, sequential_envelopes_ci_highs

def get_all_best_of_n_envelopes(all_envelopes: List[List[float]], n: int, is_maximize: bool = False) -> List[Dict[float, float]]:
    """Get all the best of n envelopes."""
    chosen_envelope_numbers = combinations(range(len(all_envelopes)), n)
    best_of_chosen_envelopes = []
    for chosen_envelope_number in chosen_envelope_numbers:
        chosen_envelopes = [all_envelopes[i] for i in chosen_envelope_number]
        best_of_envs = get_best_of_envelopes(chosen_envelopes, is_maximize=is_maximize)
        values = list(best_of_envs.values())
        # When lower is better, envelope is non-increasing; when higher is better, non-decreasing
        for i in range(1, len(values)):
            if is_maximize and values[i] < values[i - 1]:
                assert False, "Y values must be non-decreasing when is_maximize=True"
            if not is_maximize and values[i] > values[i - 1]:
                assert False, "Y values must be non-increasing when is_maximize=False"
        best_of_chosen_envelopes.append(best_of_envs)
    return best_of_chosen_envelopes
        

def get_best_of_envelopes(envelopes: List[List[float]], is_maximize: bool = False) -> Dict[float, float]: # sim_num -> y value
    x = []
    y = []
    num_simulations = 0
    best_score = float("-inf") if is_maximize else float("inf")
    max_length = max(len(envelope) for envelope in envelopes)
    for i in range(max_length):
        existing_envelopes = [envelope[i] for envelope in envelopes if i < len(envelope)]
        num_simulations += len(existing_envelopes)
        if len(existing_envelopes) > 0:
            candidate = max(existing_envelopes) if is_maximize else min(existing_envelopes)
            best_score = max(best_score, candidate) if is_maximize else min(best_score, candidate)
            x.append(num_simulations)
            y.append(best_score)
    
    # Make x values consecutive and fill y with latest actual y so far
    if not x:
        return {}
    max_x = max(x)
    new_x = list(range(1, max_x + 1))
    output_dict = {}
    current_best_y = y[0]
    x_idx = 0
    for sim_num in new_x:
        if x_idx < len(x) and x[x_idx] == sim_num:
            current_best_y = y[x_idx]
            output_dict[sim_num] = current_best_y
            x_idx += 1
        else:
            output_dict[sim_num] = current_best_y
    return output_dict


def get_average_of_Dicts(
    dicts: List[Dict[float, float]],
    skip_bootstrap: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float], List[Dict[float, float]]]:
    if not dicts:
        # Callers unpack 5 values; return empty lists instead of {} to avoid unpacking errors.
        return [], [], [], [], []
    
    # Find the maximum simulation number across all dictionaries
    max_x = max(max(dict.keys()) for dict in dicts)
    x_values = []
    y_values = []
    ci_lows = []
    ci_highs = []
    
    # Extend shorter dictionaries by repeating their last y value
    extended_dicts = []
    for dict in dicts:
        extended_dict = dict.copy()
        if dict:
            max_dict_x = max(dict.keys())
            last_y = dict[max_dict_x]
            # Fill in missing simulation numbers with the last y value
            for sim_num in range(max_dict_x + 1, max_x + 1):
                extended_dict[sim_num] = last_y
        extended_dicts.append(extended_dict)
    
    for sim_num in range(1, max_x + 1):
        if sim_num % 100 == 0:
            print(f"Computing average for sim_num={sim_num}...")
        # Collect all y values for this simulation number
        existing_ys = []
        for dict in extended_dicts:
            if sim_num in dict:
                existing_ys.append(dict[sim_num])
        
        if existing_ys:
            # Calculate mean
            if skip_bootstrap:
                mean_y = sum(existing_ys) / len(existing_ys)
                ci_low = 0
                ci_high = 0
            else:
                mean_y, ci_low, ci_high = bootstrap_ci_mean(existing_ys)
            x_values.append(sim_num)
            y_values.append(mean_y)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            # mean_y = sum(existing_ys) / len(existing_ys)
            
            # # Calculate standard deviation
            # if len(existing_ys) > 1:
            #     variance = sum((y - mean_y) ** 2 for y in existing_ys) / (len(existing_ys) - 1)
            #     std_dev = variance ** 0.5
            # else:
            #     std_dev = 0.0
            
            # output_dict[sim_num] = (mean_y, ci_low, ci_high)
    
    return x_values, y_values, ci_lows, ci_highs, extended_dicts


def plot_best_of_n_envelope(data: Dict[str, Any], save_path: str, max_num_sims: int = 1000, is_maximize: bool = False, multi_context_glia_x: List[float] = None, multi_context_glia_y: List[float] = None, multi_context_glia_ci_lows: List[float] = None, multi_context_glia_ci_highs: List[float] = None):
    """Plot the round-robin envelope of the methods."""
    if DISABLE_PLOT_PAR:
        print("Skipping best-of-n plot because DISABLE_PLOT_PAR=True")
        return None, None, None, None

    best_of_4_x = None
    best_of_4_y = None
    best_of_4_ci_lows = None
    best_of_4_ci_highs = None
    best_of_1_x = None
    best_of_1_y = None
    best_of_1_ci_lows = None
    best_of_1_ci_highs = None
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, max_num_sims)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.2)
    all_envelopes = []
    all_baselines = []
    for method_name, method_data in data.items():
        base_name = re.sub(r"\s*Run\s*\d+\s*$", "", method_name)
        if base_name != "Glia (o3)":
            continue
        color = METHOD_COLORS[base_name]
        try:
            scores, costs, _ = extract_scores_and_costs(method_data)
            score_generations = get_envelope_curve(scores, is_maximize)
            all_envelopes.append(score_generations)
            all_baselines.append(extract_baseline_scores(method_data))
        except ValueError as e:
            print(f"Warning: Could not plot envelope for {method_name}: {e}")
            continue

    if not all_envelopes:
        print(
            "Warning: plot_best_of_n_envelope() found 0 runs for base_name == 'Glia (o3)'. "
            "Skipping best-of-n envelope."
        )
        plt.close(fig)
        return None, None, None, None

    # for n in range(1, len(all_envelopes) + 1):
    for n in [1, 2, 4, 6]:
        if len(all_envelopes) < n:
            print(
                f"Warning: Not enough 'Glia (o3)' envelopes for best-of-n with n={n} "
                f"(have {len(all_envelopes)}). Skipping n={n}."
            )
            continue
        if n == 1:
            label = METHOD_NAMES["Glia (o3)"]
            color = METHOD_COLORS["Glia (o3)"]
            linestyle = METHOD_LINES["Glia (o3)"]
        if n == 2:
            label = METHOD_NAMES["Glia Best of 4 (o3)"].replace("4", "2")
            color = "mediumblue"
            linestyle = (0, (1, 1))
        if n == 4:
            label = METHOD_NAMES["Glia Best of 4 (o3)"]
            color = METHOD_COLORS["Glia Best of 4 (o3)"]
            # two dot one dash
            linestyle = METHOD_LINES["Glia Best of 4 (o3)"]
        if os.path.exists(f"{save_path}_best_of_n_x_{n}.json"):
            with open(f"{save_path}_best_of_n_x_{n}.json", "r") as f:
                x = json.load(f)
            with open(f"{save_path}_best_of_n_y_{n}.json", "r") as f:
                y = json.load(f)
            with open(f"{save_path}_best_of_n_ci_lows_{n}.json", "r") as f:
                ci_lows = json.load(f)
            with open(f"{save_path}_best_of_n_ci_highs_{n}.json", "r") as f:
                ci_highs = json.load(f)
        else:
            chosen_envelopes = get_all_best_of_n_envelopes(all_envelopes, n, is_maximize=is_maximize)
            if not chosen_envelopes:
                print(f"Warning: No chosen envelopes produced for n={n}; skipping.")
                continue
            print(f"Computing average for n={n}...")
            start_time = time.time()
            x, y, _, _, extended_dicts = get_average_of_Dicts(chosen_envelopes, skip_bootstrap=True)
            if not x:
                print(f"Warning: Empty x returned for n={n}; skipping.")
                continue
            # compute the confidence intervals using bootstrap
            ci_lows = []
            ci_highs = []
            for index, i in enumerate(x):
                inputs = [extended_dicts[j][i] for j in range(len(extended_dicts))]
                _, ci_low, ci_high = bootstrap_ci_best_of_n(inputs, n, higher_is_better=is_maximize)
                band = (ci_high - ci_low) / 2
                ci_lows.append(y[index] - band)
                ci_highs.append(y[index] + band)

            end_time = time.time()
            print(f"Time taken to average the stats for n={n}: {end_time - start_time} seconds")
            with open(f"{save_path}_best_of_n_x_{n}.json", "w") as f:
                json.dump(np.array(x).tolist(), f)
            with open(f"{save_path}_best_of_n_y_{n}.json", "w") as f:
                json.dump(np.array(y).tolist(), f)
            with open(f"{save_path}_best_of_n_ci_lows_{n}.json", "w") as f:
                json.dump(np.array(ci_lows).tolist(), f)
            with open(f"{save_path}_best_of_n_ci_highs_{n}.json", "w") as f:
                json.dump(np.array(ci_highs).tolist(), f)
        if n == 4:
            best_of_4_x = x
            best_of_4_y = y
            best_of_4_ci_lows = ci_lows
            best_of_4_ci_highs = ci_highs
        if n == 1:
            # cut the x values at CUT_LENGTH
            x = x[:CUT_LENGTH]
            y = y[:CUT_LENGTH]
            ci_lows = ci_lows[:CUT_LENGTH]
            ci_highs = ci_highs[:CUT_LENGTH]
            best_of_1_x = x
            best_of_1_y = y
            best_of_1_ci_lows = ci_lows
            best_of_1_ci_highs = ci_highs
        if n != 1:
            ax.plot(x, y, color=color, linewidth=2, label=label, linestyle=linestyle)
            ax.fill_between(x, np.array(ci_lows), np.array(ci_highs), color=color, alpha=0.1)

    if multi_context_glia_x is not None and multi_context_glia_y is not None and multi_context_glia_ci_lows is not None and multi_context_glia_ci_highs is not None:
        # align the stds for < CUT_LENGTH with n = 1
        with open(f"{save_path}_best_of_n_ci_lows_{1}.json", "r") as f:
            ci_lows_1 = json.load(f)
        with open(f"{save_path}_best_of_n_ci_highs_{1}.json", "r") as f:
            ci_highs_1 = json.load(f)
        # Only update up to the minimum of CUT_LENGTH and the actual length of the lists
        min_length = min(CUT_LENGTH, len(multi_context_glia_ci_lows), len(multi_context_glia_ci_highs), len(ci_lows_1), len(ci_highs_1))
        for i in range(min_length):
            multi_context_glia_ci_lows[i] = ci_lows_1[i]
            multi_context_glia_ci_highs[i] = ci_highs_1[i]
        ax.plot(multi_context_glia_x, multi_context_glia_y, color=METHOD_COLORS["Sequential Glia (o3)"], linewidth=2, linestyle=METHOD_LINES["Sequential Glia (o3)"], label=METHOD_NAMES["Sequential Glia (o3)"])
        ax.fill_between(multi_context_glia_x, np.array(multi_context_glia_ci_lows), np.array(multi_context_glia_ci_highs), color=METHOD_COLORS["Sequential Glia (o3)"], linestyle=METHOD_LINES["Sequential Glia (o3)"], alpha=0.1)

    # if best_of_1_x is not None and best_of_1_y is not None and best_of_1_ci_lows is not None and best_of_1_ci_highs is not None:
    #     ax.plot(best_of_1_x, best_of_1_y, color=METHOD_COLORS["Glia (o3)"], linewidth=2, label=METHOD_NAMES["Glia (o3)"], linestyle=METHOD_LINES["Glia (o3)"])
    #     ax.fill_between(best_of_1_x, np.array(best_of_1_ci_lows), np.array(best_of_1_ci_highs), color=METHOD_COLORS["Glia (o3)"], alpha=0.1)
    add_baseline_lines(ax, BASELINE_SCORES)

    ax.set_xlabel("Num Simulations")
    ax.set_ylabel(f"Best {METRIC_NAME} So Far")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.savefig(f"{save_path}_best_of_n_envelope.pdf", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{save_path}_best_of_n_envelope.png", dpi=300, bbox_inches="tight")
    plt.close()
    return best_of_4_x, best_of_4_y, best_of_4_ci_lows, best_of_4_ci_highs


def parse_method_pairs(value: str) -> Tuple[str, str]:
    """Parse a method pair in the format 'json_path:method_name'."""
    try:
        json_path, method_name = value.split(":")
        if not os.path.exists(json_path):
            raise click.BadParameter(f"JSON file not found: {json_path}")
        return json_path.strip(), method_name.strip()
    except ValueError:
        raise click.BadParameter("Each method must be specified as 'json_path:method_name'")


def process_all_methods_json(json_path: str) -> List[Tuple[str, str]]:
    """Process the all_methods.json format and return a list of method pairs."""
    with open(json_path, "r") as f:
        methods_dict = json.load(f)

    method_pairs = []
    for method_name, file_paths in methods_dict.items():
        if method_name.startswith("#"):
            continue
        for i, file_path in enumerate(file_paths, 1):
            # Create a unique name for each run
            run_name = f"{method_name} Run {i}"
            if file_path.startswith("#"):
                continue
            method_pairs.append((file_path, run_name))

    return method_pairs


def is_code_cheating(code: str) -> bool:
    # # if the code uses "num_decode_tokens", it is cheating
    # if "\"num_decode_tokens\"" in code:
    #     return True
    return False

@click.command()
@click.argument("methods", nargs=-1)
@click.option("--all-methods-json", "-a", type=click.Path(exists=True), help="JSON file in all_methods.json format")
@click.option(
    "--output-dir", "-o", required=True, type=click.Path(file_okay=False, dir_okay=True), help="Directory to save plots"
)
@click.option(
    "--show-raw-scores",
    is_flag=True,
    help="If set, add raw scores to the plots",
)
@click.option(
    "--max-num-sims",
    type=int,
    default=100,
    help="Maximum number of simulations to plot",
)
@click.option(
    "--problem-name",
    "problem_name_arg",
    type=str,
    default=None,
    help="Problem name (e.g., 'prism', 'vidur'). If omitted, auto-detected from --output-dir.",
)
@click.option(
    "--baseline-cache",
    type=click.Path(exists=True),
    default=None,
    help="Path to baseline_cache.json file. Extracts top baselines and shows them on plots.",
)
@click.option(
    "--top-baselines",
    type=int,
    default=3,
    help="Number of top baselines to show from baseline-cache (default: 3).",
)
@click.option(
    "--skip-multi",
    is_flag=True,
    help="If set, skip the multi-context glia plot",
)
@click.option(
    "--cut-length",
    type=int,
    default=None,
    help="Cut length for the plots",
)

def main(methods: List[str],
         all_methods_json: str,
         output_dir: str, show_raw_scores: bool, max_num_sims: int, problem_name_arg: str, baseline_cache: str, top_baselines: int, skip_multi: bool, cut_length: int):
    """Create plots comparing multiple methods.
    
    Each method should be specified as 'json_path:method_name'
    
    Example usage:
    python plot_methods.py \
        path/to/best_shot.json:"Best Shot" \
        path/to/evolution.json:"Evolution" \
        -o plots

    Or use the all-methods-json format:
    python plot_methods.py --all-methods-json path/to/all_methods.json -o plots
    Example all_methods.json:
        {
            "Best Shot (o4-mini)": [
                "o4-mini-vidur-best-shot_1000gen.json",
                "o4-mini-vid,ur-best-shot_801gen.json"
            ],
            "Evolution (o3)": [
                "o3-vidur-evolution_9gen.json",
                "o3-vidur-evolution_9gen.json"
            ],
            ...
        }
    """
    if not methods and not all_methods_json:
        raise click.UsageError("Either methods or --all-methods-json must be specified")

    # Set problem_name from CLI or auto-detect from output_dir if not provided
    global problem_name, METRIC_NAME, BASELINE_NAMES, BASELINE_SCORES, SKIP_GLIA_O3_PLOTS, DISABLE_PLOT_SEQ, DISABLE_PLOT_PAR, CUT_LENGTH
    if skip_multi:
        DISABLE_PLOT_SEQ = True
        DISABLE_PLOT_PAR = True
        SKIP_GLIA_O3_PLOTS = False

    if cut_length is not None:
        CUT_LENGTH = int(cut_length)

    if problem_name_arg is None:
        # Try to extract problem name from output_dir (e.g., "all_plots/prism" -> "prism")
        output_basename = os.path.basename(os.path.normpath(output_dir))
        # Check if it matches a known problem name
        if output_basename in HIGHER_IS_BETTER:
            problem_name = output_basename
        else:
            # Try parent directory (e.g., "all_plots/prism" -> check "prism")
            parent_dir = os.path.basename(os.path.dirname(os.path.normpath(output_dir)))
            if parent_dir in HIGHER_IS_BETTER:
                problem_name = parent_dir
            else:
                # Default to cloudcast if can't detect; override HIGHER_IS_BETTER there if needed.
                problem_name = input("Problem name (exact name): ")
                if problem_name not in HIGHER_IS_BETTER:
                    raise ValueError(f"Problem name '{problem_name}' not found in HIGHER_IS_BETTER")
    else:
        problem_name = problem_name_arg
        if problem_name not in HIGHER_IS_BETTER:
            raise ValueError(f"Problem name '{problem_name}' not found in HIGHER_IS_BETTER")

    # Update global METRIC_NAME and BASELINE_* based on detected problem_name
    METRIC_NAME = get_metric_name(problem_name)
    BASELINE_NAMES = get_baseline_names(problem_name)
    BASELINE_SCORES = get_baseline_scores(problem_name)

    # Override baselines from baseline_cache.json if provided
    if baseline_cache:
        BASELINE_SCORES = extract_top_baselines_from_cache(baseline_cache, top_baselines)
        BASELINE_NAMES = {f"baseline_v{i}": name for i, name in enumerate(BASELINE_SCORES.keys())}

    # Parse method pairs
    method_pairs = []
    if methods:
        method_pairs.extend([parse_method_pairs(m) for m in methods])
    if all_methods_json:
        method_pairs.extend(process_all_methods_json(all_methods_json))

    os.makedirs(output_dir, exist_ok=True)

    # Load JSON files
    data = {}
    for json_path, method_name in method_pairs:
        with open(json_path, "r") as f:
            data[method_name] = json.load(f)

    # Register any new method keys from the data so they get colors, names, line styles, and order.
    # Preserve JSON / CLI order: derive base names in the same order as data.keys().
    all_base_names: List[str] = []
    for k in data.keys():
        base = re.sub(r"\s*Run\s*\d+\s*$", "", k)
        if base not in all_base_names:
            all_base_names.append(base)
    ensure_all_method_styles(all_base_names)

    for method_name, method_data in data.items():
        for key, value in method_data.items():
            if key == "best_solution":
                if problem_name == "llm_sql":
                    # No inversion for llm_sql
                    pass
                elif problem_name == "vidur":
                    data[method_name][key]['score'] = float(20 / value['score'])
                elif problem_name in ["cloudcast"]:
                    if value['score'] == 0 or value['score'] == float("-inf"):
                        data[method_name][key]['score'] = float("inf")
                    else:
                        print(f"Converting cost to score: {value['score']} -> {1 / value['score']}")
                        data[method_name][key]['score'] = float(1 / value['score'] - 1)
            elif key == "all_iterations":
                for iteration in range(len(data[method_name][key])):
                    try:
                        code = data[method_name][key][iteration]['code']
                        if is_code_cheating(code):
                            data[method_name][key][iteration]['score'] = float("-inf") if is_maximize_for_problem(problem_name) else float("inf")
                            continue
                        score = data[method_name][key][iteration]['score']
                        if problem_name == "llm_sql":
                            # No inversion for llm_sql
                            if score == 0 or score == float("-inf"):
                                data[method_name][key][iteration]['score'] = float("-inf")
                        elif problem_name == "vidur":
                            if score == 0 or score == float("-inf"):
                                data[method_name][key][iteration]['score'] = float("inf")
                            else:
                                data[method_name][key][iteration]['score'] = float(20 / score)
                        elif problem_name in ["cloudcast"]:
                            if score == 0 or score == float("-inf"):
                                data[method_name][key][iteration]['score'] = float("inf")
                            else:
                                data[method_name][key][iteration]['score'] = float(1 / score - 1)
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Iteration: {iteration}")
                        print(f"Method name: {method_name}")
                        print(f"Key: {key}")
                        pass
    # Create plot
    base_name = os.path.basename(os.path.normpath(output_dir))
    save_path = os.path.join(output_dir, f"{base_name}_method_comparison")

    is_maximize = is_maximize_for_problem(problem_name)
    print(f"Using problem_name='{problem_name}', is_maximize={is_maximize}")

    # Save the single best code snippet per base method (e.g., best Handoff, best OpenEvolve, etc.).
    save_best_code_per_base_method(data, save_path, is_maximize=is_maximize, max_num_sims=max_num_sims)

    plot_comparison(data, save_path, show_raw_scores=show_raw_scores, max_num_sims=max_num_sims, is_maximize=is_maximize)
    plot_aggregated_methods(data, save_path, band=None, max_num_sims=max_num_sims, is_maximize=is_maximize)

    if os.path.exists(f"{save_path}_multi_context_glia_x.json"):
        with open(f"{save_path}_multi_context_glia_x.json", "r") as f:
            multi_context_glia_x = json.load(f)
        with open(f"{save_path}_multi_context_glia_y.json", "r") as f:
            multi_context_glia_y = json.load(f)
        with open(f"{save_path}_multi_context_glia_ci_lows.json", "r") as f:
            multi_context_glia_ci_lows = json.load(f)
        with open(f"{save_path}_multi_context_glia_ci_highs.json", "r") as f:
            multi_context_glia_ci_highs = json.load(f)
    else:
        print("Computing aggregated methods...")
        multi_context_glia_x, multi_context_glia_y, multi_context_glia_ci_lows, multi_context_glia_ci_highs = plot_aggregated_methods(data, f"{save_path}_std", band="std", max_num_sims=max_num_sims, is_maximize=is_maximize)
        with open(f"{save_path}_multi_context_glia_x.json", "w") as f:
            json.dump(np.array(multi_context_glia_x).tolist(), f)
        with open(f"{save_path}_multi_context_glia_y.json", "w") as f:
            json.dump(np.array(multi_context_glia_y).tolist(), f)
        with open(f"{save_path}_multi_context_glia_ci_lows.json", "w") as f:
            json.dump(np.array(multi_context_glia_ci_lows).tolist(), f)
        with open(f"{save_path}_multi_context_glia_ci_highs.json", "w") as f:
            json.dump(np.array(multi_context_glia_ci_highs).tolist(), f)
    # Optional extras: best-of-n / sequential envelopes are only defined for "Glia (o3)" runs.
    # Do not let these optional steps prevent the round-robin aggregated plot + bar plot.
    best_of_4_x = best_of_4_y = best_of_4_ci_lows = best_of_4_ci_highs = None
    sequential_envelopes_x = sequential_envelopes_y = sequential_envelopes_ci_lows = sequential_envelopes_ci_highs = None

    start_time = time.time()
    if not DISABLE_PLOT_PAR:
        try:
            print("Computing best of n envelope...")
            # x, y = plot_round_robin_envelope(data, save_path, max_num_sims=max_num_sims)
            best_of_4_x, best_of_4_y, best_of_4_ci_lows, best_of_4_ci_highs = plot_best_of_n_envelope(
                data,
                f"{save_path}_best_of_n",
                max_num_sims=max_num_sims,
                is_maximize=is_maximize,
                multi_context_glia_x=multi_context_glia_x,
                multi_context_glia_y=multi_context_glia_y,
                multi_context_glia_ci_lows=multi_context_glia_ci_lows,
                multi_context_glia_ci_highs=multi_context_glia_ci_highs,
            )
        except Exception as e:
            print(f"Warning: best-of-n envelope failed (continuing): {e}")
    else:
        print("Skipping best-of-n envelope computation because DISABLE_PLOT_PAR=True")

    if not DISABLE_PLOT_SEQ:
        try:
            print(f"Sequential envelopes start time: {time.time()}")
            if os.path.exists(f"{save_path}_sequential_envelopes_x.json"):
                with open(f"{save_path}_sequential_envelopes_x.json", "r") as f:
                    sequential_envelopes_x = json.load(f)
                with open(f"{save_path}_sequential_envelopes_y.json", "r") as f:
                    sequential_envelopes_y = json.load(f)
                with open(f"{save_path}_sequential_envelopes_ci_lows.json", "r") as f:
                    sequential_envelopes_ci_lows = json.load(f)
                with open(f"{save_path}_sequential_envelopes_ci_highs.json", "r") as f:
                    sequential_envelopes_ci_highs = json.load(f)
            else:
                sequential_envelopes_x, sequential_envelopes_y, sequential_envelopes_ci_lows, sequential_envelopes_ci_highs = plot_sequential_envelopes(
                    data,
                    f"{save_path}_sequential_envelopes",
                    max_num_sims=max_num_sims,
                    is_maximize=is_maximize,
                    multi_context_glia_x=multi_context_glia_x,
                    multi_context_glia_y=multi_context_glia_y,
                    multi_context_glia_ci_lows=multi_context_glia_ci_lows,
                    multi_context_glia_ci_highs=multi_context_glia_ci_highs,
                    best_of_4_x=best_of_4_x,
                    best_of_4_y=best_of_4_y,
                    best_of_4_ci_lows=best_of_4_ci_lows,
                    best_of_4_ci_highs=best_of_4_ci_highs,
                )
                if sequential_envelopes_x is not None:
                    with open(f"{save_path}_sequential_envelopes_x.json", "w") as f:
                        json.dump(np.array(sequential_envelopes_x).tolist(), f)
                    with open(f"{save_path}_sequential_envelopes_y.json", "w") as f:
                        json.dump(np.array(sequential_envelopes_y).tolist(), f)
                    with open(f"{save_path}_sequential_envelopes_ci_lows.json", "w") as f:
                        json.dump(np.array(sequential_envelopes_ci_lows).tolist(), f)
                    with open(f"{save_path}_sequential_envelopes_ci_highs.json", "w") as f:
                        json.dump(np.array(sequential_envelopes_ci_highs).tolist(), f)
        except Exception as e:
            print(f"Warning: sequential envelopes failed (continuing): {e}")
    else:
        print("Skipping sequential envelopes computation because DISABLE_PLOT_SEQ=True")

    print("Computing aggregated methods with round robin...")
    plot_aggregated_methods_with_round_robin(
        data,
        f"{save_path}_with_round_robin",
        band="std",
        max_num_sims=max_num_sims,
        is_maximize=is_maximize,
        best_of_4_x=best_of_4_x,
        best_of_4_y=best_of_4_y,
        best_of_4_ci_lows=best_of_4_ci_lows,
        best_of_4_ci_highs=best_of_4_ci_highs,
        sequential_envelopes_x=sequential_envelopes_x,
        sequential_envelopes_y=sequential_envelopes_y,
        sequential_envelopes_ci_lows=sequential_envelopes_ci_lows,
        sequential_envelopes_ci_highs=sequential_envelopes_ci_highs,
    )

    # --- Fastness table (time-to-threshold + normalized AUC) ---
    # This captures "reaches max score faster" in a reportable way.
    try:
        fastness_save_path = f"{save_path}_with_round_robin"
        compute_and_save_fastness_table(
            data,
            save_path=fastness_save_path,
            is_maximize=is_maximize,
            alphas=[0.95, 0.99],
            max_num_sims=max_num_sims,
        )
        # Also append a row for Multi-Context Glia based on the aggregated best-of-4 envelope, if available.
        if best_of_4_y is not None:
            append_multi_context_glia_fastness_row(
                out_csv=f"{fastness_save_path}_fastness_metrics.csv",
                mcg_envelope=best_of_4_y[:max_num_sims],
                is_maximize=is_maximize,
                alphas=[0.95, 0.99],
            )
    except Exception as e:
        print(f"Warning: fastness table computation (or MCG append) failed (continuing): {e}")

    if sequential_envelopes_x is not None and sequential_envelopes_y is not None:
        try:
            print(f"Best of n with sequential envelopes start time: {time.time()}")
            _ = plot_best_of_n_envelope(
                data,
                f"{save_path}_best_of_n_with_sequential",
                max_num_sims=max_num_sims,
                is_maximize=is_maximize,
                multi_context_glia_x=sequential_envelopes_x,
                multi_context_glia_y=sequential_envelopes_y,
                multi_context_glia_ci_lows=sequential_envelopes_ci_lows,
                multi_context_glia_ci_highs=sequential_envelopes_ci_highs,
            )
        except Exception as e:
            print(f"Warning: best-of-n-with-sequential failed (continuing): {e}")

    print(f"Post-processing took {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()