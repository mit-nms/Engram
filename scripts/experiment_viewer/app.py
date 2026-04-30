"""
Experiment Results Viewer - Streamlit UI.

Run: streamlit run scripts/experiment_viewer/app.py
  (from repo root)
Or:  python -m streamlit run app.py  (from scripts/experiment_viewer/)

Config: set VIEWER_RESULT_ROOTS to comma-separated paths, or use defaults in config.py.
"""
import sys
from pathlib import Path

# Ensure scripts/ is on path so experiment_viewer can be imported (scripts has no __init__.py)
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import streamlit as st
import plotly.graph_objects as go

from experiment_viewer.config import get_result_roots
from experiment_viewer.loaders import discover_all_runs, load_run
from experiment_viewer.utils import MAX_LOG_CHARS, MAX_ITERATIONS_DISPLAY

st.set_page_config(page_title="Experiment Results Viewer", layout="wide")

st.title("Experiment Results Viewer")
st.caption("Tree, OpenEvolve, Handoff runs — no aggregated_results.json required.")

# Config: roots
roots = get_result_roots()
if not roots:
    st.warning("No result roots found. Set VIEWER_RESULT_ROOTS (comma-separated paths) or add default paths in config.py.")
    st.stop()

with st.sidebar:
    st.subheader("Result roots")
    for r in roots:
        st.text(r)
    if st.button("Refresh runs"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data(ttl=60)
def get_runs():
    return discover_all_runs(roots)

runs = get_runs()
if not runs:
    st.info("No runs discovered under the configured roots. Check that paths contain Tree (cloudcast/logs/*-deepagents_tree_*rounds.json), OpenEvolve (checkpoints/ + logs/openevolve_*.log), or Handoff (cloudcast/logs/*-agentic_handoff_*iterations.json) directories.")
    st.stop()

# Run selector
run_options = {f"{r['name']} ({r['type']})": r for r in runs}
selected_label = st.selectbox("Select run", list(run_options.keys()), index=0)
run_spec = run_options[selected_label]

# Load run data (cache by path + type; spec dict is not hashable)
@st.cache_data(ttl=30)
def get_run_data(path: str, run_type: str, canonical_json: str = ""):
    spec = {"path": path, "type": run_type, "canonical_json": canonical_json or None}
    return load_run(spec)

data = get_run_data(
    run_spec["path"],
    run_spec["type"],
    run_spec.get("canonical_json", ""),
)

if data.get("error"):
    st.error(data["error"])
    st.stop()

# ---- Summary cards ----
best = data.get("best_solution") or {}
scores = data.get("scores") or []
envelope = data.get("envelope") or []
total = data.get("total_iterations", 0)
baselines = data.get("baselines") or {}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Best score", f"{best.get('score'):.6f}" if best.get('score') is not None else "—")
with col2:
    st.metric("Total iterations", total)
with col3:
    success_count = sum(1 for i in (data.get("all_iterations") or []) if i.get("success"))
    st.metric("Successful", f"{success_count} / {total}" if total else "—")
with col4:
    st.metric("Run type", data.get("run_type", "—"))

# ---- Score evolution chart: raw + max envelope + baselines ----
if scores or envelope:
    st.subheader("Score evolution")
    fig = go.Figure()
    x = list(range(1, len(scores) + 1))
    n = len(scores)
    if scores:
        fig.add_trace(go.Scatter(x=x, y=scores, mode="lines+markers", name="Raw score", line=dict(dash="dot"), marker=dict(size=4)))
    if envelope:
        fig.add_trace(go.Scatter(x=x, y=envelope, mode="lines", name="Best so far (envelope)", line=dict(width=2)))
    # Add baselines as horizontal reference lines
    for bl_name, bl_val in baselines.items():
        bl_score = bl_val.get("score") if isinstance(bl_val, dict) else bl_val
        if bl_score is not None and n > 0:
            fig.add_trace(go.Scatter(
                x=[1, n], y=[bl_score, bl_score],
                mode="lines", name=f"Baseline: {bl_name}",
                line=dict(dash="dash", width=1),
            ))
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Score", hovermode="x unified", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---- Baselines comparison ----
if baselines:
    st.subheader("Baselines")
    bl_names = list(baselines.keys())
    bl_scores = []
    for k in bl_names:
        v = baselines[k]
        if isinstance(v, dict):
            bl_scores.append(v.get("score"))
        else:
            bl_scores.append(v)
    if bl_scores:
        import pandas as pd
        df = pd.DataFrame({"Baseline": bl_names, "Score": bl_scores})
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---- Best code ----
st.subheader("Best solution code")
best_code = best.get("code") or ""
if best_code:
    st.code(best_code, language="python")
else:
    st.info("No code in best_solution.")

# ---- Log / conversation ----
log_path = data.get("log_path")
if log_path:
    st.subheader("Log")
    path = Path(log_path)
    if path.exists():
        try:
            text = path.read_text(errors="replace")
            if len(text) > MAX_LOG_CHARS:
                st.caption(f"Showing last {MAX_LOG_CHARS:,} characters (total {len(text):,}).")
                text = text[-MAX_LOG_CHARS:]
            st.text_area("Log content", text, height=300, key="log_content")
        except Exception as e:
            st.error(str(e))
    else:
        st.caption(f"File not found: {log_path}")
else:
    st.caption("No log path for this run.")

# ---- Iterations table (optional, collapsed) ----
iters = data.get("all_iterations") or []
expander_label = (
    f"Iterations (showing {MAX_ITERATIONS_DISPLAY} of {len(iters)})"
    if len(iters) > MAX_ITERATIONS_DISPLAY
    else f"Iterations ({len(iters)})"
)
with st.expander(expander_label):
    if iters:
        import pandas as pd
        rows = []
        for i, it in enumerate(iters[:MAX_ITERATIONS_DISPLAY]):
            row = {"#": i + 1, "score": it.get("score"), "success": it.get("success")}
            if it.get("round") is not None:
                row["round"] = it["round"]
            if it.get("node_id"):
                row["node_id"] = it["node_id"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if len(iters) > MAX_ITERATIONS_DISPLAY:
            st.caption(f"… and {len(iters) - MAX_ITERATIONS_DISPLAY} more iterations not shown.")
    else:
        st.caption("No iterations.")
