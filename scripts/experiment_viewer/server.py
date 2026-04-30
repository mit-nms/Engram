"""
Flask backend for Experiment Results Viewer.
Serves API for config (result_roots), runs list, run data, and log content.
Enter result_roots in the UI; they are saved to config.json.
"""

import json
from math import isfinite
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, render_template
from werkzeug.exceptions import HTTPException

# Ensure scripts/ is on path so experiment_viewer is importable
_here = Path(__file__).resolve().parent
import sys
_scripts = _here.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from experiment_viewer.loaders import discover_all_runs, load_run
from experiment_viewer.utils import MAX_LOG_CHARS

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

CONFIG_PATH = _here / "config.json"
DEFAULT_PORT = 5005


@app.errorhandler(Exception)
def _handle_api_error(err):
    """Return JSON error for API routes, re-raise for non-API endpoints."""
    if request.path.startswith("/api/"):
        code = err.code if isinstance(err, HTTPException) else 500
        return jsonify({"error": str(err)}), code
    raise err


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"result_roots": [], "port": DEFAULT_PORT}


def save_config(c: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(c, f, indent=2)


def get_result_roots() -> list:
    roots = load_config().get("result_roots") or []
    return [r for r in roots if isinstance(r, str) and r.strip()]


def _sanitize(obj):
    """Iteratively replace non-finite float values with None for JSON compatibility."""
    stack = [(obj, None, None)]  # (value, parent_container, key_or_index)
    root_result = [None]

    def _visit(value, parent, key):
        if isinstance(value, float):
            result = value if isfinite(value) else None
            if parent is None:
                root_result[0] = result
            elif isinstance(parent, dict):
                parent[key] = result
            else:
                parent[key] = result
            return
        if isinstance(value, dict):
            new_dict = {}
            if parent is None:
                root_result[0] = new_dict
            elif isinstance(parent, dict):
                parent[key] = new_dict
            else:
                parent[key] = new_dict
            for k, v in value.items():
                stack.append((v, new_dict, k))
            return
        if isinstance(value, list):
            new_list = [None] * len(value)
            if parent is None:
                root_result[0] = new_list
            elif isinstance(parent, dict):
                parent[key] = new_list
            else:
                parent[key] = new_list
            for i, v in enumerate(value):
                stack.append((v, new_list, i))
            return
        # Scalar (int, str, bool, None)
        if parent is None:
            root_result[0] = value
        elif isinstance(parent, dict):
            parent[key] = value
        else:
            parent[key] = value

    # Process iteratively (avoids Python recursion limit on deep structures)
    _visit(obj, None, None)
    while stack:
        val, par, k = stack.pop()
        _visit(val, par, k)
    return root_result[0]


def _is_safe_path(path: Path, root: Path) -> bool:
    """Return True if resolved path is a descendant of root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def api_config_get():
    c = load_config()
    return jsonify({"result_roots": c.get("result_roots", []), "port": c.get("port", DEFAULT_PORT)})


@app.route("/api/config", methods=["POST"])
def api_config_post():
    data = request.get_json() or {}
    roots = data.get("result_roots")
    if roots is not None:
        if isinstance(roots, list):
            roots = [str(r).strip() for r in roots if str(r).strip()]
        else:
            roots = []
    c = load_config()
    if roots is not None:
        c["result_roots"] = roots
    if "port" in data:
        c["port"] = int(data["port"]) if data["port"] else DEFAULT_PORT
    save_config(c)
    return jsonify({"ok": True, "result_roots": c.get("result_roots", [])})


@app.route("/api/runs")
def api_runs():
    roots = get_result_roots()
    if not roots:
        return jsonify([])
    runs = discover_all_runs(roots)
    return jsonify(runs)


@app.route("/api/run", methods=["POST"])
def api_run():
    """Load a single run. Body: { path, type, canonical_json? }."""
    data = request.get_json() or {}
    path = data.get("path")
    run_type = data.get("type")
    if not path or not run_type:
        return jsonify({"error": "path and type required"}), 400
    spec = {"path": path, "type": run_type, "canonical_json": data.get("canonical_json") or None}
    result = load_run(spec)
    return jsonify(_sanitize(result))


@app.route("/api/aggregate", methods=["POST"])
def api_aggregate():
    """Load multiple runs and return aggregate: best score/code ever and per-run envelopes.
    Body: { runs: [ { path, type, canonical_json? }, ... ] }."""
    data = request.get_json() or {}
    specs = data.get("runs") or []
    if not specs:
        return jsonify({"error": "runs list required"}), 400
    loaded = []
    best_score_ever = None
    best_code_ever = ""
    best_run_path = None
    for spec in specs:
        path = spec.get("path")
        run_type = spec.get("type")
        if not path or not run_type:
            continue
        s = {"path": path, "type": run_type, "canonical_json": spec.get("canonical_json") or None}
        result = load_run(s)
        if result.get("error"):
            continue
        run_path = result.get("run_path", path)
        best = result.get("best_solution") or {}
        score = best.get("score")
        if score is not None and (best_score_ever is None or score > best_score_ever):
            best_score_ever = score
            best_code_ever = best.get("code") or ""
            best_run_path = run_path
        run_name = Path(run_path).parent.name if Path(run_path).parent.name else run_path
        loaded.append({
            "run_path": run_path,
            "run_name": run_name,
            "envelope": result.get("envelope") or [],
            "scores": result.get("scores") or [],
            "best_solution": best,
            "baselines": result.get("baselines") or {},
            "total_iterations": result.get("total_iterations", 0),
        })
    return jsonify(_sanitize({
        "run_type": "aggregate",
        "best_score_ever": best_score_ever,
        "best_code_ever": best_code_ever,
        "best_run_path": best_run_path,
        "runs": loaded,
    }))


@app.route("/api/log")
def api_log():
    """Return log file content. Query: path=... (URL-encoded file path).

    Only serves files that are within the configured result roots to prevent
    path traversal.
    """
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "path required"}), 400
    p = Path(path).resolve()
    # Validate path is within one of the configured result roots
    roots = get_result_roots()
    if not roots or not any(_is_safe_path(p, Path(r).resolve()) for r in roots):
        return jsonify({"error": "path not allowed"}), 403
    if not p.exists() or not p.is_file():
        return jsonify({"error": "file not found", "content": ""})
    try:
        text = p.read_text(errors="replace")
        truncated = len(text) > MAX_LOG_CHARS
        if truncated:
            text = text[-MAX_LOG_CHARS:]
        return jsonify({"content": text, "truncated": truncated})
    except Exception as e:
        return jsonify({"error": str(e), "content": ""})


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    config = load_config()
    port = int(config.get("port", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port, debug=False)
