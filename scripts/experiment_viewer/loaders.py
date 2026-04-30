"""Loaders for Tree, OpenEvolve, and Handoff result directories. Do not assume aggregated_results.json exists."""
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import extract_scores_from_iterations, get_envelope_curve

# Compile regexes once at module level
_RE_SANITIZE = re.compile(r'(?<![A-Za-z0-9_])(-?Infinity|NaN)(?![A-Za-z0-9_])')
_RE_ROUNDS = re.compile(r'(\d+)rounds')
_RE_ITERATIONS = re.compile(r'(\d+)iterations')
_RE_RUN_SUFFIX = re.compile(r'_run_?\d+$')
_RE_GEN_INDEX = re.compile(r'^generated_(\d+)$')


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        text = p.read_text(errors="replace")
        text = _RE_SANITIZE.sub('null', text)
        return json.loads(text)
    except Exception:
        return None


def _is_valid_score(s: float) -> bool:
    """Return True if s is a finite, non-NaN float that can be used as a meaningful score."""
    return isinstance(s, float) and math.isfinite(s)


# ---- Discovery: do not require aggregated_results.json ----


def get_run_group_key(path: str) -> Optional[str]:
    """Compute a stable group key for runs that share the same prefix (e.g. ..._run1, ..._run2).
    Uses the run root's parent directory name with trailing _runN or _run_N stripped."""
    p = Path(path).resolve()
    run_dir = p.parent if p.name and p.parent else p
    name = run_dir.name
    normalized = _RE_RUN_SUFFIX.sub("", name)
    if normalized == name:
        return None
    parent_path = str(run_dir.parent)
    return f"{parent_path}|{normalized}"


def discover_tree_runs(root: Path) -> List[Dict[str, str]]:
    """Find Tree runs under root. A run has cloudcast/logs/*-deepagents_tree_*rounds.json."""
    runs: List[Dict[str, str]] = []
    root = Path(root)
    if not root.is_dir():
        return runs
    for logs_dir in root.rglob("logs"):
        if not logs_dir.is_dir():
            continue
        run_root = logs_dir.parent
        run_id = run_root.parent.name if run_root.parent else run_root.name
        agg_file = run_root / "aggregated_results.json"
        if agg_file.exists():
            runs.append({
                "id": run_id,
                "name": f"Tree {run_id}",
                "type": "tree",
                "path": str(run_root),
                "canonical_json": str(agg_file),
            })
            continue
        tree_files = list(logs_dir.glob("*-deepagents_tree_*rounds.json"))
        if not tree_files:
            continue

        def round_count(f: Path) -> int:
            m = _RE_ROUNDS.search(f.name)
            return int(m.group(1)) if m else 0

        best_file = max(tree_files, key=round_count)
        runs.append({
            "id": run_id,
            "name": f"Tree {run_id}",
            "type": "tree",
            "path": str(run_root),
            "canonical_json": str(best_file),
        })
    return runs


def discover_openevolve_runs(root: Path) -> List[Dict[str, str]]:
    """Find OpenEvolve runs: have checkpoints/ and logs/openevolve_*.log. No aggregated_results.json."""
    runs: List[Dict[str, str]] = []
    root = Path(root)
    if not root.is_dir():
        return runs
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        checkpoints = candidate / "checkpoints"
        logs_dir = candidate / "logs"
        log_files = list(logs_dir.glob("openevolve_*.log")) if logs_dir.is_dir() else []
        if checkpoints.is_dir() and log_files:
            runs.append({
                "id": candidate.name,
                "name": f"OpenEvolve {candidate.name}",
                "type": "openevolve",
                "path": str(candidate),
            })
    return runs


def discover_handoff_runs(root: Path) -> List[Dict[str, str]]:
    """Find Handoff runs: cloudcast/logs/*-agentic_handoff_*iterations.json."""
    runs: List[Dict[str, str]] = []
    root = Path(root)
    if not root.is_dir():
        return runs
    for logs_dir in root.rglob("logs"):
        if not logs_dir.is_dir():
            continue
        run_root = logs_dir.parent
        run_id = run_root.parent.name if run_root.parent else run_root.name
        agg_file = run_root / "aggregated_results.json"
        if agg_file.exists():
            runs.append({
                "id": run_id,
                "name": f"Handoff {run_id}",
                "type": "handoff",
                "path": str(run_root),
                "canonical_json": str(agg_file),
            })
            continue
        handoff_files = list(logs_dir.glob("*-agentic_handoff_*iterations.json"))
        if not handoff_files:
            continue

        def iter_count(f: Path) -> int:
            m = _RE_ITERATIONS.search(f.name)
            return int(m.group(1)) if m else 0

        best_file = max(handoff_files, key=iter_count)
        runs.append({
            "id": run_id,
            "name": f"Handoff {run_id}",
            "type": "handoff",
            "path": str(run_root),
            "canonical_json": str(best_file),
        })
    return runs


def discover_all_runs(roots: List[str]) -> List[Dict[str, str]]:
    """Discover Tree, OpenEvolve, and Handoff runs from a list of root directories."""
    out: List[Dict[str, str]] = []
    seen_paths: set[str] = set()
    for r in roots:
        p = Path(r).resolve()
        for run in discover_tree_runs(p) + discover_openevolve_runs(p) + discover_handoff_runs(p):
            key = run["path"]
            if key not in seen_paths:
                seen_paths.add(key)
                run["group_key"] = get_run_group_key(run["path"])
                out.append(run)
    return out


# ---- Load single run (normalized payload) ----


def load_tree_run(run_path: str, canonical_json: Optional[str] = None) -> Dict[str, Any]:
    """Load Tree run from cloudcast dir. Primary: logs/*-deepagents_tree_*rounds.json (max rounds)."""
    base = Path(run_path)
    logs_dir = base / "logs"
    if not logs_dir.is_dir():
        return {"error": "logs dir not found", "run_type": "tree", "run_path": run_path}

    if canonical_json:
        tree_file = Path(canonical_json)
    else:
        tree_files = list(logs_dir.glob("*-deepagents_tree_*rounds.json"))
        if not tree_files:
            return {"error": "no *-deepagents_tree_*rounds.json found", "run_type": "tree", "run_path": run_path}
        tree_file = max(tree_files, key=lambda f: int(m.group(1)) if (m := _RE_ROUNDS.search(f.name)) else 0)

    data = _read_json(tree_file)
    if not data:
        return {"error": f"failed to read {tree_file}", "run_type": "tree", "run_path": run_path}

    best = data.get("best_solution") or {}
    iterations = data.get("all_iterations") or []
    baselines = data.get("baselines") or {}

    scores = extract_scores_from_iterations(data)
    envelope = get_envelope_curve(scores, is_maximize=True) if scores else []

    log_path = None
    round1 = logs_dir / "round_1"
    if round1.is_dir():
        for task in round1.iterdir():
            if not task.is_dir():
                continue
            for sub in task.iterdir():
                if not sub.is_dir():
                    continue
                logs2 = sub / "logs"
                if not logs2.is_dir():
                    continue
                console = logs2 / "console_output.log"
                if console.exists():
                    log_path = str(console)
                    break
                args_file = logs2 / "args.json"
                if args_file.exists():
                    log_path = str(args_file)
                    break
            if log_path:
                break

    return {
        "run_type": "tree",
        "run_path": run_path,
        "best_solution": {"code": best.get("code", ""), "score": best.get("score"), "node_id": best.get("node_id")},
        "all_iterations": iterations,
        "baselines": baselines,
        "scores": scores,
        "envelope": envelope,
        "log_path": log_path,
        "total_iterations": len(iterations),
    }


def _openevolve_build_from_generated_programs(openevolve_dir: Path) -> Optional[Dict[str, Any]]:
    """Build Vidur-like payload from generated_programs/ (no aggregated_results.json)."""
    gen_dir = openevolve_dir / "generated_programs"
    if not gen_dir.is_dir():
        return None
    programs: List[Dict[str, Any]] = []
    for gdir in sorted(
        gen_dir.glob("generated_*"),
        key=lambda x: int(m.group(1)) if (m := _RE_GEN_INDEX.match(x.name)) else 0,
    ):
        jsons = list(gdir.glob("*.json"))
        py_file = gdir / "program.py"
        code = None
        if py_file.exists():
            try:
                code = py_file.read_text()
            except Exception:
                pass
        for jf in jsons:
            obj = _read_json(jf)
            if not obj:
                continue
            if code is None:
                code = obj.get("code") or ""
            it = obj.get("iteration_found") or obj.get("generation")
            if it is None and gdir.name.startswith("generated_"):
                m = _RE_GEN_INDEX.match(gdir.name)
                it = int(m.group(1)) if m else len(programs)
            metrics = obj.get("metrics") or {}
            score = metrics.get("combined_score", float("-inf"))
            programs.append({"code": code, "score": score, "iteration": it or len(programs), "metrics": metrics})
            break
        if not jsons and code:
            programs.append({"code": code, "score": float("-inf"), "iteration": len(programs), "metrics": {}})
    if not programs:
        return None
    programs.sort(key=lambda x: x["iteration"])
    best_score = float("-inf")
    best_code = ""
    all_iters = []
    for i, p in enumerate(programs):
        s = p["score"]
        if _is_valid_score(s) and s > best_score:
            best_score = s
            best_code = p["code"]
        all_iters.append({
            "iteration": i + 1,
            "score": p["score"],
            "code": p["code"],
            "success": p.get("metrics", {}).get("runs_successfully", 0) > 0.5,
        })
    if not best_code and programs:
        best_code = programs[-1]["code"]
        best_score = programs[-1]["score"]
    scores = [x["score"] for x in programs]
    envelope = get_envelope_curve(scores, is_maximize=True)
    return {
        "best_solution": {"code": best_code, "score": best_score},
        "all_iterations": all_iters,
        "scores": scores,
        "envelope": envelope,
        "baselines": {},
    }


def load_openevolve_run(run_path: str) -> Dict[str, Any]:
    """Load OpenEvolve run from generated_programs/ or checkpoints/. Do not expect aggregated_results.json."""
    base = Path(run_path)
    payload = _openevolve_build_from_generated_programs(base)
    if not payload:
        payload = _openevolve_build_from_checkpoints(base)
    if not payload:
        return {"error": "no generated_programs or checkpoints data", "run_type": "openevolve", "run_path": run_path}

    log_path = None
    logs_dir = base / "logs"
    if logs_dir.is_dir():
        for f in logs_dir.glob("openevolve_*.log"):
            log_path = str(f)
            break

    return {
        "run_type": "openevolve",
        "run_path": run_path,
        "best_solution": payload["best_solution"],
        "all_iterations": payload["all_iterations"],
        "baselines": payload.get("baselines", {}),
        "scores": payload["scores"],
        "envelope": payload["envelope"],
        "log_path": log_path,
        "total_iterations": len(payload["all_iterations"]),
    }


def _openevolve_build_from_checkpoints(openevolve_dir: Path) -> Optional[Dict[str, Any]]:
    """Build from checkpoints/ (best_program_info.json, programs/*.json)."""
    cp_dir = openevolve_dir / "checkpoints"
    if not cp_dir.is_dir():
        return None
    checkpoints = sorted(
        [d for d in cp_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]) if len(d.name.split("_")) > 1 and d.name.split("_")[1].isdigit() else 0,
    )
    all_iters: List[Dict[str, Any]] = []
    best_code = ""
    best_score = float("-inf")
    for cp in checkpoints:
        info = _read_json(cp / "best_program_info.json")
        if info:
            metrics = info.get("metrics") or {}
            s = metrics.get("combined_score") or metrics.get("cost_score") or float("-inf")
            code = ""
            if (cp / "best_program.py").exists():
                try:
                    code = (cp / "best_program.py").read_text()
                except Exception:
                    pass
            if not code and info.get("code"):
                code = info["code"]
            all_iters.append({"iteration": len(all_iters) + 1, "score": s, "code": code, "success": True})
            if _is_valid_score(s) and s > best_score:
                best_score = s
                best_code = code
    if not all_iters:
        return None
    if not best_code:
        best_code = all_iters[-1].get("code", "")
        best_score = all_iters[-1].get("score", float("-inf"))
    scores = [x["score"] for x in all_iters]
    envelope = get_envelope_curve(scores, is_maximize=True)
    return {
        "best_solution": {"code": best_code, "score": best_score},
        "all_iterations": all_iters,
        "scores": scores,
        "envelope": envelope,
        "baselines": {},
    }


def load_handoff_run(run_path: str, canonical_json: Optional[str] = None) -> Dict[str, Any]:
    """Load Handoff run. Primary: logs/*-agentic_handoff_*iterations.json (max iterations)."""
    base = Path(run_path)
    logs_dir = base / "logs"
    if not logs_dir.is_dir():
        return {"error": "logs dir not found", "run_type": "handoff", "run_path": run_path}

    if canonical_json:
        handoff_file = Path(canonical_json)
    else:
        handoff_files = list(logs_dir.glob("*-agentic_handoff_*iterations.json"))
        if not handoff_files:
            return {"error": "no *-agentic_handoff_*iterations.json found", "run_type": "handoff", "run_path": run_path}
        handoff_file = max(handoff_files, key=lambda f: int(m.group(1)) if (m := _RE_ITERATIONS.search(f.name)) else 0)

    data = _read_json(handoff_file)
    if not data:
        return {"error": f"failed to read {handoff_file}", "run_type": "handoff", "run_path": run_path}

    best = data.get("best_solution") or {}
    iterations = data.get("all_iterations") or []
    baselines = data.get("baselines") or {}
    scores = extract_scores_from_iterations(data)
    envelope = get_envelope_curve(scores, is_maximize=True) if scores else []

    log_path = None
    console = logs_dir / "console_output.log"
    if console.exists():
        log_path = str(console)

    return {
        "run_type": "handoff",
        "run_path": run_path,
        "best_solution": {"code": best.get("code", ""), "score": best.get("score")},
        "all_iterations": iterations,
        "baselines": baselines,
        "scores": scores,
        "envelope": envelope,
        "log_path": log_path,
        "total_iterations": len(iterations),
    }


def load_run(run_spec: Dict[str, str]) -> Dict[str, Any]:
    """Load a single run by spec (from discover_*)."""
    run_type = run_spec.get("type", "")
    path = run_spec.get("path", "")
    canonical = run_spec.get("canonical_json")
    if run_type == "tree":
        return load_tree_run(path, canonical_json=canonical)
    if run_type == "openevolve":
        return load_openevolve_run(path)
    if run_type == "handoff":
        return load_handoff_run(path, canonical_json=canonical)
    return {"error": f"unknown run type {run_type}", "run_type": run_type, "run_path": path}
