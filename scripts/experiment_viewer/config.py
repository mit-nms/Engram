"""Config: result roots from env or YAML. Used by app.py."""
import os
from pathlib import Path
from typing import List


def get_result_roots() -> List[str]:
    """Result root directories. Env VIEWER_RESULT_ROOTS (comma-separated), then viewer_config.yaml, then defaults."""
    env = os.environ.get("VIEWER_RESULT_ROOTS", "").strip()
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    # Optional YAML
    yaml_path = get_viewer_config_path()
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "result_roots" in cfg:
                roots = cfg["result_roots"]
                if isinstance(roots, list):
                    return [str(p) for p in roots]
        except Exception:
            pass
    # Defaults: common Glia result locations (user can override via env)
    defaults = [
        "/data2/projects/pantea-work/Glia/results/prompt_ablation_results_gpt_5.2",
        "/data2/projects/pantea-work/Glia/results/prompt_ablation_results",
        "/data2/projects/pantea-work/Glia/results/OpenEvolve_Results",
    ]
    return [d for d in defaults if Path(d).is_dir()]


def get_viewer_config_path() -> Path:
    """Optional viewer_config.yaml next to this package."""
    return Path(__file__).resolve().parent / "viewer_config.yaml"
