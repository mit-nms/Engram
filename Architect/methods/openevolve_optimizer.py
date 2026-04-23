"""
OpenEvolve Optimization Method
https://github.com/algorithmicsuperintelligence/openevolve
"""

import json
import os
import sys
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path

from Architect.pricing_table import get_pricing

from .common import OptimizationMethod


class OpenEvolveOptimizer(OptimizationMethod):
    """Optimizer that uses OpenEvolve for evolutionary code optimization."""

    def __init__(
        self,
        *args,
        openevolve_config_path: str,
        initial_program_path: str,
        evaluator_path: str,
        **kwargs,
    ):
        """
        Initialize the OpenEvolve optimizer.

        Args:
            *args: Arguments to pass to parent class
            openevolve_config_path: Path to OpenEvolve config YAML file (required)
            initial_program_path: Path to initial program file (required)
            evaluator_path: Path to evaluator .py file or directory containing evaluator.py / evaluate.py
            **kwargs: Keyword arguments to pass to parent class. `resume_from` (if set)
                is interpreted as an OpenEvolve checkpoint directory rather than a JSON log;
                we strip it from kwargs so the parent class doesn't try to load it as JSON.
        """
        # Intercept resume_from: parent expects a JSON log file, but for OpenEvolve
        # this is a checkpoint directory. Stash it for our own use.
        self.checkpoint_path = kwargs.pop("resume_from", None)

        super().__init__(*args, **kwargs)

        self.openevolve_config_path = openevolve_config_path
        self.initial_program_path = initial_program_path
        # Resolve evaluator_path to a concrete .py file (mirrors load_task_from_paths)
        self.evaluator_file = self._resolve_evaluator_file(evaluator_path)

        if not os.path.exists(self.openevolve_config_path):
            raise ValueError(f"Config file not found: {self.openevolve_config_path}")
        if not os.path.exists(self.initial_program_path):
            raise ValueError(f"Initial program file not found: {self.initial_program_path}")
        if self.checkpoint_path and not os.path.isdir(self.checkpoint_path):
            raise ValueError(f"Checkpoint directory not found: {self.checkpoint_path}")
        
        # Initialize tracking
        self.all_iterations = []
        self.output_data = {}
        
        # OpenEvolve submodule path
        glia_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._glia_root = glia_root
        self.openevolve_path = os.path.join(glia_root, "Architect", "openevolve")
        
        if not os.path.exists(self.openevolve_path):
            raise RuntimeError(
                f"OpenEvolve submodule not found at {self.openevolve_path}. "
                "Please run: git submodule update --init --recursive"
            )

    @staticmethod
    def _resolve_evaluator_file(evaluator_path: str) -> str:
        """Resolve evaluator_path to an evaluator .py file.

        Accepts a file (used as-is) or a directory (looks for evaluator.py, then evaluate.py).
        """
        if os.path.isfile(evaluator_path):
            return evaluator_path
        for fname in ("evaluator.py", "evaluate.py"):
            candidate = os.path.join(evaluator_path, fname)
            if os.path.isfile(candidate):
                return candidate
        raise ValueError(
            f"Could not resolve evaluator file from {evaluator_path}: "
            "expected a .py file or a directory containing evaluator.py / evaluate.py"
        )

    def _run_openevolve(self, initial_program_path: str, evaluator_path: str, config_path: str, output_dir: Optional[str] = None, checkpoint_path: Optional[str] = None) -> str:
        """
        Run OpenEvolve evolution process.

        Args:
            initial_program_path: Path to initial program file
            evaluator_path: Path to evaluator bridge file
            config_path: Path to config file
            output_dir: Output directory for results (optional)
            checkpoint_path: Path to checkpoint directory to resume from (optional)

        Returns:
            Path to OpenEvolve output directory
        """
        # Find openevolve-run.py
        openevolve_run = os.path.join(self.openevolve_path, "openevolve-run.py")

        # Build command - standard OpenEvolve CLI format
        cmd = [
            sys.executable,
            openevolve_run,
            initial_program_path,
            evaluator_path,
            "--config",
            config_path,
        ]

        # Add output directory if specified
        if output_dir:
            cmd.extend(["--output", output_dir])

        # Add checkpoint path if specified
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])

        if self.debug:
            print(f"Running OpenEvolve: {' '.join(cmd)}")

        # Propagate usage-log path to OpenEvolve's LLM client (and its pool workers).
        # Also put the repo root on PYTHONPATH so SystemBench evaluators that do
        # `from Architect... import ...` resolve — the subprocess cwd is
        # self.openevolve_path, which is below the repo root.
        existing_pp = os.environ.get("PYTHONPATH", "")
        pypath = self._glia_root + (os.pathsep + existing_pp if existing_pp else "")
        env = {
            **os.environ,
            "OPENEVOLVE_USAGE_LOG": self._usage_log_path,
            "PYTHONPATH": pypath,
        }

        # Run OpenEvolve
        try:
            result = subprocess.run(
                cmd,
                cwd=self.openevolve_path,
                capture_output=not self.debug,
                text=True,
                check=True,
                env=env,
            )
            if self.debug:
                print("OpenEvolve completed successfully")
        except subprocess.CalledProcessError as e:
            if self.debug:
                print(f"OpenEvolve error: {e}")
                if e.stdout:
                    print(f"stdout: {e.stdout}")
                if e.stderr:
                    print(f"stderr: {e.stderr}")
            raise
        
        # Return the actual output directory used
        if output_dir:
            return output_dir
        # If not specified, OpenEvolve will use default from config or create openevolve_output
        return os.path.join(os.path.dirname(config_path), "openevolve_output")

    def optimize(self) -> Dict[str, Any]:
        """Run OpenEvolve optimization."""
        initial_program_path = os.path.abspath(self.initial_program_path)
        config_path = os.path.abspath(self.openevolve_config_path)
        evaluator_path = os.path.abspath(self.evaluator_file)

        # OpenEvolve writes its output (checkpoints/, best/, ...) alongside the
        # other per-task artifacts. Must be absolute so it isn't resolved relative
        # to the OpenEvolve subprocess cwd.
        output_dir = os.path.abspath(os.path.join(os.path.dirname(self.log_dir), "openevolve_output"))
        os.makedirs(output_dir, exist_ok=True)
        self._usage_log_path = os.path.join(output_dir, "openevolve_usage.jsonl")

        if self.debug:
            print(f"Running OpenEvolve with:")
            print(f"  Initial program: {initial_program_path}")
            print(f"  Evaluator: {evaluator_path}")
            print(f"  Config: {config_path}")
            print(f"  Output: {output_dir}")
            print(f"  Usage log: {self._usage_log_path}")
            if self.checkpoint_path:
                print(f"  Checkpoint: {self.checkpoint_path}")

        # Run OpenEvolve
        actual_output_dir = self._run_openevolve(
            initial_program_path,
            evaluator_path,
            config_path,
            output_dir,
            self.checkpoint_path,
        )

        # Read the best program saved by OpenEvolve under {output_dir}/best/
        best_code = None
        best_score = None
        if actual_output_dir and os.path.isdir(actual_output_dir):
            best_dir = os.path.join(actual_output_dir, "best")
            info_path = os.path.join(best_dir, "best_program_info.json")
            if os.path.isfile(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                metrics = info.get("metrics", {})
                combined_score = metrics.get("combined_score")
                best_score = combined_score if combined_score is not None else metrics.get("score")
            # Find the best program source file (any extension)
            if os.path.isdir(best_dir):
                for fname in os.listdir(best_dir):
                    if fname.startswith("best_program") and not fname.endswith(".json"):
                        with open(os.path.join(best_dir, fname)) as f:
                            best_code = f.read()
                        break

        usage_stats = self._aggregate_usage(self._usage_log_path)

        output_data = {
            "best_solution": {"code": best_code, "score": best_score},
            "output_dir": actual_output_dir,
            "all_iterations": [],
            "usage_stats": usage_stats,
        }

        self.save_results(output_data, method_name="openevolve")
        return output_data

    def _aggregate_usage(self, jsonl_path: str) -> Dict[str, Any]:
        """Aggregate the JSONL side-channel written by the patched OpenEvolve LLM client.

        Applies per-row pricing (rows carry their own model name, which matters
        for ensembles) and returns the canonical usage_stats schema plus a
        per-model breakdown and cumulative-by-iteration series for transparency.
        """
        total_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        per_model: Dict[str, Dict[str, Any]] = {}
        # Bucket per-iteration usage. Keys are 1-indexed iteration numbers;
        # rows with no iteration (e.g., logged outside the iteration loop) go to None.
        per_iteration: Dict[Any, Dict[str, Any]] = {}

        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    model = row.get("model", "unknown")
                    prompt = int(row.get("prompt_tokens", 0) or 0)
                    completion = int(row.get("completion_tokens", 0) or 0)
                    tokens = int(row.get("total_tokens", prompt + completion) or 0)
                    price = get_pricing(model)
                    cost = prompt / 1_000_000 * price["input"] + completion / 1_000_000 * price["output"]

                    total_cost += cost
                    total_prompt_tokens += prompt
                    total_completion_tokens += completion
                    total_tokens += tokens

                    slot = per_model.setdefault(
                        model,
                        {"calls": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
                    )
                    slot["calls"] += 1
                    slot["cost"] += cost
                    slot["prompt_tokens"] += prompt
                    slot["completion_tokens"] += completion

                    raw_iter = row.get("iteration")
                    # OpenEvolve emits 1-based iteration numbers — use as-is.
                    iter_key = raw_iter if isinstance(raw_iter, int) else None
                    bucket = per_iteration.setdefault(
                        iter_key,
                        {"calls": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
                    )
                    bucket["calls"] += 1
                    bucket["cost"] += cost
                    bucket["prompt_tokens"] += prompt
                    bucket["completion_tokens"] += completion

        for slot in per_model.values():
            slot["cost"] = round(slot["cost"], 6)

        # Build cumulative series over known iteration numbers, skipping None.
        iter_numbers = sorted(k for k in per_iteration.keys() if isinstance(k, int))
        cumulative: List[Dict[str, Any]] = []
        running = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        for n in iter_numbers:
            b = per_iteration[n]
            running["cost"] += b["cost"]
            running["prompt_tokens"] += b["prompt_tokens"]
            running["completion_tokens"] += b["completion_tokens"]
            running["calls"] += b["calls"]
            cumulative.append(
                {
                    "iteration": n,
                    "total_cost": round(running["cost"], 6),
                    "total_prompt_tokens": running["prompt_tokens"],
                    "total_completion_tokens": running["completion_tokens"],
                    "calls": running["calls"],
                }
            )

        return {
            "total_cost": round(total_cost, 6),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "model": self.model,
            "per_model_breakdown": per_model,
            "cumulative_by_iteration": cumulative,
        }

    def cost_at_iteration(self, k: int, jsonl_path: Optional[str] = None) -> Dict[str, Any]:
        """Return cumulative usage up to and including iteration `k` (1-indexed)."""
        stats = self._aggregate_usage(jsonl_path or self._usage_log_path)
        series = stats.get("cumulative_by_iteration", [])
        if not series:
            return {"iteration": k, "total_cost": 0.0, "total_prompt_tokens": 0,
                    "total_completion_tokens": 0, "calls": 0}
        # Last row with iteration <= k
        snapshot = None
        for row in series:
            if row["iteration"] <= k:
                snapshot = row
            else:
                break
        return snapshot or {"iteration": k, "total_cost": 0.0, "total_prompt_tokens": 0,
                            "total_completion_tokens": 0, "calls": 0}


