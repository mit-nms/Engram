"""
Frontier-CS Evaluator Wrapper for Glia Integration

Wraps the frontier_cs.SingleEvaluator to conform to Glia's Evaluator ABC,
supporting both algorithmic (C++) and research (Python) problem tracks.
"""

import os
import sys
import time
import yaml
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from SystemBench.evaluator import Evaluator
from Architect.types import DesignConfig, Scenario, CodeBlock


class FrontierCSEvaluator(Evaluator):
    """
    Evaluator for Frontier-CS benchmark problems (algorithmic and research tracks).

    Delegates evaluation to frontier_cs.SingleEvaluator which handles
    Docker containers, go-judge, and GPU resource management.
    """

    def __init__(
        self,
        track: str,
        problem_id: Union[str, int],
        target_name: str = None,
        backend: str = "docker",
        timeout: Optional[int] = None,
        judge_url: str = "http://localhost:8081",
        submodule_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Frontier-CS evaluator.

        Args:
            track: "algorithmic" or "research"
            problem_id: Problem identifier (e.g., "flash_attn" or 42)
            target_name: Name of the code block to optimize.
                         Defaults to "Solution" for research, "solution" for algorithmic.
            backend: Evaluation backend ("docker" or "skypilot")
            timeout: Timeout in seconds for evaluation. Auto-discovered from problem config if None.
            judge_url: URL for the go-judge server (algorithmic track only)
            submodule_path: Path to the Frontier-CS repo. Defaults to frontier_cs_repo/ beside this file.
        """
        super().__init__(**kwargs)

        self.track = track
        self.problem_id = problem_id
        self.backend = backend
        self.judge_url = judge_url

        # Auto-set target name based on track
        if target_name is None:
            self.target_name = "Solution" if track == "research" else "solution"
        else:
            self.target_name = target_name

        # Resolve submodule path
        if submodule_path is None:
            self.submodule_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "frontier_cs_repo",
            )
        else:
            self.submodule_path = os.path.abspath(submodule_path)

        # Verify submodule exists
        if not os.path.isdir(self.submodule_path):
            raise FileNotFoundError(
                f"Frontier-CS submodule not found at {self.submodule_path}. "
                "Run: git submodule update --init"
            )

        # Set timeout from problem config or default
        self._timeout_seconds = timeout if timeout is not None else self._discover_timeout()

        # Lazily initialized SingleEvaluator
        self._fcs_evaluator = None

        # Cached problem statement
        self._problem_statement: Optional[str] = None

        # Set up the target code block
        self._setup_code_block()

    @property
    def fcs_evaluator(self):
        """Lazily create the SingleEvaluator to avoid Docker overhead at import time."""
        if self._fcs_evaluator is None:
            try:
                from frontier_cs import SingleEvaluator
            except ImportError:
                # Try importing from the submodule source directly
                src_path = os.path.join(self.submodule_path, "src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from frontier_cs import SingleEvaluator

            self._fcs_evaluator = SingleEvaluator(
                backend=self.backend,
                base_dir=Path(self.submodule_path),
                judge_url=self.judge_url,
                timeout=self._timeout_seconds,
                register_cleanup=False,
            )

            # Fix GPU detection: nvidia-smi exits with code 14 on infoROM corruption
            # even when the GPU is fully functional. The runner's has_gpu property uses
            # returncode == 0, so it incorrectly reports no GPU. Patch it here using
            # --query-gpu which exits 0 when GPUs are present regardless of infoROM state.
            if self.track == "research" and self.backend == "docker":
                import subprocess as _sp
                try:
                    r = _sp.run(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, timeout=5,
                    )
                    self._fcs_evaluator.docker_runner._has_gpu = (
                        r.returncode == 0 and bool(r.stdout.strip())
                    )
                except Exception:
                    pass  # Leave _has_gpu as None (auto-detected by runner)

        return self._fcs_evaluator

    def _discover_timeout(self) -> int:
        """Discover timeout from problem config.yaml or use defaults."""
        config_path = self._get_problem_config_path()
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if self.track == "algorithmic":
                    time_str = config.get("time", "2s")
                    seconds = int(time_str.replace("s", "").replace("m", ""))
                    return max(30, seconds * 10)
                else:
                    runtime = config.get("runtime", {})
                    if isinstance(runtime, dict):
                        return runtime.get("timeout_seconds", 600)
            except Exception:
                pass
        return 30 if self.track == "algorithmic" else 600

    def _get_problem_config_path(self) -> Optional[str]:
        """Get path to the problem's config.yaml in the Frontier-CS repo."""
        if self.track == "algorithmic":
            return os.path.join(
                self.submodule_path, "algorithmic", "problems",
                str(self.problem_id), "config.yaml",
            )
        else:
            return os.path.join(
                self.submodule_path, "research", "problems",
                str(self.problem_id), "config.yaml",
            )

    def _get_default_implementation(self) -> str:
        """Get a baseline implementation from the Frontier-CS solutions directory."""
        if self.track == "research":
            solutions_dir = os.path.join(
                self.submodule_path, "research", "solutions",
                str(self.problem_id),
            )
            ext = ".py"
        else:
            solutions_dir = os.path.join(
                self.submodule_path, "algorithmic", "solutions",
                str(self.problem_id),
            )
            ext = ".cpp"

        if os.path.isdir(solutions_dir):
            files = sorted([f for f in os.listdir(solutions_dir) if f.endswith(ext)])
            if files:
                with open(os.path.join(solutions_dir, files[0]), "r") as f:
                    return f.read()

        # Fallback: minimal stub
        if self.track == "research":
            return (
                "class Solution:\n"
                "    def solve(self, spec_path=None):\n"
                "        return {}\n"
            )
        else:
            return (
                "#include <bits/stdc++.h>\n"
                "using namespace std;\n"
                "int main() {\n"
                "    return 0;\n"
                "}\n"
            )

    def _setup_code_block(self):
        """Set up the target code block for optimization."""
        initial_code = self._get_default_implementation()
        self._code_blocks[self.target_name] = CodeBlock(
            name=self.target_name,
            description=f"Optimizable code for Frontier-CS {self.track}/{self.problem_id}",
            evolvable_code=initial_code,
            helper_code="",
        )

    def get_system_model(self) -> str:
        """Get the problem description from Frontier-CS."""
        if self._problem_statement is None:
            # Try reading directly from the submodule first (avoids Docker init)
            readme_path = self._get_readme_path()
            if readme_path and os.path.exists(readme_path):
                with open(readme_path, "r") as f:
                    self._problem_statement = f.read()
            else:
                # Fallback to SingleEvaluator API
                try:
                    self._problem_statement = self.fcs_evaluator.get_problem_statement(
                        self.track, self.problem_id
                    )
                except Exception:
                    self._problem_statement = (
                        f"Frontier-CS {self.track} problem: {self.problem_id}\n"
                        f"Optimize the {self.target_name} to maximize the score (0-100)."
                    )
        return self._problem_statement

    def _get_readme_path(self) -> Optional[str]:
        """Get path to the problem's readme/statement file."""
        if self.track == "algorithmic":
            return os.path.join(
                self.submodule_path, "algorithmic", "problems",
                str(self.problem_id), "statement.txt",
            )
        else:
            # Research problems use "readme" (no extension)
            path = os.path.join(
                self.submodule_path, "research", "problems",
                str(self.problem_id), "readme",
            )
            if os.path.exists(path):
                return path
            # Some may use readme.md
            path_md = path + ".md"
            if os.path.exists(path_md):
                return path_md
            return path

    def _format_result_summary(self, result) -> str:
        """Build a judge-style summary from per-case results.

        Shows only what a competitive programmer would see after submitting
        to an online judge: total score and per-case verdict/points/time/memory.
        """
        cases = getattr(result, "metadata", {}).get("cases", [])
        if not cases:
            return ""

        score = result.score if result.score is not None else 0.0

        lines = [f"Score: {score:.2f} / 100", ""]

        for i, c in enumerate(cases):
            status = c.get("status", "Unknown")
            case_score = c.get("scoreRatio", 0.0)
            time_ns = c.get("time", "")
            memory = c.get("memory", "")

            # Format time as ms if available
            time_str = ""
            if time_ns:
                try:
                    time_str = f"{int(time_ns) / 1_000_000:.0f} ms"
                except (ValueError, TypeError):
                    time_str = str(time_ns)

            # Format memory as MB if available
            mem_str = ""
            if memory:
                try:
                    mem_str = f"{int(memory) / (1024 * 1024):.1f} MB"
                except (ValueError, TypeError):
                    mem_str = str(memory)

            parts = [f"Test #{i+1}: {status}, {case_score:.2f} pts"]
            if time_str:
                parts.append(time_str)
            if mem_str:
                parts.append(mem_str)
            lines.append(", ".join(parts))

        return "\n".join(lines)

    def run_simulation(self, design_config: DesignConfig, scenario: Scenario) -> Dict[str, Any]:
        """Run evaluation via Frontier-CS SingleEvaluator."""
        try:
            # Extract code from design config
            algorithm_code = None
            for cb in getattr(design_config, "code_blocks", []) or []:
                if cb.name == self.target_name:
                    algorithm_code = str(cb.implementation)
                    break

            if algorithm_code is None:
                raise ValueError(f"No implementation found for {self.target_name}")

            # Call Frontier-CS evaluator
            start_time = time.time()
            result = self.fcs_evaluator.evaluate(
                track=self.track,
                problem_id=self.problem_id,
                code=algorithm_code,
            )
            elapsed = time.time() - start_time

            # Track runtime for adaptive timeout
            self.runtime_history.append(elapsed)

            # Map EvaluationResult to Glia's expected format
            score = result.score if result.score is not None else 0.0
            success = result.success

            return {
                "success": success,
                "score": score,
                "metrics": {
                    "combined_score": score,
                    "score_unbounded": result.score_unbounded if result.score_unbounded is not None else score,
                    "duration_seconds": result.duration_seconds if result.duration_seconds is not None else elapsed,
                    "runs_successfully": 1.0 if success else 0.0,
                },
                "info": {
                    "raw_result": {
                        "score": result.score,
                        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                        "message": result.message or "",
                    },
                    "track": self.track,
                    "problem_id": str(self.problem_id),
                },
                "error": (result.message or "") if not success else "",
                "error_type": self._map_status_to_error_type(result) if not success else "",
                "sim_dir": None,
                "stdout": self._format_result_summary(result) or result.logs or "",
                "stderr": "",
            }

        except Exception as e:
            return {
                "success": False,
                "score": 0.0,
                "metrics": {"combined_score": 0.0, "runs_successfully": 0.0},
                "info": {},
                "error": str(e),
                "error_type": self._get_error_type(e),
                "sim_dir": None,
            }

    def _map_status_to_error_type(self, result) -> str:
        """Map Frontier-CS EvaluationStatus to Glia error type string."""
        try:
            from frontier_cs.runner.base import EvaluationStatus
        except ImportError:
            src_path = os.path.join(self.submodule_path, "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from frontier_cs.runner.base import EvaluationStatus

        status = result.status
        if status == EvaluationStatus.TIMEOUT:
            return "timeout"
        elif status == EvaluationStatus.ERROR:
            return "evaluation_error"
        elif status == EvaluationStatus.SKIPPED:
            return "skipped"
        return "unknown"

    def _get_error_type(self, e: Exception) -> str:
        """Classify error type from exception."""
        if isinstance(e, TimeoutError):
            return "timeout"
        elif isinstance(e, SyntaxError):
            return "syntax"
        elif isinstance(e, (RuntimeError, AssertionError)):
            return "runtime"
        elif isinstance(e, FileNotFoundError):
            return "file_not_found"
        elif isinstance(e, ConnectionError):
            return "docker_connection"
        return "unknown"

    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Print analysis of evaluation results."""
        if not results["success"]:
            print(f"Evaluation failed: {results['error']}")
            return

        metrics = results["metrics"]
        print(f"\n=== FRONTIER-CS {self.track.upper()}/{self.problem_id} RESULTS ===")
        print(f"Score: {metrics.get('combined_score', 0):.2f}/100")
        if metrics.get("score_unbounded"):
            print(f"Score (unbounded): {metrics.get('score_unbounded', 0):.2f}")
        print(f"Duration: {metrics.get('duration_seconds', 0):.1f}s")

    def get_baseline_cache_dir(self) -> str:
        """Return a shared directory for caching baseline results (per-problem, across runs)."""
        return os.path.join(
            self.submodule_path, self.track, "solutions", str(self.problem_id),
        )

    def get_baselines(self) -> List[Tuple[str, str]]:
        """Get baseline solutions from Frontier-CS solutions directory."""
        baselines = []

        if self.track == "research":
            solutions_dir = os.path.join(
                self.submodule_path, "research", "solutions",
                str(self.problem_id),
            )
            ext = ".py"
        else:
            solutions_dir = os.path.join(
                self.submodule_path, "algorithmic", "solutions",
                str(self.problem_id),
            )
            ext = ".cpp"

        if os.path.isdir(solutions_dir):
            for filename in sorted(os.listdir(solutions_dir)):
                if filename.endswith(ext):
                    filepath = os.path.join(solutions_dir, filename)
                    with open(filepath, "r") as f:
                        content = f.read()
                    label = os.path.splitext(filename)[0]
                    baselines.append((f"{label} baseline", content))

        return baselines
