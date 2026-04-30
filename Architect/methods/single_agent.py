"""Single-agent optimization (ablation: no handoff, no research journal, no knowledge base).

This module implements a single-agent optimizer with the same core capabilities as
agentic_handoff (run_simulation, shell, timeout, early stopping, continue messages)
but without handoff to another agent, research journal, or knowledge base. The agent
runs until wall-clock timeout or early stop (no improvement for early_stop_patience
iterations).
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from rich.console import Console

from .common import OptimizationMethod
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware import ShellToolMiddleware, DockerExecutionPolicy
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import Command

from .deepagents_utils.filesystem_utils import PathNormalizationMiddleware
from .deepagents_utils.tee_utils import Tee
from .deepagents_utils.tool_descriptions import (
    CustomSummarizationMiddleware,
    HANDOFF_RUN_SIMULATION_DESCRIPTION,
    SUMMARY_PROMPT,
)
from .handoff_utils import AgentStreamRunner


class SingleAgent(OptimizationMethod):
    """Single-agent optimizer (ablation: no handoff, no journal, no knowledge base).

    One agent runs until timeout or early stop. Same tools and flow as handoff
    (run_simulation, continue messages, early stopping) but no summary, no
    research journal, and no knowledge base.
    """

    def __init__(
        self,
        *args,
        task_prompt_path: str,
        system_prompt_path: str,
        initial_program_path: str | List[str] | None = None,
        agent_timeout_minutes: float = 30.0,
        use_summarization_middleware: bool = False,
        enable_continue_message: bool = False,
        early_stop_patience: int = 10,
        **kwargs,
    ):
        """Initialize the single-agent optimizer.

        Args:
            task_prompt_path: Path to user prompt file
            system_prompt_path: Path to system prompt file
            initial_program_path: Path(s) to initial program file(s)
            agent_timeout_minutes: Wall-clock timeout (safety limit)
            use_summarization_middleware: Whether to enable context summarization middleware
            enable_continue_message: Whether to add continue messages automatically
            early_stop_patience: Iterations without improvement before early stop (when enable_continue_message=True)
            *args, **kwargs: Passed to OptimizationMethod
        """
        super().__init__(*args, **kwargs)

        self.terminal_log_path = os.path.join(self.log_dir, "console_output.log")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_stdout = Tee(self.terminal_log_path, self._original_stdout)
        self._tee_stderr = Tee(self.terminal_log_path, self._original_stderr)
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr

        self.set_prompts_from_files(task_prompt_path, system_prompt_path)
        self.initial_program_path = initial_program_path

        self.agent_timeout_minutes = agent_timeout_minutes
        self.use_summarization_middleware = use_summarization_middleware
        self.enable_continue_message = enable_continue_message
        self.early_stop_patience = early_stop_patience
        self.experiment_counter = 0

        self.all_iterations = []
        self.best_score = float("-inf")
        self.best_code = None
        self.simulation_results: Dict[str, Any] = {}
        self.cumulative_usage: Dict[str, Any] = {}

        self.workspace_dir = None
        self.backend = None
        self.agent = None

        self._console = Console(file=sys.stdout, force_terminal=True)

        self._save_args()

    def _save_args(self) -> None:
        """Save initialization arguments to file."""
        args_dict = {
            "agent_timeout_minutes": self.agent_timeout_minutes,
            "use_summarization_middleware": self.use_summarization_middleware,
            "enable_continue_message": self.enable_continue_message,
            "early_stop_patience": self.early_stop_patience,
            "model": self.model if isinstance(self.model, str) else getattr(self, "model_name_str", str(self.model)),
            "task_name": self.task.name,
            "target_name": getattr(getattr(self.task, 'evaluator', None), 'target_name', None),
            "debug": self.debug,
        }
        with open(f"{self.log_dir}/args.json", "w") as f:
            json.dump(args_dict, f, indent=2, default=str)

    def set_prompts_from_files(self, task_prompt_path: str, system_prompt_path: str) -> None:
        """Load user and system prompts from files."""
        for attr, path in [("task_prompt", task_prompt_path), ("system_prompt", system_prompt_path)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r") as f:
                setattr(self, attr, f.read())

    def _setup_initial_workspace(self) -> None:
        """Set up workspace: experiments/ only (no research journal, no knowledge base)."""
        workspace_parent = os.path.dirname(self.log_dir)
        unique_id = str(uuid4())[:8]
        self.workspace_dir = os.path.join(workspace_parent, "workspace", f"workspace_{unique_id}")
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "experiments"), exist_ok=True)

        self._copy_initial_programs()

        print(f"Workspace initialized at: {self.workspace_dir}")

    def _copy_initial_programs(self) -> None:
        """Copy initial program files to workspace."""
        if not self.initial_program_path:
            return

        paths = [self.initial_program_path] if isinstance(self.initial_program_path, str) else self.initial_program_path

        for path_str in paths:
            host_path = Path(path_str)
            if not host_path.is_absolute():
                host_path = Path(os.getcwd()) / host_path

            if not host_path.exists():
                print(f"Warning: Initial program not found: {host_path}")
                continue

            if host_path.is_file():
                dest = os.path.join(self.workspace_dir, host_path.name)
                shutil.copy2(host_path, dest)
                print(f"Copied initial file: {host_path.name}")
            elif host_path.is_dir():
                dest = os.path.join(self.workspace_dir, host_path.name)
                shutil.copytree(host_path, dest, dirs_exist_ok=True)
                print(f"Copied initial directory: {host_path.name}")

    def build_agent(self) -> None:
        """Build the single agent."""
        self.model_name_str = self.model if isinstance(self.model, str) else getattr(self, "model_name_str", "unknown")

        if isinstance(self.model, str):
            is_reasoning_model = any(
                prefix in self.model.lower() for prefix in ["o1", "o3", "o4", "gpt-5.2-2025-12-11", "gpt-5.2", "gpt-5.2-xhigh"]
            )
            if is_reasoning_model:
                if "gpt-5.2-xhigh" in self.model.lower():
                    print("Using GPT-5.2 XHigh model, xhigh reasoning")
                    self.model = init_chat_model(f"openai:gpt-5.2", reasoning={"summary": "auto", "effort": "xhigh"})
                else:
                    print(f"Using {self.model_name_str} model, default reasoning")
                    self.model = init_chat_model(f"openai:{self.model_name_str}", reasoning={"summary": "auto"})
            else:
                self.model = init_chat_model(f"openai:{self.model_name_str}")

        self.backend = FilesystemBackend(root_dir=self.workspace_dir, virtual_mode=True)
        self.experiment_counter = 0

        current_uid = os.getuid()
        current_gid = os.getgid()

        shell_middleware = ShellToolMiddleware(
            workspace_root=self.workspace_dir,
            execution_policy=DockerExecutionPolicy(
                image="python:3.11",
                network_enabled=True,
                read_only_rootfs=False,
                remove_container_on_exit=True,
                user=f"{current_uid}:{current_gid}",
                command_timeout=300.0,
                memory_bytes=4 * 1024 * 1024 * 1024,
            ),
            env={**os.environ, "PYTHONPATH": "/tmp/.packages", "HOME": "/tmp"},
            startup_commands=[
                f"cd {self.workspace_dir}",
                "pip install --no-cache-dir --target=/tmp/.packages numpy pandas",
            ],
        )

        path_normalization_middleware = PathNormalizationMiddleware()
        run_sim_tool = self._create_run_simulation_tool()
        middleware_list = [path_normalization_middleware, shell_middleware]

        if self.use_summarization_middleware:
            middleware_list.append(
                CustomSummarizationMiddleware(
                    model=self.model_name_str,
                    max_tokens_before_summary=16000,
                    messages_to_keep=10,
                    summary_prompt=SUMMARY_PROMPT,
                )
            )

        # Single-agent prompt has no {{AGENT_NUMBER}}; use as-is or replace if present
        agent_system_prompt = self.system_prompt.replace("{{AGENT_NUMBER}}", "1")

        self.agent = create_deep_agent(
            model=self.model,
            system_prompt=agent_system_prompt,
            backend=self.backend,
            tools=[run_sim_tool],
            middleware=middleware_list,
        )

        print("\nSingle agent built successfully")

    def _create_run_simulation_tool(self) -> BaseTool:
        """Create run_simulation tool with single-agent description (no journal/kb)."""
        optimizer = self

        @tool(description=HANDOFF_RUN_SIMULATION_DESCRIPTION)
        def run_simulation(file_path: str, runtime: ToolRuntime) -> str | Command:
            """Run simulation and auto-archive to experiments folder."""
            try:
                if not file_path.startswith("/"):
                    return f"Error: file_path must be absolute (start with /). Got: {file_path}"

                try:
                    file_content = optimizer.backend.read(file_path, offset=0, limit=10000)
                    if file_content.startswith("Error:"):
                        return file_content

                    lines = []
                    for line in file_content.split("\n"):
                        if line.strip():
                            parts = line.split("\t", 1)
                            if len(parts) == 2 and re.match(r"^\s*\d+(?:\.\d+)?\s*$", parts[0]):
                                lines.append(parts[1])
                            elif not re.match(r"^\s*\d+(?:\.\d+)?\s*$", line):
                                lines.append(line)
                    code = "\n".join(lines)
                except Exception as e:
                    return f"Error reading file {file_path}: {str(e)}"

                if not code or not code.strip():
                    return f"Error: File {file_path} is empty."

                optimizer.experiment_counter += 1
                exp_id = f"exp_{optimizer.experiment_counter:03d}"
                exp_dir = f"/experiments/{exp_id}"

                optimizer.backend.write(f"{exp_dir}/snapshot.py", code)

                score, sim_dirs, results = optimizer.evaluate_code(code)
                success, error_message = optimizer.summarize_results(results)

                score_content = f"score: {score}\nsuccess: {success}\n"
                if not success:
                    score_content += f"error: {error_message}\n"
                optimizer.backend.write(f"{exp_dir}/score.txt", score_content)

                if sim_dirs and len(sim_dirs) > 0:
                    for sim_dir in sim_dirs:
                        for file in os.listdir(sim_dir):
                            if file.endswith(".csv"):
                                src_path = os.path.join(sim_dir, file)
                                content = Path(src_path).read_text(encoding="utf-8")
                                optimizer.backend.write(f"{exp_dir}/results/{file}", content)

                tool_call_id = runtime.tool_call_id
                optimizer.simulation_results[tool_call_id] = {
                    "code": code,
                    "score": score,
                    "sim_dirs": sim_dirs,
                    "results": results,
                    "success": success,
                    "error_message": error_message,
                    "file_path": file_path,
                    "experiment_id": exp_id,
                }

                if success and score > optimizer.best_score:
                    optimizer.best_score = score
                    optimizer.best_code = code
                    print(f"New best score: {score:.4f} (experiment {exp_id})")

                stdout = ""
                stderr = ""
                for scenario_result in results.values():
                    if isinstance(scenario_result, dict):
                        stdout += scenario_result.get("stdout", "")
                        stderr += scenario_result.get("stderr", "")

                if success:
                    result_msg = f"Experiment {exp_id} completed.\n"
                    result_msg += f"Score: {score:.6f}\n"
                    result_msg += f"Archived to: {exp_dir}/\n"
                    result_msg += f"  - snapshot.py (your code)\n"
                    result_msg += f"  - score.txt\n"
                    result_msg += f"  - results/ (simulation output)\n"
                    if stdout and stdout.strip():
                        result_msg += f"\n=== Stdout ===\n{(stdout[:2000] + ('...' if len(stdout) > 2000 else ''))}\n"
                    if stderr and stderr.strip():
                        result_msg += f"\n=== Simulation Logs ===\n{(stderr[:1000] + ('...' if len(stderr) > 1000 else ''))}\n"
                    return result_msg
                else:
                    error_result = f"Experiment {exp_id} failed.\nError: {error_message}\nArchived to: {exp_dir}/"
                    if stdout and stdout.strip():
                        error_result += f"\n\n=== Stdout ===\n{(stdout[:2000] + ('...' if len(stdout) > 2000 else ''))}"
                    if stderr and stderr.strip():
                        error_result += f"\n\n=== Simulation Logs ===\n{(stderr[:1000] + ('...' if len(stderr) > 1000 else ''))}"
                    return error_result

            except Exception as e:
                return f"Error running simulation: {str(e)}"

        return run_simulation

    def _prepare_initial_files(self) -> Dict[str, Any]:
        """Prepare initial files for agent filesystem (no knowledgebase to skip)."""
        from deepagents.backends.utils import create_file_data

        result = {}
        for root, dirs, files in os.walk(self.workspace_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.workspace_dir)
                agent_path = f"/{rel_path}"
                try:
                    content = Path(file_path).read_text(encoding="utf-8")
                    file_data = create_file_data(content)
                    result[agent_path] = file_data
                except Exception:
                    pass
        return result

    def _run_agent(self) -> Dict[str, Any]:
        """Run the single agent until timeout or early stop (no summary requested)."""
        initial_files = self._prepare_initial_files()
        initial_user_message = HumanMessage(content=self.task_prompt)

        # Dummy validate/summary_prompt (require_summary=False so they are not used)
        def _validate_summary(_summary: str) -> bool:
            return False

        dummy_summary_prompt = ""

        runner = AgentStreamRunner(
            simulation_results=self.simulation_results,
            all_iterations=self.all_iterations,
            debug=self.debug,
            timeout_minutes=self.agent_timeout_minutes,
            on_new_best=self._on_new_best_score,
            periodic_save_callback=self._save_intermediate_results,
            console=self._console,
            best_score=self.best_score,
            best_code=self.best_code,
            model=getattr(self, "model_name_str", self.model if isinstance(self.model, str) else "o3"),
            cumulative_usage=self.cumulative_usage,
        )

        def rebuild_agent_callback() -> bool:
            try:
                self.build_agent()
                return True
            except Exception:
                return False

        result = runner.run_single_agent(
            agent=self.agent,
            initial_messages=[initial_user_message],
            initial_files=initial_files,
            validate_summary=_validate_summary,
            summary_prompt=dummy_summary_prompt,
            current_agent_number=1,
            rebuild_agent_callback=rebuild_agent_callback,
            enable_continue_message=self.enable_continue_message,
            early_stop_patience=self.early_stop_patience,
            require_summary=False,
        )

        if runner.best_score > self.best_score:
            self.best_score = runner.best_score
            self.best_code = runner.best_code

        return {
            "messages": result.messages,
            "files": result.files,
            "steps": result.steps,
            "elapsed_minutes": result.elapsed_minutes,
            "timed_out": result.timed_out,
            "early_stopped": result.early_stopped,
        }

    def _get_usage_stats(self) -> Dict[str, Any]:
        """Return cumulative usage stats."""
        return {
            "total_cost": round(self.cumulative_usage.get("total_cost", 0.0), 6),
            "total_prompt_tokens": self.cumulative_usage.get("total_prompt_tokens", 0),
            "total_completion_tokens": self.cumulative_usage.get("total_completion_tokens", 0),
            "total_tokens": self.cumulative_usage.get("total_tokens", 0),
            "model": self.cumulative_usage.get(
                "model", self.model if isinstance(self.model, str) else getattr(self, "model_name_str", "o3")
            ),
        }

    def _on_new_best_score(self, score: float, code: str) -> None:
        """Callback when a new best score is found."""
        self.best_score = score
        self.best_code = code

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file."""
        output_data = {
            "best_solution": {"code": self.best_code, "score": self.best_score},
            "all_iterations": self.all_iterations,
            "total_simulations": len(self.all_iterations),
            "usage_stats": self._get_usage_stats(),
        }
        original_model = self.model
        self.model = self.model_name_str
        try:
            self.save_results(output_data, "single_agent", f"_{len(self.all_iterations)}iterations")
        finally:
            self.model = original_model

    def _compile_final_results(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compile and save final results."""
        if agent_result.get("timed_out"):
            convergence_reason = "timeout"
        elif agent_result.get("early_stopped"):
            convergence_reason = "early_stop_no_improvement"
        else:
            convergence_reason = "stopped"

        output_data = {
            "best_solution": {"code": self.best_code, "score": self.best_score},
            "all_iterations": self.all_iterations,
            "total_simulations": len(self.all_iterations),
            "workspace_dir": str(self.workspace_dir),
            "convergence_reason": convergence_reason,
            "usage_stats": self._get_usage_stats(),
        }

        original_model = self.model
        self.model = self.model_name_str
        try:
            history_file, plot_path, _ = self.save_results(
                output_data, "single_agent", f"_{len(self.all_iterations)}iterations"
            )
        finally:
            self.model = original_model

        print(f"\n{'='*60}")
        print("Single-agent optimization completed!")
        print(f"Total simulations: {len(self.all_iterations)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Results saved to: {history_file}")
        print(f"{'='*60}")

        return output_data

    def _cleanup_terminal_capture(self) -> None:
        """Restore original stdout/stderr."""
        try:
            if hasattr(self, "_tee_stdout"):
                self._tee_stdout.close()
            if hasattr(self, "_tee_stderr"):
                self._tee_stderr.close()
            if hasattr(self, "_original_stdout"):
                sys.stdout = self._original_stdout
            if hasattr(self, "_original_stderr"):
                sys.stderr = self._original_stderr
        except Exception:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def optimize(self) -> Dict[str, Any]:
        """Run single-agent optimization: one agent until timeout or early stop."""
        try:
            self._setup_initial_workspace()
            self.build_agent()
            agent_result = self._run_agent()
            return self._compile_final_results(agent_result)
        finally:
            self._cleanup_terminal_capture()
