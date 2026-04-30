"""Handoff agent architecture for optimization.

This module implements a handoff-based agent architecture where multiple sequential agents
work on an optimization problem. Each agent runs until naturally exhausted, archives its
work to a knowledgebase, and a fresh agent continues with access to all previous work.
"""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
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

from .deepagents_utils.filesystem_utils import add_file_to_agent_filesystem, PathNormalizationMiddleware
from .deepagents_utils.tee_utils import Tee
from .deepagents_utils.tool_descriptions import (
    HANDOFF_RUN_SIMULATION_DESCRIPTION,
    HANDOFF_SUMMARY_REQUEST_PROMPT,
    CONTINUE_MESSAGE,
)
from .handoff_utils import (
    AgentStreamRunner,
    AgentRunResult,
    extract_text_content,
    clean_message_content,
)


class AgenticHandoff(OptimizationMethod):
    """Handoff-based agent architecture for optimization.

    Multiple agents run sequentially, each exploring until exhausted. When an agent
    completes, its workspace is archived to a knowledgebase, and a fresh agent starts
    with access to all previous work.
    """

    def __init__(
        self,
        *args,
        task_prompt_path: str,
        system_prompt_path: str,
        initial_program_path: str | List[str] | None = None,
        max_agents: int = 5,
        agent_timeout_minutes: float = 30.0,
        enable_continue_message: bool = False,
        early_stop_patience: int = 10,
        reasoning_effort: str = "xhigh",
        **kwargs,
    ):
        """Initialize the handoff optimizer.

        Args:
            task_prompt_path: Path to user prompt file
            system_prompt_path: Path to system prompt file
            initial_program_path: Path(s) to initial program file(s)
            max_agents: Maximum number of sequential agents to run
            agent_timeout_minutes: Wall-clock timeout per agent (safety limit)
            enable_continue_message: Whether to add continue messages automatically instead of waiting for natural completion
            early_stop_patience: Number of iterations without improvement before early stopping (only used when enable_continue_message=True)
            reasoning_effort: Reasoning effort level for gpt-5.2 models (default "xhigh")
            *args, **kwargs: Passed to OptimizationMethod
        """
        super().__init__(*args, **kwargs)

        # Set up terminal output capture to log file
        self.terminal_log_path = os.path.join(self.log_dir, "console_output.log")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_stdout = Tee(self.terminal_log_path, self._original_stdout)
        self._tee_stderr = Tee(self.terminal_log_path, self._original_stderr)
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr

        # Load prompts
        self.set_prompts_from_files(task_prompt_path, system_prompt_path)
        self.initial_program_path = initial_program_path

        # Handoff-specific configuration
        self.max_agents = max_agents
        self.agent_timeout_minutes = agent_timeout_minutes
        self.enable_continue_message = enable_continue_message
        self.early_stop_patience = early_stop_patience
        self.reasoning_effort = reasoning_effort
        self.current_agent_number = 0
        self.experiment_counter = 0  # Reset per agent

        # Tracking across all agents
        self.all_iterations = []
        self.best_score = float("-inf")
        self.best_code = None
        self.agent_histories = []  # Records each agent's results
        self.simulation_results = {}  # Dict keyed by tool_call_id
        self.cumulative_usage: Dict[str, Any] = {}  # Token/cost tracking (accumulated across all agents)

        # Workspace paths (set during build_agent)
        self.workspace_dir = None
        self.backend = None
        self.agent = None

        # Console for pretty-printing
        self._console = Console(file=sys.stdout, force_terminal=True)

        # Save args
        self._save_args()

    def _save_args(self):
        """Save initialization arguments to file."""
        args_dict = {
            "max_agents": self.max_agents,
            "agent_timeout_minutes": self.agent_timeout_minutes,
            "model": self.model if isinstance(self.model, str) else getattr(self, 'model_name_str', str(self.model)),
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
        """Set up the initial workspace structure."""
        # Create workspace directory
        workspace_parent = os.path.dirname(self.log_dir)
        unique_id = str(uuid4())[:8]
        self.workspace_dir = os.path.join(workspace_parent, "workspace", f"workspace_{unique_id}")
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(self.workspace_dir, "experiments"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "knowledgebase"), exist_ok=True)

        # Create initial research journal
        journal_path = os.path.join(self.workspace_dir, "research_journal.md")
        with open(journal_path, "w") as f:
            f.write("# Research Journal\n\n")

        # Copy initial program(s)
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
                # Copy single file
                dest = os.path.join(self.workspace_dir, host_path.name)
                shutil.copy2(host_path, dest)
                print(f"Copied initial file: {host_path.name}")
            elif host_path.is_dir():
                # Copy directory
                dest = os.path.join(self.workspace_dir, host_path.name)
                shutil.copytree(host_path, dest, dirs_exist_ok=True)
                print(f"Copied initial directory: {host_path.name}")

    def build_agent(self) -> None:
        """Build a fresh agent for the current handoff iteration."""
        # Store original model name for filename construction
        self.model_name_str = self.model if isinstance(self.model, str) else getattr(self, 'model_name_str', 'unknown')

        # Convert model name to chat model if needed
        if isinstance(self.model, str):
            is_reasoning_model = any(prefix in self.model.lower() for prefix in ['o1', 'o3', 'o4', 'gpt-5.2-2025-12-11', 'gpt-5.2'])
            is_gpt52 = any(prefix in self.model.lower() for prefix in ['gpt-5.2-2025-12-11', 'gpt-5.2'])
            if is_reasoning_model:
                reasoning_cfg = {"summary": "auto"}
                if is_gpt52:
                    reasoning_cfg["effort"] = self.reasoning_effort
                self.model = init_chat_model(f"openai:{self.model_name_str}", reasoning=reasoning_cfg)
            else:
                self.model = init_chat_model(f"openai:{self.model_name_str}")

        # Create FilesystemBackend for this agent
        self.backend = FilesystemBackend(root_dir=self.workspace_dir, virtual_mode=True)

        # Reset experiment counter for new agent
        self.experiment_counter = 0

        # Set up shell middleware
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
                memory_bytes=4 * 1024 * 1024 * 1024,  # 4 GB limit to prevent OOM kills
            ),
            env={**os.environ, "PYTHONPATH": "/tmp/.packages", "HOME": "/tmp"},
            startup_commands=[
                f"cd {self.workspace_dir}",
                "pip install --no-cache-dir --target=/tmp/.packages numpy pandas",
            ],
        )

        path_normalization_middleware = PathNormalizationMiddleware()

        # Create tools
        run_sim_tool = self.create_run_simulation_tool()

        # Inject agent number into system prompt
        agent_system_prompt = self.system_prompt.replace(
            "{{AGENT_NUMBER}}", str(self.current_agent_number)
        )
        print(f"Agent system prompt: {agent_system_prompt}")
        # Create the agent (no checkpointer - we handle message deduplication ourselves)
        self.agent = create_deep_agent(
            model=self.model,
            system_prompt=agent_system_prompt,
            backend=self.backend,
            tools=[run_sim_tool],
            middleware=[path_normalization_middleware, shell_middleware],
        )

        print(f"\nAgent {self.current_agent_number} built successfully")

    def create_run_simulation_tool(self) -> BaseTool:
        """Create run_simulation tool with auto-archiving."""
        optimizer = self

        @tool(description=HANDOFF_RUN_SIMULATION_DESCRIPTION)
        def run_simulation(file_path: str, runtime: ToolRuntime) -> str | Command:
            """Run simulation and auto-archive to experiments folder."""
            try:
                # Validate file path
                if not file_path.startswith('/'):
                    return f"Error: file_path must be absolute (start with /). Got: {file_path}"

                # Read code from file
                try:
                    file_content = optimizer.backend.read(file_path, offset=0, limit=10000)
                    if file_content.startswith('Error:'):
                        return file_content

                    # Strip line numbers if present
                    lines = []
                    for line in file_content.split('\n'):
                        if line.strip():
                            parts = line.split('\t', 1)
                            if len(parts) == 2 and re.match(r'^\s*\d+(?:\.\d+)?\s*$', parts[0]):
                                lines.append(parts[1])
                            elif not re.match(r'^\s*\d+(?:\.\d+)?\s*$', line):
                                lines.append(line)
                    code = "\n".join(lines)
                except Exception as e:
                    return f"Error reading file {file_path}: {str(e)}"

                if not code or not code.strip():
                    return f"Error: File {file_path} is empty."

                # Create experiment folder
                optimizer.experiment_counter += 1
                exp_id = f"exp_{optimizer.experiment_counter:03d}"
                exp_dir = f"/experiments/{exp_id}"

                # Archive snapshot
                snap_ext = ".cpp" if "#include" in code else ".py"
                optimizer.backend.write(f"{exp_dir}/snapshot{snap_ext}", code)

                # Run simulation
                score, sim_dirs, results = optimizer.evaluate_code(code)
                success, error_message = optimizer.summarize_results(results)

                # Write score.txt
                score_content = f"score: {score}\nsuccess: {success}\n"
                if not success:
                    score_content += f"error: {error_message}\n"
                optimizer.backend.write(f"{exp_dir}/score.txt", score_content)

                # Copy results to experiment folder if available
                if sim_dirs and len(sim_dirs) > 0:
                    for sim_dir in sim_dirs:
                        for file in os.listdir(sim_dir):
                            if file.endswith(".csv"):
                                src_path = os.path.join(sim_dir, file)
                                content = Path(src_path).read_text(encoding='utf-8')
                                optimizer.backend.write(f"{exp_dir}/results/{file}", content)

                # Track simulation result
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

                # Update tracking
                if success and score > optimizer.best_score:
                    optimizer.best_score = score
                    optimizer.best_code = code
                    print(f"New best score: {score:.4f} (experiment {exp_id})")

                # Build result message
                # Extract stdout/stderr from results (results is keyed by scenario name)
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
                    result_msg += f"  - snapshot{snap_ext} (your code)\n"
                    result_msg += f"  - score.txt\n"
                    result_msg += f"  - results/ (simulation output)\n"
                    # Include stdout/stderr if present (truncate to avoid overwhelming output)
                    if stdout and stdout.strip():
                        stdout_truncated = stdout[:2000] + ("..." if len(stdout) > 2000 else "")
                        result_msg += f"\n=== Stdout ===\n{stdout_truncated}\n"
                    if stderr and stderr.strip():
                        stderr_truncated = stderr[:1000] + ("..." if len(stderr) > 1000 else "")
                        result_msg += f"\n=== Simulation Logs ===\n{stderr_truncated}\n"
                    result_msg += f"\nKeep track of this result for your final summary."
                    return result_msg
                else:
                    error_result = f"Experiment {exp_id} failed.\nError: {error_message}\nArchived to: {exp_dir}/"
                    # Include stdout/stderr on failure too for debugging
                    if stdout and stdout.strip():
                        stdout_truncated = stdout[:2000] + ("..." if len(stdout) > 2000 else "")
                        error_result += f"\n\n=== Stdout ===\n{stdout_truncated}"
                    if stderr and stderr.strip():
                        stderr_truncated = stderr[:1000] + ("..." if len(stderr) > 1000 else "")
                        error_result += f"\n\n=== Simulation Logs ===\n{stderr_truncated}"
                    return error_result

            except Exception as e:
                return f"Error running simulation: {str(e)}"

        return run_simulation

    def _prepare_initial_files(self) -> Dict[str, Any]:
        """Prepare initial files for agent filesystem."""
        from deepagents.backends.utils import create_file_data

        result = {}

        # Add all files from workspace to agent's virtual filesystem
        for root, dirs, files in os.walk(self.workspace_dir):
            # Skip knowledgebase directory from virtual filesystem view
            # (agent can still access via shell)
            rel_root = os.path.relpath(root, self.workspace_dir)
            if rel_root.startswith("knowledgebase"):
                continue

            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.workspace_dir)
                agent_path = f"/{rel_path}"

                try:
                    content = Path(file_path).read_text(encoding='utf-8')
                    file_data = create_file_data(content)
                    result[agent_path] = file_data
                except Exception:
                    pass  # Skip binary files

        return result

    def run_single_agent(self) -> Dict[str, Any]:
        """Run a single agent until completion or timeout.

        Uses the AgentStreamRunner for clean message handling with
        event-based architecture for debug output and result tracking.
        """
        # Prepare initial files
        initial_files = self._prepare_initial_files()

        # Build input message
        initial_user_message = HumanMessage(content=self.task_prompt)
        print(f"Initial user message: {initial_user_message}")

        # Create summary request prompt
        summary_prompt = HANDOFF_SUMMARY_REQUEST_PROMPT.replace(
            "{{AGENT_NUMBER}}", str(self.current_agent_number)
        )

        # Create stream runner with callbacks (model_name_str set in build_agent)
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

        # Define rebuild callback
        def rebuild_agent_callback() -> bool:
            try:
                self.build_agent()
                return True
            except Exception:
                return False

        # Run the agent
        result = runner.run_single_agent(
            agent=self.agent,
            initial_messages=[initial_user_message],
            initial_files=initial_files,
            validate_summary=self._validate_summary,
            summary_prompt=summary_prompt,
            current_agent_number=self.current_agent_number,
            rebuild_agent_callback=rebuild_agent_callback,
            enable_continue_message=self.enable_continue_message,
            early_stop_patience=self.early_stop_patience,
        )

        # Update best score tracking from runner
        if runner.best_score > self.best_score:
            self.best_score = runner.best_score
            self.best_code = runner.best_code

        # Convert AgentRunResult to dict for compatibility
        return {
            "messages": result.messages,
            "files": result.files,
            "steps": result.steps,
            "elapsed_minutes": result.elapsed_minutes,
            "timed_out": result.timed_out,
        }

    def _get_usage_stats(self) -> Dict[str, Any]:
        """Return cumulative usage stats from streamed AIMessages (token/cost tracking)."""
        stats = {
            "total_cost": round(self.cumulative_usage.get("total_cost", 0.0), 6),
            "total_prompt_tokens": self.cumulative_usage.get("total_prompt_tokens", 0),
            "total_completion_tokens": self.cumulative_usage.get("total_completion_tokens", 0),
            "total_tokens": self.cumulative_usage.get("total_tokens", 0),
            "model": self.cumulative_usage.get("model", self.model if isinstance(self.model, str) else getattr(self, "model_name_str", "o3")),
        }
        return stats

    def _on_new_best_score(self, score: float, code: str) -> None:
        """Callback for when a new best score is found."""
        self.best_score = score
        self.best_code = code
        # Reset iterations_since_improvement when we get a new best
        # This will be tracked per-agent in AgentStreamRunner

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file."""
        output_data = {
            "best_solution": {"code": self.best_code, "score": self.best_score},
            "all_iterations": self.all_iterations,
            "total_simulations": len(self.all_iterations),
            "current_agent": self.current_agent_number,
            "usage_stats": self._get_usage_stats(),
        }

        # Temporarily restore model name
        original_model = self.model
        self.model = self.model_name_str
        try:
            self.save_results(output_data, "agentic_handoff", f"_{len(self.all_iterations)}iterations")
        finally:
            self.model = original_model

    def archive_workspace_to_knowledgebase(self, agent_number: int) -> None:
        """Archive current workspace to knowledgebase.

        Archives workspace contents except:
        - knowledgebase/ (avoid recursive copying)
        - research_journal.md (main journal accumulates across agents)
        """
        kb_dir = os.path.join(self.workspace_dir, "knowledgebase", f"agent_{agent_number}")
        os.makedirs(kb_dir, exist_ok=True)

        # Copy workspace contents (except knowledgebase/ and research_journal.md)
        for item in os.listdir(self.workspace_dir):
            if item == "knowledgebase":
                continue
            if item == "research_journal.md":
                continue  # Main journal accumulates - no need to copy

            src = os.path.join(self.workspace_dir, item)
            dst = os.path.join(kb_dir, item)

            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        # Copy console log from log_dir
        console_log_src = self.terminal_log_path
        if os.path.exists(console_log_src):
            console_log_dst = os.path.join(kb_dir, "console.log")
            shutil.copy2(console_log_src, console_log_dst)

        print(f"Archived Agent {agent_number} workspace to {kb_dir}")

    def prepare_workspace_for_next_agent(self) -> None:
        """Prepare workspace for the next agent.

        Clears all files except:
        - knowledgebase/ (read-only archive of previous agents)
        - research_journal.md (accumulated summaries)

        Then re-creates experiments/ and copies initial program files.
        """
        # Items to preserve
        preserve = {"knowledgebase", "research_journal.md"}

        # Remove everything except preserved items
        for item in os.listdir(self.workspace_dir):
            if item in preserve:
                continue
            item_path = os.path.join(self.workspace_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        # Recreate experiments directory
        os.makedirs(os.path.join(self.workspace_dir, "experiments"), exist_ok=True)

        # Copy fresh initial programs
        self._copy_initial_programs()

        print("Workspace prepared for next agent")

    def _extract_agent_summary(self, messages: List[Any]) -> str:
        """Extract the last AI message as the agent's summary.

        Uses the extract_text_content function from handoff_utils module
        for consistent content extraction across different formats.
        """
        from langchain_core.messages import AIMessage

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = getattr(msg, 'content', None)
                text = extract_text_content(content)
                if text.strip():
                    return text.strip()
        return ""

    def _validate_summary(self, summary: str) -> bool:
        """Check if the summary contains expected sections."""
        required_sections = ['Best Result', 'What I Tried', 'Key Insights']
        summary_lower = summary.lower()
        return all(section.lower() in summary_lower for section in required_sections)

    def _clean_message_content(self, content: Any) -> str:
        """Extract clean text content from a message, stripping reasoning blocks and IDs.

        Delegates to clean_message_content from handoff_utils module.
        """
        return clean_message_content(content)

    def _clean_messages_for_llm(self, messages: List[Any]) -> List[Any]:
        """Clean messages for re-submission to LLM by removing reasoning IDs.

        Also filters out orphaned tool calls (those without corresponding responses),
        which can occur when an agent times out mid-tool-call.
        """
        from langchain_core.messages import AIMessage, HumanMessage as HM, ToolMessage

        answered_tool_calls = {
            msg.tool_call_id for msg in messages
            if isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id')
        }

        cleaned = []
        for msg in messages:
            if isinstance(msg, HumanMessage) or isinstance(msg, HM):
                content = self._clean_message_content(msg.content)
                if content:
                    cleaned.append(HM(content=content))

            elif isinstance(msg, AIMessage):
                content = self._clean_message_content(msg.content)
                tool_calls = [
                    tc for tc in (getattr(msg, 'tool_calls', None) or [])
                    if tc.get('id') in answered_tool_calls
                ]
                if content or tool_calls:
                    cleaned.append(AIMessage(content=content, tool_calls=tool_calls))

            elif isinstance(msg, ToolMessage):
                cleaned.append(msg)

        return cleaned

    def _generate_fallback_summary(self, messages: List[Any], agent_number: int) -> str:
        """Generate a summary by feeding the LLM the full conversation history.

        This is a silent operation - no debug output is printed.

        Note: The messages passed in already include the summary request from run_single_agent,
        so we don't need to add it again.
        """
        from langchain_core.messages import SystemMessage

        # Clean messages to remove reasoning block IDs that cause duplicate errors
        cleaned_messages = self._clean_messages_for_llm(messages)

        # Add system prompt at the beginning (it's not included in message history)
        agent_system_prompt = self.system_prompt.replace(
            "{{AGENT_NUMBER}}", str(agent_number)
        )
        messages_with_system = [SystemMessage(content=agent_system_prompt)] + cleaned_messages

        try:
            response = self.model.invoke(messages_with_system)

            # Extract string content from response using extract_text_content
            content = response.content if hasattr(response, 'content') else str(response)
            text = extract_text_content(content)

            return text if text else str(content)
        except Exception as e:
            return f"## Agent {agent_number} Summary\n\n(Failed to generate summary: {e})\n"

    def _append_summary_to_journal(self, summary: str, agent_number: int) -> None:
        """Append agent summary to the research journal."""
        journal_path = os.path.join(self.workspace_dir, "research_journal.md")

        try:
            # Read current journal
            if os.path.exists(journal_path):
                with open(journal_path, "r") as f:
                    current_content = f.read()
            else:
                current_content = "# Research Journal\n"

            # Ensure summary starts with proper header if not present
            if not summary.strip().startswith("## Agent"):
                summary = f"## Agent {agent_number} Summary\n\n{summary}"

            # Append the summary
            updated_content = current_content.rstrip() + "\n\n---\n\n" + summary + "\n"

            # Write back
            with open(journal_path, "w") as f:
                f.write(updated_content)

            print(f"Appended Agent {agent_number} summary to research journal")

        except Exception as e:
            print(f"Warning: Failed to update journal: {e}")

    def _should_stop_handoffs(self) -> bool:
        """Check if we should stop creating new agents."""
        # Could add more sophisticated stopping criteria here
        # For now, just run until max_agents
        return False

    def optimize(self) -> Dict[str, Any]:
        """Run the handoff optimization loop."""
        try:
            # Set up initial workspace
            self._setup_initial_workspace()

            # Main handoff loop
            while self.current_agent_number < self.max_agents:
                self.current_agent_number += 1

                # Build fresh agent
                self.build_agent()

                # Run agent to natural completion (NO continue messages)
                agent_result = self.run_single_agent()

                # Extract and append summary to journal
                summary = self._extract_agent_summary(agent_result.get("messages", []))
                if self._validate_summary(summary):
                    self._append_summary_to_journal(summary, self.current_agent_number)
                else:
                    # Use fallback summary if agent still didn't provide one
                    if self.debug:
                        print(f"\n[DEBUG] Agent {self.current_agent_number} did not provide valid summary. Generating fallback summary...")
                    fallback_summary = self._generate_fallback_summary(
                        agent_result.get("messages", []), self.current_agent_number
                    )
                    self._append_summary_to_journal(fallback_summary, self.current_agent_number)

                # Record agent history
                self.agent_histories.append({
                    "agent_number": self.current_agent_number,
                    "steps": agent_result.get("steps", 0),
                    "elapsed_minutes": agent_result.get("elapsed_minutes", 0),
                    "experiments": self.experiment_counter,
                    "best_score_after": self.best_score,
                    "timed_out": agent_result.get("timed_out", False),
                    "early_stopped": agent_result.get("early_stopped", False),
                })

                # Archive workspace to knowledgebase
                self.archive_workspace_to_knowledgebase(self.current_agent_number)

                # Check stopping criteria
                if self._should_stop_handoffs():
                    print("Stopping handoffs due to stopping criteria")
                    break

                # Prepare for next agent
                if self.current_agent_number < self.max_agents:
                    self.prepare_workspace_for_next_agent()

            # Compile final results
            return self._compile_final_results()

        finally:
            self._cleanup_terminal_capture()

    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile and save final results."""
        # Determine convergence reason
        if self.current_agent_number >= self.max_agents:
            convergence_reason = "max_agents"
        elif any(hist.get("early_stopped", False) for hist in self.agent_histories):
            # Check if any agent stopped due to early stopping
            early_stopped_agent = next((hist for hist in self.agent_histories if hist.get("early_stopped", False)), None)
            if early_stopped_agent:
                convergence_reason = f"early_stop_no_improvement_agent_{early_stopped_agent['agent_number']}"
            else:
                convergence_reason = "early_stop_no_improvement"
        else:
            convergence_reason = "stopped"

        output_data = {
            "best_solution": {"code": self.best_code, "score": self.best_score},
            "all_iterations": self.all_iterations,
            "total_simulations": len(self.all_iterations),
            "total_agents": self.current_agent_number,
            "agent_histories": self.agent_histories,
            "workspace_dir": str(self.workspace_dir),
            "convergence_reason": convergence_reason,
            "usage_stats": self._get_usage_stats(),
        }

        # Save results
        original_model = self.model
        self.model = self.model_name_str
        try:
            history_file, plot_path, _ = self.save_results(
                output_data, "agentic_handoff", f"_{len(self.all_iterations)}iterations"
            )
        finally:
            self.model = original_model

        print(f"\n{'='*60}")
        print("Handoff optimization completed!")
        print(f"Total agents: {self.current_agent_number}")
        print(f"Total simulations: {len(self.all_iterations)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Results saved to: {history_file}")
        print(f"{'='*60}")

        return output_data

    def _cleanup_terminal_capture(self) -> None:
        """Restore original stdout/stderr."""
        try:
            if hasattr(self, '_tee_stdout'):
                self._tee_stdout.close()
            if hasattr(self, '_tee_stderr'):
                self._tee_stderr.close()
            if hasattr(self, '_original_stdout'):
                sys.stdout = self._original_stdout
            if hasattr(self, '_original_stderr'):
                sys.stderr = self._original_stderr
        except Exception:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


