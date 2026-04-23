"""Handoff agent architecture without research journal (ablation).

This module implements the same sequential handoff flow as agentic_handoff, but
with no research journal: no summaries are generated or passed between agents.
Agents have access to the knowledge base (archived workspaces from previous
agents) and initial programs only. Use the system prompt from
handoff_system_prompt_no_journal.txt with this class.
"""

import os
import shutil
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import HumanMessage

from .agentic_handoff import AgenticHandoff
from .deepagents_utils.tool_descriptions import HANDOFF_SUMMARY_REQUEST_PROMPT
from .handoff_utils import AgentStreamRunner


class AgenticHandoffNoJournal(AgenticHandoff):
    """Handoff-based optimizer with no research journal (ablation).

    Same as AgenticHandoff except:
    - No research_journal.md; no summaries are generated or passed.
    - Knowledge base is kept: workspace is archived to knowledgebase/ on handoff.
    - On handoff, next agent sees only knowledgebase/ and fresh initial programs
      (no journal file). Agents are not asked for a summary; runner exits after
      one stream or on timeout/early stop.
    """

    def _setup_initial_workspace(self) -> None:
        """Set up the initial workspace structure (no research journal)."""
        workspace_parent = os.path.dirname(self.log_dir)
        unique_id = str(uuid4())[:8]
        self.workspace_dir = os.path.join(workspace_parent, "workspace", f"workspace_{unique_id}")
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Create subdirectories (no research_journal.md)
        os.makedirs(os.path.join(self.workspace_dir, "experiments"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "knowledgebase"), exist_ok=True)

        # Copy initial program(s)
        self._copy_initial_programs()

        print(f"Workspace initialized at: {self.workspace_dir}")

    def prepare_workspace_for_next_agent(self) -> None:
        """Prepare workspace for the next agent.

        Clears all files except knowledgebase/. No research journal is preserved.
        Then re-creates experiments/ and copies initial program files.
        """
        preserve = {"knowledgebase"}

        for item in os.listdir(self.workspace_dir):
            if item in preserve:
                continue
            item_path = os.path.join(self.workspace_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        os.makedirs(os.path.join(self.workspace_dir, "experiments"), exist_ok=True)
        self._copy_initial_programs()

        print("Workspace prepared for next agent")

    def run_single_agent(self) -> Dict[str, Any]:
        """Run a single agent until completion or timeout (no summary requested)."""
        initial_files = self._prepare_initial_files()
        initial_user_message = HumanMessage(content=self.task_prompt)
        print(f"Initial user message: {initial_user_message}")

        summary_prompt = HANDOFF_SUMMARY_REQUEST_PROMPT.replace(
            "{{AGENT_NUMBER}}", str(self.current_agent_number)
        )

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
            validate_summary=self._validate_summary,
            summary_prompt=summary_prompt,
            current_agent_number=self.current_agent_number,
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
        }

    def optimize(self) -> Dict[str, Any]:
        """Run the handoff optimization loop (no journal; skip summary extract/append)."""
        try:
            self._setup_initial_workspace()

            while self.current_agent_number < self.max_agents:
                self.current_agent_number += 1

                self.build_agent()
                agent_result = self.run_single_agent()

                # No summary extraction or journal append

                self.agent_histories.append({
                    "agent_number": self.current_agent_number,
                    "steps": agent_result.get("steps", 0),
                    "elapsed_minutes": agent_result.get("elapsed_minutes", 0),
                    "experiments": self.experiment_counter,
                    "best_score_after": self.best_score,
                    "timed_out": agent_result.get("timed_out", False),
                    "early_stopped": agent_result.get("early_stopped", False),
                })

                self.archive_workspace_to_knowledgebase(self.current_agent_number)

                if self._should_stop_handoffs():
                    print("Stopping handoffs due to stopping criteria")
                    break

                if self.current_agent_number < self.max_agents:
                    self.prepare_workspace_for_next_agent()

            return self._compile_final_results()

        finally:
            self._cleanup_terminal_capture()

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file (method name: agentic_handoff_no_journal)."""
        output_data = {
            "best_solution": {"code": self.best_code, "score": self.best_score},
            "all_iterations": self.all_iterations,
            "total_simulations": len(self.all_iterations),
            "current_agent": self.current_agent_number,
            "usage_stats": self._get_usage_stats(),
        }

        original_model = self.model
        self.model = self.model_name_str
        try:
            self.save_results(output_data, "agentic_handoff_no_journal", f"_{len(self.all_iterations)}iterations")
        finally:
            self.model = original_model

    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile and save final results (method name: agentic_handoff_no_journal)."""
        if self.current_agent_number >= self.max_agents:
            convergence_reason = "max_agents"
        elif any(hist.get("early_stopped", False) for hist in self.agent_histories):
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

        original_model = self.model
        self.model = self.model_name_str
        try:
            history_file, plot_path, _ = self.save_results(
                output_data, "agentic_handoff_no_journal", f"_{len(self.all_iterations)}iterations"
            )
        finally:
            self.model = original_model

        print(f"\n{'='*60}")
        print("Handoff (no-journal) optimization completed!")
        print(f"Total agents: {self.current_agent_number}")
        print(f"Total simulations: {len(self.all_iterations)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Results saved to: {history_file}")
        print(f"{'='*60}")

        return output_data
