"""Handoff agent architecture without knowledge base (ablation).

This module implements the same sequential handoff flow as agentic_handoff, but
with no knowledge base: each agent has the research journal and its own experiments
folder during its run; when an agent ends, all experiments are deleted and the
next agent sees only the accumulated research journal (and fresh initial programs).
Use the system prompt from handoff_system_prompt_no_kb.txt with this class.
"""

import os
import shutil
from typing import Any, Dict
from uuid import uuid4

from .agentic_handoff import AgenticHandoff


class AgenticHandoffNoKB(AgenticHandoff):
    """Handoff-based optimizer with no knowledge base (ablation).

    Same as AgenticHandoff except:
    - No knowledgebase/ directory; no archiving of workspace to KB.
    - Research journal is kept and accumulated across agents.
    - On handoff, all experiments are deleted so the next agent starts with
      an empty experiments/ folder and only the journal from previous agents.
    """

    def _setup_initial_workspace(self) -> None:
        """Set up the initial workspace structure (no knowledgebase)."""
        # Create workspace directory
        workspace_parent = os.path.dirname(self.log_dir)
        unique_id = str(uuid4())[:8]
        self.workspace_dir = os.path.join(workspace_parent, "workspace", f"workspace_{unique_id}")
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Create subdirectories (no knowledgebase)
        os.makedirs(os.path.join(self.workspace_dir, "experiments"), exist_ok=True)

        # Create initial research journal
        journal_path = os.path.join(self.workspace_dir, "research_journal.md")
        with open(journal_path, "w") as f:
            f.write("# Research Journal\n\n")

        # Copy initial program(s)
        self._copy_initial_programs()

        print(f"Workspace initialized at: {self.workspace_dir}")

    def archive_workspace_to_knowledgebase(self, agent_number: int) -> None:
        """No-op: this ablation does not use a knowledge base."""
        return

    def prepare_workspace_for_next_agent(self) -> None:
        """Prepare workspace for the next agent.

        Clears all files except research_journal.md. Experiments are deleted
        so the next agent does not see prior agents' experiments.
        Then re-creates experiments/ and copies initial program files.
        """
        # Items to preserve (no knowledgebase in this ablation)
        preserve = {"research_journal.md"}

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

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file (method name: agentic_handoff_no_kb)."""
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
            self.save_results(output_data, "agentic_handoff_no_kb", f"_{len(self.all_iterations)}iterations")
        finally:
            self.model = original_model

    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile and save final results (method name: agentic_handoff_no_kb)."""
        # Determine convergence reason
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
                output_data, "agentic_handoff_no_kb", f"_{len(self.all_iterations)}iterations"
            )
        finally:
            self.model = original_model

        print(f"\n{'='*60}")
        print("Handoff (no-KB) optimization completed!")
        print(f"Total agents: {self.current_agent_number}")
        print(f"Total simulations: {len(self.all_iterations)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Results saved to: {history_file}")
        print(f"{'='*60}")

        return output_data
