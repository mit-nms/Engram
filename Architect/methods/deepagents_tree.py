"""
DeepAgents Tree Search Optimization Method

This module implements a tree-based search wrapper around AgenticDeepAgents.
Each round:
1. Select a node (algorithm) from the existing tree using an age-decay greedy heuristic
2. Run AgenticDeepAgents fresh from that node's code to produce a new child
3. Add the child to the tree and update statistics

The selection heuristic is:
    selection_score = score - λ*age + μ*max(0, max_child_delta) - ν*times_selected

Where:
- score: the node's evaluated score
- age: current_round - round_created
- max_child_delta: best_child_score - score (or 0 if no children)
- times_selected: how many times this node has been selected for expansion

Usage
-----
CLI:
    python -m Architect.main \\
        --method deepagents_tree \\
        --model gpt-4o \\
        --task_prompt_path /path/to/task_prompt.txt \\
        --evaluator_path /path/to/evaluator/ \\
        --results_dir ./results \\
        --tree_rounds 10 \\
        --tree_initial_program_path /path/to/initial_program.py \\
        --tree_task_prompt_path /path/to/task_prompt.txt \\
        --tree_system_prompt_path /path/to/system_prompt.txt \\
        --tree_max_review_iterations 40 \\
        --debug

Required Arguments:
    --tree_initial_program_path: Path to the initial program file (root node)
    --tree_task_prompt_path: Path to user prompt file for AgenticDeepAgents
    --tree_system_prompt_path: Path to system prompt file for AgenticDeepAgents

Optional Arguments:
    --tree_rounds: Number of tree expansion rounds (default: 10)
    --tree_max_review_iterations: Max iterations per AgenticDeepAgents run (default: 40)
    --tree_lambda_age: Age penalty weight (default: 0.01)
    --tree_mu_child_improvement: Child improvement bonus weight (default: 0.1)
    --tree_nu_times_selected: Times-selected penalty weight (default: 0.05)
    --tree_save_every_n_rounds: How often to save intermediate results (default: 1)

Output Structure
----------------
The output JSON contains:
- best_solution: {node_id, code, score} - the global best algorithm found
- nodes: dict of node_id -> {node_id, parent_id, round_created, times_selected, score,
                             code, selection_score, children_ids, deepagents_run, metadata}
- rounds: list of {round_idx, selected_parent_id, parent_score, child_id, child_score,
                   selection_snapshot, usage_stats}
- total_rounds, total_nodes: summary statistics
- tree_stats: {max_depth, avg_children}
- config: the hyperparameters used

Each round's AgenticDeepAgents logs are stored in:
    {results_dir}/{task_name}/logs/round_{i}/
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, field, asdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import random

from .common import OptimizationMethod
from .agentic_deepagents import AgenticDeepAgents


@dataclass
class Compaction:
    """
    Summary of what an agent learned, passed to the next agent.

    This enables infinite sequential discovery by letting each agent
    quickly understand what previous agents tried.

    The key field is `previous_agent_summaries` which contains the actual
    text summaries written by previous agents (their lessons learned,
    what worked, what didn't, and recommendations).
    """
    best_score: float
    approaches_tried: List[Dict[str, Any]]  # [{name, score, status, notes}, ...] - fallback only
    insights: List[str]  # fallback only
    next_directions: List[str]  # fallback only
    dead_ends: List[Dict[str, str]]  # [{approach, reason}, ...] - fallback only
    previous_agent_summaries: List[Dict[str, Any]] = field(default_factory=list)  # [{round, score, summary}, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Compaction":
        # Handle backward compatibility
        if "previous_agent_summaries" not in data:
            data["previous_agent_summaries"] = []
        return cls(**data)

    @classmethod
    def empty(cls, initial_score: float = 0.0) -> "Compaction":
        """Create an empty compaction for the first agent."""
        return cls(
            best_score=initial_score,
            approaches_tried=[],
            insights=[],
            next_directions=["Start by understanding the problem and trying a basic approach"],
            dead_ends=[],
            previous_agent_summaries=[]
        )

    def to_markdown(self) -> str:
        """Format compaction as markdown for inclusion in prompts.

        PRIORITY: Use actual agent summaries if available.
        These contain rich lessons learned, not just scores.
        """
        lines = ["## Previous Agents' Summaries", ""]

        lines.append(f"**Best Score Achieved So Far**: {self.best_score:.6f}")
        lines.append("")

        # PRIORITY: Show actual agent summaries (the valuable part!)
        if self.previous_agent_summaries:
            for i, summary_entry in enumerate(self.previous_agent_summaries):
                round_num = summary_entry.get("round", "?")
                score = summary_entry.get("score", 0)
                summary_text = summary_entry.get("summary", "")

                lines.append(f"### Agent from Round {round_num} (score: {score:.6f})")
                lines.append("")
                lines.append(summary_text)
                lines.append("")
                if i < len(self.previous_agent_summaries) - 1:
                    lines.append("---")
                    lines.append("")

            return "\n".join(lines)

        # FALLBACK: Use synthetic summary if no agent summaries available
        # (This happens when agents don't generate "Summary for Next Agent")
        if self.approaches_tried:
            lines.append("### Approaches Tried")
            for approach in self.approaches_tried:
                status_emoji = {"working": "✓", "abandoned": "✗", "promising": "→"}.get(approach.get("status", ""), "•")
                score_val = approach.get('score', 'N/A')
                score_str = f"{score_val:.6f}" if isinstance(score_val, (int, float)) else str(score_val)
                lines.append(f"- {status_emoji} **{approach.get('name', 'Unknown')}**: score={score_str} ({approach.get('status', 'unknown')})")
                if approach.get('notes'):
                    lines.append(f"  - {approach.get('notes')}")
            lines.append("")

        if self.insights:
            lines.append("### Key Insights")
            for insight in self.insights:
                lines.append(f"- {insight}")
            lines.append("")

        if self.next_directions:
            lines.append("### Recommended Next Steps")
            for direction in self.next_directions:
                lines.append(f"- {direction}")
            lines.append("")

        if self.dead_ends:
            lines.append("### Dead Ends (Do Not Repeat)")
            for dead_end in self.dead_ends:
                lines.append(f"- **{dead_end.get('approach', 'Unknown')}**: {dead_end.get('reason', 'No reason given')}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class TreeNode:
    """Represents a node in the algorithm tree."""
    node_id: str
    parent_id: Optional[str]
    round_created: int
    times_selected: int
    score: float
    code: str
    selection_score: float
    children_ids: List[str]
    deepagents_run: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    compaction: Optional[Dict[str, Any]] = None  # Compaction summary for this node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """Create a TreeNode from a dictionary."""
        # Handle missing compaction field for backward compatibility
        if "compaction" not in data:
            data["compaction"] = None
        return cls(**data)


@dataclass
class RoundRecord:
    """Records what happened in a single expansion round."""
    round_idx: int
    selected_parent_id: str
    parent_score: float
    child_id: str
    child_score: float
    selection_snapshot: List[Dict[str, Any]]
    usage_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


class DeepAgentsTreeOptimizer(OptimizationMethod):
    """
    Tree-based optimization using AgenticDeepAgents for node expansion.

    Each round selects an existing node using an age-decay greedy heuristic,
    runs AgenticDeepAgents from that node's code, and adds the result as a
    new child node.
    """

    def __init__(
        self,
        *args,
        # Tree-specific parameters
        tree_rounds: int = 10,
        initial_program_path: List[str] | str = None,
        task_prompt_path: str = None,
        system_prompt_path: str = None,
        # AgenticDeepAgents parameters (forwarded)
        max_review_iterations: int = 40,
        early_stop_patience: int = 10,
        capture_simulation_output: bool = False,
        # Selection heuristic weights (age-decay greedy)
        lambda_age: float = 0.01,  # Penalty for age
        mu_child_improvement: float = 0.1,  # Bonus for child improvement
        nu_times_selected: float = 0.05,  # Penalty for times selected
        # Saving
        save_every_n_rounds: int = 1,
        enable_continue_message: bool = False,
        **kwargs,
    ):
        """
        Initialize the DeepAgents Tree optimizer.

        Args:
            *args: Arguments passed to OptimizationMethod
            tree_rounds: Number of tree expansion rounds
            initial_program_path: Path to the initial program file (root node) or list of paths to initial program files
            task_prompt_path: Path to user prompt file for AgenticDeepAgents
            system_prompt_path: Path to system prompt file for AgenticDeepAgents
            max_review_iterations: Max iterations per AgenticDeepAgents run
            early_stop_patience: Stop AgenticDeepAgents if no improvement for this many iterations
            capture_simulation_output: Capture simulation output
            lambda_age: Weight for age penalty in selection score
            mu_child_improvement: Weight for child improvement bonus
            nu_times_selected: Weight for times-selected penalty
            save_every_n_rounds: How often to save intermediate results
            enable_continue_message: Whether to add the continue message to the agent's messages
            **kwargs: Additional arguments passed to OptimizationMethod
        """
        # Don't run baselines automatically - we'll handle our own initialization
        kwargs["run_baselines"] = False
        super().__init__(*args, **kwargs)

        # Store tree-specific parameters
        self.tree_rounds = tree_rounds
        self.initial_program_path = initial_program_path
        self.task_prompt_path = task_prompt_path
        self.system_prompt_path = system_prompt_path
        self.max_review_iterations = max_review_iterations
        self.early_stop_patience = early_stop_patience
        self.capture_simulation_output = capture_simulation_output
        # Selection heuristic weights
        self.lambda_age = lambda_age
        self.mu_child_improvement = mu_child_improvement
        self.nu_times_selected = nu_times_selected
        self.enable_continue_message = enable_continue_message # Whether to add the continue message to the agent's messages

        self.save_every_n_rounds = save_every_n_rounds
        # print all the parameters
        print(f"tree_rounds: {self.tree_rounds}")
        print(f"initial_program_path: {self.initial_program_path}")
        print(f"task_prompt_path: {self.task_prompt_path}")
        print(f"system_prompt_path: {self.system_prompt_path}")
        print(f"max_review_iterations: {self.max_review_iterations}")
        print(f"early_stop_patience: {self.early_stop_patience}")
        print(f"capture_simulation_output: {self.capture_simulation_output}")

        # Tree data structures
        self.nodes: Dict[str, TreeNode] = {}  # node_id -> TreeNode
        self.rounds: List[RoundRecord] = []
        self.best_node_id: Optional[str] = None
        self.best_score: float = float("-inf")

        # Validate required paths
        if not self.initial_program_path:
            raise ValueError("initial_program_path is required for deepagents_tree")
        if not self.task_prompt_path:
            raise ValueError("task_prompt_path is required for deepagents_tree")
        if not self.system_prompt_path:
            raise ValueError("system_prompt_path is required for deepagents_tree")

        # Save configuration
        self._save_config()

    def _save_config(self) -> None:
        """Save the optimizer configuration to a JSON file."""
        config = {
            "tree_rounds": self.tree_rounds,
            "initial_program_path": self.initial_program_path,
            "task_prompt_path": self.task_prompt_path,
            "system_prompt_path": self.system_prompt_path,
            "max_review_iterations": self.max_review_iterations,
            "early_stop_patience": self.early_stop_patience,
            "capture_simulation_output": self.capture_simulation_output,
            "lambda_age": self.lambda_age,
            "mu_child_improvement": self.mu_child_improvement,
            "nu_times_selected": self.nu_times_selected,
            "save_every_n_rounds": self.save_every_n_rounds,
            "model": self.model,
            "target_name": getattr(getattr(self.task, 'evaluator', None), 'target_name', None),
            "task_name": self.task.name,
        }
        config_path = os.path.join(self.log_dir, "tree_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _load_code(self, path: str) -> str:
        """Load code from a file path."""
        with open(path, "r") as f:
            return f.read()

    def _load_task_prompt(self) -> str:
        """Load the user prompt from file."""
        with open(self.task_prompt_path, "r") as f:
            return f.read()

    def _build_compaction_for_node(self, node: TreeNode) -> Compaction:
        """
        Build a compaction summary for a node by aggregating its ancestors' history.

        PRIORITY: Use actual agent summaries if available.
        These contain rich lessons learned written by the agents themselves.

        FALLBACK: Generate synthetic summary from scores/metadata.
        """
        # Collect agent summaries from ancestors (the valuable part!)
        previous_agent_summaries = []
        approaches_tried = []
        insights = []
        dead_ends = []

        # Walk up the tree to collect history
        current = node
        visited_nodes = []
        while current is not None:
            visited_nodes.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                current = None

        # Process nodes from root to current (reverse order)
        for ancestor in reversed(visited_nodes):
            if ancestor.compaction and isinstance(ancestor.compaction, dict):
                # PRIORITY: Collect agent summaries
                for summary in ancestor.compaction.get("previous_agent_summaries", []):
                    # Avoid duplicates
                    if summary not in previous_agent_summaries:
                        previous_agent_summaries.append(summary)

                # Also collect fallback info
                for insight in ancestor.compaction.get("insights", []):
                    if insight not in insights:
                        insights.append(insight)
                for de in ancestor.compaction.get("dead_ends", []):
                    if de not in dead_ends:
                        dead_ends.append(de)

            # Add synthetic approach entry if no agent summary
            if ancestor.parent_id is not None:  # Skip root
                improvement = ancestor.metadata.get("improvement", 0)
                parent_score = ancestor.metadata.get("parent_score", 0)

                if improvement > 0.0001:
                    status = "working"
                elif improvement < -0.0001:
                    status = "abandoned"
                else:
                    status = "promising"

                approach_entry = {
                    "name": f"Round {ancestor.round_created} approach",
                    "score": ancestor.score,
                    "status": status,
                    "notes": f"Improved from {parent_score:.6f} to {ancestor.score:.6f}" if improvement > 0 else f"No improvement from {parent_score:.6f}"
                }
                approaches_tried.append(approach_entry)

        # Determine next directions based on current state (used only as fallback)
        next_directions = []
        if node.score < 0.001:
            next_directions.append("Current score is low - try a fundamentally different approach")
            # next_directions.append("Consider MILP optimization if not tried yet")
        elif node.score < 0.002:
            next_directions.append("Score is improving - continue refining current approach")
            next_directions.append("Try reducing problem size for faster optimization")
        else:
            next_directions.append("Close to target - focus on fine-tuning")
            next_directions.append("Check which configurations are worst and target those")

        # Keep only the last 3 agent summaries to avoid prompt bloat
        return Compaction(
            best_score=self.best_score,
            approaches_tried=approaches_tried[-5:] if not previous_agent_summaries else [],
            insights=insights[-10:] if not previous_agent_summaries else [],
            next_directions=next_directions if not previous_agent_summaries else [],
            dead_ends=dead_ends[-5:] if not previous_agent_summaries else [],
            previous_agent_summaries=previous_agent_summaries[-3:]
        )

    def _create_augmented_task_prompt(self, parent_node: TreeNode) -> str:
        """
        Create a user prompt augmented with compaction from the parent node.

        This lets the new agent understand what was tried before.
        """
        base_prompt = self._load_task_prompt()

        # Build compaction for the parent
        compaction = self._build_compaction_for_node(parent_node)
        compaction_markdown = compaction.to_markdown()

        # Augment the prompt with compaction
        augmented_prompt = f"""{compaction_markdown}

---

{base_prompt}

---

## Important: You are continuing from previous work

The summary above shows what previous agents have tried. Use this information to:
1. Avoid repeating dead ends
2. Build on promising approaches
3. Try new directions that haven't been explored

**CRITICAL: If a previous agent recommended a fix (in "Recommended Next Steps") but didn't implement it, YOU should try implementing that fix before trying something new. Promising directions deserve multiple fix attempts, not immediate abandonment.**

When you finish, include a "Summary for Next Agent" section in your final response with:
- What you tried and the results
- Key insights you discovered
- Recommended next steps (especially if you identified a fix but didn't try it yet!)
- Any dead ends to avoid
"""

        return augmented_prompt

    def _extract_compaction_from_output(
        self,
        deepagents_output: Dict[str, Any],
        parent_node: TreeNode,
        child_score: float,
        round_idx: int
    ) -> Compaction:
        """
        Extract compaction information from agent output.

        PRIORITY: Use the agent's own summary if available (from "Summary for Next Agent").
        This contains the agent's lessons learned, what worked, what didn't, and next steps.

        Fallback to synthetic summary only if agent didn't provide one.
        """
        # Get the agent's own summary (this is the valuable part!)
        agent_summary = deepagents_output.get("agent_summary", "")

        # Start with parent's compaction as base
        parent_compaction = parent_node.compaction or {}

        # Inherit previous agent summaries (keep last 3)
        previous_summaries = list(parent_compaction.get("previous_agent_summaries", []))

        # If we have an agent summary, use it directly
        if agent_summary:
            print(f"✅ Round {round_idx}: Agent summary captured ({len(agent_summary)} chars)")
            # Add this round's summary to the chain
            this_round_summary = {
                "round": round_idx,
                "score": child_score,
                "summary": agent_summary
            }
            previous_summaries.append(this_round_summary)
            # Keep only last 3 summaries to avoid prompt bloat
            previous_summaries = previous_summaries[-3:]

            return Compaction(
                best_score=max(self.best_score, child_score),
                approaches_tried=[],  # Not needed when we have agent summary
                insights=[],  # The agent summary contains this
                next_directions=[],  # The agent summary contains this
                dead_ends=[],  # The agent summary contains this
                previous_agent_summaries=previous_summaries  # The real gold!
            )

        # FALLBACK: Generate synthetic summary if agent didn't provide one
        # (This should rarely happen if prompts are working)
        print(f"⚠️ Round {round_idx}: Agent did not provide summary, using synthetic fallback")

        improvement = child_score - parent_node.score
        if improvement > 0.0001:
            status = "working"
            approach_notes = f"Improved from {parent_node.score:.6f} to {child_score:.6f}"
        elif improvement < -0.0001:
            status = "abandoned"
            approach_notes = f"Regressed from {parent_node.score:.6f} to {child_score:.6f}"
        else:
            status = "promising"
            approach_notes = f"Score unchanged at {child_score:.6f}"

        this_approach = {
            "name": f"Round {round_idx} (no summary provided)",
            "score": child_score,
            "status": status,
            "notes": approach_notes
        }

        parent_approaches = list(parent_compaction.get("approaches_tried", []))
        all_approaches = parent_approaches + [this_approach]

        return Compaction(
            best_score=max(self.best_score, child_score),
            approaches_tried=all_approaches[-5:],
            insights=list(parent_compaction.get("insights", []))[-5:],
            next_directions=["Agent did not provide next steps - review code and continue"],
            dead_ends=list(parent_compaction.get("dead_ends", []))[-3:],
            previous_agent_summaries=previous_summaries
        )

    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Recursively convert Path objects to strings."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_paths_to_strings(item) for item in obj)
        else:
            return obj

    def _get_all_children_scores(self, node: TreeNode) -> List[float]:
        """Get the scores of all children of a node."""
        if not node.children_ids:
            return []
        return [self.nodes[cid].score for cid in node.children_ids if cid in self.nodes]

    def _compute_selection_score(self, node: TreeNode, current_round: int) -> float:
        """
        Compute the selection score for a node using age-decay greedy heuristic.

        selection_score = score - λ*age + μ*max(0, max_child_delta) + ν*times_selected
        """
        age = current_round - node.round_created
        children_scores = self._get_all_children_scores(node)
        best_child_score = max(children_scores) if children_scores else float("-inf")
        max_child_delta = max(0, best_child_score - node.score) if best_child_score > float("-inf") else 0

        selection_score = (
            node.score
            # - self.lambda_age * age
            # + self.mu_child_improvement * max_child_delta
            # - self.nu_times_selected * node.times_selected
        )
        return selection_score

    def _update_all_selection_scores(self, current_round: int) -> None:
        """Update selection scores for all nodes."""
        for node in self.nodes.values():
            node.selection_score = self._compute_selection_score(node, current_round)

    def _select_parent(self) -> TreeNode:
        """Select the node with the highest selection score for expansion."""
        # from all the ndodes with max selection score, select randomly
        max_nodes = [n for n in self.nodes.values() if n.selection_score == max(n.selection_score for n in self.nodes.values())]
        best_node = random.choice(max_nodes)
        return best_node

    def _get_selection_snapshot(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get a snapshot of the top-k nodes by selection score for debugging."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.selection_score, reverse=True)
        return [
            {
                "node_id": n.node_id,
                "score": n.score,
                "selection_score": n.selection_score,
                "times_selected": n.times_selected,
                "round_created": n.round_created,
                "num_children": len(n.children_ids),
            }
            for n in sorted_nodes[:top_k]
        ]

    def _create_root_node(self) -> TreeNode:
        """Create the root node from the initial program."""
        if self.debug:
            print(f"Creating root node from {self.initial_program_path}")

        if isinstance(self.initial_program_path, list):
            # get the one that has initial_program.py in the name
            initial_program_path = next((p for p in self.initial_program_path if "initial_program.py" in p), None)
            if initial_program_path is None:
                if initial_program_path is None:
                    initial_program_path = next((p for p in self.initial_program_path if "cloudcast_opt.py" in p), None)
                if initial_program_path is None:
                    raise ValueError(f"No initial program path found in {self.initial_program_path}")
            code = self._load_code(initial_program_path)
        elif isinstance(self.initial_program_path, str):
            code = self._load_code(self.initial_program_path)
        else:
            raise ValueError(f"Invalid initial program path: {self.initial_program_path}")

        # Evaluate the initial code
        score, sim_dirs, results = self.evaluate_code(code)

        node_id = f"root_{uuid4().hex[:8]}"
        # Create initial compaction for root node
        initial_compaction = Compaction.empty(initial_score=score)

        root_node = TreeNode(
            node_id=node_id,
            parent_id=None,
            round_created=0,
            times_selected=0,
            score=score,
            code=code,
            selection_score=score,  # Initially just the score
            children_ids=[],
            deepagents_run={
                "best_solution": {"code": code, "score": score},
                "all_iterations": [],
                "history_file": None,
                "workspace_dir": None,
            },
            metadata={
                "is_root": True,
                "initial_program_path": self.initial_program_path,
                "sim_dirs": [str(d) for d in sim_dirs] if sim_dirs else [],
            },
            compaction=initial_compaction.to_dict(),
        )

        self.nodes[node_id] = root_node

        # Update global best
        if score > self.best_score:
            self.best_score = score
            self.best_node_id = node_id

        if self.debug:
            print(f"Root node created: {node_id} with score {score:.4f}")

        return root_node

    def _run_deepagents_for_parent(
        self, parent_node: TreeNode, round_idx: int
    ) -> Dict[str, Any]:
        """
        Run AgenticDeepAgents fresh from a parent node's code.

        Returns the deepagents output_data including best_solution.
        """
        # Create round-specific directory (must be absolute path for shell middleware)
        round_dir = os.path.abspath(os.path.join(self.log_dir, f"round_{round_idx}"))
        os.makedirs(round_dir, exist_ok=True)

        # Write parent code to a temporary file for AgenticDeepAgents to read
        temp_program_path = os.path.join(round_dir, "parent_program.py")
        with open(temp_program_path, "w") as f:
            f.write(parent_node.code)

        # Create augmented user prompt with compaction from parent
        augmented_prompt = self._create_augmented_task_prompt(parent_node)
        temp_task_prompt_path = os.path.join(round_dir, "augmented_task_prompt.txt")
        with open(temp_task_prompt_path, "w") as f:
            f.write(augmented_prompt)

        # Combine parent's code with other files from initial_program_path
        # (excluding initial_program.py since it's replaced by parent_program.py)
        combined_initial_paths = [temp_program_path]  # Start with parent's code

        if isinstance(self.initial_program_path, list):
            # Add other files from the original list (excluding initial_program.py)
            for path_str in self.initial_program_path:
                path_str_lower = str(path_str).lower()
                # Skip initial_program.py since we're using parent_program.py instead
                if "initial_program.py" in path_str_lower:
                    continue

                # Convert to Path for easier checking
                path_obj = Path(path_str)
                if not path_obj.is_absolute():
                    path_obj = Path(os.getcwd()) / path_obj

                # Only add if file/directory exists
                if path_obj.exists():
                    combined_initial_paths.append(str(path_obj))
                elif self.debug:
                    print(f"  Warning: Skipping non-existent path from initial_program_path: {path_str}")
        elif isinstance(self.initial_program_path, str):
            # If it's a string (single file), check if it's not initial_program.py
            path_str_lower = self.initial_program_path.lower()
            if "initial_program.py" not in path_str_lower:
                path_obj = Path(self.initial_program_path)
                if not path_obj.is_absolute():
                    path_obj = Path(os.getcwd()) / path_obj
                if path_obj.exists():
                    combined_initial_paths.append(str(path_obj))

        # Use combined list if we have more than just temp_program_path, otherwise just use temp_program_path
        initial_program_path_for_deepagents = combined_initial_paths if len(combined_initial_paths) > 1 else temp_program_path

        if self.debug:
            print(f"Running AgenticDeepAgents for round {round_idx}")
            print(f"  Parent node: {parent_node.node_id}")
            print(f"  Parent score: {parent_node.score:.4f}")
            print(f"  Results dir: {round_dir}")
            print(f"  Using augmented prompt with compaction")
            if isinstance(initial_program_path_for_deepagents, list):
                print(f"  Initial files: {', '.join(initial_program_path_for_deepagents)}")
            else:
                print(f"  Initial file: {initial_program_path_for_deepagents}")

        # Create and run AgenticDeepAgents with augmented prompt
        deepagents = AgenticDeepAgents(
            task=self.task,
            model=self.model,
            results_dir=round_dir,
            debug=self.debug,
            run_baselines=False,
            task_prompt_path=temp_task_prompt_path,  # Use augmented prompt
            system_prompt_path=self.system_prompt_path,
            initial_program_path=initial_program_path_for_deepagents,
            max_review_iterations=self.max_review_iterations,
            early_stop_patience=self.early_stop_patience,
            capture_simulation_output=self.capture_simulation_output,
            enable_continue_message=self.enable_continue_message,
        )

        # Run optimization
        output_data = deepagents.optimize()

        # Clean up temp files
        if os.path.exists(temp_program_path):
            os.remove(temp_program_path)
        # Keep augmented prompt for debugging (don't delete)

        return output_data

    def _load_best_solution_from_json_files(self, round_idx: int) -> Optional[Dict[str, Any]]:
        """
        Load the best solution from the last (highest iteration) JSON file in the round's log directory.

        Returns:
            Dictionary with keys:
            - 'best_solution': The best solution dict from the JSON file
            - 'all_iterations': List of all iterations from the JSON file
            - 'history_file': Path to the JSON file that was loaded
            Or None if no valid JSON file found
        """
        round_dir = os.path.join(self.log_dir, f"round_{round_idx}")
        log_subdir = os.path.join(round_dir, self.task.name, "logs")

        if not os.path.exists(log_subdir):
            return None

        # Find all JSON files matching *iterations.json pattern
        json_files = [f for f in os.listdir(log_subdir) if f.endswith(".json") and "iterations" in f and "usage_stats" not in f]
        if not json_files:
            return None

        # Extract iteration count from filename (e.g., "15" from "*_15iterations.json")
        def extract_iterations(filename: str) -> int:
            match = re.search(r'(\d+)iterations', filename)
            return int(match.group(1)) if match else 0

        # Sort by iteration count descending (highest first), then by filename for tie-breaking
        json_files_sorted = sorted(json_files, key=lambda f: (extract_iterations(f), f), reverse=True)

        # Load the first (highest iteration) JSON file
        last_json_file = json_files_sorted[0]
        history_file = os.path.join(log_subdir, last_json_file)

        try:
            with open(history_file, 'r') as f:
                json_data = json.load(f)

            best_solution = json_data.get("best_solution", {})
            if not best_solution or best_solution.get("code") is None:
                return None

            return {
                "best_solution": best_solution,
                "all_iterations": json_data.get("all_iterations", []),
                "history_file": history_file,
            }
        except (json.JSONDecodeError, IOError, KeyError) as e:
            if self.debug:
                print(f"  Warning: Failed to load JSON file {last_json_file}: {e}")
            return None

    def _create_child_node(
        self, parent_node: TreeNode, deepagents_output: Dict[str, Any], round_idx: int
    ) -> TreeNode:
        """Create a new child node from AgenticDeepAgents output."""
        best_solution = deepagents_output.get("best_solution", {})
        child_code = best_solution.get("code")
        child_score = best_solution.get("score", float("-inf"))

        # First, try to load the best solution from the last JSON file
        # (AgenticDeepAgents may have saved better results after optimize() returned)
        json_data = None
        if best_solution is None or best_solution.get("score", float("-inf")) == float("-inf"):
            json_data = self._load_best_solution_from_json_files(round_idx)
            if json_data and json_data.get("best_solution"):
                json_best_solution = json_data["best_solution"]
                json_score = json_best_solution.get("score", float("-inf"))

                # Use JSON solution if it's better than what optimize() returned
                if json_score > child_score:
                    if self.debug:
                        print(f"  Using better solution from JSON file (score: {child_score:.6f} -> {json_score:.6f})")
                    best_solution = json_best_solution
                    child_code = json_best_solution.get("code")
                    child_score = json_score
                    # Also update all_iterations from JSON if available
                    if json_data.get("all_iterations"):
                        deepagents_output["all_iterations"] = json_data["all_iterations"]

        # Handle case where AgenticDeepAgents didn't find a valid solution
        if child_code is None:
            child_code = parent_node.code
            child_score = parent_node.score

        node_id = f"node_{round_idx}_{uuid4().hex[:8]}"

        # Store a reference to the history file
        history_file = None
        if json_data and json_data.get("history_file"):
            history_file = json_data["history_file"]
        else:
            # Fallback: find any JSON file if json_data wasn't available
            round_dir = os.path.join(self.log_dir, f"round_{round_idx}")
            log_subdir = os.path.join(round_dir, self.task.name, "logs")
            if os.path.exists(log_subdir):
                json_files = [f for f in os.listdir(log_subdir) if f.endswith(".json")]
                if json_files:
                    history_file = os.path.join(log_subdir, json_files[0])

        # Build compaction for the child node
        # This captures what this agent learned
        child_compaction = self._extract_compaction_from_output(
            deepagents_output, parent_node, child_score, round_idx
        )

        child_node = TreeNode(
            node_id=node_id,
            parent_id=parent_node.node_id,
            round_created=round_idx,
            times_selected=0,
            score=child_score,
            code=child_code,
            selection_score=child_score,  # Will be updated by _update_all_selection_scores
            children_ids=[],
            deepagents_run={
                "best_solution": {"code": child_code, "score": child_score},
                "all_iterations": deepagents_output.get("all_iterations", []),  # Store all intermediate iterations!
                "total_simulations": deepagents_output.get("total_simulations", 0),
                "history_file": history_file,
                "workspace_dir": deepagents_output.get("workspace_dir"),
                "convergence_reason": deepagents_output.get("convergence_reason"),
            },
            metadata={
                "parent_score": parent_node.score,
                "improvement": child_score - parent_node.score,
            },
            compaction=child_compaction.to_dict(),
        )

        # Add to tree
        self.nodes[node_id] = child_node
        parent_node.children_ids.append(node_id)

        # Update global best
        if child_score > self.best_score:
            self.best_score = child_score
            self.best_node_id = node_id
            if self.debug:
                print(f"New global best! Node {node_id} with score {child_score:.4f}")

        return child_node

    def _build_output_data(self) -> Dict[str, Any]:
        """Build the output_data dictionary for saving."""
        # Get best node
        best_node = self.nodes.get(self.best_node_id) if self.best_node_id else None

        # Build all_iterations for compatibility with base class plot_results (stairs plot)
        # Aggregate ALL intermediate iterations from every AgenticDeepAgents run
        all_iterations = []
        iteration_counter = 0

        # Add root's initial score as first iteration
        root_nodes = [n for n in self.nodes.values() if n.parent_id is None]
        for root in root_nodes:
            all_iterations.append({
                "score": root.score,
                "node_id": root.node_id,
                "round": 0,
                "iteration": iteration_counter,
                "is_root": True,
            })
            iteration_counter += 1

        # Add ALL iterations from each round's AgenticDeepAgents run
        for round_record in self.rounds:
            child_node = self.nodes.get(round_record.child_id)
            if child_node:
                # Get the all_iterations from this node's deepagents run
                node_iterations = child_node.deepagents_run.get("all_iterations", [])
                for node_iter in node_iterations:
                    # Add round context to each iteration
                    iter_entry = {
                        "score": node_iter.get("score", float("-inf")),
                        "node_id": child_node.node_id,
                        "round": round_record.round_idx,
                        "iteration": iteration_counter,
                        "parent_id": round_record.selected_parent_id,
                        "parent_score": round_record.parent_score,
                        "success": node_iter.get("success", False),
                    }
                    all_iterations.append(iter_entry)
                    iteration_counter += 1

                # If no iterations were recorded, still add the final child score
                if not node_iterations:
                    all_iterations.append({
                        "score": child_node.score,
                        "node_id": child_node.node_id,
                        "round": round_record.round_idx,
                        "iteration": iteration_counter,
                        "parent_id": round_record.selected_parent_id,
                        "parent_score": round_record.parent_score,
                    })
                    iteration_counter += 1

        output_data = {
            "best_solution": {
                "node_id": self.best_node_id,
                "code": best_node.code if best_node else None,
                "score": self.best_score,
            },
            "all_iterations": all_iterations,  # For compatibility with base plot_results
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "rounds": [r.to_dict() for r in self.rounds],
            "total_rounds": len(self.rounds),
            "total_nodes": len(self.nodes),
            "tree_stats": {
                "max_depth": self._compute_max_depth(),
                "avg_children": self._compute_avg_children(),
            },
            "config": {
                "tree_rounds": self.tree_rounds,
                "max_review_iterations": self.max_review_iterations,
                "early_stop_patience": self.early_stop_patience,
                "capture_simulation_output": self.capture_simulation_output,
                "lambda_age": self.lambda_age,
                "mu_child_improvement": self.mu_child_improvement,
                "nu_times_selected": self.nu_times_selected,
            },
        }

        return self._convert_paths_to_strings(output_data)

    def _compute_max_depth(self) -> int:
        """Compute the maximum depth of the tree."""
        def depth(node_id: str) -> int:
            node = self.nodes.get(node_id)
            if not node or not node.children_ids:
                return 1
            return 1 + max(depth(cid) for cid in node.children_ids)

        root_nodes = [n for n in self.nodes.values() if n.parent_id is None]
        if not root_nodes:
            return 0
        return max(depth(n.node_id) for n in root_nodes)

    def _compute_avg_children(self) -> float:
        """Compute the average number of children per node."""
        if not self.nodes:
            return 0.0
        total_children = sum(len(n.children_ids) for n in self.nodes.values())
        return total_children / len(self.nodes)

    def plot_tree(self, save_path: Optional[str] = None, show: bool = False) -> str:
        """
        Plot the tree structure with simulation scores.

        Args:
            save_path: Path to save the plot (without extension). If None, uses default.
            show: Whether to display the plot interactively.

        Returns:
            Path to the saved plot file.
        """
        if not self.nodes:
            if self.debug:
                print("No nodes to plot.")
            return ""

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes and edges
        for node_id, node in self.nodes.items():
            G.add_node(node_id, score=node.score, round_created=node.round_created)
            if node.parent_id and node.parent_id in self.nodes:
                G.add_edge(node.parent_id, node_id)

        # Get scores for coloring
        scores = [self.nodes[nid].score for nid in G.nodes()]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        score_range = max_score - min_score if max_score != min_score else 1

        # Normalize scores to [0, 1] for colormap
        norm_scores = [(s - min_score) / score_range for s in scores]

        # Use a hierarchical layout for tree visualization
        pos = self._hierarchical_layout(G)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Color map: low scores = red, high scores = green
        cmap = plt.cm.RdYlGn

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1',
            alpha=0.7,
        )

        # Draw nodes with colors based on score
        node_colors = [cmap(ns) for ns in norm_scores]
        node_collection = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=800,
            alpha=0.9,
        )

        # Highlight the best node with a thick border
        if self.best_node_id and self.best_node_id in pos:
            best_pos = pos[self.best_node_id]
            ax.scatter(
                [best_pos[0]], [best_pos[1]],
                s=1200, facecolors='none', edgecolors='gold',
                linewidths=4, zorder=5
            )

        # Create labels with scores
        labels = {}
        for node_id in G.nodes():
            node = self.nodes[node_id]
            # Short label: round number and score
            if node.parent_id is None:
                label = f"R\n{node.score:.4f}"
            else:
                label = f"{node.round_created}\n{node.score:.4f}"
            labels[node_id] = label

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_weight='bold',
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_score, vmax=max_score))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Score', fontsize=12)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=3, label=f'Best (score: {self.best_score:.4f})'),
            mpatches.Patch(facecolor=cmap(1.0), label='High score'),
            mpatches.Patch(facecolor=cmap(0.0), label='Low score'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # Title and labels
        ax.set_title(
            f'DeepAgents Tree Search\n'
            f'Total nodes: {len(self.nodes)} | Rounds: {len(self.rounds)} | '
            f'Best score: {self.best_score:.4f}',
            fontsize=14, fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.plot_dir, f"{self.model}-{self.task.name}-tree")

        plot_file = f"{save_path}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor='white')

        # Also save as PDF for higher quality
        pdf_file = f"{save_path}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

        if self.debug:
            print(f"Tree plot saved to: {plot_file}")
            print(f"Tree plot (PDF) saved to: {pdf_file}")

        return plot_file

    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Compute a hierarchical tree layout with root at top.

        Args:
            G: NetworkX directed graph

        Returns:
            Dictionary mapping node_id to (x, y) positions.
        """
        # Find root nodes (no parent)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]

        if not roots:
            # Fallback to spring layout if no clear root
            return nx.spring_layout(G, seed=42)

        pos = {}

        def assign_positions(node: str, depth: int, left: float, right: float) -> None:
            """Recursively assign positions to nodes."""
            x = (left + right) / 2
            y = -depth  # Negative so root is at top
            pos[node] = (x, y)

            children = list(G.successors(node))
            if not children:
                return

            # Distribute children evenly in the horizontal space
            width = right - left
            child_width = width / len(children)

            for i, child in enumerate(children):
                child_left = left + i * child_width
                child_right = child_left + child_width
                assign_positions(child, depth + 1, child_left, child_right)

        # Start from root(s)
        total_width = len(self.nodes) * 2  # Give enough horizontal space
        if len(roots) == 1:
            assign_positions(roots[0], 0, 0, total_width)
        else:
            # Multiple roots - distribute them
            root_width = total_width / len(roots)
            for i, root in enumerate(roots):
                assign_positions(root, 0, i * root_width, (i + 1) * root_width)

        return pos

    def optimize(self) -> Dict[str, Any]:
        """
        Run the tree-based optimization.

        Returns:
            Dictionary containing optimization results including best_solution,
            all nodes, and round history.
        """
        if self.debug:
            print("\n" + "=" * 60)
            print("DeepAgents Tree Optimization")
            print("=" * 60)
            print(f"Rounds: {self.tree_rounds}")
            print(f"Max review iterations per round: {self.max_review_iterations}")
            print(f"Early stop patience: {self.early_stop_patience}")
            print(f"Capture simulation output: {self.capture_simulation_output}")
            print(f"Selection weights: λ={self.lambda_age}, μ={self.mu_child_improvement}, ν={self.nu_times_selected}")
            print("=" * 60 + "\n")

        # Create root node from initial program
        root_node = self._create_root_node()

        # Main expansion loop
        for round_idx in range(1, self.tree_rounds + 1):
            if self.debug:
                print(f"\n{'='*60}")
                print(f"Round {round_idx}/{self.tree_rounds}")
                print(f"{'='*60}")

            # Check stopping conditions
            should_continue, reason = self.should_continue()
            if not should_continue:
                if self.debug:
                    print(f"Stopping early: {reason}")
                break

            # Update selection scores for all nodes
            self._update_all_selection_scores(round_idx)

            # Get selection snapshot for debugging
            selection_snapshot = self._get_selection_snapshot(top_k=5)

            # Select parent node to expand
            parent_node = self._select_parent()
            parent_node.times_selected += 1

            if self.debug:
                print(f"Selected parent: {parent_node.node_id}")
                print(f"  Score: {parent_node.score:.4f}")
                print(f"  Selection score: {parent_node.selection_score:.4f}")
                print(f"  Times selected: {parent_node.times_selected}")

            # Run AgenticDeepAgents from parent
            try:
                deepagents_output = self._run_deepagents_for_parent(parent_node, round_idx)
            except Exception as e:
                if self.debug:
                    print(f"AgenticDeepAgents failed: {e}")
                    import traceback
                    traceback.print_exc()
                # Try to load the best solution from JSON files if they exist
                # (AgenticDeepAgents may have saved results before the exception)
                json_data = self._load_best_solution_from_json_files(round_idx)
                if json_data and json_data.get("best_solution"):
                    if self.debug:
                        print(f"  Found solution in JSON file (score: {json_data['best_solution'].get('score', float('-inf')):.6f})")
                    deepagents_output = {
                        "best_solution": json_data["best_solution"],
                        "all_iterations": json_data.get("all_iterations", []),
                        "total_simulations": len(json_data.get("all_iterations", [])),
                        "convergence_reason": f"error: {str(e)} (recovered from JSON file)",
                    }
                else:
                    # Fall back to parent's code if no JSON file found
                    deepagents_output = {
                        "best_solution": {"code": parent_node.code, "score": parent_node.score},
                        "total_simulations": 0,
                        "convergence_reason": f"error: {str(e)}",
                    }

            # Create child node
            child_node = self._create_child_node(parent_node, deepagents_output, round_idx)

            # Record round
            round_record = RoundRecord(
                round_idx=round_idx,
                selected_parent_id=parent_node.node_id,
                parent_score=parent_node.score,
                child_id=child_node.node_id,
                child_score=child_node.score,
                selection_snapshot=selection_snapshot,
            )
            self.rounds.append(round_record)

            if self.debug:
                print(f"Created child: {child_node.node_id}")
                print(f"  Score: {child_node.score:.4f}")
                print(f"  Improvement: {child_node.score - parent_node.score:+.4f}")
                print(f"  Current best: {self.best_score:.4f} ({self.best_node_id})")

            # Periodic saving
            if round_idx % self.save_every_n_rounds == 0:
                output_data = self._build_output_data()
                self.save_results(
                    output_data,
                    method_name="deepagents_tree",
                    extra_info=f"_{round_idx}rounds",
                )

        # Final save
        output_data = self._build_output_data()
        history_file, plot_path, _ = self.save_results(
            output_data,
            method_name="deepagents_tree",
            extra_info=f"_{len(self.rounds)}rounds",
        )

        # Plot the tree
        tree_plot_path = self.plot_tree()

        if self.debug:
            print("\n" + "=" * 60)
            print("DeepAgents Tree Optimization Complete!")
            print("=" * 60)
            print(f"Total rounds: {len(self.rounds)}")
            print(f"Total nodes: {len(self.nodes)}")
            print(f"Best score: {self.best_score:.4f}")
            print(f"Best node: {self.best_node_id}")
            print(f"Results saved to: {history_file}")
            if tree_plot_path:
                print(f"Tree plot saved to: {tree_plot_path}")
            print("=" * 60 + "\n")

        return output_data

