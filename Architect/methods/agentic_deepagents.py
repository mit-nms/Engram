import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Dict
from uuid import uuid4

from rich.console import Console
from rich.markdown import Markdown
from .common import OptimizationMethod
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemState
from langchain.agents.middleware import ShellToolMiddleware, DockerExecutionPolicy 
from langchain.chat_models import init_chat_model

from langchain_core.tools import BaseTool
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import Command

from .deepagents_utils.filesystem_utils import add_file_to_agent_filesystem, PathNormalizationMiddleware
from .deepagents_utils.tee_utils import Tee
from .deepagents_utils.tool_descriptions import (
    CustomSummarizationMiddleware,
    RUN_SIMULATION_TOOL_DESCRIPTION,
    SUMMARY_PROMPT,
    CONTINUE_MESSAGE,
)


class AgenticDeepAgents(OptimizationMethod):
    """Agentic deep agents that uses deep agents to optimize the code."""

    def __init__(
        self,
        *args,
        task_prompt_path: str,
        system_prompt_path: str,
        initial_program_path: str | List[str] | None = None,
        max_review_iterations: int = 40,
        use_summarization_middleware: bool = False,
        early_stop_patience: int = 10,
        capture_simulation_output: bool = False,
        enable_continue_message: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Set up terminal output capture to log file
        self.terminal_log_path = os.path.join(self.log_dir, "console_output.log")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_stdout = Tee(self.terminal_log_path, self._original_stdout)
        self._tee_stderr = Tee(self.terminal_log_path, self._original_stderr)
        
        # Replace stdout and stderr with Tee objects
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr
        
        self.set_prompts_from_files(task_prompt_path, system_prompt_path)
        self.initial_program_path = initial_program_path
        self.max_review_iterations = max_review_iterations
        self.use_summarization_middleware = use_summarization_middleware
        self.early_stop_patience = early_stop_patience
        self.capture_simulation_output = capture_simulation_output
        self.enable_continue_message = enable_continue_message # Whether to add the continue message to the agent's messages
        # Initialize tracking variables for internal logging
        self.all_iterations = []
        self.best_score = float("-inf")
        self.best_code = None
        self.iterations_since_improvement = 0  # Track iterations without improvement
        # Dict keyed by tool_call_id; entries are consumed once when logged
        self.simulation_results = {}
        
        self.build_agent()
        
        # Override args.json with AgenticDeepAgents-specific logging
        args_dict = vars(self).copy()
        # Remove unused base class attributes
        args_dict.pop('base_implement_prompt', None)
        args_dict.pop('architect', None)
        if not hasattr(self, 'system_prompt'):
            args_dict['system_prompt'] = None
        if not hasattr(self, 'task_prompt'):
            args_dict['task_prompt'] = None
        
        if hasattr(self, 'workspace_dir') and self.workspace_dir:
            args_dict['workspace_dir'] = str(self.workspace_dir)
        
        with open(f"{self.log_dir}/args.json", "w") as f:
            json.dump(args_dict, f, indent=2, default=str)

        # Console for pretty-printing markdown reasoning summaries in debug logs
        # Create console that writes to stdout (which is now our Tee object)
        # This ensures Rich output is captured to both terminal and log file
        self._console = Console(file=sys.stdout, force_terminal=True)

    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Recursively convert Path objects to strings in nested data structures."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_paths_to_strings(item) for item in obj)
        else:
            return obj

    def _process_stream_chunk(self, chunk: Any) -> tuple[Optional[str], Optional[Dict]]:
        """Extract node_name and state_update from a stream chunk.
        
        Returns:
            Tuple of (node_name, state_update) or (None, None) if chunk format is unexpected.
        """
        node_name = None
        state_update = None
        
        if isinstance(chunk, tuple):
            if len(chunk) == 2:
                node_name, state_update = chunk
            elif len(chunk) == 3:
                namespace, stream_mode, data = chunk
                if stream_mode == "updates":
                    node_name = namespace
                    state_update = data
                else:
                    return None, None
            else:
                if self.debug:
                    print(f"Warning: Unexpected tuple length {len(chunk)}, skipping...")
                return None, None
        elif isinstance(chunk, dict):
            if len(chunk) == 1:
                node_name = list(chunk.keys())[0]
                state_update = chunk[node_name]
            else:
                for key, value in chunk.items():
                    if isinstance(value, dict) and ("messages" in value or "files" in value):
                        node_name = key
                        state_update = value
                        break
                else:
                    node_name = list(chunk.keys())[0]
                    state_update = chunk[node_name]
        else:
            if self.debug:
                print(f"Warning: Unexpected chunk format: {type(chunk)}, skipping...")
            return None, None
        
        # Handle Overwrite wrapper (from LangGraph)
        if isinstance(state_update, dict):
            for key in list(state_update.keys()):
                value = state_update[key]
                if hasattr(value, 'value'):
                    state_update[key] = value.value
        
        return node_name, state_update

    def _print_debug_message(self, msg: Any) -> None:
        """Print debug information for a message."""
        if not self.debug:
            return
        
        msg_type = type(msg).__name__
        
        # Show tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"\n🔧 Tool Calls:")
            for tc in msg.tool_calls:
                tool_name = tc.get('name', 'unknown')
                tool_id = tc.get('id', 'unknown')
                print(f"   - {tool_name} (id: {tool_id})")
                if 'args' in tc:
                    args = tc['args']
                    for key, value in args.items():
                        if key in ['file_path', 'command', 'code', 'pattern', 'path']:
                            print(f"     {key.capitalize()}: {value}")
                        else:
                            print(f"     {key}: {str(value)}")
        
        # Show tool results
        if hasattr(msg, 'name') and msg.name:
            print(f"\n✅ Tool Result: {msg.name}")
            if hasattr(msg, 'content') and msg.content:
                print(f"   {msg.content}")
            if hasattr(msg, 'tool_call_id'):
                print(f"   Tool Call ID: {msg.tool_call_id}")

        # Show AI / Human messages (including reasoning summaries)
        if msg_type in ['AIMessage', 'HumanMessage'] and hasattr(msg, 'content') and msg.content:
            content = msg.content

            if isinstance(content, str):
                print(f"\n💬 {msg_type}: {content}")
                return

            if isinstance(content, list):
                for i, item in enumerate(content):
                    if not isinstance(item, dict):
                        # Non-dict items: print them directly
                        print(f"\n💬 {msg_type}: {item}")
                        continue

                    item_type = item.get('type')

                    # Reasoning summary blocks
                    if item_type == 'reasoning':
                        summaries = item.get('summary') or []
                        if summaries:
                            for s in summaries:
                                text = s.get('text') if isinstance(s, dict) else str(s)
                                # Render markdown summary nicely in the
                                # terminal using rich (will also be captured by Tee)
                                self._console.print(Markdown(text))
                    
                    # Skip tool call results (they have a 'name' attribute and are handled separately)
                    elif 'name' in item:
                        # This is a tool call result, skip it
                        continue
                    
                    # Text content and other non-tool-call content
                    elif item_type == 'text' or 'text' in item:
                        text_content = item.get('text', '')
                        if text_content:
                            print(f"\n💬 {msg_type}: {text_content}")
                    
                    # Other content types (print them)
                    else:
                        # Print other content items, excluding tool call results
                        if 'name' not in item:  # Double-check to avoid tool results
                            print(f"\n💬 {msg_type}: {item}")

            else:
                # Unexpected content type, fall back to repr
                print(f"\n💬 {msg_type}: {repr(content)}")

    def optimize(self) -> Dict[str, Any]:
        """Optimize the code using deep agents."""
        try:
            if self.debug:
                print("\n" + "=" * 60)
                print("Streaming agent execution (real-time progress)...")
                print("=" * 60 + "\n")
            
            step_count = 0
            accumulated_state = {"messages": [], "files": {}}
            
            initial_files = self._prepare_initial_files()
            if initial_files:
                accumulated_state["files"].update(initial_files)
                if self.debug:
                    print(f"Pre-populated agent filesystem with: {list(initial_files.keys())}")
            
            input_messages = {
                "messages": [{"role": "user", "content": self.task_prompt}]
            }
            if initial_files:
                input_messages["files"] = initial_files
            
            iteration = 0
            try:
                while iteration < self.max_review_iterations:
                    if self.debug:
                        print(f"\n{'='*60}")
                        print(f"Iteration {iteration + 1}/{self.max_review_iterations}")
                        print(f"{'='*60}\n")
                    
                    for chunk in self.agent.stream(input_messages, stream_mode="updates"):
                        step_count += 1
                        node_name, state_update = self._process_stream_chunk(chunk)
                        
                        if node_name is None or state_update is None:
                            continue
                        
                        if isinstance(state_update, dict) and "messages" in state_update:
                            new_messages = state_update["messages"]
                            if hasattr(new_messages, 'value'):
                                new_messages = new_messages.value
                            if new_messages:
                                if isinstance(new_messages, list):
                                    accumulated_state["messages"].extend(new_messages)
                                else:
                                    accumulated_state["messages"].append(new_messages)
                                
                                # Track simulation results when run_simulation tool is called
                                for msg in new_messages if isinstance(new_messages, list) else [new_messages]:
                                    # Check if this is a tool call for run_simulation
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            if isinstance(tool_call, dict):
                                                tool_name = tool_call.get('name', '')
                                                tool_call_id = tool_call.get('id', '')
                                                if tool_name == 'run_simulation' and tool_call_id:
                                                    # We'll process this when we get the tool result
                                                    pass
                                    
                                    # Check if this is a tool result for run_simulation.
                                    # We treat entries in simulation_results as one-shot:
                                    # once a tool_call_id has been logged, its entry is
                                    # removed so that replayed messages don't create
                                    # duplicate iterations.
                                    if hasattr(msg, 'tool_call_id'):
                                        sim_result = self.simulation_results.pop(msg.tool_call_id, None)
                                    else:
                                        sim_result = None

                                    if sim_result is not None:
                                        # Create iteration data dictionary
                                        iteration_data = {
                                            "simulation_number": len(self.all_iterations) + 1,
                                            "code": sim_result["code"],
                                            "score": sim_result["score"],
                                            "success": sim_result["success"],
                                            "sim_dirs": [str(d) for d in sim_result.get("sim_dirs", [])],
                                            "error": sim_result.get("error_message", ""),
                                            "file_path": sim_result.get("file_path", ""),
                                        }
                                            
                                        # Convert Path objects to strings for JSON serialization
                                        try:
                                            iteration_data = self._convert_paths_to_strings(iteration_data)
                                            # Test JSON serialization
                                            json.dumps(iteration_data)
                                            self.all_iterations.append(iteration_data)
                                        except (TypeError, ValueError) as json_error:
                                            if self.debug:
                                                print(f"⚠️ Failed to serialize iteration data: {json_error}")
                                            # Store minimal data if serialization fails
                                            self.all_iterations.append({
                                                "simulation_number": len(self.all_iterations) + 1,
                                                "code": sim_result["code"],
                                                "score": float(sim_result["score"]) if isinstance(sim_result["score"], (int, float)) else str(sim_result["score"]),
                                                "success": sim_result["success"],
                                                "sim_dirs": [],
                                                "error": str(sim_result.get("error_message", "")),
                                                "serialization_error": str(json_error)
                                            })
                                        
                                        if sim_result["success"] and sim_result["score"] > self.best_score:
                                            self.best_score = sim_result["score"]
                                            self.best_code = sim_result["code"]
                                            self.iterations_since_improvement = 0  # Reset counter on improvement
                                            if self.debug:
                                                print(f"🎯 New best score: {self.best_score:.4f}")
                                        else:
                                            self.iterations_since_improvement += 1
                                            if self.debug and self.iterations_since_improvement > 0:
                                                print(f"⏳ No improvement for {self.iterations_since_improvement} iterations (patience: {self.early_stop_patience})")
                                        
                                        # Periodically save intermediate results (every 5 iterations)
                                        if len(self.all_iterations) % 5 == 0:
                                            output_data = {
                                                "best_solution": {
                                                    "code": self.best_code,
                                                    "score": self.best_score,
                                                },
                                                "all_iterations": self.all_iterations,
                                                "final_code": self.best_code,
                                                "total_simulations": len(self.all_iterations),
                                            }
                                            # Temporarily restore model name string for filename construction
                                            original_model = self.model
                                            self.model = self.model_name_str
                                            try:
                                                self.save_results(
                                                    output_data,
                                                    "agentic_deepagents",
                                                    f"_{len(self.all_iterations)}iterations"
                                                )
                                            finally:
                                                # Restore the chat model object
                                                self.model = original_model
                                
                                if self.debug:
                                    for msg in new_messages if isinstance(new_messages, list) else [new_messages]:
                                        self._print_debug_message(msg)
                        
                        if isinstance(state_update, dict) and "files" in state_update:
                            files = state_update["files"]
                            if hasattr(files, 'value'):
                                files = files.value
                            if files and isinstance(files, dict):
                                accumulated_state["files"].update(files)
                                if self.debug:
                                    print(f"\n📁 Files updated: {list(files.keys())}")
                    
                    if self.debug:
                        print(f"\n{'='*60}")
                        print(f"Main agent completed iteration {iteration + 1} after {step_count} steps!")
                        print(f"{'='*60}\n")
                    
                    # Check for early stopping due to no improvement
                    if self.iterations_since_improvement >= self.early_stop_patience:
                        if self.debug:
                            print(f"\n🛑 Early stopping: No improvement for {self.iterations_since_improvement} iterations")
                            print(f"   Best score achieved: {self.best_score:.4f}")
                        break

                    if not self.enable_continue_message:
                        # Check if agent has signaled completion
                        if self._has_agent_signaled_completion(accumulated_state["messages"]):
                            if self.debug:
                                print(f"\n✅ Agent signaled completion. Stopping iteration loop.")
                            break
                    else:
                        # Automatically add continue message and proceed to next iteration
                        from langchain_core.messages import HumanMessage
                        accumulated_state["messages"].append(HumanMessage(content=CONTINUE_MESSAGE))
                        input_messages = {
                            "messages": accumulated_state["messages"],
                            "files": accumulated_state["files"]
                        }
                    iteration += 1
            
            except Exception as e:
                if self.debug:
                    print(f"\n❌ Error during streaming: {e}")
                    import traceback
                    traceback.print_exc()
                raise
            
            result = {
                "messages": accumulated_state["messages"],
                "files": accumulated_state["files"]
            }
            
            if self.debug:
                print(f"\nFinal Summary:")
                print(f"   Total steps: {step_count}")
                print(f"   Total messages: {len(accumulated_state['messages'])}")
                print(f"   Files created/updated: {len(accumulated_state['files'])}")
                if accumulated_state['files']:
                    print(f"   File paths: {list(accumulated_state['files'].keys())}")
                print()
                self.display_tool_calls(result)
            
            # Determine convergence reason
            if self.iterations_since_improvement >= self.early_stop_patience:
                convergence_reason = f"early_stop_no_improvement_{self.early_stop_patience}"
            elif iteration >= self.max_review_iterations:
                convergence_reason = "max_iterations"
            else:
                convergence_reason = "completed"

            # Extract agent's summary from the last AI message
            agent_summary = self._extract_agent_summary(accumulated_state["messages"])

            # Prepare final results
            output_data = {
                "best_solution": {
                    "code": self.best_code,
                    "score": self.best_score,
                },
                "all_iterations": self.all_iterations,
                "final_code": self.best_code,
                "total_simulations": len(self.all_iterations),
                "convergence_reason": convergence_reason,
                "workspace_dir": str(self.workspace_dir),
                "iterations_since_improvement": self.iterations_since_improvement,
                "agent_summary": agent_summary,  # The agent's own lessons learned
            }
            
            # Convert Path objects to strings before saving
            output_data = self._convert_paths_to_strings(output_data)
            
            # Temporarily restore model name string for filename construction # TODO: this is weird, we should use a different approach
            # (save_results uses self.model for filenames)
            original_model = self.model
            self.model = self.model_name_str
            
            try:
                # Save results
                history_file, plot_path, _ = self.save_results(
                    output_data,
                    "agentic_deepagents",
                    f"_{len(self.all_iterations)}iterations"
                )
            finally:
                # Restore the chat model object
                self.model = original_model
            
            if self.debug:
                print(f"\nDeep agents optimization completed!")
                print(f"Total iterations: {len(self.all_iterations)}")
                print(f"Best score achieved: {self.best_score:.4f}")
                print(f"Workspace directory: {self.workspace_dir}")
                print(f"Results saved to: {history_file}")
                print(f"Terminal output saved to: {self.terminal_log_path}")
            
            return output_data
        
        finally:
            # Always cleanup terminal capture when optimize() completes
            # This ensures the log file is properly closed
            self._cleanup_terminal_capture()
    
    def _cleanup_terminal_capture(self):
        """Restore original stdout/stderr and close log files."""
        try:
            # Flush and close Tee objects
            if hasattr(self, '_tee_stdout'):
                self._tee_stdout.close()
            if hasattr(self, '_tee_stderr'):
                self._tee_stderr.close()
            
            # Restore original streams
            if hasattr(self, '_original_stdout'):
                sys.stdout = self._original_stdout
            if hasattr(self, '_original_stderr'):
                sys.stderr = self._original_stderr
        except Exception as e:
            # If cleanup fails, at least try to restore stdout/stderr
            try:
                sys.stdout = self._original_stdout if hasattr(self, '_original_stdout') else sys.__stdout__
                sys.stderr = self._original_stderr if hasattr(self, '_original_stderr') else sys.__stderr__
            except Exception:
                pass

    def _has_agent_signaled_completion(self, messages: list) -> bool:
        """
        Check if the agent has signaled completion by returning "## Summary for Next Agent".

        Looks at the most recent AI messages (last 5 messages) to detect if the agent
        has provided a summary indicating it's done with its work.

        Returns:
            True if agent has signaled completion, False otherwise.
        """
        import re

        # Pattern to match "## Summary for Next Agent" or "### Summary for Next Agent"
        completion_pattern = re.compile(
            r'#{2,3}\s*Summary for Next Agent',
            re.IGNORECASE
        )

        # Look at the most recent AI messages (last 5 to avoid false positives)
        recent_ai_messages = []
        for msg in reversed(messages):
            msg_type = type(msg).__name__
            if msg_type != 'AIMessage':
                continue
            if not hasattr(msg, 'content') or not msg.content:
                continue

            # Extract text content from message
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "\n".join(text_parts)

            if isinstance(content, str) and len(content) > 10:
                recent_ai_messages.append(content)
                # Only check last 5 AI messages to avoid false positives
                if len(recent_ai_messages) >= 5:
                    break

        # Check if any recent message contains the completion signal
        for content in recent_ai_messages:
            if completion_pattern.search(content):
                if self.debug:
                    print(f"\n✅ Agent signaled completion with 'Summary for Next Agent'")
                return True

        return False

    def _extract_agent_summary(self, messages: list) -> str:
        """
        Extract the agent's summary with a simple two-phase approach:
        1. Try regex on ALL AI messages - if found, return immediately
        2. If regex fails, give ALL AI message content to LLM to extract/generate summary

        Returns the extracted summary text, or empty string if not found.
        """
        import re

        # Collect ALL AI message contents
        all_ai_contents = []
        for msg in messages:  # Chronological order
            msg_type = type(msg).__name__
            if msg_type != 'AIMessage':
                continue
            if not hasattr(msg, 'content') or not msg.content:
                continue

            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "\n".join(text_parts)

            if isinstance(content, str) and len(content) > 20:
                all_ai_contents.append(content)

        if not all_ai_contents:
            if self.debug:
                print("\n⚠️ No AI messages found")
            return ""

        if self.debug:
            print(f"\n🔍 Scanning {len(all_ai_contents)} AI messages for summary...")

        # PHASE 1: Try regex extraction first (fast)
        # Search in reverse order (most recent first) for the summary section
        summary_pattern = re.compile(
            r'##\s*Summary for Next Agent\s*\n(.*?)(?=\n={5,}|\n💬|\Z)',
            re.DOTALL | re.IGNORECASE
        )

        for content in reversed(all_ai_contents):
            # We want the last summary found, that is associated with the last result
            match = summary_pattern.search(content)
            if match:
                summary = match.group(1).strip()
                if len(summary) > 100:  # Valid summary found
                    if self.debug:
                        print(f"\n📋 Regex extracted summary ({len(summary)} chars)")
                    return summary

        # PHASE 2: Regex didn't find it - send to LLM
        if self.debug:
            print("\n🤖 Regex didn't find summary, sending to LLM...")

        # Combine all AI messages (limit total size to avoid token explosion)
        # Take last N messages that fit within ~30k chars
        combined_text = ""
        for content in reversed(all_ai_contents):
            if len(combined_text) + len(content) > 30000:
                break
            combined_text = content + "\n\n---\n\n" + combined_text

        # Use LLM to extract or generate summary
        extraction_prompt = """You are helping extract or generate a summary for the next AI agent.

TASK: Look through the AI agent's work below and either:
1. EXTRACT the "Summary for Next Agent" section if it exists (copy it EXACTLY, word for word)
2. GENERATE a summary if no explicit summary section exists

WHAT TO LOOK FOR (for extraction):
- A section titled "Summary for Next Agent" (may have ## or ### markers)
- Contains: Agent Mode, Best Result, What I Tried, Key Insights, Next Steps, Dead Ends

IF YOU FIND AN EXPLICIT SUMMARY:
- Copy it EXACTLY as written, preserving formatting
- Do NOT paraphrase or rewrite

IF NO EXPLICIT SUMMARY EXISTS:
- Generate a summary based on the agent's work
- Use this format:

### Agent Mode
- Mode: [EXPLORATION or EXPLOITATION based on what they did]

### Best Result
- Score: [best score achieved]
- Approach: [what approach got this score]

### What I Tried
[List approaches tried with scores and outcomes]

### Key Insights
[What the agent learned]

### Recommended Next Steps
[What to try next]

### Dead Ends
[What didn't work and should be avoided]

AI AGENT'S WORK:
---
""" + combined_text + """
---

SUMMARY (extracted or generated):"""

        try:
            # Use a fast/cheap model for extraction
            extraction_model = init_chat_model("openai:gpt-4o-mini")
            response = extraction_model.invoke(extraction_prompt)

            extracted = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            if len(extracted) < 50:
                if self.debug:
                    print("\n⚠️ LLM returned too short response")
                return ""

            if self.debug:
                print(f"\n📋 LLM extracted/generated summary ({len(extracted)} chars)")
            return extracted

        except Exception as e:
            if self.debug:
                print(f"\n⚠️ LLM extraction failed: {e}")
            return ""

    def build_agent(self) -> None:
        """Build the agent using deep agents."""
        # Store original model name for filename construction (before converting to chat model object)
        self.model_name_str = self.model
        
        # # Check if this is a reasoning model (o1, o3, o4-mini, etc.)
        is_reasoning_model = any(prefix in self.model.lower() for prefix in ['o1', 'o3', 'o4'])
        
        if is_reasoning_model:
            # Enable reasoning summaries for reasoning models
            self.model = init_chat_model(
                f"openai:{self.model}",
                reasoning={"summary": "auto"}
            )
        else:
            self.model = init_chat_model(f"openai:{self.model}")
        
        # Create workspace directory
        workspace_parent = os.path.dirname(self.log_dir)
        unique_id = str(uuid4())[:8]  
        workspace_dir = os.path.join(workspace_parent, "workspace", f"workspace_{unique_id}")
        os.makedirs(workspace_dir, exist_ok=True)
        self.workspace_dir = workspace_dir
        
        if self.debug:
            print(f"Workspace directory: {workspace_dir}")
        
        self.backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)
        
        current_uid = os.getuid()
        current_gid = os.getgid()

        shell_middleware = ShellToolMiddleware(
            workspace_root=workspace_dir,
            execution_policy=DockerExecutionPolicy(
                image="python:3.11",
                network_enabled=True,
                read_only_rootfs=False,
                remove_container_on_exit=True,
                user=f"{current_uid}:{current_gid}",
            ),
            env={**os.environ, "PYTHONPATH": "/tmp/.packages", "HOME": "/tmp"},
            startup_commands=[
                f"cd {workspace_dir}",
                "pip install --no-cache-dir --target=/tmp/.packages numpy pandas",
            ],
        )
        
        path_normalization_middleware = PathNormalizationMiddleware()
        
        run_sim_tool = self.create_run_simulation_tool()
        
        middleware_list = [
            path_normalization_middleware, 
            shell_middleware,
        ]
        
        if self.use_summarization_middleware:
            middleware_list.append(
                CustomSummarizationMiddleware(
                    model=self.model_name_str,
                    max_tokens_before_summary=16000,
                    messages_to_keep=10,
                    summary_prompt=SUMMARY_PROMPT,
                )
            )
        
        self.agent = create_deep_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            backend=self.backend,
            tools=[run_sim_tool],
            middleware=middleware_list,
        )
    
    def set_prompts_from_files(self, task_prompt_path: str, system_prompt_path: str) -> None:
        """Set user and system prompts from files given their paths (used in the constructor)."""
        for attr, path in [("task_prompt", task_prompt_path), ("system_prompt", system_prompt_path)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r") as f:
                setattr(self, attr, f.read())

    def _prepare_initial_files(self) -> Dict[str, Any]:
        """Prepare initial files to be added to the agent's filesystem.
        
        Supports both files and directories. For directories, recursively copies
        all files within them, preserving the directory structure.
        """
        if not self.initial_program_path:
            return {}
        
        paths = [self.initial_program_path] if isinstance(self.initial_program_path, str) else (self.initial_program_path or [])
        result = {}
        
        from deepagents.backends.utils import create_file_data
        
        for path_str in paths:
            host_path = Path(path_str)
            if not host_path.is_absolute():
                host_path = Path(os.getcwd()) / host_path
            
            if not host_path.exists():
                raise FileNotFoundError(f"Initial program file or directory not found: {host_path}")
            
            if host_path.is_file():
                # Handle single file
                agent_file_path = f"/{host_path.name}"
                content = host_path.read_text(encoding='utf-8')
                
                if isinstance(self.backend, FilesystemBackend):
                    relative_path = agent_file_path.lstrip('/')
                    target_path = self.backend.cwd / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(content, encoding='utf-8')
                    if self.debug:
                        print(f"Written initial file to filesystem: {target_path}")
                
                file_data = create_file_data(content)
                result[agent_file_path] = file_data
                
            elif host_path.is_dir():
                # Handle directory - recursively copy all files
                for root, dirs, files in os.walk(host_path):
                    for file_name in files:
                        file_path = Path(root) / file_name
                        # Get relative path from the directory being copied
                        relative_path = file_path.relative_to(host_path)
                        # Create agent path preserving directory structure
                        agent_file_path = f"/{host_path.name}/{relative_path.as_posix()}"
                        
                        content = file_path.read_text(encoding='utf-8')
                        
                        if isinstance(self.backend, FilesystemBackend):
                            relative_agent_path = agent_file_path.lstrip('/')
                            target_path = self.backend.cwd / relative_agent_path
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            target_path.write_text(content, encoding='utf-8')
                            if self.debug:
                                print(f"Written initial file to filesystem: {target_path}")
                        
                        file_data = create_file_data(content)
                        result[agent_file_path] = file_data
            else:
                raise ValueError(f"Path is neither a file nor a directory: {host_path}")
        
        return result

    def create_run_simulation_tool(self) -> BaseTool:
        """Create a tool to run the simulation."""
        
        # Capture self in a local variable to ensure proper closure
        optimizer_instance = self

        @tool(description=RUN_SIMULATION_TOOL_DESCRIPTION)
        def run_simulation(file_path: str, runtime: ToolRuntime[None, FilesystemState]) -> str | Command:
            """Run the simulation with code from a file and return the results."""
            try:
                if not file_path.startswith('/'):
                    return f"Error: file_path must be an absolute path starting with /. Got: {file_path}"
                
                try:
                    file_content = optimizer_instance.backend.read(file_path, offset=0, limit=10000)
                    if file_content.startswith('Error:'):
                        return file_content
                    
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
                    return f"Error reading file {file_path}: {str(e)}. Make sure the file exists in your filesystem."
                
                if not code or not code.strip():
                    return f"Error: File {file_path} is empty or contains no code."
                
                # Capture stdout/stderr during evaluation if flag is enabled
                if optimizer_instance.capture_simulation_output:
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
                        score, sim_dirs, results = optimizer_instance.evaluate_code(code)
                    simulation_output = captured_output.getvalue()
                else:
                    score, sim_dirs, results = optimizer_instance.evaluate_code(code)
                    simulation_output = ""

                success, error_message = optimizer_instance.summarize_results(results)
                
                tool_call_id = runtime.tool_call_id
                optimizer_instance.simulation_results[tool_call_id] = {
                    "code": code,
                    "score": score,
                    "sim_dirs": sim_dirs,
                    "results": results,
                    "success": success,
                    "error_message": error_message,
                    "file_path": file_path,
                }
                
                # Prepare output prefix/suffix for return messages
                output_prefix = f"Simulation Output:\n{simulation_output}\n\n" if simulation_output else ""

                if success:
                    if sim_dirs and len(sim_dirs) > 0:
                        random_number = str(uuid4())
                        csv_file = None
                        for file in os.listdir(sim_dirs[0]):
                            if file.endswith(".csv"):
                                csv_file = os.path.join(sim_dirs[0], file)
                                break
                        base_name, ext = os.path.splitext(os.path.basename(csv_file))
                        simulation_file_name = f"/{base_name}_{random_number}{ext}"
                        file_result = add_file_to_agent_filesystem(
                            csv_file,
                            backend=optimizer_instance.backend,
                            agent_path=simulation_file_name,
                            tool_call_id=runtime.tool_call_id
                        )
                        return f"{output_prefix}{file_result}" if output_prefix else file_result
                    else:
                        return f"{output_prefix}Simulation completed successfully with score: {score:.6f}\n(No simulation logs available)"
                else:
                    output_suffix = f"\n\nSimulation Output:\n{simulation_output}" if simulation_output else ""
                    return f"Simulation failed: {error_message}{output_suffix}"
            except Exception as e:
                return f"Error running simulation: {str(e)}"

        return run_simulation

    def display_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Display the tool calls and their results."""
        if "messages" not in result:
            return []
        
        messages = result["messages"]
        tool_calls = []
        tool_results = {}
        
        for i, msg in enumerate(messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        'id': tool_call.get('id', 'unknown'),
                        'name': tool_call.get('name', 'unknown'),
                        'args': tool_call.get('args', {}),
                        'message_index': i
                    })
            
            if hasattr(msg, 'name') and hasattr(msg, 'tool_call_id'):
                tool_results[msg.tool_call_id] = {
                    'name': msg.name,
                    'content': str(msg.content) if hasattr(msg, 'content') else '',
                }
        
        print(f"\nTotal tool calls: {len(tool_calls)}\n")
        
        for idx, tool_call in enumerate(tool_calls, 1):
            tool_id = tool_call['id']
            tool_name = tool_call['name']
            
            print(f"{idx}. {tool_name} (ID: {tool_id})")
            
            if tool_call['args']:
                print("   Args:", tool_call['args'])
            
            if tool_id in tool_results:
                result_info = tool_results[tool_id]
                content_preview = result_info['content'][:200] + "..." if len(result_info['content']) > 200 else result_info['content']
                print(f"Result: {content_preview}")
            else:
                print("Result: (not yet received)")
            print()
        
        return tool_calls
