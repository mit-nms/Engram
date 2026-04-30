"""Integration layer for agent stream processing.

Provides AgentStreamRunner as the main entry point for running agent streams
with the new message handling architecture.
"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# --- Monkey-patch ShellSession to fix two bugs in langchain's shell middleware ---
#
# Bug 1 (CRITICAL): _kill_process() uses os.killpg() which kills the ENTIRE process
#   group. When Docker is started with start_new_session=False, the docker client
#   shares our process group. So killing the process group kills our Python process.
#   Fix: use self._process.kill() to kill only the docker client process.
#
# Bug 2: restart() calls stop() then start() with no overall timeout. If Docker is
#   slow/overloaded, start() -> subprocess.Popen("docker run ...") hangs forever.
#   Fix: wrap restart() in a thread with a 60s timeout.

_RESTART_TIMEOUT = 60


def _safe_kill_process(self):
    """Kill only the subprocess, not the entire process group."""
    if not self._process:
        return
    import contextlib
    with contextlib.suppress(ProcessLookupError, OSError):
        self._process.kill()


def _safe_shell_restart(self):
    """Restart shell session with timeout to prevent indefinite Docker hangs."""
    result = [None]

    def _do_restart():
        try:
            self.stop(self._policy.termination_timeout)
            self.start()
        except Exception as e:
            result[0] = e

    t = threading.Thread(target=_do_restart, daemon=True)
    t.start()
    t.join(timeout=_RESTART_TIMEOUT)

    if t.is_alive():
        # restart() is hung (likely Docker spawn). Mark session as dead
        # so the error propagates and the agent can recover.
        self._terminated = True
        self._process = None
        raise RuntimeError(
            f"Shell session restart timed out after {_RESTART_TIMEOUT}s "
            "(Docker may be overloaded). Session is now unavailable."
        )

    if result[0] is not None:
        raise result[0]


try:
    from langchain.agents.middleware.shell_tool import ShellSession
    ShellSession._kill_process = _safe_kill_process
    ShellSession.restart = _safe_shell_restart
except ImportError:
    pass  # langchain not installed or different version
# --- End monkey-patch ---

from ..deepagents_utils.tool_descriptions import CONTINUE_MESSAGE
from .callbacks import (
    DebugDisplayCallback,
    HandoffUsageCallbackHandler,
    ResultTrackingCallback,
    ToolCallStopCallback,
    TimeoutCallback,
    UsageTrackingCallback,
)
from .events import EventEmitter, StreamEvent, StreamEventType
from .extractors import extract_text_content
from .handlers import HandlerContext, MessageDispatcher
from .processor import AccumulatedState, ChunkParser, StreamProcessor


@dataclass
class AgentRunResult:
    """Result of running a single agent.

    Attributes:
        messages: All messages from the agent run
        files: All files from the agent run
        steps: Number of steps/chunks processed
        elapsed_minutes: Time spent in minutes
        timed_out: Whether the agent timed out
        summary: Extracted summary from agent (if valid)
        summary_valid: Whether the summary was valid
        early_stopped: Whether the agent stopped due to early stopping (no improvement)
    """

    messages: List[Any] = field(default_factory=list)
    files: Dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    elapsed_minutes: float = 0.0
    timed_out: bool = False
    summary: str = ""
    summary_valid: bool = False
    early_stopped: bool = False


class AgentStreamRunner:
    """Main entry point for running agent streams.

    Coordinates stream processing, event handling, and callbacks
    to provide a clean interface for running agents.

    Example usage:
        runner = AgentStreamRunner(
            simulation_results=optimizer.simulation_results,
            all_iterations=optimizer.all_iterations,
            debug=True,
            timeout_minutes=30.0,
        )

        result = runner.run_single_agent(
            agent=agent,
            initial_messages=[initial_user_message],
            initial_files=initial_files,
            validate_summary=validate_summary_func,
            summary_prompt="Please provide a summary...",
            current_agent_number=1,
        )
    """

    def __init__(
        self,
        simulation_results: Dict[str, Any],
        all_iterations: List[Dict[str, Any]],
        debug: bool = False,
        timeout_minutes: float = 30.0,
        on_new_best: Optional[Callable[[float, str], None]] = None,
        periodic_save_callback: Optional[Callable[[], None]] = None,
        console: Optional[Any] = None,
        stream_config: Optional[Dict[str, Any]] = None,
        best_score: float = float("-inf"),
        best_code: Optional[str] = None,
        model: str = "o3",
        cumulative_usage: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the runner.

        Args:
            simulation_results: Dict mapping tool_call_id to simulation result data
            all_iterations: List to append iteration data to
            debug: Whether debug output is enabled
            timeout_minutes: Timeout per agent in minutes
            on_new_best: Callback when new best score is found
            periodic_save_callback: Callback for periodic saves
            console: Rich Console for output (optional)
            stream_config: Config for agent.stream() including thread_id for checkpointer
            best_score: Current best score from previous agents
            best_code: Code for current best score
        """
        self.simulation_results = simulation_results
        self.all_iterations = all_iterations
        self.debug = debug
        self.timeout_minutes = timeout_minutes
        self.on_new_best = on_new_best
        self.periodic_save_callback = periodic_save_callback
        self.console = console
        self.stream_config = stream_config or {}

        # Best score tracking (updated by ResultTrackingCallback)
        self.best_score = best_score
        self.best_code = best_code

        # Usage tracking (accumulated across streamed AIMessages)
        self.model = model
        self.cumulative_usage = cumulative_usage if cumulative_usage is not None else {}

        # Early stopping tracking (per agent)
        self.iterations_since_improvement = 0

    def run_single_agent(
        self,
        agent: Any,
        initial_messages: List[Any],
        initial_files: Dict[str, Any],
        validate_summary: Callable[[str], bool],
        summary_prompt: str,
        current_agent_number: int,
        rebuild_agent_callback: Optional[Callable[[], bool]] = None,
        enable_continue_message: bool = False,
        early_stop_patience: int = 10,
        require_summary: bool = True,
    ) -> AgentRunResult:
        """Run a single agent until completion or timeout.

        Args:
            agent: The agent to run
            initial_messages: Initial messages to send to agent
            initial_files: Initial files for agent filesystem
            validate_summary: Function to validate summary content
            summary_prompt: Prompt to request summary if not provided
            current_agent_number: Current agent number for tracking
            rebuild_agent_callback: Optional callback to rebuild agent on error
            enable_continue_message: Whether to add continue messages automatically instead of waiting for natural completion
            early_stop_patience: Number of iterations without improvement before early stopping (only used when enable_continue_message=True)
            require_summary: If False, never request or validate summary; exit only on timeout, early stop, or stream error.

        Returns:
            AgentRunResult with all data from the agent run
        """
        print(f"\n{'='*60}")
        print(f"Starting Agent {current_agent_number}")
        print(f"Timeout: {self.timeout_minutes} minutes")
        print(f"{'='*60}\n")

        start_time = time.time()
        timeout_seconds = self.timeout_minutes * 60

        # Set up event emitter and callbacks
        emitter = EventEmitter()

        # Debug display callback
        debug_callback = DebugDisplayCallback(
            enabled=self.debug,
            console=self.console,
        )
        debug_callback.register(emitter)

        # Result tracking callback
        result_callback = ResultTrackingCallback(
            simulation_results=self.simulation_results,
            all_iterations=self.all_iterations,
            best_score=self.best_score,
            best_code=self.best_code,
            current_agent_number=current_agent_number,
            on_new_best=self._handle_new_best,
            periodic_save_callback=self.periodic_save_callback,
            debug=self.debug,
        )
        result_callback.register(emitter)

        # Tool call stop callback (for detecting tool calls after summary request)
        stop_callback = ToolCallStopCallback(
            summary_requested=False,
            debug=self.debug,
        )
        stop_callback.register(emitter)

        # Timeout callback
        timeout_callback = TimeoutCallback(
            start_time=start_time,
            timeout_seconds=timeout_seconds,
            debug=self.debug,
            current_agent_number=current_agent_number,
        )
        timeout_callback.register(emitter)

        # Usage tracking callback (accumulates token usage from AIMessages)
        usage_callback = UsageTrackingCallback(
            cumulative_usage=self.cumulative_usage,
            model=self.model,
        )
        usage_callback.register(emitter)

        # Set up handler context
        context = HandlerContext(
            simulation_results=self.simulation_results,
            current_agent_number=current_agent_number,
            debug=self.debug,
        )

        # Set up processor
        processor = StreamProcessor(emitter=emitter, debug=self.debug)

        # Initialize accumulated state
        accumulated_state = AccumulatedState(
            messages=list(initial_messages),
            files=initial_files.copy(),
        )

        step_count = 0
        timed_out = False
        early_stopped = False
        summary_requested = False
        stream_error_retries = 0
        max_stream_error_retries = 5

        # Reset iterations_since_improvement for this agent
        self.iterations_since_improvement = 0
        # Track best score at start of each stream iteration to detect improvements
        iteration_start_best_score = self.best_score

        try:
            # Main loop: first iteration is exploration, second (if needed) is summary request
            while True:
                # Always pass full state - duplicate detection in processor.py
                # filters out messages we've already seen
                stream_input = accumulated_state.to_dict()
                if self.debug:
                    print(f"\n[DEBUG] Stream call - passing {len(stream_input.get('messages', []))} messages")

                try:
                    # Pass LangChain callback to capture usage in on_llm_end
                    # (streamed AIMessages lack usage in response_metadata)
                    stream_config = dict(self.stream_config)
                    existing = stream_config.get("callbacks") or []
                    usage_handler = HandoffUsageCallbackHandler(
                        cumulative_usage=self.cumulative_usage,
                        model=self.model,
                    )
                    stream_config["callbacks"] = [usage_handler] + list(existing)

                    stream_iterator = agent.stream(
                        stream_input,
                        config=stream_config,
                        stream_mode="updates"
                    )
                except Exception as stream_init_error:
                    print(f"\nFailed to initialize stream: {stream_init_error}")
                    break

                stream_error_occurred = False
                try:
                    for chunk in stream_iterator:
                        step_count += 1

                        # Check timeout
                        elapsed = time.time() - start_time
                        if elapsed > timeout_seconds:
                            print(f"\nAgent {current_agent_number} timed out after {elapsed/60:.1f} minutes")
                            timed_out = True
                            break

                        # Process chunk using the processor
                        accumulated_state, new_messages = processor.process_single_chunk(
                            chunk, accumulated_state, context
                        )

                        # Check if any new AIMessage contains a valid summary (only when summary is required)
                        # This allows us to stop BEFORE tools execute (important because
                        # models sometimes include tool calls after their summary text)
                        found_summary = False
                        if require_summary:
                            for msg in new_messages:
                                if isinstance(msg, AIMessage):
                                    content = getattr(msg, "content", None)
                                    text = extract_text_content(content)
                                    if text.strip() and validate_summary(text.strip()):
                                        print(f"\nDetected valid summary in AIMessage, stopping before tool execution")
                                        found_summary = True
                                        break

                        if found_summary:
                            # Clean exit - we have a valid summary
                            break

                        # Check if we should stop (tool calls after summary request)
                        if stop_callback.should_stop:
                            timed_out = True
                            break

                        # Break if timed out
                        if timeout_callback.timed_out:
                            timed_out = True
                            break

                except Exception as stream_error:
                    # All stream errors are treated as recoverable up to the retry limit.
                    # Common causes: shell timeout, Docker restart failure, tool RuntimeError.
                    stream_error_retries += 1
                    error_type = type(stream_error).__name__
                    print(f"\nStream error ({stream_error_retries}/{max_stream_error_retries}): {error_type}: {stream_error}")
                    if self.debug:
                        traceback.print_exc()

                    if stream_error_retries >= max_stream_error_retries:
                        print(f"\nMax stream error retries reached. Stopping agent.")
                        break

                    # IMPORTANT: Check if we already have a valid summary before continuing (only when summary required)
                    # This handles the case where agent provided summary but then an error occurred
                    if require_summary:
                        summary = self._extract_summary(accumulated_state.messages)
                        if validate_summary(summary):
                            print("\nAgent already provided valid summary before error. Stopping agent.")
                            break

                    stream_error_occurred = True

                    # Patch dangling tool calls: if an AIMessage has tool_calls without
                    # corresponding ToolMessages, inject error ToolMessages so the LLM
                    # API doesn't reject the message history.
                    agent_facing_error = (
                        "A shell command or tool timed out. The session was restarted."
                    )
                    self._patch_dangling_tool_calls(accumulated_state.messages, agent_facing_error)

                    # Add error message to conversation
                    error_content = (
                        f"[System] {agent_facing_error} "
                        "You can continue working. Consider breaking long operations into "
                        "smaller steps or using the run_simulation tool for evaluations."
                    )
                    error_msg = HumanMessage(content=error_content)
                    accumulated_state.messages.append(error_msg)

                    # Rebuild agent if callback provided
                    if rebuild_agent_callback:
                        try:
                            if not rebuild_agent_callback():
                                print(f"\nFailed to rebuild agent")
                                break
                        except Exception as rebuild_error:
                            print(f"\nFailed to rebuild agent: {rebuild_error}")
                            break

                    continue  # Retry with rebuilt agent

                # Stream completed successfully - reset retry counter
                stream_error_retries = 0

                if self.debug:
                    print(f"[DEBUG] Accumulated state AFTER stream: {len(accumulated_state.messages)} messages")

                # Break outer loop if timed out
                if timed_out:
                    break

                # If stream ended due to error and we're retrying, don't process summary yet
                if stream_error_occurred:
                    continue

                # When summary is not required, exit after stream unless we're in continue-message mode
                if not require_summary:
                    if enable_continue_message:
                        # Fall through to the existing early-stop and continue-message logic below
                        pass
                    else:
                        # No summary needed and no continue message: we're done after one stream
                        break

                # Handle continue message mode
                if enable_continue_message:
                    # Check for early stopping due to no improvement
                    if self.iterations_since_improvement >= early_stop_patience:
                        if self.debug:
                            print(f"\n🛑 Early stopping: No improvement for {self.iterations_since_improvement} iterations (patience: {early_stop_patience})")
                            print(f"   Best score: {self.best_score:.4f}")
                        early_stopped = True
                        break

                    # Check if we had improvement in this stream iteration
                    if self.best_score > iteration_start_best_score:
                        # New best found - reset counter
                        self.iterations_since_improvement = 0
                        if self.debug:
                            print(f"\n🎯 New best score: {self.best_score:.4f}, resetting early stop counter")
                    else:
                        # No improvement - increment counter
                        self.iterations_since_improvement += 1
                        if self.debug:
                            print(f"\n⏳ No improvement for {self.iterations_since_improvement} iterations (patience: {early_stop_patience})")

                    # Update iteration_start_best_score for next iteration
                    iteration_start_best_score = self.best_score

                    # Automatically add continue message and proceed to next iteration
                    accumulated_state.messages.append(HumanMessage(content=CONTINUE_MESSAGE))
                    if self.debug:
                        print("\n[DEBUG] Added continue message, continuing loop...")
                    # Loop back to stream again with continue message
                    continue

                # When require_summary is False we already broke above (or we're in continue mode)
                if not require_summary:
                    break

                # Check if agent provided a valid summary
                summary = self._extract_summary(accumulated_state.messages)
                if validate_summary(summary):
                    # Valid summary provided, we're done
                    break

                # If we already requested a summary and it's still not valid, stop trying
                if summary_requested:
                    print("\nSummary request did not produce a valid summary. Stopping agent.")
                    break

                # No valid summary - request one
                print("\nAgent did not provide a valid summary. Requesting summary...")

                accumulated_state.messages.append(HumanMessage(content=summary_prompt))
                summary_requested = True
                stop_callback.summary_requested = True
                # Loop back to stream again (next iteration will pass only the summary request)

        except Exception as e:
            error_type = type(e).__name__
            print(f"\nAgent {current_agent_number} error: {error_type}: {e}")
            traceback.print_exc()

        elapsed = time.time() - start_time
        print(f"\nAgent {current_agent_number} completed in {elapsed/60:.1f} minutes")
        print(f"Total steps: {step_count}")

        # Update best score from callback
        self.best_score = result_callback.best_score
        self.best_code = result_callback.best_code

        # Extract and validate final summary
        final_summary = self._extract_summary(accumulated_state.messages)
        summary_valid = validate_summary(final_summary)

        return AgentRunResult(
            messages=accumulated_state.messages,
            files=accumulated_state.files,
            steps=step_count,
            elapsed_minutes=elapsed / 60,
            timed_out=timed_out,
            summary=final_summary,
            summary_valid=summary_valid,
            early_stopped=early_stopped,
        )

    def _extract_summary(self, messages: List[Any]) -> str:
        """Extract the last AI message as the agent's summary.

        Args:
            messages: List of messages

        Returns:
            Summary text (may be empty)
        """
        from langchain_core.messages import AIMessage

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = getattr(msg, "content", None)
                text = extract_text_content(content)
                if text.strip():
                    return text.strip()

        return ""

    @staticmethod
    def _patch_dangling_tool_calls(messages: List[Any], error_msg: str) -> None:
        """Patch dangling tool calls in the message history.

        When an exception occurs mid-tool-execution (e.g., first of 4 parallel tool calls
        succeeds, second throws RuntimeError), the remaining tool calls never produce
        ToolMessages. The LLM API rejects message histories with AIMessage.tool_calls
        that lack corresponding ToolMessages.

        This method scans for such dangling tool calls and injects error ToolMessages.
        """
        # Collect all tool_call_ids that already have responses
        answered_ids = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    answered_ids.add(tool_call_id)

        # Find AIMessages with unanswered tool_calls and inject error ToolMessages
        patched = False
        insert_positions = []  # (index, tool_messages_to_insert)
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    missing = []
                    for tc in tool_calls:
                        tc_id = tc.get("id")
                        if tc_id and tc_id not in answered_ids:
                            missing.append(ToolMessage(
                                content=f"[System] Tool call failed due to error: {error_msg}",
                                name=tc.get("name", "unknown"),
                                tool_call_id=tc_id,
                                status="error",
                            ))
                            answered_ids.add(tc_id)
                    if missing:
                        insert_positions.append((i, missing))
                        patched = True

        # Insert in reverse order to preserve indices
        for idx, tool_msgs in reversed(insert_positions):
            for j, tm in enumerate(tool_msgs):
                messages.insert(idx + 1 + j, tm)

        if patched:
            print(f"[DEBUG] Patched dangling tool calls in message history")

    def _handle_new_best(self, score: float, code: str) -> None:
        """Handle new best score event.

        Updates internal tracking and calls user callback if provided.
        Also resets iterations_since_improvement counter.

        Args:
            score: New best score
            code: Code that achieved the score
        """
        self.best_score = score
        self.best_code = code
        # Reset early stopping counter when new best is found
        # (The counter will be checked/updated in the main loop)
        if self.on_new_best:
            self.on_new_best(score, code)
