"""Callbacks for handling stream events.

Provides:
- DebugDisplayCallback: Prints debug output with Rich formatting
- ResultTrackingCallback: Tracks simulation results to all_iterations list
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from .events import EventEmitter, StreamEvent, StreamEventType

# LangChain callback for on_llm_end - usage is available there, not on streamed AIMessages
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import ChatGeneration, LLMResult
    _HAS_LANGCHAIN_CALLBACK = True
except ImportError:
    BaseCallbackHandler = object  # type: ignore
    ChatGeneration = None  # type: ignore
    LLMResult = None  # type: ignore
    _HAS_LANGCHAIN_CALLBACK = False


class HandoffUsageCallbackHandler(BaseCallbackHandler if _HAS_LANGCHAIN_CALLBACK else object):
    """LangChain callback that captures token usage in on_llm_end.

    The streamed AIMessage response_metadata does NOT contain usage (as seen in logs).
    Usage is available in on_llm_end via the LLMResult/ChatGeneration message.usage_metadata.
    """

    def __init__(self, cumulative_usage: Dict[str, Any], model: str = "o3"):
        if _HAS_LANGCHAIN_CALLBACK:
            super().__init__()
        self.cumulative_usage = cumulative_usage
        self.model = model
        self._pricing = get_pricing(model)

    def on_llm_end(self, response: "LLMResult", **kwargs: Any) -> None:
        """Capture usage from LLMResult when LLM finishes."""
        if not _HAS_LANGCHAIN_CALLBACK or not response.generations:
            return
        try:
            gen = response.generations[0][0]
        except (IndexError, KeyError):
            return
        if ChatGeneration is None or not isinstance(gen, ChatGeneration):
            return
        msg = getattr(gen, "message", None)
        if msg is None:
            return
        usage = getattr(msg, "usage_metadata", None)
        if usage is None:
            return
        # usage_metadata can be a dict or UsageMetadata object
        if hasattr(usage, "input_tokens"):
            prompt_tokens = getattr(usage, "input_tokens", 0) or 0
            completion_tokens = getattr(usage, "output_tokens", 0) or 0
        elif isinstance(usage, dict):
            prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0) or 0
        else:
            return
        if (prompt_tokens or completion_tokens) == 0:
            return
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        prompt_cost = (prompt_tokens / 1_000_000) * self._pricing["input"]
        completion_cost = (completion_tokens / 1_000_000) * self._pricing["output"]
        cost = prompt_cost + completion_cost

        self.cumulative_usage["total_prompt_tokens"] = (
            self.cumulative_usage.get("total_prompt_tokens", 0) + prompt_tokens
        )
        self.cumulative_usage["total_completion_tokens"] = (
            self.cumulative_usage.get("total_completion_tokens", 0) + completion_tokens
        )
        self.cumulative_usage["total_tokens"] = (
            self.cumulative_usage.get("total_tokens", 0) + total_tokens
        )
        self.cumulative_usage["total_cost"] = round(
            self.cumulative_usage.get("total_cost", 0.0) + cost, 6
        )
        self.cumulative_usage["model"] = self.model


@dataclass
class DebugDisplayCallback:
    """Callback for displaying debug output during streaming.

    Registers with EventEmitter to receive events and prints
    formatted debug information using Rich console.

    Attributes:
        enabled: Whether debug output is enabled
        console: Rich Console for output
    """

    enabled: bool = True
    console: Optional[Console] = None

    def __post_init__(self):
        """Initialize console if not provided."""
        if self.console is None:
            import sys
            self.console = Console(file=sys.stdout, force_terminal=True)

    def register(self, emitter: EventEmitter) -> None:
        """Register event handlers with the emitter.

        Args:
            emitter: EventEmitter to register with
        """
        if not self.enabled:
            return

        emitter.on(StreamEventType.REASONING_SUMMARY, self._on_reasoning_summary)
        emitter.on(StreamEventType.TOOL_CALL_INITIATED, self._on_tool_call)
        emitter.on(StreamEventType.TOOL_CALL_COMPLETED, self._on_tool_result)
        emitter.on(StreamEventType.TEXT_RESPONSE, self._on_text_response)

    def _on_reasoning_summary(self, event: StreamEvent) -> None:
        """Handle reasoning summary events.

        Renders reasoning summaries using Rich Markdown.
        """
        summary = event.get("summary", "")
        if summary:
            self.console.print(Markdown(summary))

    def _on_tool_call(self, event: StreamEvent) -> None:
        """Handle tool call initiated events.

        Prints tool name and key arguments.
        """
        tool_name = event.get("tool_name", "unknown")
        args = event.get("args", {})

        print(f"\n Tool Calls:")
        print(f"   - {tool_name}")

        # Print key arguments
        for key, value in args.items():
            if key in ["file_path", "command", "pattern", "path"]:
                print(f"     {key}: {value}")

    def _on_tool_result(self, event: StreamEvent) -> None:
        """Handle tool call completed events.

        Prints tool name and result content.
        """
        tool_name = event.get("tool_name", "unknown")
        result = event.get("result", "")

        print(f"\n Tool Result: {tool_name}")
        if result:
            print(f"   {result}")

    def _on_text_response(self, event: StreamEvent) -> None:
        """Handle text response events.

        Prints AI text responses with prefix.
        """
        text = event.get("text", "")
        if text.strip():
            print(f"\n AI: {text}")


@dataclass
class ResultTrackingCallback:
    """Callback for tracking simulation results.

    Monitors SIMULATION_COMPLETED events and populates the all_iterations
    list with simulation data. Also tracks best scores.

    Attributes:
        simulation_results: Dict mapping tool_call_id to simulation result data
        all_iterations: List to append iteration data to
        best_score: Current best score (updated on new best)
        best_code: Code for best score (updated on new best)
        current_agent_number: Current agent number for tracking
        on_new_best: Optional callback when new best score is found
        periodic_save_callback: Optional callback for periodic saves
        save_interval: Number of iterations between periodic saves
    """

    simulation_results: Dict[str, Any] = field(default_factory=dict)
    all_iterations: List[Dict[str, Any]] = field(default_factory=list)
    best_score: float = float("-inf")
    best_code: Optional[str] = None
    current_agent_number: int = 0
    on_new_best: Optional[Callable[[float, str], None]] = None
    periodic_save_callback: Optional[Callable[[], None]] = None
    save_interval: int = 5
    debug: bool = False

    def register(self, emitter: EventEmitter) -> None:
        """Register event handlers with the emitter.

        Args:
            emitter: EventEmitter to register with
        """
        emitter.on(StreamEventType.TOOL_MESSAGE, self._on_tool_message)

    def _on_tool_message(self, event: StreamEvent) -> None:
        """Handle tool message events.

        Extracts simulation results and updates tracking.
        """
        message = event.message
        if message is None:
            return

        tool_call_id = getattr(message, "tool_call_id", None)
        if not tool_call_id:
            return

        # Check if this tool_call_id has a pending simulation result
        sim_result = self.simulation_results.pop(tool_call_id, None)
        if sim_result is None:
            return

        # Build iteration data
        iteration_data = {
            "simulation_number": len(self.all_iterations) + 1,
            "agent_number": self.current_agent_number,
            "experiment_id": sim_result.get("experiment_id", ""),
            "code": sim_result["code"],
            "score": sim_result["score"],
            "success": sim_result["success"],
            "sim_dirs": [str(d) for d in sim_result.get("sim_dirs", [])],
            "error": sim_result.get("error_message", ""),
            "file_path": sim_result.get("file_path", ""),
        }
        self.all_iterations.append(iteration_data)

        # Check for new best score
        if sim_result["success"] and sim_result["score"] > self.best_score:
            self.best_score = sim_result["score"]
            self.best_code = sim_result["code"]
            if self.debug:
                exp_id = sim_result.get("experiment_id", "")
                print(f"New best score: {self.best_score:.4f} (experiment {exp_id})")
            if self.on_new_best:
                self.on_new_best(self.best_score, self.best_code)

        # Periodic save
        if self.periodic_save_callback and len(self.all_iterations) % self.save_interval == 0:
            self.periodic_save_callback()


from Architect.pricing_table import get_pricing


def _extract_usage_from_aimessage(message: Any) -> Optional[Dict[str, Any]]:
    """Extract token usage from AIMessage response_metadata.

    LangChain/OpenAI may use: usage, input_tokens, output_tokens, token_usage, etc.
    """
    if message is None:
        return None
    meta = getattr(message, "response_metadata", None) or getattr(message, "usage_metadata", None)
    if not meta or not isinstance(meta, dict):
        return None
    # Try various formats
    usage = meta.get("usage") or meta.get("token_usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)
    else:
        prompt_tokens = meta.get("input_tokens") or meta.get("prompt_tokens", 0)
        completion_tokens = meta.get("output_tokens") or meta.get("completion_tokens", 0)
    if (prompt_tokens or completion_tokens) == 0:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
    }


@dataclass
class UsageTrackingCallback:
    """Callback for accumulating token usage from AIMessages.

    Monitors AI_MESSAGE events and extracts usage from response_metadata,
    accumulating cost and token counts for the handoff run.

    Attributes:
        cumulative_usage: Mutable dict to accumulate usage (modified in place)
        model: Model name for pricing lookup
    """

    cumulative_usage: Dict[str, Any] = field(default_factory=dict)
    model: str = "o3"

    def register(self, emitter: EventEmitter) -> None:
        emitter.on(StreamEventType.AI_MESSAGE, self._on_ai_message)

    def _on_ai_message(self, event: StreamEvent) -> None:
        message = event.message
        usage = _extract_usage_from_aimessage(message)
        if usage is None:
            return
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]

        pricing = get_pricing(self.model)
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["output"]
        cost = prompt_cost + completion_cost

        self.cumulative_usage["total_prompt_tokens"] = (
            self.cumulative_usage.get("total_prompt_tokens", 0) + prompt_tokens
        )
        self.cumulative_usage["total_completion_tokens"] = (
            self.cumulative_usage.get("total_completion_tokens", 0) + completion_tokens
        )
        self.cumulative_usage["total_tokens"] = (
            self.cumulative_usage.get("total_tokens", 0) + total_tokens
        )
        self.cumulative_usage["total_cost"] = (
            round(self.cumulative_usage.get("total_cost", 0.0) + cost, 6)
        )
        self.cumulative_usage["model"] = self.model


@dataclass
class ToolCallStopCallback:
    """Callback for stopping stream when tool calls are made after summary request.

    Used to detect when an agent makes tool calls instead of providing a summary.

    Attributes:
        summary_requested: Whether a summary has been requested
        should_stop: Set to True when tool calls detected after summary request
    """

    summary_requested: bool = False
    should_stop: bool = False
    debug: bool = False

    def register(self, emitter: EventEmitter) -> None:
        """Register event handlers with the emitter.

        Args:
            emitter: EventEmitter to register with
        """
        emitter.on(StreamEventType.TOOL_CALL_INITIATED, self._on_tool_call)

    def _on_tool_call(self, event: StreamEvent) -> None:
        """Handle tool call initiated events.

        If summary was requested and we see tool calls, signal to stop.
        """
        if self.summary_requested:
            self.should_stop = True
            if self.debug:
                print("\nAgent made tool calls instead of providing summary. Stopping.")

    def reset(self) -> None:
        """Reset the callback state for a new stream."""
        self.should_stop = False


@dataclass
class TimeoutCallback:
    """Callback for tracking timeout during streaming.

    Attributes:
        start_time: Time when streaming started
        timeout_seconds: Timeout in seconds
        timed_out: Whether timeout has occurred
    """

    start_time: float = 0.0
    timeout_seconds: float = 1800.0  # 30 minutes default
    timed_out: bool = False
    debug: bool = False
    current_agent_number: int = 0

    def register(self, emitter: EventEmitter) -> None:
        """Register event handlers with the emitter.

        Args:
            emitter: EventEmitter to register with
        """
        emitter.on(StreamEventType.CHUNK_RECEIVED, self._on_chunk)

    def _on_chunk(self, event: StreamEvent) -> None:
        """Check timeout on each chunk received."""
        import time
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            self.timed_out = True
            if self.debug:
                print(f"\nAgent {self.current_agent_number} timed out after {elapsed/60:.1f} minutes")

    def reset(self, start_time: float) -> None:
        """Reset timeout state for a new stream.

        Args:
            start_time: New start time
        """
        self.start_time = start_time
        self.timed_out = False
