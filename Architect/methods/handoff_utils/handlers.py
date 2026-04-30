"""Message type handlers using the Strategy pattern.

Each handler knows how to process a specific message type and emit
appropriate events. The MessageDispatcher routes messages to the
correct handler based on message type.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .events import EventEmitter, StreamEvent, StreamEventType
from .extractors import extract_full_content, extract_reasoning


@dataclass
class HandlerContext:
    """Context passed to message handlers.

    Provides access to shared state and configuration needed
    for message processing.

    Attributes:
        simulation_results: Dict mapping tool_call_id to simulation result data
        current_agent_number: The current agent number in handoff sequence
        debug: Whether debug mode is enabled
    """

    simulation_results: Dict[str, Any] = field(default_factory=dict)
    current_agent_number: int = 0
    debug: bool = False


class MessageHandler(ABC):
    """Base class for message type handlers."""

    @abstractmethod
    def can_handle(self, message: Any) -> bool:
        """Check if this handler can process the given message.

        Args:
            message: The message to check

        Returns:
            True if this handler can process the message
        """
        pass

    @abstractmethod
    def handle(
        self,
        message: Any,
        emitter: EventEmitter,
        context: HandlerContext,
        node_name: Optional[str] = None,
    ) -> None:
        """Process a message and emit appropriate events.

        Args:
            message: The message to process
            emitter: EventEmitter for emitting events
            context: HandlerContext with shared state
            node_name: The graph node that produced this message
        """
        pass


class AIMessageHandler(MessageHandler):
    """Handler for AI/assistant messages.

    Extracts and emits events for:
    - Reasoning summaries (v0 or v1 format)
    - Tool calls
    - Text responses
    """

    def can_handle(self, message: Any) -> bool:
        """Check if message is an AIMessage."""
        return isinstance(message, AIMessage)

    def handle(
        self,
        message: Any,
        emitter: EventEmitter,
        context: HandlerContext,
        node_name: Optional[str] = None,
    ) -> None:
        """Process an AIMessage and emit events.

        Emits:
        - AI_MESSAGE for the overall message
        - REASONING_SUMMARY for each reasoning summary (if present)
        - TOOL_CALL_INITIATED for each tool call (if present)
        - TEXT_RESPONSE for text content (if present)
        """
        # Extract all content
        extracted = extract_full_content(message)

        # Emit overall AI message event
        emitter.emit(StreamEvent(
            event_type=StreamEventType.AI_MESSAGE,
            data={
                "text": extracted.text,
                "has_reasoning": extracted.has_reasoning,
                "has_tool_calls": len(extracted.tool_calls) > 0,
            },
            message=message,
            node_name=node_name,
        ))

        # Emit reasoning summaries (if present)
        if extracted.reasoning and extracted.reasoning.has_reasoning:
            for summary in extracted.reasoning.summaries:
                emitter.emit(StreamEvent(
                    event_type=StreamEventType.REASONING_SUMMARY,
                    data={
                        "summary": summary,
                        "format_version": extracted.reasoning.format_version,
                        "reasoning_id": extracted.reasoning.reasoning_id,
                    },
                    message=message,
                    node_name=node_name,
                ))

        # Emit tool calls (if present)
        for tc in extracted.tool_calls:
            emitter.emit(StreamEvent(
                event_type=StreamEventType.TOOL_CALL_INITIATED,
                data={
                    "tool_call_id": tc["id"],
                    "tool_name": tc["name"],
                    "args": tc["args"],
                },
                message=message,
                node_name=node_name,
            ))

        # Emit text response (if present and non-empty)
        if extracted.text.strip():
            emitter.emit(StreamEvent(
                event_type=StreamEventType.TEXT_RESPONSE,
                data={"text": extracted.text},
                message=message,
                node_name=node_name,
            ))


class ToolMessageHandler(MessageHandler):
    """Handler for tool result messages.

    Emits TOOL_CALL_COMPLETED events and handles simulation result tracking.
    """

    def can_handle(self, message: Any) -> bool:
        """Check if message is a ToolMessage."""
        return isinstance(message, ToolMessage)

    def handle(
        self,
        message: Any,
        emitter: EventEmitter,
        context: HandlerContext,
        node_name: Optional[str] = None,
    ) -> None:
        """Process a ToolMessage and emit events.

        Emits:
        - TOOL_MESSAGE for the overall message
        - TOOL_CALL_COMPLETED with tool result
        - SIMULATION_COMPLETED if this was a run_simulation result
        """
        tool_call_id = getattr(message, "tool_call_id", None)
        tool_name = getattr(message, "name", "unknown")
        content = getattr(message, "content", "")

        # Emit overall tool message event
        emitter.emit(StreamEvent(
            event_type=StreamEventType.TOOL_MESSAGE,
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": content,
            },
            message=message,
            node_name=node_name,
        ))

        # Emit tool call completed event
        emitter.emit(StreamEvent(
            event_type=StreamEventType.TOOL_CALL_COMPLETED,
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": content,
            },
            message=message,
            node_name=node_name,
        ))

        # Check if this is a simulation result
        if tool_call_id and tool_call_id in context.simulation_results:
            sim_result = context.simulation_results[tool_call_id]
            emitter.emit(StreamEvent(
                event_type=StreamEventType.SIMULATION_COMPLETED,
                data={
                    "tool_call_id": tool_call_id,
                    "simulation_result": sim_result,
                },
                message=message,
                node_name=node_name,
            ))


class HumanMessageHandler(MessageHandler):
    """Handler for human/user messages."""

    def can_handle(self, message: Any) -> bool:
        """Check if message is a HumanMessage."""
        return isinstance(message, HumanMessage)

    def handle(
        self,
        message: Any,
        emitter: EventEmitter,
        context: HandlerContext,
        node_name: Optional[str] = None,
    ) -> None:
        """Process a HumanMessage and emit events.

        Emits:
        - HUMAN_MESSAGE for the message
        """
        content = getattr(message, "content", "")
        if isinstance(content, list):
            # Handle structured content
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            content = "\n".join(text_parts)

        emitter.emit(StreamEvent(
            event_type=StreamEventType.HUMAN_MESSAGE,
            data={"content": content},
            message=message,
            node_name=node_name,
        ))


class MessageDispatcher:
    """Routes messages to appropriate handlers.

    Example usage:
        dispatcher = MessageDispatcher()
        dispatcher.register(AIMessageHandler())
        dispatcher.register(ToolMessageHandler())
        dispatcher.register(HumanMessageHandler())

        # Dispatch a message
        dispatcher.dispatch(message, emitter, context)
    """

    def __init__(self):
        """Initialize the dispatcher with empty handler list."""
        self._handlers: List[MessageHandler] = []

    def register(self, handler: MessageHandler) -> "MessageDispatcher":
        """Register a message handler.

        Args:
            handler: The handler to register

        Returns:
            Self for method chaining
        """
        self._handlers.append(handler)
        return self

    def dispatch(
        self,
        message: Any,
        emitter: EventEmitter,
        context: HandlerContext,
        node_name: Optional[str] = None,
    ) -> bool:
        """Dispatch a message to the appropriate handler.

        Iterates through registered handlers and uses the first one
        that can handle the message.

        Args:
            message: The message to dispatch
            emitter: EventEmitter for emitting events
            context: HandlerContext with shared state
            node_name: The graph node that produced this message

        Returns:
            True if a handler processed the message, False otherwise
        """
        for handler in self._handlers:
            if handler.can_handle(message):
                handler.handle(message, emitter, context, node_name)
                return True
        return False

    @classmethod
    def create_default(cls) -> "MessageDispatcher":
        """Create a dispatcher with default handlers registered.

        Returns:
            MessageDispatcher with AI, Tool, and Human handlers
        """
        dispatcher = cls()
        dispatcher.register(AIMessageHandler())
        dispatcher.register(ToolMessageHandler())
        dispatcher.register(HumanMessageHandler())
        return dispatcher
