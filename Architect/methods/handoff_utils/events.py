"""Event system for decoupled message handling.

Provides event types and an event emitter for publishing/subscribing
to stream events, enabling clean separation between event generation
and event handling (display, tracking, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


class StreamEventType(Enum):
    """Types of events that can occur during agent streaming."""

    # Message events (when a message is received)
    AI_MESSAGE = auto()
    TOOL_MESSAGE = auto()
    HUMAN_MESSAGE = auto()

    # Content events (specific content within messages)
    REASONING_SUMMARY = auto()
    TOOL_CALL_INITIATED = auto()
    TOOL_CALL_COMPLETED = auto()
    TEXT_RESPONSE = auto()

    # Simulation events (from run_simulation tool)
    SIMULATION_COMPLETED = auto()
    NEW_BEST_SCORE = auto()

    # Lifecycle events
    STREAM_STARTED = auto()
    STREAM_COMPLETED = auto()
    STREAM_ERROR = auto()
    CHUNK_RECEIVED = auto()

    # Summary events
    SUMMARY_REQUESTED = auto()
    SUMMARY_VALIDATED = auto()
    SUMMARY_INVALID = auto()


@dataclass
class StreamEvent:
    """An event emitted during stream processing.

    Attributes:
        event_type: The type of event
        data: Event-specific data dictionary
        message: The original message object (if applicable)
        node_name: The graph node that generated this event (if applicable)
        timestamp: When the event occurred (auto-set if not provided)
    """

    event_type: StreamEventType
    data: Dict[str, Any] = field(default_factory=dict)
    message: Optional[Any] = None
    node_name: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the event data."""
        return self.data.get(key, default)


# Type alias for event handlers
EventHandler = Callable[[StreamEvent], None]


class EventEmitter:
    """Event emitter for publishing and subscribing to stream events.

    Example usage:
        emitter = EventEmitter()

        # Subscribe to events
        emitter.on(StreamEventType.AI_MESSAGE, lambda e: print(e.data))
        emitter.on(StreamEventType.TOOL_CALL_INITIATED, handle_tool_call)

        # Emit events
        emitter.emit(StreamEvent(
            event_type=StreamEventType.AI_MESSAGE,
            data={'text': 'Hello world'},
            message=original_message
        ))
    """

    def __init__(self):
        """Initialize the event emitter."""
        self._handlers: Dict[StreamEventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

    def on(self, event_type: StreamEventType, handler: EventHandler) -> "EventEmitter":
        """Subscribe to a specific event type.

        Args:
            event_type: The event type to subscribe to
            handler: Callback function that receives StreamEvent

        Returns:
            Self for method chaining
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        return self

    def on_all(self, handler: EventHandler) -> "EventEmitter":
        """Subscribe to all events.

        Args:
            handler: Callback function that receives StreamEvent

        Returns:
            Self for method chaining
        """
        self._global_handlers.append(handler)
        return self

    def off(self, event_type: StreamEventType, handler: EventHandler) -> "EventEmitter":
        """Unsubscribe from a specific event type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove

        Returns:
            Self for method chaining
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass  # Handler not found
        return self

    def off_all(self, handler: EventHandler) -> "EventEmitter":
        """Remove a global handler.

        Args:
            handler: The handler to remove

        Returns:
            Self for method chaining
        """
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            pass  # Handler not found
        return self

    def emit(self, event: StreamEvent) -> None:
        """Emit an event to all subscribed handlers.

        Args:
            event: The event to emit
        """
        # Call type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Don't let one handler's error stop others
                pass

        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
        self._global_handlers.clear()

    def handler_count(self, event_type: Optional[StreamEventType] = None) -> int:
        """Get the number of handlers registered.

        Args:
            event_type: If provided, count only handlers for this type.
                       If None, count all handlers including global.

        Returns:
            Number of handlers
        """
        if event_type is not None:
            return len(self._handlers.get(event_type, []))
        total = sum(len(handlers) for handlers in self._handlers.values())
        total += len(self._global_handlers)
        return total
