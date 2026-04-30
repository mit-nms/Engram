"""Message handling architecture for agent streaming.

This module provides a clean, extensible architecture for handling all message types
during agent streaming, including reasoning model summaries, tool calls/results,
AI responses, and summary messages for handoff.

Key components:
- extractors: Pure functions for content extraction from messages
- events: Event types and EventEmitter for decoupled event handling
- handlers: Message type handlers using Strategy pattern
- processor: StreamProcessor and ChunkParser for stream normalization
- callbacks: DebugDisplayCallback and ResultTrackingCallback
- integration: AgentStreamRunner main entry point
"""

from .extractors import (
    ExtractedContent,
    ReasoningContent,
    extract_text_content,
    extract_reasoning,
    extract_tool_calls,
    extract_full_content,
    clean_message_content,
)

from .events import (
    StreamEventType,
    StreamEvent,
    EventEmitter,
)

from .handlers import (
    HandlerContext,
    MessageHandler,
    AIMessageHandler,
    ToolMessageHandler,
    HumanMessageHandler,
    MessageDispatcher,
)

from .processor import (
    ChunkParseResult,
    ChunkParser,
    AccumulatedState,
    StreamProcessor,
)

from .callbacks import (
    DebugDisplayCallback,
    ResultTrackingCallback,
)

from .integration import (
    AgentRunResult,
    AgentStreamRunner,
)

__all__ = [
    # Extractors
    "ExtractedContent",
    "ReasoningContent",
    "extract_text_content",
    "extract_reasoning",
    "extract_tool_calls",
    "extract_full_content",
    "clean_message_content",
    # Events
    "StreamEventType",
    "StreamEvent",
    "EventEmitter",
    # Handlers
    "HandlerContext",
    "MessageHandler",
    "AIMessageHandler",
    "ToolMessageHandler",
    "HumanMessageHandler",
    "MessageDispatcher",
    # Processor
    "ChunkParseResult",
    "ChunkParser",
    "AccumulatedState",
    "StreamProcessor",
    # Callbacks
    "DebugDisplayCallback",
    "ResultTrackingCallback",
    # Integration
    "AgentRunResult",
    "AgentStreamRunner",
]
