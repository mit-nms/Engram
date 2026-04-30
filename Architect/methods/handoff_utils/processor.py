"""Stream processing and chunk parsing.

Provides:
- ChunkParser: Normalizes various chunk formats from agent.stream()
- AccumulatedState: Tracks messages and files during streaming
- StreamProcessor: Main class for processing agent streams
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .events import EventEmitter, StreamEvent, StreamEventType
from .handlers import HandlerContext, MessageDispatcher


@dataclass
class ChunkParseResult:
    """Result of parsing a stream chunk.

    Attributes:
        node_name: The graph node that produced this chunk
        state_update: The state update dict (messages, files, etc.)
        is_valid: Whether the chunk was successfully parsed
    """

    node_name: Optional[str] = None
    state_update: Optional[Dict[str, Any]] = None
    is_valid: bool = False


class ChunkParser:
    """Parser for normalizing stream chunk formats.

    Handles various chunk formats from LangGraph agent.stream():
    - 2-tuple: (node_name, state_update)
    - 3-tuple: (namespace, stream_mode, data)
    - dict: {node_name: state_update}
    - Overwrite wrapper objects
    """

    def parse(self, chunk: Any) -> ChunkParseResult:
        """Parse a stream chunk into normalized format.

        Args:
            chunk: The raw chunk from agent.stream()

        Returns:
            ChunkParseResult with node_name and state_update
        """
        node_name = None
        state_update = None

        if isinstance(chunk, tuple):
            result = self._parse_tuple(chunk)
            if result:
                node_name, state_update = result
        elif isinstance(chunk, dict):
            result = self._parse_dict(chunk)
            if result:
                node_name, state_update = result
        else:
            # Unknown chunk type
            return ChunkParseResult(is_valid=False)

        if node_name is None or state_update is None:
            return ChunkParseResult(is_valid=False)

        # Unwrap Overwrite wrappers
        state_update = self._unwrap_overwrite(state_update)

        return ChunkParseResult(
            node_name=node_name,
            state_update=state_update,
            is_valid=True,
        )

    def _parse_tuple(self, chunk: tuple) -> Optional[Tuple[str, Dict]]:
        """Parse tuple-format chunks.

        Handles:
        - 2-tuple: (node_name, state_update)
        - 3-tuple: (namespace, stream_mode, data)
        """
        if len(chunk) == 2:
            node_name, state_update = chunk
            return node_name, state_update
        elif len(chunk) == 3:
            namespace, stream_mode, data = chunk
            if stream_mode == "updates":
                return namespace, data
            else:
                return None
        return None

    def _parse_dict(self, chunk: dict) -> Optional[Tuple[str, Dict]]:
        """Parse dict-format chunks.

        Handles:
        - Single-key dict: {node_name: state_update}
        - Multi-key dict: finds key with messages/files
        """
        if len(chunk) == 1:
            node_name = list(chunk.keys())[0]
            state_update = chunk[node_name]
            return node_name, state_update
        else:
            # Multi-key dict - find the one with state data
            for key, value in chunk.items():
                if isinstance(value, dict) and ("messages" in value or "files" in value):
                    return key, value
            # Fallback to first key
            if chunk:
                node_name = list(chunk.keys())[0]
                state_update = chunk[node_name]
                return node_name, state_update
        return None

    def _unwrap_overwrite(self, state_update: Dict) -> Dict:
        """Unwrap LangGraph Overwrite wrapper objects.

        Args:
            state_update: Dict that may contain Overwrite wrappers

        Returns:
            Dict with Overwrite wrappers replaced by their values
        """
        if not isinstance(state_update, dict):
            return state_update

        result = {}
        for key, value in state_update.items():
            if hasattr(value, "value"):
                # Unwrap Overwrite object
                result[key] = value.value
            else:
                result[key] = value
        return result


@dataclass
class AccumulatedState:
    """Accumulated state during stream processing.

    Tracks messages and files as they're received from the agent stream.

    Attributes:
        messages: List of all messages received
        files: Dict of file path -> file data
        _known_message_ids: Set of message IDs we've already seen (for deduplication)
    """

    messages: List[Any] = field(default_factory=list)
    files: Dict[str, Any] = field(default_factory=dict)
    _known_message_ids: set = field(default_factory=set)

    def add_messages(self, new_messages: Any, debug: bool = False) -> List[Any]:
        """Add messages to the accumulated state.

        Handles both single messages and lists of messages.
        Filters out duplicate messages based on message ID to handle
        LangGraph echoing back restored checkpoint state.

        Args:
            new_messages: Message or list of messages to add
            debug: Whether to print debug info

        Returns:
            List of messages that were actually added (excluding duplicates)
        """
        if new_messages is None:
            return []

        # Unwrap Overwrite if needed
        if hasattr(new_messages, "value"):
            new_messages = new_messages.value

        # Normalize to list
        if isinstance(new_messages, list):
            messages_to_add = new_messages
        else:
            messages_to_add = [new_messages]

        # Debug: show what messages are being received from stream
        if debug and messages_to_add:
            print(f"\n[DEBUG] Stream returned {len(messages_to_add)} message(s):")
            for i, msg in enumerate(messages_to_add):
                msg_type = type(msg).__name__
                content = getattr(msg, 'content', '')
                if isinstance(content, list):
                    # Handle structured content (reasoning models)
                    text_parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
                    content_preview = ' '.join(text_parts)[:80]
                elif isinstance(content, str):
                    content_preview = content[:80]
                else:
                    content_preview = str(content)[:80]
                print(f"[DEBUG]   [{i}] {msg_type}: {content_preview}...")

        # Filter out duplicates using message ID
        actually_added = []
        duplicates_skipped = 0
        for msg in messages_to_add:
            msg_id = self._get_message_id(msg)
            if msg_id and msg_id in self._known_message_ids:
                duplicates_skipped += 1
                continue
            if msg_id:
                self._known_message_ids.add(msg_id)
            self.messages.append(msg)
            actually_added.append(msg)

        if debug and duplicates_skipped > 0:
            print(f"[DEBUG] Skipped {duplicates_skipped} duplicate message(s), added {len(actually_added)} new message(s)")

        return actually_added

    def _get_message_id(self, msg: Any) -> Optional[str]:
        """Get unique ID for a message.

        Uses LangChain message ID if available, falls back to content hash.
        """
        # Try to get message ID (LangChain messages have this)
        msg_id = getattr(msg, 'id', None)
        if msg_id:
            return str(msg_id)

        # Fallback: hash content + type for deduplication
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', '')
        if isinstance(content, list):
            # Handle structured content
            content = str(content)
        elif not isinstance(content, str):
            content = str(content)

        # Create a hash of type + content
        import hashlib
        content_hash = hashlib.md5(f"{msg_type}:{content}".encode()).hexdigest()[:16]
        return f"hash_{content_hash}"

    def update_files(self, new_files: Any) -> None:
        """Update files in the accumulated state.

        Args:
            new_files: Dict of file path -> file data
        """
        if new_files is None:
            return

        # Unwrap Overwrite if needed
        if hasattr(new_files, "value"):
            new_files = new_files.value

        if isinstance(new_files, dict):
            self.files.update(new_files)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for passing to agent.stream().

        Returns:
            Dict with 'messages' and 'files' keys
        """
        return {
            "messages": self.messages.copy(),
            "files": self.files.copy(),
        }

    def copy(self) -> "AccumulatedState":
        """Create a copy of this state.

        Returns:
            New AccumulatedState with copied messages and files
        """
        return AccumulatedState(
            messages=self.messages.copy(),
            files=self.files.copy(),
        )


class StreamProcessor:
    """Processes agent streams and emits events.

    Coordinates chunk parsing, state accumulation, and message dispatching.

    Example usage:
        processor = StreamProcessor(emitter=emitter)
        result = processor.process_stream(
            stream=agent.stream(input_messages, stream_mode="updates"),
            initial_state=AccumulatedState(messages=initial_messages),
            context=HandlerContext(simulation_results=sim_results),
        )
    """

    def __init__(
        self,
        emitter: EventEmitter,
        dispatcher: Optional[MessageDispatcher] = None,
        debug: bool = False,
    ):
        """Initialize the stream processor.

        Args:
            emitter: EventEmitter for emitting events
            dispatcher: MessageDispatcher for routing messages (default: create default)
            debug: Whether to print debug info about messages
        """
        self.emitter = emitter
        self.parser = ChunkParser()
        self.dispatcher = dispatcher or MessageDispatcher.create_default()
        self.debug = debug

    def process_stream(
        self,
        stream: Iterator,
        initial_state: AccumulatedState,
        context: HandlerContext,
    ) -> AccumulatedState:
        """Process an agent stream.

        Iterates through stream chunks, accumulates state, and dispatches
        messages to handlers.

        Args:
            stream: Iterator from agent.stream()
            initial_state: Initial accumulated state
            context: HandlerContext for message handlers

        Returns:
            Final accumulated state after processing stream
        """
        state = initial_state.copy()

        # Emit stream started event
        self.emitter.emit(StreamEvent(
            event_type=StreamEventType.STREAM_STARTED,
            data={"initial_message_count": len(state.messages)},
        ))

        try:
            for chunk in stream:
                # Emit chunk received event
                self.emitter.emit(StreamEvent(
                    event_type=StreamEventType.CHUNK_RECEIVED,
                    data={"chunk": chunk},
                ))

                # Parse chunk
                result = self.parser.parse(chunk)
                if not result.is_valid:
                    continue

                node_name = result.node_name
                state_update = result.state_update

                # Process state update
                if isinstance(state_update, dict):
                    # Handle messages
                    if "messages" in state_update:
                        added_messages = state.add_messages(state_update["messages"])

                        # Dispatch each message to handlers
                        for msg in added_messages:
                            self.dispatcher.dispatch(msg, self.emitter, context, node_name)

                    # Handle files
                    if "files" in state_update:
                        state.update_files(state_update["files"])

            # Emit stream completed event
            self.emitter.emit(StreamEvent(
                event_type=StreamEventType.STREAM_COMPLETED,
                data={"final_message_count": len(state.messages)},
            ))

        except Exception as e:
            # Emit stream error event
            self.emitter.emit(StreamEvent(
                event_type=StreamEventType.STREAM_ERROR,
                data={"error": str(e), "exception": e},
            ))
            raise

        return state

    def process_single_chunk(
        self,
        chunk: Any,
        state: AccumulatedState,
        context: HandlerContext,
    ) -> Tuple[AccumulatedState, List[Any]]:
        """Process a single stream chunk.

        Useful for manual iteration where you need to check conditions
        between chunks.

        Args:
            chunk: The chunk to process
            state: Current accumulated state (modified in place)
            context: HandlerContext for message handlers

        Returns:
            Tuple of (updated state, list of new messages)
        """
        result = self.parser.parse(chunk)
        if not result.is_valid:
            return state, []

        node_name = result.node_name
        state_update = result.state_update
        new_messages = []

        if isinstance(state_update, dict):
            # Handle messages
            if "messages" in state_update:
                new_messages = state.add_messages(state_update["messages"], debug=self.debug)

                # Dispatch each message to handlers
                for msg in new_messages:
                    self.dispatcher.dispatch(msg, self.emitter, context, node_name)

            # Handle files
            if "files" in state_update:
                state.update_files(state_update["files"])

        return state, new_messages
