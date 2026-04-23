"""Pure functions for extracting content from messages.

Handles all content formats including:
- String content
- List content (mixed text/reasoning blocks)
- v0 reasoning format (in additional_kwargs)
- v1 reasoning format (inline in content)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningContent:
    """Extracted reasoning content from a message."""

    summaries: List[str] = field(default_factory=list)
    format_version: str = "unknown"  # "v0" (additional_kwargs) or "v1" (inline)
    reasoning_id: Optional[str] = None

    @property
    def has_reasoning(self) -> bool:
        return len(self.summaries) > 0


@dataclass
class ExtractedContent:
    """Fully extracted content from a message."""

    text: str = ""
    reasoning: Optional[ReasoningContent] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.reasoning and not self.tool_calls

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None and self.reasoning.has_reasoning


def extract_text_content(content: Any) -> str:
    """Extract text content from various content formats.

    Handles:
    - String content: returns as-is
    - List content: extracts 'text' type items, skips 'reasoning' items
    - Dict content: extracts 'text' field if present

    Args:
        content: The content to extract text from

    Returns:
        Extracted text as a string
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                # Skip reasoning blocks - they're handled separately
                if item_type == "reasoning":
                    continue
                # Extract text from 'text' type items
                if item_type == "text" or "text" in item:
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)
        return "\n".join(text_parts)

    if isinstance(content, dict):
        return content.get("text", "")

    return str(content)


def extract_reasoning(message: Any) -> Optional[ReasoningContent]:
    """Extract reasoning content from a message.

    Handles two formats from OpenAI reasoning models (o1, o3):

    Format v0 (Legacy) - reasoning in additional_kwargs:
        AIMessage(
            content=[{"type": "text", "text": "Hello"}],
            additional_kwargs={
                "reasoning": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "..."}]
                }
            }
        )

    Format v1 (Responses API) - reasoning inline in content:
        AIMessage(
            content=[
                {"type": "reasoning", "id": "rs_123", "summary": [{"type": "summary_text", "text": "..."}]},
                {"type": "text", "text": "Hello", "id": "msg_123"}
            ]
        )

    Args:
        message: The message to extract reasoning from

    Returns:
        ReasoningContent if reasoning found, None otherwise
    """
    if message is None:
        return None

    # Try v0 format first (additional_kwargs)
    additional_kwargs = getattr(message, "additional_kwargs", {})
    if additional_kwargs and isinstance(additional_kwargs, dict):
        reasoning_data = additional_kwargs.get("reasoning")
        if reasoning_data and isinstance(reasoning_data, dict):
            summary_list = reasoning_data.get("summary", [])
            summaries = _extract_summary_texts(summary_list)
            if summaries:
                return ReasoningContent(
                    summaries=summaries,
                    format_version="v0",
                    reasoning_id=reasoning_data.get("id"),
                )

    # Try v1 format (inline in content)
    content = getattr(message, "content", None)
    if content and isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                summary_list = item.get("summary", [])
                summaries = _extract_summary_texts(summary_list)
                if summaries:
                    return ReasoningContent(
                        summaries=summaries,
                        format_version="v1",
                        reasoning_id=item.get("id"),
                    )

    return None


def _extract_summary_texts(summary_list: Any) -> List[str]:
    """Extract text strings from a summary list.

    Summary is always a list of {type: "summary_text", text: "..."} objects.

    Args:
        summary_list: List of summary objects

    Returns:
        List of summary text strings
    """
    if not summary_list or not isinstance(summary_list, list):
        return []

    texts = []
    for item in summary_list:
        if isinstance(item, dict):
            # Handle {type: "summary_text", text: "..."} format
            text = item.get("text", "")
            if text:
                texts.append(text)
        elif isinstance(item, str):
            # Direct string
            texts.append(item)

    return texts


def extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from a message.

    Args:
        message: The message to extract tool calls from

    Returns:
        List of tool call dicts with 'id', 'name', 'args' keys
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []

    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            result.append({
                "id": tc.get("id", ""),
                "name": tc.get("name", "unknown"),
                "args": tc.get("args", {}),
            })
        else:
            # Handle object-style tool calls
            result.append({
                "id": getattr(tc, "id", ""),
                "name": getattr(tc, "name", "unknown"),
                "args": getattr(tc, "args", {}),
            })

    return result


def extract_full_content(message: Any) -> ExtractedContent:
    """Extract all content from a message.

    Combines text extraction, reasoning extraction, and tool call extraction
    into a single ExtractedContent object.

    Args:
        message: The message to extract content from

    Returns:
        ExtractedContent with text, reasoning, and tool_calls
    """
    content = getattr(message, "content", None)

    return ExtractedContent(
        text=extract_text_content(content),
        reasoning=extract_reasoning(message),
        tool_calls=extract_tool_calls(message),
    )


def clean_message_content(content: Any) -> str:
    """Extract clean text content, stripping reasoning blocks and IDs.

    Used when preparing messages for re-submission to LLM.

    Args:
        content: The content to clean

    Returns:
        Clean text content without reasoning blocks
    """
    # This is essentially the same as extract_text_content but named
    # differently for clarity of intent
    return extract_text_content(content)
