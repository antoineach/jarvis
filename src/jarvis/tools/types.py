"""Common types and result classes for tools."""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ToolExecutionResult:
    """Result object for tool execution."""
    success: bool
    reply_text: Optional[str]
    error_message: Optional[str] = None
    missing_context: Optional[str] = None
    resource_references: tuple[dict[str, Any], ...] = ()
    result_metadata: Optional[dict[str, Any]] = None
    # Nothing was retrieved (e.g. every search provider failed): the payload is a short
    # instruction to the reply model, which a fact-note digest would only garble, slowly.
    skip_digest: bool = False
