"""
cli_to_tools/_model.py – data types shared by the shell layer and the translator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TranslationError(ValueError):
    """Raised when a command line cannot be fully translated (fail closed).

    Attributes:
        reason: Why the command line was rejected.
        source: The offending fragment of the command line.
    """

    def __init__(self, reason: str, source: str = "") -> None:
        self.reason = reason
        self.source = source
        super().__init__(f"{reason}: {source!r}" if source else reason)


@dataclass
class Word:
    """One argv word after quote removal."""

    text: str
    static: bool = True
    """False when the shell would expand the word (``$VAR``, globs, ``$(...)``, ``~``)."""


@dataclass
class CommandNode:
    """One simple command, in execution order, with its position in the chain."""

    argv: List[Word]
    source: str = ""
    operator: Optional[str] = None
    """Connector to the previous command: ``;`` ``&&`` ``||`` ``|`` ``$()`` ``wrap``, or None."""
    pipeline: Optional[int] = None
    conditional: bool = False
    """True when the command may be skipped at run time (right-hand side of ``&&`` / ``||``)."""
    depth: int = 0
    redirects: List[Dict[str, str]] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    wrapper: Optional[str] = None
    """Name of the wrapper command (``sudo``, ``xargs``, ``bash -c`` ...) this one runs under."""


@dataclass
class ToolCall:
    """A structured tool call derived from one simple command."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the call in AgentLTL's trace dict format, with CLI metadata under ``cli``."""
        return {"tool_name": self.name, "arguments": self.args, "id": self.id, "cli": self.meta}
