"""
cli_to_tools/agentltl.py – AgentLTL glue (optional, needs ``agentltl`` installed).

:class:`CliConstraintEnforcer` is a drop-in ``ConstraintEnforcer`` that expands
calls to a shell tool (``bash(command=...)``) into the structured calls of the
command line, so constraints are written over ``git_push``, ``rm`` ... instead
of over one opaque ``bash`` tool. A chain is approved or rejected as a whole,
before anything runs.

Usage::

    from agentltl import Before, Constraint
    from cli_to_tools.agentltl import CliConstraintEnforcer

    enforcer = CliConstraintEnforcer(
        constraints=[Constraint("commit_first", Before("git_commit", "git_push"))],
        shell_tools={"bash": "command"},
    )
    enforcer.check("bash", {"command": "git push && git commit -m x"}, step_number=1)
    # ("soft_block" | ..., feedback)  – or raises on HARD_STOP

    # With the native backend:
    agent._enforcer = CliConstraintEnforcer.from_enforcer(agent._enforcer)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from agentltl._enforcement_engine import ConstraintEnforcer, Decision

from ._model import ToolCall, TranslationError
from ._translate import Translator

DEFAULT_SHELL_TOOLS: Dict[str, str] = {"bash": "command"}


class CliConstraintEnforcer(ConstraintEnforcer):
    """``ConstraintEnforcer`` that checks shell command lines call by call.

    Args:
        *args: Forwarded to :class:`ConstraintEnforcer`.
        translator: Translator to use. Defaults to all bundled packs.
        shell_tools: Maps each shell tool name to the argument holding the command
            line. Calls to any other tool are checked unchanged.
        **kwargs: Forwarded to :class:`ConstraintEnforcer`.

    Attributes:
        untranslatable: Command lines rejected because they could not be translated.
    """

    def __init__(
        self, *args: Any, translator: Optional[Translator] = None,
        shell_tools: Optional[Mapping[str, str]] = None, **kwargs: Any,
    ) -> None:
        self._init_cli(translator, shell_tools)
        super().__init__(*args, **kwargs)

    def _init_cli(
        self, translator: Optional[Translator], shell_tools: Optional[Mapping[str, str]],
    ) -> None:
        self._translator = translator or Translator()
        self._shell_tools = dict(DEFAULT_SHELL_TOOLS if shell_tools is None else shell_tools)
        self._approved: Dict[str, List[ToolCall]] = {}
        self._blocked_command: Optional[str] = None
        self.untranslatable: List[Dict[str, Any]] = []

    @classmethod
    def from_enforcer(
        cls, enforcer: ConstraintEnforcer, translator: Optional[Translator] = None,
        shell_tools: Optional[Mapping[str, str]] = None,
    ) -> CliConstraintEnforcer:
        """Wrap an existing enforcer, keeping its constraints, settings and state."""
        new = cls.__new__(cls)
        new.__dict__.update(enforcer.__dict__)
        new._init_cli(translator, shell_tools)
        return new

    def reset(self) -> None:
        super().reset()
        self._approved = {}
        self._blocked_command = None
        self.untranslatable = []

    def translate(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[List[ToolCall]]:
        """Return the structured calls of a shell tool call, or None for other tools."""
        if tool_name not in self._shell_tools:
            return None
        return self._translator.translate((tool_args or {}).get(self._shell_tools[tool_name]))

    def check(self, tool_name: str, tool_args: Dict[str, Any], step_number: int) -> Decision:
        """Check a call; a shell command line is checked call by call, all or nothing."""
        if tool_name not in self._shell_tools:
            return super().check(tool_name, tool_args, step_number)
        command = (tool_args or {}).get(self._shell_tools[tool_name])
        try:
            calls = self._translator.translate(command)
        except TranslationError as exc:
            self.untranslatable.append({"step": step_number, "command": command, "reason": str(exc)})
            return ("persistent_block", (
                "[COMMAND REJECTED]\nThe command line could not be analysed and was NOT executed.\n"
                f"Reason: {exc}\n"
                "Rewrite it as plain commands chained with ;, &&, || or | "
                "(no eval, background jobs, function definitions or dynamic command names)."
            ))

        retry = command == self._blocked_command
        start = len(self._completed_tool_calls)
        try:
            for call in calls:
                pointer = self._last_blocked_call
                overrides = len(self._block_and_warn_overrides)
                decision = super().check(call.name, call.args, step_number)
                if decision != "allow":
                    self._blocked_command = command
                    return (decision[0], (
                        f"[In command line: {command}]\n[Blocked at: {call.meta['source']}]\n"
                        f"No part of the command line was executed.\n{decision[1]}"
                    ))
                # An allowed call clears the BLOCK_AND_WARN pointer. When the model re-issues
                # the identical command line, keep it alive until the blocked call is reached.
                if (retry and pointer is not None and self._last_blocked_call is None
                        and len(self._block_and_warn_overrides) == overrides):
                    self._last_blocked_call = pointer
                # Tentative: later calls of the chain must see this one in the trace.
                self._completed_tool_calls.append(call.to_dict())
        finally:
            del self._completed_tool_calls[start:]
        self._blocked_command = None
        self._approved[command] = calls
        return "allow"

    def record_completed(
        self, tool_name: str, tool_args: Dict[str, Any], tool_id: str, result: str,
    ) -> None:
        """Record an executed call; a shell command line is recorded as its structured calls."""
        if tool_name not in self._shell_tools:
            return super().record_completed(tool_name, tool_args, tool_id, result)
        command = (tool_args or {}).get(self._shell_tools[tool_name])
        calls = self._approved.pop(command, None) or self._translator.translate(command)
        for i, call in enumerate(calls):
            entry = call.to_dict()
            entry["cli"]["tool_call_id"] = tool_id
            # The output belongs to the command line as a whole; attach it to the last call.
            entry["result"] = result if i == len(calls) - 1 else None
            self._completed_tool_calls.append(entry)


def expand_tool_calls(
    tool_calls: List[Dict[str, Any]], translator: Optional[Translator] = None,
    shell_tools: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Expand shell tool calls in a recorded trace, for post-hoc ``verify_trace``.

    Args:
        tool_calls: ``metrics["tool_calls"]`` as produced by an AgentLTL backend.
        translator: Translator to use. Defaults to all bundled packs.
        shell_tools: Maps each shell tool name to its command-line argument.

    Returns:
        A new list where each shell call is replaced by its structured calls.
        Untranslatable command lines are kept as they were.
    """
    translator = translator or Translator()
    tools = dict(DEFAULT_SHELL_TOOLS if shell_tools is None else shell_tools)
    out: List[Dict[str, Any]] = []
    for tc in tool_calls:
        name = tc.get("tool_name", tc.get("name"))
        args = tc.get("tool_args", tc.get("arguments", {})) or {}
        if name not in tools:
            out.append(tc)
            continue
        try:
            calls = translator.translate(args.get(tools[name]))
        except TranslationError:
            out.append(tc)
            continue
        result = tc.get("tool_result", tc.get("result"))
        for i, call in enumerate(calls):
            out.append({**call.to_dict(), "tool_result": result if i == len(calls) - 1 else None})
    return out
