"""
cli_to_tools/agentltl.py – AgentLTL glue (optional, needs ``agentltl`` installed).

Constraints are written over ``git_push``, ``rm`` ... instead of over one opaque ``bash``
tool: a call to a shell tool (``bash(command=...)``) is checked as the structured calls
of its command line, all or nothing, before anything runs
(:meth:`agentltl.Enforcer.check_chain`).

- :class:`ShellEnforcer` is an :class:`agentltl.Enforcer`: ``check()`` returns an
  :class:`agentltl.Decision` whose ``index`` names the refused command of the line.
- :class:`CliConstraintEnforcer` is the same with the original ``ConstraintEnforcer``
  return values (``"allow"`` or ``(kind, feedback)``, raising on a stop).

Usage::

    from agentltl import Before, Constraint
    from cli_to_tools.agentltl import ShellEnforcer

    enforcer = ShellEnforcer([Constraint("commit_first", Before("git_commit", "git_push"))],
                             shell_tools={"bash": "command"})
    decision = enforcer.check("bash", {"command": "git push && git commit -m x"})
    decision.action, decision.index        # ("stop", 0): git push is refused

    # With the native backend:
    agent._enforcer = CliConstraintEnforcer.from_enforcer(agent._enforcer)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from agentltl import Decision as _Decision
from agentltl import Enforcer
from agentltl._enforcement_engine import ConstraintEnforcer, Decision

from ._model import ToolCall, TranslationError
from ._translate import Translator

DEFAULT_SHELL_TOOLS: Dict[str, str] = {"bash": "command"}

_REJECTED = ("[COMMAND REJECTED]\nThe command line could not be analysed and was NOT executed.\n"
             "Reason: {reason}\n"
             "Rewrite it as plain commands chained with ;, &&, || or | "
             "(no eval, background jobs, function definitions or dynamic command names).")


class _Shell:
    """What both enforcers add: shell tools are checked as their command lines."""

    def _init_cli(
        self, translator: Optional[Translator], shell_tools: Optional[Mapping[str, str]],
    ) -> None:
        self._translator = translator or Translator()
        self._shell_tools = dict(DEFAULT_SHELL_TOOLS if shell_tools is None else shell_tools)
        self._approved: Dict[str, List[ToolCall]] = {}
        self.untranslatable: List[Dict[str, Any]] = []

    @classmethod
    def from_enforcer(cls, enforcer: Enforcer, translator: Optional[Translator] = None,
                      shell_tools: Optional[Mapping[str, str]] = None) -> Any:
        """Wrap an existing enforcer, keeping its constraints, settings and state."""
        new = cls.__new__(cls)
        new.__dict__.update(enforcer.__dict__)
        new._init_cli(translator, shell_tools)
        return new

    def reset(self) -> None:
        super().reset()  # type: ignore[misc]
        self._approved = {}
        self.untranslatable = []

    def translate(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[List[ToolCall]]:
        """Return the structured calls of a shell tool call, or None for other tools."""
        if tool_name not in self._shell_tools:
            return None
        return self._translator.translate((tool_args or {}).get(self._shell_tools[tool_name]))

    def _check_shell(self, tool_name: str, tool_args: Dict[str, Any],
                     step_number: Optional[int], generation: Any) -> Optional[_Decision]:
        """The decision for a shell tool call; None for any other tool."""
        if tool_name not in self._shell_tools:
            return None
        command = (tool_args or {}).get(self._shell_tools[tool_name])
        try:
            calls = self._translator.translate(command)
        except TranslationError as exc:
            self.untranslatable.append({"step": step_number, "command": command,
                                        "reason": str(exc)})
            return _Decision("block", feedback=_REJECTED.format(reason=exc))
        decision = Enforcer.check_chain(self, [(c.name, c.args) for c in calls], step_number,
                                        generation=generation, chain_key=command)
        if decision.allowed:
            self._approved[command] = calls
        elif decision.index is not None:
            decision.feedback = (f"[In command line: {command}]\n"
                                 f"[Blocked at: {calls[decision.index].meta['source']}]\n"
                                 f"No part of the command line was executed.\n{decision.feedback}")
        return decision

    def record_completed(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None,
                         tool_id: str = "", result: Any = None, *,
                         status: Optional[int] = None) -> None:
        """Record an executed call; a shell command line is recorded as its structured calls.
        Its output and exit status belong to the line as a whole: they go on the last call."""
        if tool_name not in self._shell_tools:
            return super().record_completed(tool_name, tool_args, tool_id,  # type: ignore[misc]
                                            result, status=status)
        command = (tool_args or {}).get(self._shell_tools[tool_name])
        calls = self._approved.pop(command, None) or self._translator.translate(command)
        for i, call in enumerate(calls):
            entry = call.to_dict()
            entry["cli"]["tool_call_id"] = tool_id
            last = i == len(calls) - 1
            entry["result"] = result if last else None
            if last and status is not None:
                entry["status"] = status
            self.record_entry(entry)  # type: ignore[attr-defined]


class ShellEnforcer(_Shell, Enforcer):
    """:class:`agentltl.Enforcer` that checks shell command lines call by call.

    Args:
        *args: Forwarded to :class:`agentltl.Enforcer`.
        translator: Translator to use. Defaults to all bundled packs.
        shell_tools: Maps each shell tool name to the argument holding the command
            line. Calls to any other tool are checked unchanged.
        **kwargs: Forwarded to :class:`agentltl.Enforcer`.

    Attributes:
        untranslatable: Command lines refused because they could not be translated.
    """

    def __init__(self, *args: Any, translator: Optional[Translator] = None,
                 shell_tools: Optional[Mapping[str, str]] = None, **kwargs: Any) -> None:
        self._init_cli(translator, shell_tools)
        super().__init__(*args, **kwargs)

    def check(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None,
              step_number: Optional[int] = None, *, generation: Any = None) -> _Decision:
        decision = self._check_shell(tool_name, tool_args or {}, step_number, generation)
        if decision is None:
            decision = Enforcer.check(self, tool_name, tool_args, step_number,
                                      generation=generation)
        return decision


class CliConstraintEnforcer(_Shell, ConstraintEnforcer):
    """:class:`ShellEnforcer` with the original ``ConstraintEnforcer`` interface:
    ``check()`` returns ``"allow"`` or ``(kind, feedback)``, and raises on a stop."""

    def __init__(self, *args: Any, translator: Optional[Translator] = None,
                 shell_tools: Optional[Mapping[str, str]] = None, **kwargs: Any) -> None:
        self._init_cli(translator, shell_tools)
        super().__init__(*args, **kwargs)

    def check(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None,  # type: ignore[override]
              step_number: Optional[int] = None, **kwargs: Any) -> Decision:
        decision = self._check_shell(tool_name, tool_args or {}, step_number,
                                     kwargs.get("generation"))
        if decision is None:
            return ConstraintEnforcer.check(self, tool_name, tool_args, step_number, **kwargs)
        return self.legacy(decision)


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
