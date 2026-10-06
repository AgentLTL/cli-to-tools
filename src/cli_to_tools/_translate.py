"""
cli_to_tools/_translate.py – command line → ordered tool calls.

Usage::

    from cli_to_tools import Translator

    calls = Translator().translate('git commit -am "fix" && git push origin main')
    [c.name for c in calls]        # ["git_commit", "git_push"]
    calls[0].args                  # {"message": "fix", "all": True, ...}
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional

from ._model import CommandNode, ToolCall
from ._effects import PATH_NAMES, Effects, Paths
from ._shell import parse_command_line
from ._spec import SpecRegistry, normalize_tool_name


class Translator:
    """Translates shell command lines into structured tool calls.

    Args:
        registry: Command specs to use. Defaults to all bundled packs.
        strict: Reject control flow that can only be over-approximated (``if``,
            ``case``, ``while``, ``for`` over a run-time list) instead of listing
            its commands once, flagged ``conditional`` / ``repeated``.
        shell_effects: Also record what each command does to files beyond its own
            arguments: redirections, paths made absolute (following ``cd``), and
            arguments only known at run time (see :mod:`cli_to_tools._effects`).
        paths: Where relative paths are resolved from (with *shell_effects*). None
            leaves paths as written.
    """

    def __init__(self, registry: Optional[SpecRegistry] = None, strict: bool = False,
                 shell_effects: bool = False, paths: Optional[Paths] = None) -> None:
        self.registry = registry if registry is not None else SpecRegistry()
        self.strict = strict
        self.shell_effects = shell_effects
        self.paths = paths
        self._effects: Optional[Effects] = None

    def translate(self, command: str) -> List[ToolCall]:
        """Return one :class:`ToolCall` per simple command, in execution order.

        Raises:
            TranslationError: If any part of the command line cannot be translated.
        """
        batch = uuid.uuid4().hex[:8]
        self._effects = Effects(self.paths) if self.shell_effects else None
        try:
            return [
                self._call(node, command, f"cli_{batch}_{i}", i)
                for i, node in enumerate(parse_command_line(command, self.strict))
            ]
        finally:
            self._effects = None

    def path_keys(self, executable: str, tool_name: str) -> List[str]:
        """The arguments of *tool_name* that hold file paths."""
        spec = self.registry.get(executable)
        sub = spec.find(tool_name) if spec is not None else None
        return sub.path_keys() if sub is not None else list(PATH_NAMES)

    def _call(self, node: CommandNode, command: str, call_id: str, index: int) -> ToolCall:
        executable = os.path.basename(node.argv[0].text)
        argv = [w.text for w in node.argv[1:]]
        spec = self.registry.get(executable)
        if spec is not None:
            name, args, matched = spec.parse(argv)
        else:
            name, args, matched = normalize_tool_name(executable), {"argv": argv}, False
        meta = {
            "command": command,
            "source": node.source,
            "index": index,
            "operator": node.operator,
            "pipeline": node.pipeline,
            "conditional": node.conditional,
            "repeated": node.repeated,
            "depth": node.depth,
            "wrapper": node.wrapper,
            "redirects": node.redirects,
            "env": node.env,
            "dynamic_args": [w.text for w in node.argv[1:] if not w.static],
            "spec_matched": matched,
        }
        if self._effects is not None:
            args = self._effects.apply(name, args, meta, self.path_keys(executable, name))
        return ToolCall(name, args, call_id, meta)
