"""
cli_to_tools – translate shell command lines into ordered, structured tool calls.

Example::

    from cli_to_tools import Translator

    for call in Translator().translate("git add . && git commit -m 'x' | tee log"):
        print(call.name, call.args)
"""

from ._model import CommandNode, ToolCall, TranslationError, Word
from ._shell import parse_command_line
from ._spec import CommandSpec, SpecRegistry, available_packs, normalize_tool_name
from ._translate import Translator

__all__ = [
    "CommandNode",
    "CommandSpec",
    "SpecRegistry",
    "ToolCall",
    "TranslationError",
    "Translator",
    "Word",
    "available_packs",
    "normalize_tool_name",
    "parse_command_line",
]
