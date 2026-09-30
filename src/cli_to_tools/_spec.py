"""
cli_to_tools/_spec.py – command specs: argv → (tool name, named arguments).

A :class:`CommandSpec` is a thin layer over :mod:`argparse`. Specs can be
written in Python or loaded from YAML; the bundled packs in ``packs/`` are
plain YAML files loaded through the same path.

Example::

    git = CommandSpec("git")
    commit = git.subcommand("commit")            # -> tool "git_commit"
    commit.add_argument("-m", "--message")
    commit.add_argument("-a", "--all", action="store_true")

    git.parse(["commit", "-am", "fix"])
    # ("git_commit", {"message": "fix", "all": True}, True)
"""

from __future__ import annotations

import argparse
import os
import re
from importlib import resources
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

_OPTION_TYPES = {"str", "bool", "count", "list", "int"}


def normalize_tool_name(name: str) -> str:
    """Lowercase *name* and collapse non-alphanumerics to ``_`` (OpenAI-safe tool name)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class _SpecError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of printing usage and exiting."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _SpecError(message)

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:  # type: ignore[override]
        raise _SpecError(message or "exit")


class CommandSpec:
    """Declares how one executable's argv maps to a tool name and named arguments.

    Args:
        executable: Command name as typed (``git``).
        tool_name: Tool name override; defaults to the normalised command path.
        posix: Stop option parsing at the first positional, so everything after it
            is positional (``docker run IMAGE CMD...``, ``python script.py --flag``).
    """

    def __init__(self, executable: str, tool_name: Optional[str] = None, posix: bool = False) -> None:
        self.executable = executable
        self.tool_name = tool_name or normalize_tool_name(executable)
        self.posix = posix
        self.subcommands: Dict[str, CommandSpec] = {}
        self._parser = _Parser(prog=executable, add_help=False, allow_abbrev=False)

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Declare an option or positional; same signature as ``ArgumentParser.add_argument``."""
        return self._parser.add_argument(*args, **kwargs)

    def subcommand(
        self, name: str, tool_name: Optional[str] = None, posix: bool = False,
        aliases: Iterable[str] = (),
    ) -> CommandSpec:
        """Declare a subcommand and return its spec (tool name ``<parent>_<name>``)."""
        sub = CommandSpec(name, tool_name or f"{self.tool_name}_{normalize_tool_name(name)}", posix)
        for key in (name, *aliases):
            self.subcommands[key] = sub
        return sub

    def parse(self, argv: List[str]) -> Tuple[str, Dict[str, Any], bool]:
        """Translate *argv* (without the executable).

        Returns:
            ``(tool_name, arguments, matched)``. Tokens the spec does not know end
            up in ``arguments["extra_args"]``; if argv does not fit the spec at all,
            ``arguments`` is ``{"argv": [...]}``. *matched* is False in both cases.
        """
        options, rest = self._partition(argv, posix=True)
        # After a literal `--` the next word is an operand, never a subcommand.
        dispatch = bool(self.subcommands) and bool(rest) and "--" not in argv[:len(argv) - len(rest)]
        if dispatch and rest[0] in self.subcommands:
            try:
                own, extra = self._parser.parse_known_args(options)
            except _SpecError:
                return self.subcommands[rest[0]].tool_name, {"argv": list(argv)}, False
            name, args, matched = self.subcommands[rest[0]].parse(rest[1:])
            if "argv" in args and not matched and len(args) == 1:
                return name, {"argv": list(argv)}, False
            merged = {**self._clean(own), **args}
            if extra:
                merged["extra_args"] = extra + merged.get("extra_args", [])
            return name, merged, matched and not extra
        if dispatch and re.fullmatch(r"[a-z][a-z0-9_-]*", rest[0]):
            # Undeclared subcommand: keep the naming scheme so specs stay predictable.
            return f"{self.tool_name}_{normalize_tool_name(rest[0])}", {"argv": list(argv)}, False
        try:
            if not self.posix:
                options, rest = self._partition(argv, posix=False)
            own, extra = self._parser.parse_known_args(options + (["--"] + rest if rest else []))
            if rest and not self._parser._get_positional_actions():
                extra.remove("--")  # argparse hands the separator back when nothing consumes it
        except _SpecError:
            return self.tool_name, {"argv": list(argv)}, False
        args = self._clean(own)
        if extra:
            args["extra_args"] = extra
        return self.tool_name, args, not extra

    @staticmethod
    def _clean(namespace: argparse.Namespace) -> Dict[str, Any]:
        return {k: v for k, v in vars(namespace).items() if v is not None}

    def _takes_value(self, flag: str) -> bool:
        action = self._parser._option_string_actions.get(flag)
        return action is not None and action.nargs != 0

    def _partition(self, argv: List[str], posix: bool) -> Tuple[List[str], List[str]]:
        """Split *argv* into option tokens (with their values) and positionals.

        With *posix*, everything from the first positional on is positional.
        A literal ``--`` always ends option parsing.
        """
        options: List[str] = []
        positionals: List[str] = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok == "--":
                positionals.extend(argv[i + 1:])
                break
            if not tok.startswith("-") or tok == "-":
                if posix:
                    positionals.extend(argv[i:])
                    break
                positionals.append(tok)
                i += 1
                continue
            if "=" in tok or tok in self._parser._option_string_actions or tok.startswith("--"):
                width = 2 if "=" not in tok and self._takes_value(tok) else 1
            else:
                # Short bundle (-am): a value-taking flag in last position consumes the next token.
                width = 1
                for j, char in enumerate(tok[1:], start=1):
                    if self._takes_value("-" + char):
                        width = 2 if j == len(tok) - 1 else 1
                        break
            options.extend(argv[i:i + width])
            i += width
        return options, positionals

    # ── YAML / dict loading ───────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, executable: str, data: Optional[Dict[str, Any]], *,
                  _parent: Optional[CommandSpec] = None) -> CommandSpec:
        """Build a spec from its YAML/dict form (see ``packs/*.yaml`` for the schema)."""
        data = data or {}
        unknown = set(data) - {"tool", "posix", "aliases", "options", "positionals", "subcommands"}
        if unknown:
            raise ValueError(f"spec '{executable}': unknown keys {sorted(unknown)}")
        posix = bool(data.get("posix", False))
        if _parent is None:
            spec = cls(executable, data.get("tool"), posix)
        else:
            spec = _parent.subcommand(executable, data.get("tool"), posix, data.get("aliases", ()))
        for opt in data.get("options", []):
            kind = opt.get("type", "str")
            if kind not in _OPTION_TYPES:
                raise ValueError(f"spec '{executable}': unknown option type '{kind}'")
            kwargs: Dict[str, Any] = {}
            if "dest" in opt:
                kwargs["dest"] = opt["dest"]
            if kind == "bool":
                kwargs["action"] = "store_true"
            elif kind == "count":
                kwargs.update(action="count", default=0)
            elif kind == "list":
                kwargs["action"] = "append"
            elif kind == "int":
                kwargs["type"] = int
            spec.add_argument(*opt["flags"], **kwargs)
        for pos in data.get("positionals", []):
            kwargs = {"nargs": pos["nargs"]} if "nargs" in pos else {}
            spec.add_argument(pos["name"], **kwargs)
        for name, sub in (data.get("subcommands") or {}).items():
            cls.from_dict(name, sub, _parent=spec)
        return spec


class SpecRegistry:
    """Lookup table from executable name to :class:`CommandSpec`.

    Args:
        packs: Names of bundled packs to load. ``None`` loads all of them;
            an empty list loads none.
    """

    def __init__(self, packs: Optional[Iterable[str]] = None) -> None:
        self._specs: Dict[str, CommandSpec] = {}
        for pack in (available_packs() if packs is None else packs):
            self.load_pack(pack)

    def register(self, spec: CommandSpec, aliases: Iterable[str] = ()) -> CommandSpec:
        """Add or replace the spec for ``spec.executable``."""
        for key in (spec.executable, *aliases):
            self._specs[key] = spec
        return spec

    def get(self, executable: str) -> Optional[CommandSpec]:
        """Return the spec for *executable* (a path is reduced to its basename)."""
        return self._specs.get(os.path.basename(executable))

    def load_dict(self, data: Dict[str, Any]) -> None:
        for executable, body in data.items():
            self.register(CommandSpec.from_dict(executable, body), (body or {}).get("aliases", ()))

    def load_yaml(self, path: str) -> None:
        """Load specs from a YAML file; later definitions override earlier ones."""
        with open(path, encoding="utf-8") as fh:
            self.load_dict(yaml.safe_load(fh) or {})

    def load_pack(self, name: str) -> None:
        """Load a bundled pack by name (``shell``, ``git``, ``docker``, ...)."""
        if name not in available_packs():
            raise ValueError(f"unknown pack '{name}'; available: {available_packs()}")
        text = resources.files("cli_to_tools").joinpath("packs", f"{name}.yaml").read_text("utf-8")
        self.load_dict(yaml.safe_load(text) or {})


def available_packs() -> List[str]:
    """Names of the spec packs bundled with the package."""
    folder = resources.files("cli_to_tools").joinpath("packs")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".yaml"))
