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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

_OPTION_TYPES = {"str", "bool", "count", "list", "int", "attached"}


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
        self.command = executable
        """The command as typed, including parent commands (``git commit``)."""
        self.subcommands: Dict[str, CommandSpec] = {}
        self._parser = _Parser(prog=executable, add_help=False, allow_abbrev=False)
        self._calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self._attached: Dict[str, str] = {}        # flag -> dest, value glued to the flag
        self._unless: Dict[str, List[str]] = {}    # positional -> option dests that drop it
        self._variants: Dict[frozenset, _Parser] = {}
        self.after_dashdash: Optional[str] = None
        """Argument that collects the words after a literal ``--`` (git's paths)."""
        self.key_value = False
        """Operands are ``key=value`` words (``dd if=a of=b``)."""

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Declare an option or positional; same signature as ``ArgumentParser.add_argument``."""
        self._calls.append((args, kwargs))
        self._variants.clear()
        return self._parser.add_argument(*args, **kwargs)

    def add_attached(self, *flags: str, dest: Optional[str] = None) -> None:
        """Declare a flag whose optional value must be glued to it: ``-i``, ``-i.bak``,
        ``--in-place=.bak`` (GNU sed, perl). The call gets ``<dest>: True`` and, when a
        value is given, ``<dest>_suffix``. A detached next word is never taken as the value."""
        name = dest or flags[-1].lstrip("-").replace("-", "_")
        for flag in flags:
            self._attached[flag] = name

    def add_unless(self, positional: str, dests: Iterable[str]) -> None:
        """Drop *positional* when any of the option *dests* is given (``sed -e`` replaces the
        script operand; ``cp -t DIR`` leaves no destination operand)."""
        self._unless[positional] = list(dests)
        self._variants.clear()

    def subcommand(
        self, name: str, tool_name: Optional[str] = None, posix: bool = False,
        aliases: Iterable[str] = (),
    ) -> CommandSpec:
        """Declare a subcommand and return its spec (tool name ``<parent>_<name>``)."""
        sub = CommandSpec(name, tool_name or f"{self.tool_name}_{normalize_tool_name(name)}", posix)
        sub.command = f"{self.command} {name}"
        sub._parent = self
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
        if self.key_value:
            return self._parse_key_value(argv)
        argv, attached = self._take_attached(argv)
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
            for key, value in self._clean(own).items():   # a global list option given twice
                if isinstance(value, list) and isinstance(args.get(key), list) and key != "argv":
                    merged[key] = value + args[key]
            if extra:
                merged["extra_args"] = extra + merged.get("extra_args", [])
            return name, merged, matched and not extra
        if dispatch and re.fullmatch(r"[a-z][a-z0-9_-]*", rest[0]):
            # Undeclared subcommand: keep the naming scheme so specs stay predictable.
            return f"{self.tool_name}_{normalize_tool_name(rest[0])}", {"argv": list(argv)}, False
        tail: Optional[List[str]] = None
        if self.after_dashdash and "--" in argv:
            cut = argv.index("--")
            argv, tail = argv[:cut], argv[cut + 1:]
        try:
            if not self.posix:
                options, rest = self._partition(argv, posix=False)
            parser = self._parser_for(options)
            own, extra = parser.parse_known_args(options + (["--"] + rest if rest else []))
            if rest and not parser._get_positional_actions():
                extra.remove("--")  # argparse hands the separator back when nothing consumes it
        except _SpecError:
            return self.tool_name, {"argv": list(argv) + (["--"] + tail if tail else [])}, False
        args = self._clean(own)
        args.update(attached)
        if tail is not None:
            args[self.after_dashdash] = args.get(self.after_dashdash, []) + tail
        if extra:
            args["extra_args"] = extra
        return self.tool_name, args, not extra

    def _parse_key_value(self, argv: List[str]) -> Tuple[str, Dict[str, Any], bool]:
        args: Dict[str, Any] = {}
        extra = []
        for tok in argv:
            key, eq, value = tok.partition("=")
            if eq and key and not key.startswith("-"):
                args[normalize_tool_name(key)] = value
            else:
                extra.append(tok)
        if extra:
            args["extra_args"] = extra
        return self.tool_name, args, not extra

    def _take_attached(self, argv: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Pull glued-value flags (see :meth:`add_attached`) out of *argv*."""
        if not self._attached:
            return argv, {}
        out: List[str] = []
        found: Dict[str, Any] = {}

        def hit(dest: str, value: str) -> None:
            found[dest] = True
            if value:
                found[f"{dest}_suffix"] = value

        for i, tok in enumerate(argv):
            if tok == "--":
                out.extend(argv[i:])
                break
            flag, eq, value = tok.partition("=")
            if tok.startswith("--") and flag in self._attached:
                hit(self._attached[flag], value)
                continue
            if tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
                kept = "-"
                for j, char in enumerate(tok[1:], start=1):
                    if "-" + char in self._attached:
                        hit(self._attached["-" + char], tok[j + 1:])
                        break
                    kept += char
                else:
                    out.append(tok)
                    continue
                if kept != "-":
                    out.append(kept)
                continue
            out.append(tok)
        return out, found

    def _parser_for(self, options: List[str]) -> "_Parser":
        """The parser to use given the options present: positionals named in
        :meth:`add_unless` are dropped when one of their options is given."""
        if not self._unless:
            return self._parser
        given = set()
        for tok in options:
            flag = tok.split("=", 1)[0]
            flags = [flag] if flag.startswith("--") or len(flag) == 2 else [
                "-" + c for c in flag[1:]]
            for f in flags:
                action = self._parser._option_string_actions.get(f)
                if action is not None:
                    given.add(action.dest)
        dropped = frozenset(p for p, dests in self._unless.items() if given & set(dests))
        if not dropped:
            return self._parser
        if dropped not in self._variants:
            parser = _Parser(prog=self.executable, add_help=False, allow_abbrev=False)
            for args, kwargs in self._calls:
                if not (args and not args[0].startswith("-") and args[0] in dropped):
                    parser.add_argument(*args, **kwargs)
            self._variants[dropped] = parser
        return self._variants[dropped]

    def _properties(self) -> Dict[str, Dict[str, Any]]:
        parent = getattr(self, "_parent", None)
        props = parent._properties() if parent is not None else {}
        for action in self._parser._actions:
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                schema: Dict[str, Any] = {"type": "boolean"}
            elif isinstance(action, argparse._CountAction):
                schema = {"type": "integer"}
            elif isinstance(action, argparse._AppendAction) or action.nargs in ("*", "+"):
                schema = {"type": "array", "items": {"type": "string"}}
            else:
                schema = {"type": "integer" if action.type is int else "string"}
            # Only what the spec author stated. A generated filler ("-name option")
            # would read as documentation to a constraint generator and license it to
            # assert free-text values against a parameter that documents nothing.
            if action.choices:
                schema["enum"] = list(action.choices)
            if action.help:
                schema["description"] = action.help
            props[action.dest] = schema
        for dest in dict.fromkeys(self._attached.values()):
            props[dest] = {"type": "boolean"}
            props[f"{dest}_suffix"] = {"type": "string"}
        if self.after_dashdash:
            props[self.after_dashdash] = {"type": "array", "items": {"type": "string"}}
        return props

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """JSON-schema description of every tool this spec can produce.

        One entry per command and subcommand, in the OpenAI function format minus
        the ``{"type": "function"}`` envelope. Use it to tell a constraint author
        (human or LLM) which tool names and arguments exist.
        """
        out = [{
            "name": self.tool_name,
            "description": f"The `{self.command}` command.",
            "parameters": {"type": "object", "properties": self._properties()},
        }]
        seen = {self.tool_name}
        for sub in self.subcommands.values():
            for schema in sub.tool_schemas():
                if schema["name"] not in seen:
                    seen.add(schema["name"])
                    out.append(schema)
        return out

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
                  _parent: Optional[CommandSpec] = None,
                  _globals: Sequence[Dict[str, Any]] = ()) -> CommandSpec:
        """Build a spec from its YAML/dict form (see ``packs/*.yaml`` for the schema).

        An option with ``global: true`` is also accepted after any subcommand
        (``kubectl delete pod x -n prod``), as kubectl, gh, helm and aws allow.
        """
        data = data or {}
        unknown = set(data) - {"tool", "posix", "aliases", "options", "positionals", "subcommands",
                               "after_dashdash", "key_value"}
        if unknown:
            raise ValueError(f"spec '{executable}': unknown keys {sorted(unknown)}")
        posix = bool(data.get("posix", False))
        if _parent is None:
            spec = cls(executable, data.get("tool"), posix)
        else:
            spec = _parent.subcommand(executable, data.get("tool"), posix, data.get("aliases", ()))
        spec.after_dashdash = data.get("after_dashdash")
        spec.key_value = bool(data.get("key_value", False))
        own_flags = {f for opt in data.get("options", []) for f in opt.get("flags", [])}
        inherited = [g for g in _globals if not own_flags & set(g["flags"])]
        for opt in inherited:
            _add_option(spec, opt, executable, inherited=True)
        for opt in data.get("options", []):
            _add_option(spec, opt, executable)
        globals_ = [*inherited, *(o for o in data.get("options", []) if o.get("global"))]
        for pos in data.get("positionals", []):
            kwargs = {key: pos[key] for key in ("nargs", "help", "choices") if key in pos}
            spec.add_argument(pos["name"], **kwargs)
            if pos.get("unless"):
                spec.add_unless(pos["name"], pos["unless"])
        for name, sub in (data.get("subcommands") or {}).items():
            cls.from_dict(name, sub, _parent=spec, _globals=globals_)
        return spec


def _add_option(spec: CommandSpec, opt: Dict[str, Any], executable: str,
                inherited: bool = False) -> None:
    """Declare one option of the dict form. An *inherited* (global) option leaves no value
    when absent, so it never hides the one given before the subcommand."""
    kind = opt.get("type", "str")
    if kind not in _OPTION_TYPES:
        raise ValueError(f"spec '{executable}': unknown option type '{kind}'")
    unknown = set(opt) - {"flags", "type", "dest", "help", "choices", "global"}
    if unknown:
        raise ValueError(f"spec '{executable}': option {opt.get('flags')}: unknown keys "
                         f"{sorted(unknown)}")
    if kind == "attached":
        spec.add_attached(*opt["flags"], dest=opt.get("dest"))
        return
    kwargs: Dict[str, Any] = {}
    for key in ("dest", "help", "choices"):
        if key in opt:
            kwargs[key] = opt[key]
    if kind == "bool":
        kwargs["action"] = "store_true"
    elif kind == "count":
        kwargs.update(action="count", default=0)
    elif kind == "list":
        kwargs["action"] = "append"
    elif kind == "int":
        kwargs["type"] = int
    if inherited:
        kwargs["default"] = argparse.SUPPRESS
    spec.add_argument(*opt["flags"], **kwargs)


def merge_spec_dicts(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """*base* extended by *extra*: options are merged by flag (an option of *extra* replaces
    the one of *base* sharing a flag), subcommands recursively, other keys replaced."""
    base, extra = dict(base or {}), dict(extra or {})
    out = {**base, **{k: v for k, v in extra.items() if k not in ("options", "subcommands")}}
    if "options" in extra:
        flags = {f for opt in extra["options"] for f in opt.get("flags", [])}
        out["options"] = [o for o in base.get("options", [])
                          if not flags & set(o.get("flags", []))] + list(extra["options"])
    if "subcommands" in extra:
        subs = dict(base.get("subcommands") or {})
        for name, body in (extra["subcommands"] or {}).items():
            subs[name] = merge_spec_dicts(subs.get(name), body)
        out["subcommands"] = subs
    return out


class SpecRegistry:
    """Lookup table from executable name to :class:`CommandSpec`.

    Args:
        packs: Names of bundled packs to load. ``None`` loads all of them;
            an empty list loads none.
    """

    def __init__(self, packs: Optional[Iterable[str]] = None) -> None:
        self._specs: Dict[str, CommandSpec] = {}
        self._raw: Dict[str, Dict[str, Any]] = {}
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

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """JSON-schema description of every tool the registered specs can produce."""
        out: Dict[str, Dict[str, Any]] = {}
        for spec in self._specs.values():
            for schema in spec.tool_schemas():
                out.setdefault(schema["name"], schema)
        return sorted(out.values(), key=lambda s: s["name"])

    def load_dict(self, data: Dict[str, Any], extend: bool = False) -> None:
        """Register the specs of *data*, replacing earlier specs for the same executables.

        An entry with ``extend: true`` (or every entry, with *extend*, unless it says
        ``extend: false``) is merged into the earlier spec instead (see :func:`merge_spec_dicts`),
        so adding one subcommand to ``kubectl`` keeps the bundled ones.
        """
        for executable, body in data.items():
            body = dict(body or {})
            if body.pop("extend", extend) and executable in self._raw:
                body = merge_spec_dicts(self._raw[executable], body)
            self._raw[executable] = body
            self.register(CommandSpec.from_dict(executable, body), body.get("aliases", ()))

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
