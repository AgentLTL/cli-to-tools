"""
cli_to_tools/_shell.py – bash source → ordered list of simple commands.

The command line is parsed with tree-sitter-bash and the parse tree is walked
in execution order. Anything that cannot be analysed statically raises
:class:`TranslationError` (fail closed).

Usage::

    from cli_to_tools import parse_command_line

    for cmd in parse_command_line('git add . && git commit -m "x"'):
        print(cmd.operator, [w.text for w in cmd.argv])
"""

from __future__ import annotations

import codecs
import os
import re
import shlex
from typing import Any, Callable, Dict, List, Optional

import tree_sitter_bash
from tree_sitter import Language, Parser

from ._model import CommandNode, TranslationError, Word

_PARSER = Parser(Language(tree_sitter_bash.language()))

# ── Node classes ──────────────────────────────────────────────────────────────

_UNSUPPORTED = {
    "for_statement": "loop",
    "c_style_for_statement": "loop",
    "while_statement": "loop",
    "if_statement": "if statement",
    "case_statement": "case statement",
    "function_definition": "function definition",
}
_STATEMENTS = {
    "command", "list", "pipeline", "subshell", "compound_statement", "negated_command",
    "redirected_statement", "variable_assignment", "variable_assignments",
    "declaration_command", "unset_command", "test_command", *_UNSUPPORTED,
}
_SUBSTITUTIONS = {"command_substitution", "process_substitution"}
_REDIRECTS = {"file_redirect", "heredoc_redirect", "herestring_redirect"}
_GROUPS = {"subshell", "compound_statement"}

# Commands that execute text the parser cannot see.
_FORBIDDEN = {"eval", "source", "."}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_PYTHON = re.compile(r"python(\d(\.\d+)?)?")
_FIND_EXEC = {"-exec", "-execdir", "-ok", "-okdir"}

_GLOB = re.compile(r"(?<!\\)[*?\[]")
_BRACE = re.compile(r"\{[^{}]*(,|\.\.)[^{}]*\}")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


# ── Wrapper commands ──────────────────────────────────────────────────────────

def _skip_options(
    with_value: frozenset = frozenset(), assignments: bool = False, positional: int = 0,
) -> Callable[[List[Word]], Optional[int]]:
    """Build a function returning the index where a wrapper's inner command starts."""

    def start(args: List[Word]) -> Optional[int]:
        i = 0
        while i < len(args):
            tok = args[i].text
            if tok == "--":
                i += 1
                break
            if assignments and _ASSIGNMENT.match(tok):
                i += 1
            elif tok.startswith("-") and len(tok) > 1:
                i += 2 if tok in with_value else 1
            else:
                break
        return i + positional

    return start


def _command_builtin(args: List[Word]) -> Optional[int]:
    # `command -v name` only looks the name up; nothing is executed.
    if any(a.text in ("-v", "-V") for a in args):
        return None
    return _skip_options()(args)


_WRAPPERS: Dict[str, Callable[[List[Word]], Optional[int]]] = {
    "sudo": _skip_options(frozenset({
        "-u", "-g", "-p", "-h", "-C", "-D", "-R", "-T", "-U", "-r", "-t",
        "--user", "--group", "--prompt", "--host", "--chdir",
    }), assignments=True),
    "doas": _skip_options(frozenset({"-u", "-C"})),
    "env": _skip_options(frozenset({"-u", "--unset", "-C", "--chdir"}), assignments=True),
    "time": _skip_options(frozenset({"-o", "--output", "-f", "--format"})),
    "timeout": _skip_options(frozenset({"-s", "--signal", "-k", "--kill-after"}), positional=1),
    "nohup": _skip_options(),
    "setsid": _skip_options(),
    "nice": _skip_options(frozenset({"-n", "--adjustment"})),
    "stdbuf": _skip_options(frozenset({"-i", "-o", "-e"})),
    "watch": _skip_options(frozenset({"-n", "--interval"})),
    "exec": _skip_options(frozenset({"-a"})),
    "builtin": _skip_options(),
    "command": _command_builtin,
    "xargs": _skip_options(frozenset({
        "-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a",
        "--max-args", "--max-procs", "--max-lines", "--delimiter", "--arg-file", "--eof",
    })),
}


# ── Word resolution ───────────────────────────────────────────────────────────

def _text(node: Any) -> str:
    return node.text.decode()


def _resolve(node: Any) -> Word:
    """Return the word a node denotes after quote removal, and whether it is static."""
    kind = node.type
    raw = _text(node)
    if kind == "word":
        static = not (_GLOB.search(raw) or raw.startswith("~"))
        return Word(re.sub(r"\\(.)", r"\1", raw.replace("\\\n", ""), flags=re.S), bool(static))
    if kind in ("number", "variable_name", "test_operator"):
        return Word(raw)
    if kind == "raw_string":
        return Word(raw[1:-1])
    if kind == "ansi_c_string":
        try:
            return Word(codecs.decode(raw[2:-1], "unicode_escape"))
        except UnicodeDecodeError:
            return Word(raw, False)
    if kind in ("string", "translated_string"):
        return _resolve_string(node)
    if kind == "concatenation":
        parts = [_resolve(c) for c in node.children]
        text = "".join(p.text for p in parts)
        return Word(text, all(p.static for p in parts) and not _BRACE.search(text))
    # Expansions, substitutions, arithmetic: the value is only known at run time.
    return Word(raw, False)


def _resolve_string(node: Any) -> Word:
    """Resolve a double-quoted string, keeping expansions as literal source text."""
    def unescape(chunk: bytes) -> str:
        return re.sub(r'\\([$`"\\\n])', lambda m: "" if m.group(1) == "\n" else m.group(1),
                      chunk.decode())

    source = node.text
    base = node.start_byte
    pos = source.index(b'"') + 1
    end = len(source) - 1
    out: List[str] = []
    static = True
    for child in node.children:
        start, stop = child.start_byte - base, child.end_byte - base
        if child.type == '"' or start < pos or stop > end:
            continue
        out.append(unescape(source[pos:start]))
        if child.type == "string_content":
            out.append(unescape(source[start:stop]))
        else:
            out.append(_text(child))
            static = False
        pos = stop
    out.append(unescape(source[pos:end]))
    return Word("".join(out), static)


# ── Tree walk ─────────────────────────────────────────────────────────────────

class _Walker:
    """Walks a bash parse tree and collects simple commands in execution order."""

    def __init__(self) -> None:
        self.out: List[CommandNode] = []
        self._pipelines = 0

    def _new_pipeline(self) -> int:
        self._pipelines += 1
        return self._pipelines

    def statements(
        self, nodes: List[Any], op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        """Visit a ``;`` / newline separated sequence; the first statement inherits *op*."""
        first = True
        for node in nodes:
            if node.type == "&":
                raise TranslationError("background execution (&) is not supported")
            if not node.is_named or node.type == "comment" or node.type in _REDIRECTS:
                continue
            self.visit(node, op if first else ";", cond, depth, pipe)
            first = False

    def visit(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        kind = node.type
        if kind == "command":
            self.command(node, op, cond, depth, pipe)
        elif kind == "list":
            left, connector, right = node.children[0], node.children[1], node.children[-1]
            self.visit(left, op, cond, depth, pipe)
            self.visit(right, connector.type, True, depth, pipe)
        elif kind == "pipeline":
            self.pipeline(node, op, cond, depth, self._new_pipeline())
        elif kind in _GROUPS:
            self.statements(node.children, op, cond, depth + 1, pipe)
        elif kind == "negated_command":
            self.statements(node.children, op, cond, depth, pipe)
        elif kind == "redirected_statement":
            self.redirected(node, op, cond, depth, pipe)
        elif kind in ("variable_assignment", "variable_assignments"):
            self.substitutions(node, depth)
        elif kind in ("declaration_command", "unset_command", "test_command"):
            self.builtin(node, op, cond, depth, pipe)
        elif kind in _UNSUPPORTED:
            raise TranslationError(f"{_UNSUPPORTED[kind]} is not supported", _text(node))
        else:
            raise TranslationError(f"unsupported shell construct '{kind}'", _text(node))

    def pipeline(self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: int) -> None:
        first = True
        for child in node.children:
            if child.is_named:
                self.visit(child, op if first else "|", cond, depth, pipe)
                first = False

    def substitutions(self, node: Any, depth: int) -> None:
        """Emit the commands of every ``$(...)`` / ``<(...)`` nested in *node*."""
        for child in node.children:
            if child.type in _SUBSTITUTIONS:
                self.statements(child.children, "$()", False, depth + 1, None)
            elif child.type not in _STATEMENTS:
                self.substitutions(child, depth)

    def redirect(self, node: Any) -> Dict[str, str]:
        info: Dict[str, str] = {}
        if node.type == "heredoc_redirect":
            info["op"] = "<<"
            for child in node.children:
                if child.type == "heredoc_start":
                    info["target"] = _text(child)
            return info
        for child in node.children:
            if child.type == "file_descriptor":
                info["fd"] = _text(child)
            elif not child.is_named:
                info["op"] = _text(child)
            else:
                info["target"] = _resolve(child).text
        return info

    def heredoc_tail(self, node: Any, cond: bool, depth: int) -> None:
        """Visit commands that follow a heredoc marker (``cat <<EOF | grep x``)."""
        for child in node.children:
            if child.type == "pipeline":
                if self.out and self.out[-1].pipeline is None:
                    self.out[-1].pipeline = self._new_pipeline()
                pipe = self.out[-1].pipeline if self.out else self._new_pipeline()
                for sub in child.children:
                    if sub.is_named:
                        self.visit(sub, "|", cond, depth, pipe)
            elif child.type in _STATEMENTS:
                raise TranslationError("unsupported command after heredoc marker", _text(node))

    def redirected(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        nodes = [c for c in node.children if c.type in _REDIRECTS]
        for child in nodes:
            self.substitutions(child, depth)
        start = len(self.out)
        body = None
        for child in node.children:
            if child.type not in _REDIRECTS and child.is_named:
                body = child
                self.visit(child, op, cond, depth, pipe)
        redirects = [self.redirect(c) for c in nodes]
        targets = self.out[start:] if body is not None and body.type in _GROUPS else self.out[start:][-1:]
        for cmd in targets:
            cmd.redirects.extend(redirects)
        for child in nodes:
            self.heredoc_tail(child, cond, depth)

    def builtin(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        """Emit ``export`` / ``unset`` / ``[ ... ]`` style statements as plain commands."""
        self.substitutions(node, depth)
        if node.type == "test_command":
            name, rest = "test", [c for c in node.children if c.is_named]
        else:
            name, rest = _text(node.children[0]), node.children[1:]
        argv = [Word(name)]
        for child in rest:
            text = _text(child)
            argv.append(Word(text, not re.search(r"[$`]", text)))
        self.out.append(CommandNode(argv, _text(node), op, pipe, cond, depth))

    def command(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        self.substitutions(node, depth)
        env: Dict[str, str] = {}
        argv: List[Word] = []
        redirects: List[Dict[str, str]] = []
        tails: List[Any] = []
        for child in node.children:
            if child.type == "variable_assignment":
                value = child.children[2:]
                env[_text(child.children[0])] = _resolve(value[0]).text if value else ""
            elif child.type == "command_name":
                argv.append(_resolve(child.children[0]))
            elif child.type in _REDIRECTS:
                redirects.append(self.redirect(child))
                tails.append(child)
            elif child.is_named:
                argv.append(_resolve(child))
        if not argv:
            return
        cmd = CommandNode(argv, _text(node), op, pipe, cond, depth, redirects, env)
        self.emit(cmd)
        for child in tails:
            self.heredoc_tail(child, cond, depth)

    def emit(self, cmd: CommandNode) -> None:
        """Append *cmd*, then any command it wraps (``sudo``, ``xargs``, ``bash -c`` ...)."""
        head = cmd.argv[0]
        if not head.static:
            raise TranslationError("command name is not static", cmd.source)
        name = os.path.basename(head.text)
        if name in _FORBIDDEN:
            raise TranslationError(f"'{name}' executes text that cannot be analysed", cmd.source)
        self.out.append(cmd)
        args = cmd.argv[1:]

        if name in _SHELLS:
            script = self._shell_script(args)
            if script is not None:
                if not script.static:
                    raise TranslationError(f"dynamic script passed to '{name} -c'", cmd.source)
                self._nested(script.text, cmd, name)
            elif not any(not a.text.startswith(("-", "+")) for a in args):
                # `curl ... | sh`, `bash <<< "$X"`: the script arrives on stdin.
                raise TranslationError(f"'{name}' reads commands from stdin", cmd.source)
        elif name in _WRAPPERS:
            start = _WRAPPERS[name](args)
            if start is not None and start < len(args):
                self._wrapped(args[start:], cmd, name)
        elif _PYTHON.fullmatch(name):
            # `python -m pip install x` runs pip: surface the module as a command.
            for i, arg in enumerate(args):
                if arg.text == "-m" and i + 1 < len(args):
                    self._wrapped(args[i + 1:], cmd, "python -m")
                if arg.text in ("-m", "-c") or not arg.text.startswith("-"):
                    break
        elif name == "find":
            i = 0
            while i < len(args):
                if args[i].text in _FIND_EXEC:
                    stop = i + 1
                    while stop < len(args) and args[stop].text not in (";", "+"):
                        stop += 1
                    if stop > i + 1:
                        self._wrapped(args[i + 1:stop], cmd, "find")
                    i = stop
                i += 1

    @staticmethod
    def _shell_script(args: List[Word]) -> Optional[Word]:
        """Return the script word of ``bash -c <script>``, or None when there is none."""
        i = 0
        while i < len(args):
            tok = args[i].text
            if tok in ("-o", "-O", "+o", "+O"):
                i += 2
            elif tok.startswith("-") and not tok.startswith("--") and "c" in tok[1:]:
                return args[i + 1] if i + 1 < len(args) else None
            elif tok.startswith(("-", "+")) and len(tok) > 1:
                i += 1
            else:
                return None
        return None

    def _wrapped(self, argv: List[Word], parent: CommandNode, wrapper: str) -> None:
        inner = CommandNode(
            argv, shlex.join(w.text for w in argv), "wrap", parent.pipeline,
            parent.conditional, parent.depth + 1, wrapper=wrapper,
        )
        self.emit(inner)

    def _nested(self, script: str, parent: CommandNode, wrapper: str) -> None:
        pipelines: Dict[int, int] = {}
        for i, inner in enumerate(parse_command_line(script)):
            if inner.pipeline is not None:
                inner.pipeline = pipelines.setdefault(inner.pipeline, self._new_pipeline())
            if i == 0:
                inner.operator = "wrap"
            inner.depth += parent.depth + 1
            inner.conditional = inner.conditional or parent.conditional
            inner.wrapper = inner.wrapper or wrapper
            self.out.append(inner)


def _first_error(node: Any) -> Any:
    if node.type == "ERROR" or node.is_missing:
        return node
    for child in node.children:
        if child.has_error or child.is_missing:
            return _first_error(child)
    return node


def parse_command_line(source: str) -> List[CommandNode]:
    """Parse a bash command line into its simple commands, in execution order.

    Args:
        source: The command line, possibly a chain (``;`` ``&&`` ``||`` ``|``).

    Returns:
        One :class:`CommandNode` per simple command. Commands inside ``$(...)``
        come before the command that consumes them.

    Raises:
        TranslationError: On a syntax error or a construct that cannot be
            analysed statically (loops, ``eval``, background jobs, ...).
    """
    if not isinstance(source, str):
        raise TranslationError("command line must be a string", repr(source))
    root = _PARSER.parse(source.encode()).root_node
    if root.has_error:
        raise TranslationError("shell syntax error", _text(_first_error(root)))
    walker = _Walker()
    walker.statements(root.children, None, False, 0, None)
    return walker.out
