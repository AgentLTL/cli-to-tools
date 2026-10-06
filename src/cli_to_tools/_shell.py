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
    "function_definition": "function definition",
}
_LOOPS = {"for_statement", "c_style_for_statement", "while_statement"}
_CONTROL = {*_LOOPS, "if_statement", "case_statement"}
_STATEMENTS = {
    "command", "list", "pipeline", "subshell", "compound_statement", "negated_command",
    "redirected_statement", "variable_assignment", "variable_assignments",
    "declaration_command", "unset_command", "test_command", *_CONTROL, *_UNSUPPORTED,
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
_VAR_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_BRACE = re.compile(r"\{[^{}]*(,|\.\.)[^{}]*\}")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Loop values that can be pasted into the loop body without changing how it parses.
_SAFE_VALUE = r"[A-Za-z0-9_./:@%+=-]+"
_BRACE_LIST = re.compile(rf"({_SAFE_VALUE})?\{{({_SAFE_VALUE}(?:,{_SAFE_VALUE})+)\}}({_SAFE_VALUE})?")
_BRACE_RANGE = re.compile(rf"({_SAFE_VALUE})?\{{(-?\d+)\.\.(-?\d+)\}}({_SAFE_VALUE})?")
_MAX_UNROLL = 64
_LOOP_EXITS = {"break", "continue"}


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
        # `~` and `~/x` are the home directory; `~user`, `~+` and `~-` depend on the system
        static = not (_GLOB.search(raw) or (raw.startswith("~") and raw != "~"
                                             and not raw.startswith("~/")))
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

    def __init__(self, strict: bool = False) -> None:
        self.out: List[CommandNode] = []
        self.strict = strict
        self._pipelines = 0
        self._repeated = 0  # > 0 while inside a loop that could not be unrolled
        # Shell variables assigned a literal earlier in the command line (`F=a.txt; rm $F`).
        # None marks one whose value is not known statically.
        self.vars: Dict[str, Optional[str]] = {}

    def _subst(self, word: Word) -> Word:
        """Replace ``$NAME`` / ``${NAME}`` by values assigned earlier in the command line."""
        if word.static or "$" not in word.text:
            return word

        def value(m: "re.Match[str]") -> str:
            known = self.vars.get(m.group(1) or m.group(2))
            return m.group(0) if known is None else known

        text = _VAR_REF.sub(value, word.text)
        if text == word.text:
            return word
        static = not (re.search(r"[$`]", text) or _GLOB.search(text) or text.startswith("~"))
        return Word(text, static)

    def assign(self, node: Any, cond: bool) -> None:
        """Record ``NAME=value`` statements; a value known only at run time is recorded as None."""
        nodes = [node] if node.type == "variable_assignment" else [
            c for c in node.children if c.type == "variable_assignment"]
        for child in nodes:
            name = _text(child.children[0])
            parts = child.children[2:]
            word = self._subst(_resolve(parts[0])) if parts else Word("")
            new = word.text if word.static else None
            if self._repeated or (cond and self.vars.get(name, new) != new):
                new = None   # set in a loop, or on some paths only: no single value
            self.vars[name] = new

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
            self.assign(node, cond)
        elif kind in ("declaration_command", "unset_command", "test_command"):
            self.builtin(node, op, cond, depth, pipe)
        elif kind == "for_statement":
            self.for_loop(node, op, cond, depth, pipe)
        elif kind in _LOOPS:
            self.loop_once(node, op, cond, depth, pipe)
        elif kind == "if_statement":
            self._approximate("if statement", node)
            self.branch(node, op, cond, depth, pipe, in_body=False)
        elif kind == "case_statement":
            self._approximate("case statement", node)
            self.substitutions(node, depth)
            for item in node.children:
                if item.type == "case_item":
                    self.statements(
                        [c for c in item.children if c.type in _STATEMENTS], ";", True, depth, pipe)
        elif kind in _UNSUPPORTED:
            raise TranslationError(f"{_UNSUPPORTED[kind]} is not supported", _text(node))
        else:
            raise TranslationError(f"unsupported shell construct '{kind}'", _text(node))

    # ── Control flow ──────────────────────────────────────────────────────────

    def _approximate(self, what: str, node: Any) -> None:
        """Gate for constructs that are listed as an over-approximation of what runs."""
        if self.strict:
            raise TranslationError(f"{what} cannot be expanded statically (strict mode)", _text(node))

    def branch(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
        in_body: bool,
    ) -> None:
        """Visit an ``if`` / ``elif`` / ``else``: conditions, then every branch as conditional."""
        first = True
        for child in node.children:
            if child.type == "then":
                in_body = True
            elif child.type == "elif_clause":
                self.branch(child, ";", True, depth, pipe, in_body=False)
            elif child.type == "else_clause":
                self.branch(child, ";", True, depth, pipe, in_body=True)
            elif child.is_named and child.type != "comment":
                self.visit(child, op if first else ";", cond or in_body, depth, pipe)
                first = False

    def loop_once(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        """List a loop's condition and body once, flagged ``repeated`` (domain unknown)."""
        self._approximate("loop", node)
        for child in node.children:     # the loop variable takes values only known at run time
            if child.type == "variable_name":
                self.vars[_text(child)] = None
        self._repeated += 1
        try:
            first = True
            for child in node.children:
                if child.type in ("do_group", "compound_statement"):
                    self.statements(child.children, op if first else ";", True, depth, pipe)
                elif child.type in _STATEMENTS:
                    self.visit(child, op if first else ";", cond, depth, pipe)
                else:
                    # A `for` list (`in $(ls)`) is evaluated once, before the loop.
                    self._repeated -= 1
                    self.substitutions(child, depth, include_self=True)
                    self._repeated += 1
                    continue
                first = False
        finally:
            self._repeated -= 1

    def for_loop(
        self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: Optional[int],
    ) -> None:
        """Unroll ``for x in <literal list>``; fall back to :meth:`loop_once` otherwise."""
        var = next(_text(c) for c in node.children if c.type == "variable_name")
        body = next(c for c in node.children if c.type == "do_group")
        types = [c.type for c in node.children]
        values = node.children[types.index("in") + 1:types.index("do_group")] if "in" in types else None
        domain = _loop_domain([v for v in values if v.is_named]) if values is not None else None
        if domain is None or len(domain) > _MAX_UNROLL or _rebinds(body, var):
            return self.loop_once(node, op, cond, depth, pipe)

        # `break` / `continue` can cut an iteration short: later calls become conditional.
        exits = _has_command(body, _LOOP_EXITS)
        start, stop = body.children[0].end_byte, body.children[-1].start_byte
        spans = sorted(_references(body, var))
        for i, value in enumerate(domain):
            source, pos = b"", start
            for a, b in spans:
                source += body.text[pos - body.start_byte:a - body.start_byte] + value.encode()
                pos = b
            source += body.text[pos - body.start_byte:stop - body.start_byte]
            root = _PARSER.parse(source).root_node
            if root.has_error:
                raise TranslationError("loop body cannot be expanded", _text(node))
            self.statements(root.children, op if i == 0 else ";", cond or exits, depth, pipe)

    def pipeline(self, node: Any, op: Optional[str], cond: bool, depth: int, pipe: int) -> None:
        first = True
        for child in node.children:
            if child.is_named:
                self.visit(child, op if first else "|", cond, depth, pipe)
                first = False

    def substitutions(self, node: Any, depth: int, include_self: bool = False) -> None:
        """Emit the commands of every ``$(...)`` / ``<(...)`` nested in *node*."""
        for child in ([node] if include_self else node.children):
            if child.type in _SUBSTITUTIONS:
                self.statements(child.children, "$()", False, depth + 1, None)
            elif child.type not in _STATEMENTS:
                self.substitutions(child, depth)

    def redirects(self, node: Any) -> List[Dict[str, str]]:
        """The redirections of one redirect node. A heredoc carries the redirections written
        after its marker (``cat <<EOF > f.txt``) as children of its own."""
        if node.type == "heredoc_redirect":
            info = {"op": "<<"}
            nested: List[Dict[str, str]] = []
            for child in node.children:
                if child.type == "heredoc_start":
                    info["target"] = _text(child)
                elif child.type in ("file_redirect", "herestring_redirect"):
                    nested.extend(self.redirects(child))
            return [info] + nested
        info = {}
        for child in node.children:
            if child.type == "file_descriptor":
                info["fd"] = _text(child)
            elif not child.is_named:
                info["op"] = info.get("op", "") + _text(child)
            else:
                info["target"] = self._subst(_resolve(child)).text
        return [info]

    def heredoc_tail(self, node: Any, cond: bool, depth: int) -> None:
        """Visit commands that follow a heredoc marker (``cat <<EOF | grep x``,
        ``cat <<EOF && ls``)."""
        op: Optional[str] = None
        for child in node.children:
            if not child.is_named and child.type in ("&&", "||"):
                op = child.type
            elif child.type == "pipeline":
                if self.out and self.out[-1].pipeline is None:
                    self.out[-1].pipeline = self._new_pipeline()
                pipe = self.out[-1].pipeline if self.out else self._new_pipeline()
                for sub in child.children:
                    if sub.is_named:
                        self.visit(sub, "|", cond, depth, pipe)
            elif child.type in _STATEMENTS:
                if op is None:
                    raise TranslationError("unsupported command after heredoc marker",
                                           _text(node))
                self.visit(child, op, True, depth, None)

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
        redirects = [r for c in nodes for r in self.redirects(c)]
        whole = body is not None and body.type in (_GROUPS | _CONTROL)
        targets = self.out[start:] if whole else self.out[start:][-1:]
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
            if node.type == "declaration_command":   # export F=x / local F=x
                self.assign(node, cond)
        argv = [Word(name)]
        for child in rest:
            text = _text(child)
            argv.append(Word(text, not re.search(r"[$`]", text)))
        self.out.append(
            CommandNode(argv, _text(node), op, pipe, cond, depth, repeated=self._repeated > 0))

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
                argv.append(self._subst(_resolve(child.children[0])))
            elif child.type in _REDIRECTS:
                redirects.extend(self.redirects(child))
                tails.append(child)
            elif child.is_named:
                argv.append(self._subst(_resolve(child)))
        if not argv:
            return
        cmd = CommandNode(
            argv, _text(node), op, pipe, cond, depth, redirects, env, repeated=self._repeated > 0)
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
        if name in _LOOP_EXITS:
            return  # loop control, not a call; its effect is the `conditional` flag
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
            # `-exec CMD {} ;` hands CMD to find: emit CMD as its own call, and give find
            # the segment as one `-exec "CMD {} ;"` word so it does not read as paths.
            kept: List[Word] = [cmd.argv[0]]
            i = 0
            while i < len(args):
                if args[i].text in _FIND_EXEC:
                    stop = i + 1
                    while stop < len(args) and args[stop].text not in (";", "+"):
                        stop += 1
                    if stop > i + 1:
                        self._wrapped(args[i + 1:stop], cmd, "find")
                    segment = " ".join(w.text for w in args[i + 1:stop + 1])
                    kept += [args[i], Word(segment, all(w.static for w in args[i + 1:stop]))]
                    i = stop + 1
                    continue
                kept.append(args[i])
                i += 1
            cmd.argv = kept

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
            parent.conditional, parent.depth + 1, wrapper=wrapper, repeated=parent.repeated,
        )
        self.emit(inner)

    def _nested(self, script: str, parent: CommandNode, wrapper: str) -> None:
        pipelines: Dict[int, int] = {}
        for i, inner in enumerate(parse_command_line(script, self.strict)):
            if inner.pipeline is not None:
                inner.pipeline = pipelines.setdefault(inner.pipeline, self._new_pipeline())
            if i == 0:
                inner.operator = "wrap"
            inner.depth += parent.depth + 1
            inner.conditional = inner.conditional or parent.conditional
            inner.repeated = inner.repeated or parent.repeated
            inner.wrapper = inner.wrapper or wrapper
            self.out.append(inner)


def _loop_domain(nodes: List[Any]) -> Optional[List[str]]:
    """Values of a ``for`` list when they are all literal, else None."""
    out: List[str] = []
    for node in nodes:
        raw = _text(node)
        word = _resolve(node)
        if word.static and re.fullmatch(_SAFE_VALUE, word.text):
            out.append(word.text)
        elif node.type in ("concatenation", "brace_expression") and (m := _BRACE_RANGE.fullmatch(raw)):
            lo, hi = int(m.group(2)), int(m.group(3))
            if abs(hi - lo) >= _MAX_UNROLL:
                return None
            step = 1 if hi >= lo else -1
            out.extend(f"{m.group(1) or ''}{n}{m.group(4) or ''}" for n in range(lo, hi + step, step))
        elif node.type == "concatenation" and (m := _BRACE_LIST.fullmatch(raw)):
            out.extend(f"{m.group(1) or ''}{item}{m.group(3) or ''}" for item in m.group(2).split(","))
        else:
            return None
    return out


def _walk(node: Any) -> Any:
    yield node
    for child in node.children:
        yield from _walk(child)


def _references(body: Any, var: str) -> List[tuple]:
    """Byte spans of plain ``$var`` / ``${var}`` references inside *body*."""
    spans = []
    for node in _walk(body):
        kinds = [c.type for c in node.children]
        if ((node.type == "simple_expansion" and kinds == ["$", "variable_name"])
                or (node.type == "expansion" and kinds == ["${", "variable_name", "}"])):
            if _text(node.children[1]) == var:
                spans.append((node.start_byte, node.end_byte))
    return spans


def _rebinds(body: Any, var: str) -> bool:
    """True if *body* assigns *var* or reuses it as an inner loop / ``read`` variable."""
    for node in _walk(body):
        if node.type in ("variable_assignment", "for_statement", "c_style_for_statement"):
            if any(c.type == "variable_name" and _text(c) == var for c in _walk(node)
                   if c.parent is not None and c.parent.type != "simple_expansion"
                   and c.parent.type != "expansion"):
                return True
        if node.type == "command" and node.children and _text(node.children[0]) == "read":
            if any(_text(c) == var for c in node.children[1:]):
                return True
    return False


def _has_command(body: Any, names: set) -> bool:
    return any(n.type == "command_name" and _text(n) in names for n in _walk(body))


def _first_error(node: Any) -> Any:
    if node.type == "ERROR" or node.is_missing:
        return node
    for child in node.children:
        if child.has_error or child.is_missing:
            return _first_error(child)
    return node


# A redirection written after a heredoc marker, before the rest of the line:
# `cmd <<'EOF' 2>&1 | tail`. tree-sitter-bash cannot parse it when a pipe or another
# command follows.
_HEREDOC_THEN_REDIRECTS = re.compile(
    r"""(<<-?[ \t]*(?:'[^'\n]*'|"[^"\n]*"|[^\s|;&<>()]+))"""
    r"((?:[ \t]+\d*(?:&>>?|>>|>&|<&|>\||>|<)[ \t]*[^\s|;&<>()]+)+)")


_HEREDOC_THEN_LIST = re.compile(
    r"""<<(-?)[ \t]*('([^'\n]*)'|"([^"\n]*)"|([^\s|;&<>()]+))[ \t]*;[ \t]*([^\n]*)""")


def _defer_after_heredoc(source: str) -> str:
    """``cat <<EOF; ls`` runs ls after cat, so it is the same as writing ls on the line
    after the heredoc's end marker -- which tree-sitter-bash can parse."""
    lines = source.split("\n")
    for i, line in enumerate(lines):
        m = _HEREDOC_THEN_LIST.search(line)
        if not m:
            continue
        word = m.group(3) if m.group(3) is not None else (
            m.group(4) if m.group(4) is not None else m.group(5))
        rest = m.group(6)
        for j in range(i + 1, len(lines)):
            if (lines[j].lstrip("\t") if m.group(1) else lines[j]) == word:
                head = line[:m.start()] + line[m.start():m.start(6)].rstrip().rstrip(";").rstrip()
                return "\n".join(lines[:i] + [head] + lines[i + 1:j + 1] + [rest]
                                 + lines[j + 1:])
        break
    return source


def _hoist_heredoc_redirects(source: str) -> str:
    """Move redirections written after a heredoc marker in front of it. Each applies to a
    different file descriptor than the heredoc (stdin), so the meaning is unchanged; one
    that reads stdin (`<`, `<&`) is left in place."""
    def swap(m: "re.Match[str]") -> str:
        if re.search(r"(^|[ \t])\d*<", m.group(2)):
            return m.group(0)
        return m.group(2).lstrip() + " " + m.group(1)
    return _HEREDOC_THEN_REDIRECTS.sub(swap, source)


def parse_command_line(source: str, strict: bool = False) -> List[CommandNode]:
    """Parse a bash command line into its simple commands, in execution order.

    Args:
        source: The command line, possibly a chain (``;`` ``&&`` ``||`` ``|``).
        strict: Reject control flow that can only be over-approximated (``if``,
            ``case``, ``while``, ``for`` over a run-time list). ``for`` loops over
            a literal list are unrolled exactly either way.

    Returns:
        One :class:`CommandNode` per simple command. Commands inside ``$(...)``
        come before the command that consumes them.

    Raises:
        TranslationError: On a syntax error or a construct that cannot be
            analysed statically (``eval``, background jobs, function definitions, ...).
    """
    if not isinstance(source, str):
        raise TranslationError("command line must be a string", repr(source))
    root = _PARSER.parse(source.encode()).root_node
    if root.has_error:
        hoisted = _defer_after_heredoc(_hoist_heredoc_redirects(source))
        retry = _PARSER.parse(hoisted.encode()).root_node if hoisted != source else root
        if retry.has_error:
            raise TranslationError("shell syntax error", _text(_first_error(root)))
        root = retry
    walker = _Walker(strict)
    walker.statements(root.children, None, False, 0, None)
    return walker.out
