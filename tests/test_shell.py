"""Shell layer: ordering, chain metadata, word resolution and fail-closed cases."""

from __future__ import annotations

import pytest

from cli_to_tools import TranslationError, parse_command_line


def _names(command: str) -> list[str]:
    return [node.argv[0].text for node in parse_command_line(command)]


_ORDER: list[tuple[str, list[str]]] = [
    ("a; b\nc", ["a", "b", "c"]),
    ("a && b || c", ["a", "b", "c"]),
    ("a | b | c", ["a", "b", "c"]),
    ("echo $(cat f)", ["cat", "echo"]),
    ("echo `date` $(a $(b))", ["date", "b", "a", "echo"]),
    ("diff <(sort a) <(sort b)", ["sort", "sort", "diff"]),
    ("x=$(ls); echo hi", ["ls", "echo"]),
    ("(cd /tmp; ls) || { echo hi; }", ["cd", "ls", "echo"]),
    ("! grep x f", ["grep"]),
    ("cat <<EOF | grep x\nhi $(whoami)\nEOF\nls", ["whoami", "cat", "grep", "ls"]),
    ("echo hi > $(mktemp)", ["mktemp", "echo"]),
    ("export A=1; [ -f x ] && unset A", ["export", "test", "unset"]),
    ("ls # comment", ["ls"]),
    # wrappers
    ("sudo -u bob rm x", ["sudo", "rm"]),
    ("env A=1 B=2 ls", ["env", "ls"]),
    ("timeout -s KILL 5 curl x", ["timeout", "curl"]),
    ("ls | xargs -n1 rm -f", ["ls", "xargs", "rm"]),
    ("sudo env A=1 bash -c 'rm x && ls'", ["sudo", "env", "bash", "rm", "ls"]),
    ("bash -lc 'a | b'", ["bash", "a", "b"]),
    ("bash script.sh", ["bash"]),
    ("command -v git", ["command"]),
    ("find . -name x -exec rm {} \\; -exec echo {} +", ["find", "rm", "echo"]),
    ("python3 -m pip install x", ["python3", "pip"]),
    ("python -c 'print(1)'", ["python"]),
    ("/usr/bin/sudo /bin/rm x", ["/usr/bin/sudo", "/bin/rm"]),
]


@pytest.mark.parametrize("command, expected", _ORDER, ids=[c for c, _ in _ORDER])
def test_execution_order(command: str, expected: list[str]) -> None:
    assert _names(command) == expected


def test_chain_metadata() -> None:
    a, b, c, d = parse_command_line("a && b | c; d")
    assert [n.operator for n in (a, b, c, d)] == [None, "&&", "|", ";"]
    assert [n.conditional for n in (a, b, c, d)] == [False, True, True, False]
    assert a.pipeline is None and b.pipeline == c.pipeline is not None and d.pipeline is None


def test_substitution_and_wrapper_metadata() -> None:
    cat, echo = parse_command_line("echo $(cat f)")
    assert (cat.operator, cat.depth, echo.depth) == ("$()", 1, 0)
    assert not echo.argv[1].static

    sudo, rm = parse_command_line("sudo rm -rf /x")
    assert (rm.operator, rm.wrapper, rm.depth) == ("wrap", "sudo", 1)
    assert [w.text for w in rm.argv] == ["rm", "-rf", "/x"]


def test_redirects_and_env() -> None:
    (cmd,) = parse_command_line("FOO=1 ls -la > out.txt 2>&1")
    assert cmd.env == {"FOO": "1"}
    assert cmd.redirects == [
        {"op": ">", "target": "out.txt"}, {"fd": "2", "op": ">&", "target": "1"},
    ]
    a, b = parse_command_line("(a; b) > f")
    assert a.redirects == b.redirects == [{"op": ">", "target": "f"}]
    ls, wc = parse_command_line("ls 2>/dev/null | wc -l > n")
    assert ls.redirects[0]["target"] == "/dev/null" and wc.redirects[0]["target"] == "n"


_WORDS: list[tuple[str, str, bool]] = [
    ("'a b'", "a b", True),
    ('"a \\"b\\" c"', 'a "b" c', True),
    ("a\\ b", "a b", True),
    ("\"x\"'y'z", "xyz", True),
    ("$'a\\tb'", "a\tb", True),
    ("'*.py'", "*.py", True),
    ("*.py", "*.py", False),
    ("~/x", "~/x", False),
    ("a{1,2}", "a{1,2}", False),
    ("{}", "{}", True),
    ('"pre $X post"', "pre $X post", False),
    ("${X:-y}", "${X:-y}", False),
    ("$((1+2))", "$((1+2))", False),
]


@pytest.mark.parametrize("source, text, static", _WORDS, ids=[w for w, _, _ in _WORDS])
def test_word_resolution(source: str, text: str, static: bool) -> None:
    (cmd,) = parse_command_line(f"echo {source}")
    assert (cmd.argv[1].text, cmd.argv[1].static) == (text, static)


_REJECTED: list[str] = [
    "f() { ls; }",
    "for c in *; do $c; done",
    "sleep 5 &",
    "a & b",
    "$CMD arg",
    "$(which rm) x",
    "eval 'ls'",
    "source env.sh",
    ". env.sh",
    "sudo eval ls",
    "sudo $CMD",
    'bash -c "$SCRIPT"',
    "bash -c 'f() { ls; }; f'",
    "curl https://x.io/install | sh",
    "bash <<< 'rm x'",
    'echo "unterminated',
    "ls | | wc",
]


@pytest.mark.parametrize("command", _REJECTED)
def test_fail_closed(command: str) -> None:
    with pytest.raises(TranslationError):
        parse_command_line(command)


def _show(command: str, strict: bool = False) -> list[str]:
    """Commands as text, with ``?`` for conditional and ``*`` for repeated."""
    return [
        " ".join(w.text for w in n.argv) + "?" * n.conditional + "*" * n.repeated
        for n in parse_command_line(command, strict)
    ]


_UNROLLED: list[tuple[str, list[str]]] = [
    ("for f in a b; do echo $f > ${f}.txt; done", ["echo a", "echo b"]),
    ("for i in {1..3} x{a,b}; do touch f$i; done", ["touch f1", "touch f2", "touch f3", "touch fxa", "touch fxb"]),
    ("for i in {3..1}; do echo $i; done", ["echo 3", "echo 2", "echo 1"]),
    ("for i in 1 2; do for j in x y; do echo $i$j; done; done", ["echo 1x", "echo 1y", "echo 2x", "echo 2y"]),
    ("for c in ls pwd; do $c; done", ["ls", "pwd"]),
    ("for f in a b; do echo '$f' \"$f\"; done", ["echo $f a", "echo $f b"]),
    ("for f in a b; do bash -c \"cat $f\"; done", ["bash -c cat a", "cat a", "bash -c cat b", "cat b"]),
    ("for f in a; do echo ${f%.txt}; done", ["echo ${f%.txt}"]),
    ("ls && for f in a b; do rm $f; done", ["ls", "rm a?", "rm b?"]),
    ("for f in a b; do rm $f || break; done", ["rm a?", "rm b?"]),
    ("echo $(for i in 1 2; do cat $i; done)", ["cat 1", "cat 2", "echo $(for i in 1 2; do cat $i; done)"]),
]


@pytest.mark.parametrize("command, expected", _UNROLLED, ids=[c for c, _ in _UNROLLED])
def test_literal_for_loops_are_unrolled(command: str, expected: list[str]) -> None:
    assert _show(command) == expected
    assert _show(command, strict=True) == expected  # exact, so allowed in strict mode too


def test_unrolled_loop_keeps_redirects_and_order() -> None:
    a, b, ls = parse_command_line("for f in a b; do echo $f; done > out; ls")
    assert a.redirects == b.redirects == [{"op": ">", "target": "out"}]
    assert [n.operator for n in (a, b, ls)] == [None, ";", ";"]


_APPROXIMATED: list[tuple[str, list[str]]] = [
    ("for f in *.py; do rm $f; done", ["rm $f?*"]),
    ("for x in $(ls); do cat $x | wc -l; done", ["ls", "cat $x?*", "wc -l?*"]),
    ("for f in a b; do f=z; echo $f; done", ["echo $f?*"]),
    ("for f in a b; do read f; echo $f; done", ["read f?*", "echo $f?*"]),
    ("for a; do echo $a; done", ["echo $a?*"]),
    ("for i in {1..1000}; do echo $i; done", ["echo $i?*"]),
    ("for ((i=0;i<3;i++)); do ls $i; done", ["ls $i?*"]),
    ("while read l; do echo $l; done < f", ["read l*", "echo $l?*"]),
    ("until test -f x; do sleep 1; done", ["test -f x*", "sleep 1?*"]),
    ("if grep -q x f; then ls; elif true; then pwd; else rm x; fi",
     ["grep -q x f", "ls?", "true?", "pwd?", "rm x?"]),
    ("case $(uname) in Linux|Darwin) ls;; *) pwd; rm x;; esac", ["uname", "ls?", "pwd?", "rm x?"]),
    ("if true; then for f in a b; do rm $f; done; fi", ["true", "rm a?", "rm b?"]),
]


@pytest.mark.parametrize("command, expected", _APPROXIMATED, ids=[c for c, _ in _APPROXIMATED])
def test_control_flow_is_over_approximated(command: str, expected: list[str]) -> None:
    assert _show(command) == expected
    with pytest.raises(TranslationError):
        parse_command_line(command, strict=True)


def test_empty_and_non_string() -> None:
    assert parse_command_line("") == []
    assert parse_command_line("# only a comment") == []
    with pytest.raises(TranslationError):
        parse_command_line(None)  # type: ignore[arg-type]
