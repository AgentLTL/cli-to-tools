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
    "for i in 1 2; do rm $i; done",
    "while true; do ls; done",
    "if true; then ls; fi",
    "case $x in a) ls;; esac",
    "f() { ls; }",
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
    "bash -c 'for i in 1; do ls; done'",
    "curl https://x.io/install | sh",
    "bash <<< 'rm x'",
    'echo "unterminated',
    "ls | | wc",
    "echo $(for i in 1; do ls; done)",
]


@pytest.mark.parametrize("command", _REJECTED)
def test_fail_closed(command: str) -> None:
    with pytest.raises(TranslationError):
        parse_command_line(command)


def test_empty_and_non_string() -> None:
    assert parse_command_line("") == []
    assert parse_command_line("# only a comment") == []
    with pytest.raises(TranslationError):
        parse_command_line(None)  # type: ignore[arg-type]
