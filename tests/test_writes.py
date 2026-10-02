"""Commands that write, overwrite or delete files expose their targets as named arguments.

Each case comes from a review of what a rule like "never touch migrations/" needs to see.
"""

from __future__ import annotations

from typing import Any

import pytest

from cli_to_tools import Translator, parse_command_line

T = Translator()


def _calls(command: str) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, c.args) for c in T.translate(command)]


def _subset(want: dict[str, Any], args: dict[str, Any]) -> bool:
    return all(args.get(k) == v for k, v in want.items())


_CASES = [
    # in-place editors: the script is not a path, -i takes a glued suffix only
    ("sed -i 's/a/b/' f.txt g.txt", [("sed", {"in_place": True, "script": "s/a/b/",
                                               "paths": ["f.txt", "g.txt"]})]),
    ("sed -i.bak -e 's/a/b/' f.txt", [("sed", {"in_place": True, "in_place_suffix": ".bak",
                                                "expression": ["s/a/b/"], "paths": ["f.txt"]})]),
    ("sed --in-place=.orig 's/a/b/' f.txt", [("sed", {"in_place": True, "in_place_suffix": ".orig",
                                                      "script": "s/a/b/", "paths": ["f.txt"]})]),
    ("sed -n p f.txt", [("sed", {"quiet": True, "script": "p", "paths": ["f.txt"]})]),
    ("sed -ni 's/a/b/' f", [("sed", {"quiet": True, "in_place": True, "paths": ["f"]})]),
    ("awk -i inplace '{print}' f.txt", [("awk", {"include": ["inplace"], "program": "{print}",
                                                  "paths": ["f.txt"]})]),
    ("awk -f prog.awk a b", [("awk", {"program_file": ["prog.awk"], "paths": ["a", "b"]})]),
    ("perl -pi -e 's/a/b/' f.txt", [("perl", {"print_loop": True, "in_place": True,
                                               "expression": ["s/a/b/"], "paths": ["f.txt"]})]),
    ("perl -pi.bak -e 's/a/b/' f.txt", [("perl", {"in_place": True, "in_place_suffix": ".bak",
                                                   "paths": ["f.txt"]})]),
    ("perl script.pl data.txt", [("perl", {"script": "script.pl", "paths": ["data.txt"]})]),
    ("ed -s f.txt", [("ed", {"quiet": True, "paths": "f.txt"})]),
    ("vim -c 'wq' f", [("vim", {"commands": ["wq"], "paths": ["f"]})]),
    ("ex -s f", [("vim", {"silent": True, "paths": ["f"]})]),
    # writers with no spec before
    ("dd if=/dev/zero of=disk.img bs=1M", [("dd", {"if": "/dev/zero", "of": "disk.img", "bs": "1M"})]),
    ("install -m 644 a.conf /etc/a.conf", [("install", {"mode": "644", "sources": ["a.conf"],
                                                        "destination": "/etc/a.conf"})]),
    ("install -t /usr/bin a b", [("install", {"destination_dir": "/usr/bin", "sources": ["a", "b"]})]),
    ("truncate -s 0 log.txt", [("truncate", {"size": "0", "paths": ["log.txt"]})]),
    ("shred -u secret.txt", [("shred", {"remove": True, "paths": ["secret.txt"]})]),
    ("split -l 10 big.txt part_", [("split", {"file": "big.txt", "prefix": "part_"})]),
    ("echo hi | sponge f.txt", [("echo", {}), ("sponge", {"file": "f.txt"})]),
    ("patch -p1 -i fix.diff", [("patch", {"strip": "1", "input": "fix.diff"})]),
    ("git apply -p1 fix.diff", [("git_apply", {"strip": "1", "patches": ["fix.diff"]})]),
    # output-path options
    ("curl --output-dir d -O https://x/f", [("curl", {"output_dir": "d", "remote_name": True,
                                                      "urls": ["https://x/f"]})]),
    ("unzip a.zip -d out", [("unzip", {"archive": "a.zip", "directory": "out"})]),
    ("zip -r a.zip src", [("zip", {"recursive": True, "archive": "a.zip", "paths": ["src"]})]),
    ("7z x a.7z -oout", [("7z_x", {"archive": "a.7z", "directory": "out"})]),
    ("install -d build/bin", [("install", {"directories": True, "sources": ["build/bin"]})]),
    # cp / mv with -t and -T
    ("cp -t dest a b", [("cp", {"destination_dir": "dest", "sources": ["a", "b"]})]),
    ("cp -T a b", [("cp", {"no_target_directory": True, "sources": ["a"], "destination": "b"})]),
    ("mv -t dest a", [("mv", {"destination_dir": "dest", "sources": ["a"]})]),
    ("mv a b c dest/", [("mv", {"sources": ["a", "b", "c"], "destination": "dest/"})]),
    # git: paths after `--` are kept apart from refs
    ("git checkout main -- f.py", [("git_checkout", {"targets": ["main"], "paths": ["f.py"]})]),
    ("git checkout -- f.py", [("git_checkout", {"paths": ["f.py"]})]),
    ("git restore --source HEAD~1 -- f.py", [("git_restore", {"source": "HEAD~1", "paths": ["f.py"]})]),
    ("git reset HEAD -- f.py", [("git_reset", {"targets": ["HEAD"], "paths": ["f.py"]})]),
    # find -exec: the nested command is its own call; find's paths stay clean
    ("find . -name '*.pyc' -exec rm {} +", [("find", {"paths": ["."], "exec": ["rm {} +"]}),
                                            ("rm", {"paths": ["{}"]})]),
    ("find src -exec sed -i s/a/b/ {} ';'", [("find", {"paths": ["src"]}),
                                             ("sed", {"in_place": True, "paths": ["{}"]})]),
    # cloud CLIs
    ("kubectl -n prod delete pod web-1", [("kubectl_delete", {"namespace": "prod", "resource": "pod",
                                                              "names": ["web-1"]})]),
    ("terraform apply -auto-approve", [("terraform_apply", {"auto_approve": True})]),
    ("gh pr merge 12 --admin", [("gh_pr_merge", {"pr": "12", "admin": True})]),
    ("aws --profile prod s3 rm s3://b/k --recursive", [("aws_s3_rm", {"profile": "prod",
                                                                      "path": "s3://b/k",
                                                                      "recursive": True})]),
]


@pytest.mark.parametrize("command, expected", _CASES, ids=[c for c, _ in _CASES])
def test_write_targets(command: str, expected: list[tuple[str, dict[str, Any]]]) -> None:
    actual = _calls(command)
    assert [n for n, _ in actual] == [n for n, _ in expected]
    for (name, args), (_, want) in zip(actual, expected):
        assert _subset(want, args), f"{name}: {args} does not contain {want}"


class TestRedirects:
    def test_a_redirect_written_after_a_heredoc_marker_is_kept(self):
        (node,) = parse_command_line("cat <<EOF > f.txt\nhello\nEOF")
        assert {"op": ">", "target": "f.txt"} in node.redirects

    def test_every_command_of_a_pipeline_keeps_its_own(self):
        a, b = parse_command_line("echo a > x | cat > y")
        assert a.redirects == [{"op": ">", "target": "x"}]
        assert b.redirects == [{"op": ">", "target": "y"}]

    def test_file_descriptor_redirects(self):
        (node,) = parse_command_line("cmd > out 2> err &>> all")
        assert [(r.get("fd"), r["op"], r["target"]) for r in node.redirects] == [
            (None, ">", "out"), ("2", ">", "err"), (None, "&>>", "all")]


class TestVariables:
    @pytest.mark.parametrize("command, path", [
        ("F=a.txt; rm $F", "a.txt"),
        ("F=a.txt && rm \"$F\"", "a.txt"),
        ("D=build; rm -r ${D}/out", "build/out"),
        ("export F=x; rm $F", "x"),
        ("F=a; G=$F.bak; rm $G", "a.bak"),
        ("F=a.txt; cat > $F", None),
    ])
    def test_literal_assignments_are_resolved(self, command, path):
        nodes = parse_command_line(command)
        last = nodes[-1]
        if path is None:
            assert last.redirects[-1]["target"] == "a.txt"
        else:
            assert last.argv[-1].text == path and last.argv[-1].static

    @pytest.mark.parametrize("command", [
        "rm $F",                                   # never assigned
        "F=$(cat list); rm $F",                    # assigned at run time
        "for f in $(ls); do rm $f; done",          # loop variable
        "if x; then F=a; else F=b; fi; rm $F",     # depends on the branch taken
        "F=x rm $F",                               # prefix assignment: $F expands first
    ])
    def test_unknown_values_stay_dynamic(self, command):
        last = parse_command_line(command)[-1]
        assert not last.argv[-1].static
