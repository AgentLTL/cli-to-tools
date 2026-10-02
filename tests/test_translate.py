"""Specs, registry, bundled packs and the translator."""

from __future__ import annotations

from typing import Any

import pytest

from cli_to_tools import CommandSpec, SpecRegistry, Translator, available_packs

_T = Translator()


def _calls(command: str) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, c.args) for c in _T.translate(command)]


def _subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(actual.get(k) == v for k, v in expected.items())


# ── Specs ─────────────────────────────────────────────────────────────────────

def test_python_spec() -> None:
    git = CommandSpec("git")
    git.add_argument("-C", dest="cwd")
    commit = git.subcommand("commit")
    commit.add_argument("-m", "--message")
    commit.add_argument("-a", "--all", action="store_true")

    assert git.parse(["commit", "-am", "fix"]) == ("git_commit", {"message": "fix", "all": True}, True)
    assert git.parse(["-C", "/r", "commit", "--message=x"]) == (
        "git_commit", {"cwd": "/r", "message": "x", "all": False}, True)
    # unknown flag -> extra_args; unknown subcommand -> predictable name; bad argv -> raw argv
    assert git.parse(["commit", "--weird", "-a"]) == (
        "git_commit", {"all": True, "extra_args": ["--weird"]}, False)
    assert git.parse(["frob", "x"]) == ("git_frob", {"argv": ["frob", "x"]}, False)
    assert git.parse(["commit", "-m"]) == ("git_commit", {"argv": ["commit", "-m"]}, False)


def test_yaml_spec_and_registry_override(tmp_path: Any) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        "deploy:\n"
        "  tool: ship_it\n"
        "  aliases: [dpl]\n"
        "  options:\n"
        "    - {flags: [-e, --env], dest: environment}\n"
        "    - {flags: [--force], type: bool}\n"
        "    - {flags: [-t, --tag], type: list}\n"
        "  positionals: [{name: services, nargs: '*'}]\n"
        "rm:\n"
        "  positionals: [{name: victims, nargs: '*'}]\n"
    )
    registry = SpecRegistry(packs=["shell"])
    registry.load_yaml(str(path))
    translator = Translator(registry)
    (call,) = translator.translate("dpl -e prod -t a -t b web api")
    assert call.name == "ship_it"
    assert call.args == {"environment": "prod", "force": False, "tag": ["a", "b"],
                         "services": ["web", "api"]}
    assert translator.translate("rm a")[0].args == {"victims": ["a"]}
    assert translator.translate("git push")[0].args == {"argv": ["push"]}  # pack not loaded


def test_invalid_spec_rejected() -> None:
    with pytest.raises(ValueError):
        CommandSpec.from_dict("x", {"optoins": []})
    with pytest.raises(ValueError):
        SpecRegistry(packs=["nope"])


def test_tool_schemas() -> None:
    schemas = {s["name"]: s for s in SpecRegistry(packs=["git", "docker"]).tool_schemas()}
    assert {"git", "git_commit", "git_stash_pop", "docker_compose_up"} <= set(schemas)
    props = schemas["git_commit"]["parameters"]["properties"]
    assert props["message"]["type"] == "string" and props["all"]["type"] == "boolean"
    assert props["paths"] == {"type": "array", "items": {"type": "string"}}
    assert "cwd" in props  # options of the parent command are part of the arguments
    assert schemas["git_commit"]["description"] == "The `git commit` command."
    # a description or enum appears only where the spec states one
    spec = CommandSpec("pick")
    spec.add_argument("--mode", choices=["fast", "safe"], help="How to pick.")
    assert spec.tool_schemas()[0]["parameters"]["properties"]["mode"] == {
        "type": "string", "enum": ["fast", "safe"], "description": "How to pick."}
    # every argument a translation produces is declared in the schema
    for command in ("git commit -am x", "docker compose -f a.yml up -d web", "git push -f origin"):
        (call,) = _T.translate(command)
        assert set(call.args) <= set(schemas[call.name]["parameters"]["properties"])


def test_packs_load() -> None:
    assert available_packs() == ["cloud", "docker", "files", "git", "network", "python", "shell"]
    assert SpecRegistry(packs=[]).get("git") is None


# ── Packs ─────────────────────────────────────────────────────────────────────

_CASES: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
    ("rm -rf /tmp/x y", [("rm", {"recursive": True, "force": True, "paths": ["/tmp/x", "y"]})]),
    ("rm a -f b", [("rm", {"force": True, "recursive": False, "paths": ["a", "b"]})]),
    ("/bin/rm -- -f", [("rm", {"force": False, "paths": ["-f"]})]),
    ("cp -r a b dest", [("cp", {"recursive": True, "sources": ["a", "b"], "destination": "dest"})]),
    ("grep -rn 'foo bar' src", [("grep", {"recursive": True, "line_number": True,
                                          "pattern": "foo bar", "paths": ["src"]})]),
    ("mytool --x 1", [("mytool", {"argv": ["--x", "1"]})]),
    ('git add . && git commit -am "x y" | tee log; echo $(cat f)', [
        ("git_add", {"paths": ["."]}),
        ("git_commit", {"message": "x y", "all": True}),
        ("tee", {"paths": ["log"]}),
        ("cat", {"paths": ["f"]}),
        ("echo", {"words": ["$(cat f)"]}),
    ]),
    ("git -C /repo push -f origin main", [
        ("git_push", {"cwd": "/repo", "force": True, "remote": "origin", "refspecs": ["main"]})]),
    ("git checkout -b feat -- f.py", [("git_checkout", {"new_branch": "feat", "paths": ["f.py"]})]),
    ("git reset --hard HEAD~1", [("git_reset", {"hard": True, "targets": ["HEAD~1"]})]),
    ("git stash; git stash pop; git remote rm origin", [
        ("git_stash", {}), ("git_stash_pop", {}), ("git_remote_remove", {"name": "origin"})]),
    ("docker run --rm -it -v a:/b -e X=1 ubuntu:22.04 ls -la /b", [
        ("docker_run", {"rm": True, "interactive": True, "tty": True, "volume": ["a:/b"],
                        "env": ["X=1"], "image": "ubuntu:22.04", "command": ["ls", "-la", "/b"]})]),
    ("docker compose -f a.yml up -d web", [
        ("docker_compose_up", {"file": ["a.yml"], "detach": True, "services": ["web"]})]),
    ("docker-compose down -v", [("docker_compose_down", {"volumes": True})]),
    ("docker system prune -af", [("docker_system_prune", {"all": True, "force": True})]),
    ("python3 -m pip install -U requests 'numpy>=2'", [
        ("python", {"module": "pip"}),
        ("pip_install", {"upgrade": True, "packages": ["requests", "numpy>=2"]}),
    ]),
    ("python script.py --flag", [("python", {"script": "script.py", "script_args": ["--flag"]})]),
    ("pytest -xvv tests -k a", [
        ("pytest", {"exitfirst": True, "verbose": 2, "keyword": "a", "paths": ["tests"]})]),
    ("curl -sSL -X POST -H 'A: b' -d @f https://x.io -o out", [
        ("curl", {"request": "POST", "header": ["A: b"], "data": ["@f"], "silent": True,
                  "location": True, "urls": ["https://x.io"], "output": "out"})]),
    ("ssh -p 2222 me@host ls -la", [
        ("ssh", {"port": "2222", "host": "me@host", "command": ["ls", "-la"]})]),
    ("sudo rm -rf /x", [("sudo", {}), ("rm", {"recursive": True, "paths": ["/x"]})]),
]


@pytest.mark.parametrize("command, expected", _CASES, ids=[c for c, _ in _CASES])
def test_pack_translation(command: str, expected: list[tuple[str, dict[str, Any]]]) -> None:
    actual = _calls(command)
    assert [n for n, _ in actual] == [n for n, _ in expected]
    for (name, args), (_, want) in zip(actual, expected):
        assert _subset(want, args), f"{name}: {args} does not contain {want}"


# ── Tool call shape ───────────────────────────────────────────────────────────

def test_tool_call_dict_and_meta() -> None:
    command = "FOO=1 git push > log && rm $X"
    push, rm = _T.translate(command)
    data = push.to_dict()
    assert set(data) == {"tool_name", "arguments", "id", "cli"}
    assert data["tool_name"] == "git_push" and data["id"] != rm.id
    assert data["cli"]["command"] == command
    assert data["cli"]["source"] == "FOO=1 git push"
    assert data["cli"]["env"] == {"FOO": "1"}
    assert data["cli"]["redirects"] == [{"op": ">", "target": "log"}]
    assert data["cli"]["spec_matched"] is True
    assert (rm.meta["index"], rm.meta["operator"], rm.meta["conditional"]) == (1, "&&", True)
    assert rm.meta["dynamic_args"] == ["$X"]
