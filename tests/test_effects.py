"""Translator(shell_effects=True): redirections, paths after cd, run-time-only files."""

import os

import pytest

from cli_to_tools import Paths, SpecRegistry, Translator, UNKNOWN_PATHS


@pytest.fixture(scope="module")
def registry():
    return SpecRegistry()


def calls(command, registry, cwd="/proj"):
    t = Translator(registry, shell_effects=True, paths=Paths(cwd, cwd))
    return [(c.name, c.args) for c in t.translate(command)]


def args(command, registry, cwd="/proj", i=-1):
    return calls(command, registry, cwd)[i][1]


def test_off_by_default(registry):
    call = Translator(registry).translate("echo x > a.txt")[0]
    assert "redirect_to" not in call.args


def test_redirections(registry):
    assert args("echo x > a.txt", registry)["overwrite_to"] == ["/proj/a.txt"]
    a = args("echo x >> log.txt", registry)
    assert a["redirect_to"] == ["/proj/log.txt"] and "overwrite_to" not in a
    assert args("sort < in.txt", registry)["redirect_from"] == ["/proj/in.txt"]
    assert args("cat <<EOF > out.txt\nhi\nEOF", registry)["overwrite_to"] == ["/proj/out.txt"]
    a = args("make 2>&1 >/dev/null", registry)
    assert a["redirect_to"] == ["/dev/null"] and "overwrite_to" not in a


def test_windows_path_forms_are_rooted_and_nul_is_a_device_only_on_windows(monkeypatch):
    from cli_to_tools import _effects
    monkeypatch.setattr(_effects, "WINDOWS", True)
    assert all(_effects.is_absolute(p) for p in ("C:\\foo", "c:/foo", "\\foo", "/foo", "~"))
    assert not _effects.is_absolute("foo") and not _effects.is_absolute("C:foo")
    assert _effects.is_device("NUL") and _effects.is_device("nul")

    monkeypatch.setattr(_effects, "WINDOWS", False)   # POSIX: these are ordinary file names
    assert not _effects.is_absolute("C:\\foo") and not _effects.is_absolute("\\foo")
    assert _effects.is_absolute("/foo") and _effects.is_absolute("~")
    assert not _effects.is_device("NUL") and _effects.is_device("/dev/null")


@pytest.mark.skipif(os.name == "nt", reason="paths are resolved the Windows way there")
def test_on_posix_a_backslash_name_is_relative_and_NUL_is_a_file(registry):
    assert args("cd $D; rm '\\foo'", registry)[UNKNOWN_PATHS] == ["paths"]
    assert args("echo x > NUL", registry)["overwrite_to"] == ["/proj/NUL"]


def test_cd_moves_where_relative_paths_point(registry):
    assert args("cd secrets && rm key", registry)["paths"] == ["/proj/secrets/key"]
    assert args("cd secrets; cd ..; rm key", registry)["paths"] == ["/proj/key"]
    assert args("cd /etc && rm hosts", registry)["paths"] == ["/etc/hosts"]


def test_a_cd_ends_with_its_subshell_and_does_nothing_in_a_pipeline(registry):
    assert args("(cd secrets; ls); rm key", registry)["paths"] == ["/proj/key"]
    assert args('bash -c "cd secrets && rm key"', registry)["paths"] == ["/proj/secrets/key"]
    assert args("cd secrets | rm key", registry)["paths"] == ["/proj/key"]
    assert args("pushd secrets && rm key", registry)["paths"] == ["/proj/secrets/key"]


def test_after_a_cd_to_an_unknown_directory_relative_paths_are_unknown(registry):
    assert args("cd $D && rm key", registry)[UNKNOWN_PATHS] == ["paths"]
    assert args("popd; rm key", registry)[UNKNOWN_PATHS] == ["paths"]
    assert UNKNOWN_PATHS not in args("cd $D && rm /tmp/key", registry)


def test_files_only_known_at_run_time(registry):
    assert args("ls | xargs rm", registry)[UNKNOWN_PATHS] is True
    assert args("rm $F", registry)[UNKNOWN_PATHS] == ["paths"]
    assert args("find . -exec rm {} +", registry)[UNKNOWN_PATHS]
    assert UNKNOWN_PATHS not in args("rm build/x", registry)


def test_resolved_variables_and_home(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))     # what `~` expands to on Windows
    assert args("D=secrets; rm $D/key", registry)["paths"] == ["/proj/secrets/key"]
    assert args("rm ~/notes", registry)["paths"] == [(tmp_path / "notes").as_posix()]


def test_patch_targets_come_from_the_diff(registry, tmp_path):
    (tmp_path / "fix.diff").write_text(
        "diff --git a/migrations/001.sql b/migrations/001.sql\n"
        "--- a/migrations/001.sql\n+++ b/migrations/001.sql\n@@ -1 +1 @@\n-x\n+y\n")
    assert args("git apply fix.diff", registry, str(tmp_path))["paths"] == [
        (tmp_path / "migrations/001.sql").as_posix()]
    assert args("git apply missing.diff", registry, str(tmp_path))[UNKNOWN_PATHS] is True


def test_curl_remote_name_writes_a_file(registry):
    assert args("curl --output-dir bin -O https://x.dev/tool", registry)["output"] == [
        "/proj/bin/tool"]


def test_specs_can_mark_path_arguments(registry):
    reg = SpecRegistry(packs=[])
    reg.load_dict({"deploy": {"options": [{"flags": ["--manifest"], "path": True},
                                          {"flags": ["--file"], "path": False}]}})
    a = Translator(reg, shell_effects=True, paths=Paths("/proj", "/proj")).translate(
        "deploy --manifest m.yaml --file notes")[0].args
    assert a["manifest"] == "/proj/m.yaml" and a["file"] == "notes"
    schema = {s["name"]: s for s in reg.tool_schemas()}["deploy"]["parameters"]["properties"]
    assert schema["manifest"]["format"] == "path" and "format" not in schema["file"]


@pytest.mark.parametrize("command, names", [
    ("python3 - <<'EOF' 2>&1 | tail -20\nprint(1)\nEOF", ["python", "tail"]),
    ("cat <<EOF; ls\nhi\nEOF", ["cat", "ls"]),
    ("cat <<EOF && ls\nhi\nEOF", ["cat", "ls"]),
    ('python3 - <<"EOF" 2>/dev/null; ls\nx\nEOF', ["python", "ls"]),
])
def test_heredocs_followed_by_more_of_the_line(registry, command, names):
    assert [n for n, _ in calls(command, registry)] == names
