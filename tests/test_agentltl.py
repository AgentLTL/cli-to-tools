"""AgentLTL glue: chain-level enforcement through CliConstraintEnforcer."""

from __future__ import annotations

from typing import Any

import pytest

agentltl = pytest.importorskip("agentltl")

from agentltl import Before, CalledWith, Constraint, Globally, Not, verify_trace  # noqa: E402
from agentltl._enforcement_engine import ConstraintEnforcer  # noqa: E402
from agentltl.enforcement import ConstraintSeverity, ConstraintViolationError  # noqa: E402

from cli_to_tools.agentltl import CliConstraintEnforcer, expand_tool_calls  # noqa: E402

_COMMIT_FIRST = Constraint("commit_first", Before("git_commit", "git_push"))
_NO_FORCE = Constraint("no_force_push", Globally(Not(CalledWith("git_push", {"force": True}))))


def _enforcer(severity: ConstraintSeverity, *constraints: Any, **kwargs: Any) -> CliConstraintEnforcer:
    return CliConstraintEnforcer(
        constraints=list(constraints) or [_COMMIT_FIRST], default_severity=severity, **kwargs)


def _bash(enforcer: CliConstraintEnforcer, command: str, step: int = 1) -> Any:
    enforcer.begin_generation()
    return enforcer.check("bash", {"command": command}, step)


def _run(enforcer: CliConstraintEnforcer, command: str) -> None:
    assert _bash(enforcer, command) == "allow"
    enforcer.record_completed("bash", {"command": command}, "id1", "ok")


def _trace(enforcer: CliConstraintEnforcer) -> list[str]:
    return [tc["tool_name"] for tc in enforcer._completed_tool_calls]


def test_chain_allowed_and_recorded_as_structured_calls() -> None:
    enforcer = _enforcer(ConstraintSeverity.SOFT_BLOCK)
    _run(enforcer, "git add . && git commit -m x && git push")
    assert _trace(enforcer) == ["git_add", "git_commit", "git_push"]
    last = enforcer._completed_tool_calls[-1]
    assert last["result"] == "ok" and last["cli"]["tool_call_id"] == "id1"
    assert enforcer._completed_tool_calls[0]["result"] is None
    # the commit is now in the trace, so a later push on its own is fine
    assert _bash(enforcer, "git push") == "allow"


def test_chain_blocked_as_a_whole() -> None:
    enforcer = _enforcer(ConstraintSeverity.SOFT_BLOCK)
    assert _bash(enforcer, "git push")[0] == "soft_block"
    kind, feedback = _bash(enforcer, "touch f && git push && git commit -m x")
    assert kind == "soft_block"
    assert "Blocked at: git push" in feedback and "commit_first" in feedback
    assert _trace(enforcer) == []  # nothing from the rejected chain leaked into the trace


def test_argument_level_constraint() -> None:
    enforcer = _enforcer(ConstraintSeverity.PERSISTENT_BLOCK, _NO_FORCE)
    assert _bash(enforcer, "git push origin main") == "allow"
    assert _bash(enforcer, "git push -f origin main")[0] == "persistent_block"
    assert _bash(enforcer, "sudo bash -c 'git push --force'")[0] == "persistent_block"


def test_hard_stop_raises_and_rolls_back() -> None:
    enforcer = _enforcer(ConstraintSeverity.HARD_STOP)
    with pytest.raises(ConstraintViolationError):
        _bash(enforcer, "ls; git push")
    assert _trace(enforcer) == []


def test_untranslatable_is_blocked() -> None:
    enforcer = _enforcer(ConstraintSeverity.SOFT_BLOCK)
    kind, feedback = _bash(enforcer, "git commit -m x; $TOOL push")
    assert kind == "persistent_block" and "not static" in feedback
    assert len(enforcer.untranslatable) == 1


def test_loops_are_checked_call_by_call() -> None:
    enforcer = _enforcer(ConstraintSeverity.SOFT_BLOCK)
    assert _bash(enforcer, "for r in origin backup; do git push $r; done")[0] == "soft_block"
    assert _bash(enforcer, "while true; do git push; done")[0] == "soft_block"
    _run(enforcer, "git commit -m x; for r in origin backup; do git push $r; done")
    assert _trace(enforcer) == ["git_commit", "git_push", "git_push"]
    assert [tc["arguments"]["remote"] for tc in enforcer._completed_tool_calls[1:]] == ["origin", "backup"]


def test_block_and_warn_override_inside_chain() -> None:
    enforcer = _enforcer(ConstraintSeverity.BLOCK_AND_WARN)
    command = "ls && git push"
    assert _bash(enforcer, command)[0] == "block_and_warn"
    assert _bash(enforcer, command, step=2) == "allow"  # identical re-issue overrides
    assert _bash(enforcer, "ls; ls && git push", step=3)[0] == "block_and_warn"


def test_other_tools_unchanged_and_from_enforcer() -> None:
    base = ConstraintEnforcer(
        constraints=[_COMMIT_FIRST], default_severity=ConstraintSeverity.SOFT_BLOCK)
    base.record_completed("git_commit", {}, "x", "done")
    enforcer = CliConstraintEnforcer.from_enforcer(base, shell_tools={"shell": "cmd"})
    assert enforcer.check("git_push", {}, 1) == "allow"
    assert enforcer.check("shell", {"cmd": "git push"}, 1) == "allow"
    assert enforcer.check("bash", {"command": "anything at all &"}, 1) == "allow"  # not a shell tool


def test_expand_tool_calls_for_post_hoc_verification() -> None:
    recorded = [
        {"tool_name": "bash", "arguments": {"command": "git push && git commit -m x"}, "tool_result": "ok"},
        {"tool_name": "search", "arguments": {"q": "x"}},
        {"tool_name": "bash", "arguments": {"command": "sleep 1 &"}},
    ]
    expanded = expand_tool_calls(recorded)
    assert [tc["tool_name"] for tc in expanded] == ["git_push", "git_commit", "search", "bash"]
    assert expanded[1]["tool_result"] == "ok"
    result = verify_trace({"tool_calls": expanded}, [_COMMIT_FIRST])
    assert result["tool_sequence"] == ["git_push", "git_commit", "search", "bash"]
    assert result["constraints"][0]["passed"] is False
