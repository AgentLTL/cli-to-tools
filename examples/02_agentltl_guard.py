"""Enforce AgentLTL constraints on a `bash` tool, without an LLM in the loop.

Requires AgentLTL:  pip install -e /path/to/AgentLTL

With the native backend the same enforcer is installed with::

    agent._enforcer = CliConstraintEnforcer.from_enforcer(agent._enforcer)
"""

from __future__ import annotations

from agentltl import Before, CalledWith, Constraint, Globally, Not
from agentltl.enforcement import ConstraintSeverity

from cli_to_tools.agentltl import CliConstraintEnforcer

enforcer = CliConstraintEnforcer(
    constraints=[
        Constraint(
            "commit_before_push", Before("git_commit", "git_push"),
            repair="Commit your changes before pushing.",
        ),
        Constraint(
            "no_force_push", Globally(Not(CalledWith("git_push", {"force": True}))),
            repair="Push without --force.",
        ),
    ],
    default_severity=ConstraintSeverity.SOFT_BLOCK,
    max_soft_attempts=10,
    shell_tools={"bash": "command"},
)

COMMANDS = [
    "git push",
    "git push && git commit -m x",
    "for i in 1 2; do rm $i; done",
    "git add . && git commit -m x && git push",
    "sudo bash -c 'git push --force'",
]

for step, command in enumerate(COMMANDS, start=1):
    args = {"command": command}
    enforcer.begin_generation()
    decision = enforcer.check("bash", args, step)
    print(f"$ {command}")
    if decision == "allow":
        # ... the caller executes the original command string here ...
        enforcer.record_completed("bash", args, f"call_{step}", "ok")
        print("  allowed")
    else:
        print(f"  {decision[0]}: " + decision[1].replace("\n", "\n    "))

print("trace:", [tc["tool_name"] for tc in enforcer._completed_tool_calls])
