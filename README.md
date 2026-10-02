# cli-to-tools

Translate shell command lines into ordered, structured tool calls, so that an agent working
through a single `bash(command=...)` tool can be constrained by a tool-call harness such as
[AgentLTL](https://github.com/lailanelkoussy/AgentLTL).

```python
from cli_to_tools import Translator

for call in Translator().translate('git add . && git commit -am "fix" | tee log; echo $(cat VERSION)'):
    print(call.name, call.args)
# git_add     {"paths": ["."], ...}
# git_commit  {"message": ["fix"], "all": True, ...}
# tee         {"paths": ["log"], ...}
# cat         {"paths": ["VERSION"], ...}     <- runs before the echo that consumes it
# echo        {"words": ["$(cat VERSION)"], ...}
```

The library only translates. Executing the command line stays with the caller.

## Install

```bash
pip install -e ".[dev]"
pip install -e /path/to/AgentLTL      # only for cli_to_tools.agentltl
pytest
```

## How it works

1. **Shell layer** – the command line is parsed with tree-sitter-bash and the parse tree is
   walked in execution order: `;`, `&&`, `||` and `|` left to right; commands inside `$(...)`,
   backticks and `<(...)` before the command that consumes them; subshells and `{ ...; }` groups
   flattened.
2. **Specs** – each simple command is matched against a command spec, which gives it a tool name
   (`git_commit`, `docker_compose_up`) and flat named arguments.
3. **Fallback** – a command without a spec becomes `tool_name=<executable>`,
   `arguments={"argv": [...]}`.

Each `ToolCall` has `name`, `args`, `id` and `meta`. `to_dict()` returns AgentLTL's trace format,
`{"tool_name", "arguments", "id", "cli": meta}`, where `meta` holds:

| key | meaning |
|---|---|
| `command`, `source`, `index` | the full command line, this command's text, its position |
| `operator` | connector to the previous call: `;` `&&` `\|\|` `\|` `$()` `wrap` or `None` |
| `conditional` | `True` if the call may be skipped at run time (after `&&` / `\|\|`, inside `if` / `case`, a loop body that may run zero times) |
| `repeated` | `True` if the call sits in a loop that could not be unrolled: listed once, may run any number of times |
| `pipeline`, `depth` | pipeline id; nesting depth (subshell, substitution, wrapper) |
| `wrapper` | the wrapper this call runs under (`sudo`, `xargs`, `bash`, `find`, `python -m`) |
| `redirects`, `env` | `[{op, target, fd?}]` and `VAR=value` prefixes |
| `dynamic_args` | arguments the shell would expand (`$X`, globs, `$(...)`, `~`) |
| `spec_matched` | `False` if there was no spec or the spec did not fully cover the argv |

Conditional chains are over-approximated: every command that *could* run is emitted.

### Control flow

| construct | translation |
|---|---|
| `for x in a b c`, `{1..5}`, `x{a,b}` (literal list, up to 64 values) | **unrolled exactly**: the body is emitted once per value with `$x` / `${x}` substituted |
| `for x in *.py`, `in $(ls)`, `in $LIST`, `for ((...))`, `while`, `until` | body listed **once**, flagged `conditional` and `repeated`; the loop variable stays a dynamic argument |
| `if` / `elif` / `else`, `case` | conditions, then **every** branch, flagged `conditional` |

The second and third rows are over-approximations. They are sound for constraints on names and
order ("never call X", `Before`), but not for counts (`CalledNTimes`) or exact argument values,
because the number of iterations and the run-time values are unknown. `Translator(strict=True)`
rejects those constructs instead; literal `for` loops are exact and stay allowed.

### Wrappers

Commands that run other commands are unwrapped, so they cannot hide a call: `sudo`, `doas`,
`env`, `time`, `timeout`, `nohup`, `setsid`, `nice`, `stdbuf`, `watch`, `exec`, `command`,
`builtin`, `xargs`, `find -exec`, `python -m <module>`, and `bash -c '<script>'` (the script is
parsed recursively). The wrapper is emitted first, then the wrapped call.

Not unwrapped: commands that run a command string elsewhere or in another language
(`ssh host '<cmd>'`, `docker exec ... <cmd>`, `python -c`, `make`, `npx`, scripts such as
`bash script.sh`). These appear as one call with the command visible in its arguments; forbid or
restrict them in your constraints if that matters.

### Fail closed

`translate()` raises `TranslationError` instead of guessing on: syntax errors, function
definitions, background jobs (`&`), `eval` / `source` / `.`, a non-static command name
(`$CMD args`, including a loop variable that cannot be resolved), a dynamic `bash -c "$X"`, and a
shell reading its script from stdin (`curl ... | sh`).

## Specs

Bundled packs are loaded by default: `shell`, `git`, `docker` (with `docker compose` /
`docker-compose`), `python` (`python`, `pip`, `pytest`), `network` (`curl`, `wget`, `ssh`, `scp`,
`rsync`). They cover the commonly used subcommands and flags, not complete man pages. Flags a spec
does not know are kept in `extra_args`; an undeclared subcommand still gets a predictable name
(`git frob x` -> `git_frob`, `{"argv": [...]}`).

```python
from cli_to_tools import CommandSpec, SpecRegistry, Translator

registry = SpecRegistry()                      # all packs; SpecRegistry(packs=["shell", "git"]) to pick
registry.load_yaml("my_specs.yaml")            # add or override from YAML

deploy = registry.register(CommandSpec("deploy"))   # or in Python: plain argparse arguments
deploy.add_argument("-e", "--env", dest="environment")
deploy.add_argument("--force", action="store_true")
deploy.add_argument("services", nargs="*")

translator = Translator(registry)
```

```yaml
# my_specs.yaml
deploy:
  tool: deploy                 # optional tool name override
  aliases: [dpl]
  posix: false                 # true: options stop at the first positional (docker run IMAGE CMD...)
  options:
    - {flags: [-e, --env], dest: environment}
    - {flags: [--force], type: bool}          # str (default) | bool | count | list | int
    - {flags: [--mode], choices: [fast, safe], help: "How to deploy."}   # optional, for tool schemas
    - {flags: [-n, --namespace], global: true}   # also accepted after a subcommand
  positionals:
    - {name: services, nargs: "*"}
  subcommands: {}              # nested specs, same schema -> tool "deploy_<name>"

kubectl:
  extend: true                 # add to the bundled kubectl spec instead of replacing it
  subcommands:
    rollout: {positionals: [{name: action}, {name: resource}]}
```

A later spec for the same command replaces the earlier one, unless it says `extend: true`:
then its options are merged by flag and its subcommands one by one.
`registry.load_dict(specs, extend=True)` extends by default.

Boolean flags are always present in the arguments (`False` when absent), so both
`CalledWith("rm", {"recursive": True})` and `CalledWith("git_push", {"force": False})` work.

## AgentLTL

`CliConstraintEnforcer` is a drop-in `ConstraintEnforcer`. Calls to a shell tool are expanded into
the structured calls of the command line; every other tool is checked unchanged.

```python
from agentltl import Before, CalledWith, Constraint, Globally, Not
from cli_to_tools.agentltl import CliConstraintEnforcer

enforcer = CliConstraintEnforcer(
    constraints=[
        Constraint("commit_before_push", Before("git_commit", "git_push")),
        Constraint("no_force_push", Globally(Not(CalledWith("git_push", {"force": True})))),
    ],
    shell_tools={"bash": "command"},           # tool name -> argument holding the command line
)

enforcer.check("bash", {"command": "git push && git commit -m x"}, step_number=1)   # blocked
```

- **All or nothing.** The calls of a chain are checked in order, each against a trace that
  already contains the earlier calls of the same chain. If any call is blocked, the whole command
  line is rejected and nothing is recorded, so nothing should be executed.
- **Untranslatable** command lines are blocked with feedback telling the model how to rephrase,
  and listed in `enforcer.untranslatable`.
- **`record_completed`** stores one trace entry per structured call (with the `cli` metadata, so
  `Predicate`s can read `call.raw["cli"]`). The tool result is attached to the last call.
- **Native backend:** with AgentLTL's `cli` extra, pass `shell_tools={"bash": "command"}` to
  `AgentWithConstraints(backend="native")`, `MultiTurnAgent` or `NativeOpenAIAgent`; the agent
  then uses this enforcer and emits the structured calls in `metrics["tool_calls"]`. On an
  AgentLTL without that option: `agent._enforcer = CliConstraintEnforcer.from_enforcer(agent._enforcer)`.
- **Post hoc:** `expand_tool_calls(metrics["tool_calls"])` rewrites a recorded trace before
  `verify_trace`.

See `examples/02_agentltl_guard.py` for a runnable walk-through.
