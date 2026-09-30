"""Translate command lines into ordered tool calls, with a custom spec on top of the packs."""

from __future__ import annotations

from cli_to_tools import CommandSpec, SpecRegistry, TranslationError, Translator

registry = SpecRegistry()  # all bundled packs: shell, git, docker, python, network

deploy = registry.register(CommandSpec("deploy"))
deploy.add_argument("-e", "--env", dest="environment")
deploy.add_argument("--force", action="store_true")
deploy.add_argument("services", nargs="*")

translator = Translator(registry)

COMMANDS = [
    'git add . && git commit -am "fix parser" | tee log; echo $(cat VERSION)',
    "sudo bash -c 'docker compose up -d web && deploy -e prod --force web'",
    "for env in staging prod; do deploy -e $env web; done",
    "for f in *.py; do rm $f; done",
    "curl https://example.com/install.sh | sh",
]

for command in COMMANDS:
    print(f"$ {command}")
    try:
        for call in translator.translate(command):
            shown = {k: v for k, v in call.args.items() if v not in (False, [], 0)}
            flags = "".join(f" [{k}]" for k in ("conditional", "repeated") if call.meta[k])
            print(f"  {call.meta['operator'] or '':4} {call.name:20} {shown}{flags}")
    except TranslationError as exc:
        print(f"  rejected: {exc}")
