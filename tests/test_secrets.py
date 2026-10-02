"""Commands that reveal, decrypt, create or pass credentials expose what a secrets rule needs."""

from __future__ import annotations

from typing import Any

import pytest

from cli_to_tools import Translator

T = Translator()

_CASES: list[tuple[str, str, dict[str, Any]]] = [
    # environment
    ("printenv GITHUB_TOKEN", "printenv", {"names": ["GITHUB_TOKEN"]}),
    ("export -p", "export", {"print": True}),
    ("declare -x", "declare", {"export": True}),
    ("docker inspect web", "docker_inspect", {"names": ["web"]}),
    ("docker compose config", "docker_compose_config", {}),
    # secret stores and cloud CLIs
    ("kubectl get secret db -o yaml -n prod", "kubectl_get",
     {"resource": "secret", "names": ["db"], "output": "yaml", "namespace": "prod"}),
    ("kubectl config view --raw", "kubectl_config_view", {"raw": True}),
    ("kubectl create secret generic s --from-literal=pw=x", "kubectl_create",
     {"resource": "secret", "from_literal": ["pw=x"]}),
    ("kubectl get pods -v=9", "kubectl_get", {"verbosity": "9"}),
    ("helm get values web -n prod", "helm_get_values", {"release": "web", "namespace": "prod"}),
    ("terraform output -raw db_password", "terraform_output", {"raw": True, "name": "db_password"}),
    ("terraform state pull", "terraform_state_pull", {}),
    ("aws secretsmanager get-secret-value --secret-id db --profile prod",
     "aws_secretsmanager_get_secret_value", {"secret_id": "db", "profile": "prod"}),
    ("aws ssm get-parameter --name /db/pw --with-decryption", "aws_ssm_get_parameter",
     {"name": "/db/pw", "with_decryption": True}),
    ("aws configure get aws_secret_access_key", "aws_configure_get", {"key": "aws_secret_access_key"}),
    ("aws iam create-access-key --user-name ci", "aws_iam_create_access_key", {"user_name": "ci"}),
    ("aws s3 ls --debug", "aws_s3_ls", {"debug": True}),
    ("gcloud secrets versions access latest --secret=db", "gcloud_secrets_versions_access",
     {"version": "latest", "secret": "db"}),
    ("gcloud auth print-access-token", "gcloud_auth_print_access_token", {}),
    ("gcloud iam service-accounts keys create key.json --iam-account=ci@p.iam",
     "gcloud_iam_service_accounts_keys_create", {"output_file": "key.json"}),
    ("az keyvault secret show --vault-name v -n db", "az_keyvault_secret_show",
     {"vault_name": "v", "name": "db"}),
    ("az account get-access-token", "az_account_get_access_token", {}),
    ("az ad sp create-for-rbac -n ci", "az_ad_sp_create_for_rbac", {"name": "ci"}),
    ("vault kv get -field=password secret/db", "vault_kv_get", {"field": "password", "path": "secret/db"}),
    ("vault token create -policy=admin", "vault_token_create", {"policy": ["admin"]}),
    ("gh auth token", "gh_auth_token", {}),
    ("gh auth status --show-token", "gh_auth_status", {"show_token": True}),
    ("gh gist create .env", "gh_gist_create", {"files": [".env"]}),
    ("heroku config:get DATABASE_URL -a web", "heroku_config_get", {"key": "DATABASE_URL", "app": "web"}),
    ("doppler secrets get API_KEY --plain", "doppler_secrets_get", {"names": ["API_KEY"], "plain": True}),
    ("vercel env pull .env.local", "vercel_env_pull", {"file": ".env.local"}),
    ("git credential fill", "git_credential", {"action": "fill"}),
    # decryption, password managers, keychains
    ("sops -d secrets.enc.yaml", "sops", {"decrypt": True, "files": ["secrets.enc.yaml"]}),
    ("ansible-vault view vault.yml", "ansible_vault_view", {"files": ["vault.yml"]}),
    ("gpg -d creds.gpg", "gpg", {"decrypt": True, "files": ["creds.gpg"]}),
    ("gpg --export-secret-keys -a me", "gpg", {"export_secret_keys": True}),
    ("age -d -i key.txt s.age", "age", {"decrypt": True, "identity": ["key.txt"]}),
    ("op read op://vault/db/password", "op_read", {"reference": "op://vault/db/password"}),
    ("op item get db --reveal", "op_item_get", {"item": "db", "reveal": True}),
    ("bw get password github", "bw_get", {"object": "password", "id": "github"}),
    ("pass show github", "pass_show", {"name": "github"}),
    ("security find-generic-password -s gh -w", "security_find_generic_password",
     {"service": "gh", "print_password": True}),
    ("secret-tool lookup service gh", "secret_tool_lookup", {"attributes": ["service", "gh"]}),
    ("openssl rsa -in key.pem -text -noout", "openssl_rsa", {"input": "key.pem", "text": True}),
    ("openssl pkcs12 -in c.p12 -nodes", "openssl_pkcs12", {"input": "c.p12", "nodes": True}),
    # secrets on the command line
    ("mysql -uroot -psecret db", "mysql", {"user": "root", "password_suffix": "secret"}),
    ("psql postgres://u:p@h/db", "psql", {"database": "postgres://u:p@h/db"}),
    ("redis-cli -a pw", "redis_cli", {"password": "pw"}),
    ("base64 -d key.b64", "base64", {"decode": True, "paths": ["key.b64"]}),
    ("strings id_rsa", "strings", {"paths": ["id_rsa"]}),
]


@pytest.mark.parametrize("command, name, want", _CASES, ids=[c for c, _, _ in _CASES])
def test_secret_commands(command: str, name: str, want: dict[str, Any]) -> None:
    (call,) = T.translate(command)
    assert call.name == name
    assert all(call.args.get(k) == v for k, v in want.items()), call.args
    assert "extra_args" not in call.args, call.args
