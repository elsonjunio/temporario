"""Entry point: the agent-mini CLI with the OpenCode Go and LM Studio providers."""

from __future__ import annotations

import os
from pathlib import Path

import click

from src.providers import register

# Must run before importing agent_mini.cli, which copies PROVIDERS at import time.
register()

from agent_mini.cli import cli  # noqa: E402
import agent_mini.cli as _agent_cli  # noqa: E402
from agent_mini.config import config_file, save_config  # noqa: E402

from src.policy import install as install_workspace_policy  # noqa: E402

# Ask only when a call leaves the workspace (see src/policy.py).
install_workspace_policy(_agent_cli)

PROVIDER_NAMES = ("opencode", "lmstudio", "stackspot")


@cli.command("setup")
@click.option(
    "--provider",
    "provider_name",
    type=click.Choice(PROVIDER_NAMES),
    default="opencode",
    show_default=True,
    help="Provider used by default.",
)
@click.option(
    "--workspace",
    default=None,
    help="Workspace directory (default: AGENT_MINI_WORKSPACE or the current directory).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def setup(provider_name: str, workspace: str | None, force: bool) -> None:
    """Write a config.json configured for the custom providers."""
    path = config_file()
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists; use --force to overwrite it.")

    directory = Path(
        workspace or os.getenv("AGENT_MINI_WORKSPACE") or Path.cwd()
    ).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    save_config(
        {
            "provider": provider_name,
            "providers": {
                "opencode": {
                    "baseUrl": "https://opencode.ai/inference/go/v1",
                    "model": os.getenv("OPENCODE_MODEL") or "deepseek-v4.1-flash",
                },
                "lmstudio": {
                    "baseUrl": "http://localhost:1234/v1",
                    "model": os.getenv("LMSTUDIO_MODEL") or "local-model",
                },
                "stackspot": {
                    "baseUrl": "https://genai-inference-app.stackspot.com",
                    "authUrl": "https://idm.stackspot.com",
                    "realm": os.getenv("STACKSPOT_REALM") or "stackspot-freemium",
                    "model": os.getenv("STACKSPOT_AGENT_ID") or "",
                    "knowledge": True,
                    "returnKs": False,
                },
            },
            "workspace": str(directory),
            "tools": {"confirm": ["shell", "write_file", "edit_file"]},
        }
    )
    click.echo(f"Config written to {path}")
    if provider_name == "stackspot":
        click.echo(
            "Set STACKSPOT_CLIENT_ID, STACKSPOT_CLIENT_SECRET and STACKSPOT_AGENT_ID, "
            "then run: ./run.sh doctor"
        )
    else:
        click.echo("Set OPENCODE_API_KEY or LMSTUDIO_MODEL, then run: ./run.sh doctor")


if __name__ == "__main__":
    cli()
