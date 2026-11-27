"""Entry point: the agent-mini CLI with the OpenCode Zen and LM Studio providers."""

from __future__ import annotations

import os
from pathlib import Path

import click

from src.providers import register

# Must run before importing agent_mini.cli, which copies PROVIDERS at import time.
register()

from agent_mini.cli import cli  # noqa: E402
from agent_mini.config import agent_home, config_file, save_config  # noqa: E402

PROVIDER_NAMES = ("opencode", "lmstudio")


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
    help="Workspace directory (default: <agent-mini home>/workspace).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def setup(provider_name: str, workspace: str | None, force: bool) -> None:
    """Write a config.json configured for the custom providers."""
    path = config_file()
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists; use --force to overwrite it.")

    directory = Path(workspace).expanduser() if workspace else agent_home() / "workspace"
    directory.mkdir(parents=True, exist_ok=True)

    save_config(
        {
            "provider": provider_name,
            "providers": {
                "opencode": {
                    "baseUrl": "https://opencode.ai/zen/v1",
                    "model": os.getenv("OPENCODE_MODEL") or "big-pickle",
                },
                "lmstudio": {
                    "baseUrl": "http://localhost:1234/v1",
                    "model": os.getenv("LMSTUDIO_MODEL") or "local-model",
                },
            },
            "workspace": str(directory),
            "tools": {"confirm": ["shell", "write_file", "edit_file"]},
        }
    )
    click.echo(f"Config written to {path}")
    click.echo("Set OPENCODE_API_KEY and/or LMSTUDIO_MODEL, then run: ./run.sh doctor")


if __name__ == "__main__":
    cli()
