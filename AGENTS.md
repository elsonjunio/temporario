# Agent Instructions

## Architecture
This repository implements a registry for managing multiple MCP (Model and Context Protocol) providers.

- **Provider Discovery**: The `Registry` class in `src/registry.py` scans a target directory for subdirectories containing a `run.sh` (Linux) or `run.bat` (Windows) file.
- **MCP Tools**: The `src/search/server.py` module defines tools like `search_files` and `list_files`.

## Adding a New Provider
To add a new MCP provider:
1. Create a new directory within the providers root.
2. Add a `run.sh` (for Linux) or `run.bat` (for Windows) file to that directory.
3. Ensure the script is executable and starts the desired service.

## Development & Testing
- **Dependencies**: Requires `mcp` (see `requirements.txt`).
- **Environment**: Uses a Python virtual environment (`.venv`).
