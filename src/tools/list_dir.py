from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.tools._fs import EXCLUDED_DIRS
from src.tools.base import ToolSpec


def handle_list_dir(
    path: str = ".",
    page: int = 1,
    page_size: int = 100,
    recursive: bool = False,
    max_depth: int = 3,
    extension: str | None = None,
    tree: bool = False,
) -> dict:

    root = Path(path).resolve()

    if not root.exists():

        return {
            "error": "path_not_found",
            "path": str(root),
        }

    if tree:

        tree_lines = []

        def walk(
            current: Path,
            prefix: str = "",
            depth: int = 0,
        ):

            if depth > max_depth:
                return

            try:
                children = sorted(
                    current.iterdir(),
                    key=lambda p: (
                        not p.is_dir(),
                        p.name.lower(),
                    ),
                )
            except PermissionError:
                return

            for idx, child in enumerate(children):

                if child.name in EXCLUDED_DIRS:
                    continue

                is_last = idx == len(children) - 1

                connector = "└── " if is_last else "├── "

                tree_lines.append(f"{prefix}{connector}{child.name}")

                if child.is_dir():

                    walk(
                        child,
                        prefix + ("    " if is_last else "│   "),
                        depth + 1,
                    )

        tree_lines.append(root.name)

        walk(root)

        return {
            "status": "success",
            "path": str(root),
            "max_depth": max_depth,
            "tree": tree_lines,
        }

    items: list[dict[str, Any]] = []

    if recursive:

        for current, dirs, files in os.walk(root, followlinks=False):
            # Prune dependency / cache directories so a recursive listing of a
            # project never pulls in node_modules, .git, .angular, etc.
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            relative_depth = len(Path(current).relative_to(root).parts)
            if relative_depth > max_depth:
                dirs[:] = []
                continue
            for name in dirs:
                items.append(
                    {
                        "name": name,
                        "path": str(Path(current) / name),
                        "type": "directory",
                        "size": None,
                    }
                )
            for name in files:
                if extension and Path(name).suffix.lower() != extension.lower():
                    continue
                file_path = Path(current) / name
                items.append(
                    {
                        "name": name,
                        "path": str(file_path),
                        "type": "file",
                        "size": file_path.stat().st_size,
                    }
                )

    else:

        for item in root.iterdir():

            if extension and item.is_file():

                if item.suffix.lower() != extension.lower():
                    continue

            items.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "type": ("directory" if item.is_dir() else "file"),
                    "size": (item.stat().st_size if item.is_file() else None),
                }
            )

    total = len(items)

    start = max(0, (page - 1) * page_size)
    end = start + page_size

    page_items = items[start:end]

    return {
        "status": "success",
        "path": str(root),
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": end < total,
        "items": page_items,
    }


MANUAL = (
    "list_dir: list files and directories inside a folder.\n"
    "Actions:\n"
    "  - list\n"
    "    Params:\n"
    '      path (str, default "."): folder to inspect.\n'
    "      page (int, default 1): result page.\n"
    "      page_size (int, default 100): items per page.\n"
    "      recursive (bool, default false): descend into subdirectories.\n"
    "      max_depth (int, default 3): max recursion depth when recursive is true.\n"
    '      extension (str, optional): filter files by suffix (e.g. ".py").\n'
    "      tree (bool, default false): return a tree view instead of a paginated list.\n"
    "    Returns: paginated items (name, path, type, size) or a tree."
)


SPEC = ToolSpec(
    name="list_dir",
    handlers={"list": handle_list_dir},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
