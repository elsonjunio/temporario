from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from src.tools.base import ToolSpec


def handle_search_files(
    pattern: str,
    path: str = ".",
    page: int = 1,
    page_size: int = 50,
    recursive: bool = True,
    case_sensitive: bool = False,
) -> dict:

    root = Path(path).resolve()

    if not root.exists():
        return {
            "error": "path_not_found",
            "path": str(root),
        }

    matches = []

    iterator = root.rglob("*") if recursive else root.iterdir()

    pattern_cmp = pattern if case_sensitive else pattern.lower()

    for item in iterator:

        name = item.name

        candidate = name if case_sensitive else name.lower()

        if fnmatch(candidate, pattern_cmp):

            matches.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "type": ("directory" if item.is_dir() else "file"),
                }
            )

    total = len(matches)

    start = max(0, (page - 1) * page_size)
    end = start + page_size

    page_items = matches[start:end]

    return {
        "status": "success",
        "path": str(root),
        "pattern": pattern,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": end < total,
        "items": page_items,
    }


MANUAL = (
    "search_files: search files and directories by name pattern.\n"
    "Actions:\n"
    "  - search\n"
    "    Params:\n"
    '      pattern (str, required): shell-style wildcard, e.g. "*.py", "*agent*".\n'
    '      path (str, default "."): folder to search in.\n'
    "      page (int, default 1): result page.\n"
    "      page_size (int, default 50): items per page.\n"
    "      recursive (bool, default true): search subdirectories.\n"
    "      case_sensitive (bool, default false): match case exactly.\n"
    "    Returns: paginated matches (name, path, type)."
)


SPEC = ToolSpec(
    name="search_files",
    handlers={"search": handle_search_files},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
