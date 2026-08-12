from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from src.tools.base import ToolSpec
from src.tools._fs import is_binary

DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_MATCHES_PER_FILE = 50
DEFAULT_MAX_FILE_SIZE = 1_000_000
DEFAULT_EXCLUDE_DIRS = {".git", "node_modules", ".venv", "__pycache__"}
_BINARY_SNIFF_SIZE = 8192


def _detect_encoding(path: Path) -> str:
    """Return a text encoding for ``path``: utf-8, falling back to latin-1."""
    try:
        with open(path, "rb") as f:
            head = f.read(_BINARY_SNIFF_SIZE)
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"
    except OSError:
        return "utf-8"


def _excluded(relative: Path, exclude: set[str]) -> bool:
    for part in relative.parts[:-1]:
        if part in exclude:
            return True
    return False


def _match_lines(
    path: Path,
    compiled: re.Pattern[str],
    max_matches: int,
    context_lines: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (matches, truncated) for one file. Each match carries the line
    number, the matched line and optional surrounding context."""
    try:
        encoding = _detect_encoding(path)
        if is_binary(path):
            return [], False
        if path.stat().st_size > DEFAULT_MAX_FILE_SIZE:
            return [], False
        lines = path.read_text(encoding=encoding).splitlines()
    except OSError:
        return [], False

    matches: list[dict[str, Any]] = []
    truncated = False

    for lineno, line in enumerate(lines, 1):
        if not compiled.search(line):
            continue

        context = lines[max(0, lineno - 1 - context_lines) : lineno + context_lines]
        matches.append(
            {
                "line": lineno,
                "content": line,
                "context": context,
            }
        )
        if len(matches) >= max_matches:
            truncated = True
            break

    return matches, truncated


def handle_grep_files(
    pattern: str,
    path: str = ".",
    include: str | None = None,
    exclude: str | None = None,
    case_sensitive: bool = False,
    context_lines: int = 0,
    max_matches_per_file: int = DEFAULT_MAX_MATCHES_PER_FILE,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """Search file contents with a regex, grep-style.

    Skips binary files, files larger than ``max_file_size``, and common
    dependency directories by default. Matches are paginated and capped per
    file.
    """
    if not pattern:
        return {"error": "invalid_arguments", "message": "pattern must not be empty"}

    root = Path(path).expanduser().resolve()

    if not root.exists():
        return {"error": "path_not_found", "path": str(root)}

    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        return {
            "status": "error",
            "message": f"Invalid regex pattern: {exc}",
            "pattern": pattern,
        }

    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    if exclude:
        exclude_set.update(ex.strip() for ex in exclude.split(",") if ex.strip())

    if include is not None and include:
        include_patterns = [inc.strip() for inc in include.split(",") if inc.strip()]
    else:
        include_patterns = []

    file_matches: list[dict[str, Any]] = []
    total_matches = 0

    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if _excluded(item.relative_to(root), exclude_set):
            continue
        if include_patterns and not any(
            fnmatch.fnmatch(item.name, pat) for pat in include_patterns
        ):
            continue

        matches, truncated = _match_lines(
            item, compiled, max_matches_per_file, context_lines
        )
        if matches:
            total_matches += len(matches)
            file_matches.append(
                {
                    "file": item.name,
                    "path": str(item),
                    "matches": matches,
                    "truncated": truncated,
                }
            )

    total_files = len(file_matches)
    start = max(0, (page - 1) * page_size)
    end = start + page_size

    return {
        "status": "success",
        "path": str(root),
        "pattern": pattern,
        "page": page,
        "page_size": page_size,
        "total_files": total_files,
        "total_matches": total_matches,
        "has_next": end < total_files,
        "items": file_matches[start:end],
    }


MANUAL = (
    "grep_files: search file contents with a regex, grep-style.\n"
    "Use this to find where a string or pattern appears in source code.\n"
    "Search is case-insensitive by default, recurses, skips binary files and\n"
    "large files, and ignores .git, node_modules, .venv and __pycache__.\n"
    "Actions:\n"
    "  - search\n"
    "    Params:\n"
    "      pattern (str, required): regular expression to search for.\n"
    "      path (str, default '.'): folder to search in.\n"
    "      include (str, optional): comma-separated filename globs to restrict,\n"
    "        e.g. '*.py' or '*.py,*.js'.\n"
    "      exclude (str, optional): comma-separated extra directories/globs to\n"
    "        skip on top of the defaults.\n"
    "      case_sensitive (bool, default false): match case exactly.\n"
    "      context_lines (int, default 0): lines of context around each match.\n"
    "      max_matches_per_file (int, default 50): cap on matches per file.\n"
    "      page (int, default 1): result page (pages of files with matches).\n"
    "      page_size (int, default 20): files per page.\n"
    "    Returns: paginated file matches with line numbers, matched lines and\n"
    "    optional context, plus total_files and total_matches."
)


SPEC = ToolSpec(
    name="grep_files",
    handlers={"search": handle_grep_files},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
