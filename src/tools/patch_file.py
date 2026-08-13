from __future__ import annotations

import re

from src.tools._fs import atomic_write, is_binary, resolve_path
from src.tools.base import ToolSpec

DEFAULT_ENCODING = "utf-8"


def _parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
    if m is None:
        return None
    old_start = int(m.group(1))
    old_count = int(m.group(2) or 1)
    new_start = int(m.group(3))
    new_count = int(m.group(4) or 1)
    return old_start, old_count, new_start, new_count


def _parse_unified_diff(diff: str) -> list[dict]:
    """Parse a unified diff into hunks with old/new start and prefixed lines."""
    hunks: list[dict] = []
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        header = _parse_hunk_header(lines[i])
        if header is None:
            i += 1
            continue

        old_start, old_count, new_start, new_count = header
        i += 1
        hunk_lines: list[str] = []
        while i < len(lines) and _parse_hunk_header(lines[i]) is None:
            hunk_lines.append(lines[i])
            i += 1

        hunks.append(
            {
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "lines": hunk_lines,
            }
        )
    return hunks


def _hunk_sides(hunk: dict) -> tuple[list[str], list[str]]:
    old_side = [line[1:] for line in hunk["lines"] if line[:1] in (" ", "-")]
    new_side = [line[1:] for line in hunk["lines"] if line[:1] in (" ", "+")]
    return old_side, new_side


def _find_block(lines: list[str], block: list[str], preferred_start: int) -> int | None:
    if not block:
        return preferred_start if 0 <= preferred_start <= len(lines) else None
    n = len(block)
    preferred = preferred_start - 1
    if 0 <= preferred <= len(lines) - n and lines[preferred : preferred + n] == block:
        return preferred
    for i in range(len(lines) - n + 1):
        if lines[i : i + n] == block:
            return i
    return None


def _apply_hunks(lines: list[str], hunks: list[dict]) -> list[str]:
    """Apply hunks from last to first so earlier offsets stay valid."""
    for hunk in reversed(hunks):
        old_side, new_side = _hunk_sides(hunk)
        idx = _find_block(lines, old_side, hunk["old_start"])
        if idx is None:
            raise ValueError(
                f"hunk @ -{hunk['old_start']},{hunk['old_count']} "
                f"+{hunk['new_start']},{hunk['new_count']} @@ not found in file"
            )
        lines = lines[:idx] + new_side + lines[idx + len(old_side) :]
    return lines


def handle_apply_patch(
    file_path: str,
    diff: str,
    encoding: str = DEFAULT_ENCODING,
) -> dict:
    """Apply a unified diff to an existing file using only the stdlib."""
    root = resolve_path(file_path)

    if not root.exists():
        return {"error": "File not found", "path": str(root)}

    if not root.is_file():
        return {"error": "Not a file", "path": str(root)}

    if is_binary(root):
        return {"error": "Binary file, patches not supported", "path": str(root)}

    hunks = _parse_unified_diff(diff)
    if not hunks:
        return {
            "status": "error",
            "message": "No @@ hunks found in diff.",
        }

    try:
        content = root.read_text(encoding=encoding)
    except OSError as exc:
        return {"status": "error", "message": f"Error reading file: {exc}"}

    had_newline = content.endswith("\n")
    lines = content.split("\n")
    if had_newline:
        lines = lines[:-1]

    try:
        new_lines = _apply_hunks(lines, hunks)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "path": str(root)}

    result = "\n".join(new_lines)
    if had_newline and not result.endswith("\n"):
        result += "\n"
    if not had_newline and result.endswith("\n"):
        result = result[:-1]

    if result == content:
        return {
            "status": "success",
            "path": str(root),
            "mode": "unchanged",
            "hunks_applied": len(hunks),
        }

    try:
        atomic_write(root, result, encoding)
    except OSError as exc:
        return {"status": "error", "path": str(root), "message": str(exc)}

    return {
        "status": "success",
        "path": str(root),
        "mode": "patched",
        "hunks_applied": len(hunks),
    }


def handle_replace(
    file_path: str,
    old: str,
    new: str = "",
    replace_all: bool = False,
    encoding: str = DEFAULT_ENCODING,
) -> dict:
    """Find-and-replace an exact block of text with safety guards."""
    root = resolve_path(file_path)

    if not root.exists():
        return {"error": "File not found", "path": str(root)}

    if is_binary(root):
        return {"error": "Binary file, patches not supported", "path": str(root)}

    try:
        content = root.read_text(encoding=encoding)
    except OSError as exc:
        return {"status": "error", "message": f"Error reading file: {exc}"}

    if not old:
        return {"status": "error", "message": "old must not be empty."}

    occurrences = content.count(old)
    if occurrences == 0:
        return {
            "status": "error",
            "message": (
                "old text not found in file. Copy the 'old' text EXACTLY as it "
                "appears in the current file (re-read the file first); do not "
                "guess or paraphrase it. If you cannot produce an exact anchor, "
                "use write_file with rewrite:true to write the whole corrected "
                "file instead."
            ),
            "path": str(root),
        }

    if occurrences > 1 and not replace_all:
        return {
            "status": "error",
            "message": (
                f"old text found {occurrences} times; pass replace_all=True "
                "to replace every occurrence or provide more context."
            ),
            "occurrences": occurrences,
            "path": str(root),
        }

    new_content = (
        content.replace(old, new) if replace_all else content.replace(old, new, 1)
    )
    if new_content == content:
        return {
            "status": "success",
            "path": str(root),
            "mode": "unchanged",
            "replaced": 0,
        }

    try:
        atomic_write(root, new_content, encoding)
    except OSError as exc:
        return {"status": "error", "path": str(root), "message": str(exc)}

    return {
        "status": "success",
        "path": str(root),
        "mode": "replaced",
        "replaced": occurrences if replace_all else 1,
    }


MANUAL = (
    "patch_file: modify an existing text file by applying a unified diff or a\n"
    "precise find-and-replace. Both writes are atomic.\n"
    "Actions:\n"
    "  - apply\n"
    "    Params:\n"
    "      file_path (str, required): path of the file to modify.\n"
    "      diff (str, required): unified diff (git style) with one or more\n"
    '        hunks, e.g. "@@ -1,3 +1,3 @@" followed by context lines prefixed\n'
    "        with a space, removals with '-', additions with '+'.\n"
    "    Returns: status, mode (patched/unchanged) and hunks_applied.\n"
    "  - replace\n"
    "    Params:\n"
    "      file_path (str, required): path of the file to modify.\n"
    "      old (str, required): exact text block to find (must match once).\n"
    '      new (str, default ""): replacement text.\n'
    "      replace_all (bool, default false): replace every occurrence; when\n"
    "        false the old text must appear exactly once, else an error is\n"
    "        returned to avoid ambiguous edits.\n"
    "    Returns: status, mode and the number of replacements made.\n"
    "Notes: binary files and missing paths return a clean error."
)


SPEC = ToolSpec(
    name="patch_file",
    handlers={
        "apply": handle_apply_patch,
        "replace": handle_replace,
    },
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
