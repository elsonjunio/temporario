from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.tools.base import ToolSpec

DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_LINES = 2000
_BINARY_SNIFF_SIZE = 8192


def _get_words_in_line(line: str) -> list[str]:
    """Helper to tokenize a line into words using \\S+."""
    return re.findall(r"\S+", line)


def _detect_encoding(path: Path) -> tuple[str, bool]:
    """Return (encoding, is_binary). Sniffs a NUL byte to flag binary files."""
    with open(path, "rb") as f:
        head = f.read(_BINARY_SNIFF_SIZE)

    if b"\x00" in head:
        return "", True

    try:
        head.decode("utf-8")
        return "utf-8", False
    except UnicodeDecodeError:
        return "latin-1", False


def _scan(path: Path, encoding: str) -> tuple[int, int]:
    """First pass: count total words and lines without loading the file."""
    total_words = 0
    total_lines = 0
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            total_lines += 1
            total_words += len(_get_words_in_line(line))
    return total_words, total_lines


def _read_line_range(
    path: Path,
    encoding: str,
    start_line: int,
    end_line: int,
    show_line_number: bool,
) -> str:
    lines: list[str] = []
    with open(path, "r", encoding=encoding) as f:
        for lineno, line in enumerate(f, 1):
            if lineno < start_line:
                continue
            if lineno > end_line:
                break
            clean_line = line.rstrip("\n\r")
            prefix = f"Line {lineno}: " if show_line_number else ""
            lines.append(f"{prefix}{clean_line}")
    return "\n".join(lines)


def handle_read_file(
    file_path: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    show_line_number: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
) -> dict[str, Any]:
    """Read a text file, word-paginated by default or by explicit line range.

    Detects binary files, falls back to latin-1 when utf-8 fails, and returns
    useful metadata (byte_size, encoding, total_lines) alongside the content.
    """
    root = Path(file_path)

    if not root.exists():
        return {
            "error": "File not found",
            "path": str(root),
        }

    if not root.is_file():
        return {
            "error": "Not a file",
            "path": str(root),
        }

    try:
        encoding, is_binary = _detect_encoding(root)
    except OSError as exc:
        return {"status": "error", "message": f"Error reading file: {exc}"}

    if is_binary:
        return {
            "status": "error",
            "message": "Binary file, not readable as text.",
            "path": str(root),
        }

    try:
        total_words, total_lines = _scan(root, encoding)
    except OSError as exc:
        return {"status": "error", "message": f"Error reading file: {exc}"}

    try:
        byte_size = root.stat().st_size
    except OSError:
        byte_size = None

    meta: dict[str, Any] = {
        "status": "success",
        "path": str(root),
        "byte_size": byte_size,
        "encoding": encoding,
        "total_lines": total_lines,
    }

    if start_line is not None:
        if start_line < 1:
            return {**meta, "status": "error", "message": "start_line must be >= 1."}
        if end_line is None:
            end_line = start_line
        if end_line < start_line:
            return {
                **meta,
                "status": "error",
                "message": "end_line must be >= start_line.",
            }

        selected = end_line - start_line + 1
        truncated = selected > max_lines
        if truncated:
            end_line = start_line + max_lines - 1

        try:
            content = _read_line_range(
                root, encoding, start_line, end_line, show_line_number
            )
        except OSError as exc:
            return {**meta, "status": "error", "message": f"Error reading file: {exc}"}

        return {
            **meta,
            "mode": "lines",
            "start_line": start_line,
            "end_line": end_line,
            "requested_end_line": (start_line + selected - 1) if truncated else None,
            "truncated": truncated,
            "max_lines": max_lines,
            "content": content,
        }

    if total_words == 0:
        return {
            **meta,
            "mode": "words",
            "total_words": 0,
            "total_pages": 1,
            "current_page_content": "",
        }

    total_pages = (total_words + page_size - 1) // page_size

    if page < 1 or page > total_pages:
        return {
            **meta,
            "status": "error",
            "message": f"Page must be between 1 and {total_pages}.",
            "total_words": total_words,
            "total_pages": total_pages,
        }

    start_word_idx = (page - 1) * page_size
    end_word_idx = min(total_words, start_word_idx + page_size)

    formatted_lines: list[str] = []
    current_word_count = 0
    line_number = 0
    in_page_range = False

    try:
        with open(root, "r", encoding=encoding) as f:
            for line in f:
                line_number += 1
                clean_line = line.rstrip("\n\r")

                spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", clean_line)]
                num_words_in_line = len(spans)

                line_has_valid_words = False
                line_start_word_idx = current_word_count
                line_end_word_idx = current_word_count + num_words_in_line

                if max(line_start_word_idx, start_word_idx) < min(
                    line_end_word_idx, end_word_idx
                ):
                    line_has_valid_words = True

                if line_has_valid_words:
                    in_page_range = True
                    valid_span_indices = []
                    for i, (s, e) in enumerate(spans):
                        word_idx = current_word_count + i
                        if start_word_idx <= word_idx < end_word_idx:
                            valid_span_indices.append(i)

                    if valid_span_indices:
                        i_first = valid_span_indices[0]
                        i_last = valid_span_indices[-1]
                        line_start = spans[i_first][0] if i_first > 0 else 0
                        line_end = spans[i_last][1]
                        content_to_show = clean_line[line_start:line_end]
                        prefix = f"Line {line_number}: " if show_line_number else ""
                        formatted_lines.append(f"{prefix}{content_to_show}")

                elif in_page_range:
                    if num_words_in_line == 0:
                        prefix = f"Line {line_number}: " if show_line_number else ""
                        formatted_lines.append(f"{prefix}{clean_line}")

                current_word_count += num_words_in_line

                if current_word_count >= end_word_idx:
                    break

    except OSError as exc:
        return {**meta, "status": "error", "message": f"Error reading file: {exc}"}

    return {
        **meta,
        "mode": "words",
        "total_words": total_words,
        "total_pages": total_pages,
        "current_page_content": "\n".join(formatted_lines),
        "word_count_per_page": page_size,
    }


MANUAL = (
    "read_file: read the content of a text file, word-paginated by default or\n"
    "by explicit line range. Detects binary files and reports metadata.\n"
    "Actions:\n"
    "  - read\n"
    "    Params:\n"
    "      file_path (str, required): absolute or relative path of the file.\n"
    "      page (int, default 1): page number to read (word pagination).\n"
    "      page_size (int, default 500): words per page.\n"
    "      show_line_number (bool, default false): prefix each line with its number.\n"
    "      start_line (int, optional): read a line range starting at this line\n"
    "        (takes precedence over page).\n"
    "      end_line (int, optional): last line of the range (defaults to start_line).\n"
    "      max_lines (int, default 2000): cap on lines returned in line mode.\n"
    "    Returns: content plus metadata (byte_size, encoding, total_lines,\n"
    "    total_words, total_pages). Binary files and missing paths return a\n"
    "    clean error instead of garbage."
)


SPEC = ToolSpec(
    name="read_file",
    handlers={"read": handle_read_file},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
