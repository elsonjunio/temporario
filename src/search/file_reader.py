from pathlib import Path
import re
from typing import Dict, Any, List

def _get_words_in_line(line: str) -> List[str]:
    """Helper to tokenize a line into words using \S+."""
    return re.findall(r'\S+', line)

def _count_total_words(file_path: Path) -> int:
    """First pass: Count total words in the file without loading everything into memory."""
    total_words = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_words += len(_get_words_in_line(line))
    except Exception:
        return 0
    return total_words

def handle_paginated_read(
    file_path: str,
    page: int = 1,
    page_size: int = 500,
    show_line_number: bool = False,
) -> Dict[str, Any]:
    """
    Reads a file and returns content paginated by word count while preserving formatting.
    Uses two-pass approach for memory efficiency and preserves internal/leading whitespace.
    """
    root = Path(file_path)

    if not root.exists():
        return {
            "error": "File not found",
            "path": str(root),
        }

    total_words = _count_total_words(root)

    if total_words == 0:
        return {
            "status": "success",
            "content": "",
            "total_words": 0,
            "total_pages": 1,
            "current_page_content": ""
        }

    total_pages = (total_words + page_size - 1) // page_size

    if page < 1 or page > total_pages:
        return {
            "status": "error",
            "message": f"Page must be between 1 and {total_pages}.",
            "total_words": total_words,
            "total_pages": total_pages
        }

    start_word_idx = (page - 1) * page_size
    end_word_idx = min(total_words, start_word_idx + page_size)

    formatted_lines: List[str] = []
    current_word_count = 0
    line_number = 0
    in_page_range = False

    try:
        with open(root, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                clean_line = line.rstrip('\n\r')
                
                spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', clean_line)]
                num_words_in_line = len(spans)

                # Check if this line contains words that fall into the range
                line_has_valid_words = False
                line_start_word_idx = current_word_count
                line_end_word_idx = current_word_count + num_words_in_line
                
                # Check for overlap between [line_start_word_idx, line_end_word_idx) and [start_word_idx, end_word_idx)
                if max(line_start_word_idx, start_word_idx) < min(line_end_word_idx, end_word_idx):
                    line_has_valid_words = True

                if line_has_valid_words:
                    in_page_range = True
                    # Find which words are in range
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
                    # If we are already in the page range and encounter an empty line, include it
                    if num_words_in_line == 0:
                        prefix = f"Line {line_number}: " if show_line_number else ""
                        formatted_lines.append(f"{prefix}{clean_line}")

                current_word_count += num_words_in_line
                
                # If we have passed the end of the page range, stop processing lines
                if current_word_count >= end_word_idx:
                    break

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading file: {str(e)}"
        }

    return {
        "status": "success",
        "total_words": total_words,
        "total_pages": total_pages,
        "current_page_content": "\n".join(formatted_lines),
        "word_count_per_page": page_size
    }
