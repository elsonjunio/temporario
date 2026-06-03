from pathlib import Path
from fnmatch import fnmatch


def handle_search_files(
    pattern: str,
    path: str = '.',
    page: int = 1,
    page_size: int = 50,
    recursive: bool = True,
    case_sensitive: bool = False,
) -> dict:

    root = Path(path).resolve()

    if not root.exists():
        return {
            'error': 'path_not_found',
            'path': str(root),
        }

    matches = []

    iterator = root.rglob('*') if recursive else root.iterdir()

    pattern_cmp = pattern if case_sensitive else pattern.lower()

    for item in iterator:

        name = item.name

        candidate = name if case_sensitive else name.lower()

        if fnmatch(candidate, pattern_cmp):

            matches.append(
                {
                    'name': item.name,
                    'path': str(item),
                    'type': ('directory' if item.is_dir() else 'file'),
                }
            )

    total = len(matches)

    start = max(0, (page - 1) * page_size)
    end = start + page_size

    page_items = matches[start:end]

    return {
        'path': str(root),
        'pattern': pattern,
        'page': page,
        'page_size': page_size,
        'total': total,
        'has_next': end < total,
        'items': page_items,
    }
