from pathlib import Path


def handle_list_files(
    path: str = '.',
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
            'error': 'path_not_found',
            'path': str(root),
        }

    if tree:

        tree_lines = []

        def walk(
            current: Path,
            prefix: str = '',
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

                is_last = idx == len(children) - 1

                connector = '└── ' if is_last else '├── '

                tree_lines.append(f'{prefix}{connector}{child.name}')

                if child.is_dir():

                    walk(
                        child,
                        prefix + ('    ' if is_last else '│   '),
                        depth + 1,
                    )

        tree_lines.append(root.name)

        walk(root)

        return {
            'path': str(root),
            'max_depth': max_depth,
            'tree': tree_lines,
        }

    items = []

    if recursive:

        for item in root.rglob('*'):

            relative_depth = len(item.relative_to(root).parts)

            if relative_depth > max_depth:
                continue

            if extension and item.is_file():

                if item.suffix.lower() != extension.lower():
                    continue

            items.append(
                {
                    'name': item.name,
                    'path': str(item),
                    'type': ('directory' if item.is_dir() else 'file'),
                    'size': (item.stat().st_size if item.is_file() else None),
                }
            )

    else:

        for item in root.iterdir():

            if extension and item.is_file():

                if item.suffix.lower() != extension.lower():
                    continue

            items.append(
                {
                    'name': item.name,
                    'path': str(item),
                    'type': ('directory' if item.is_dir() else 'file'),
                    'size': (item.stat().st_size if item.is_file() else None),
                }
            )

    total = len(items)

    start = max(0, (page - 1) * page_size)
    end = start + page_size

    page_items = items[start:end]

    return {
        'path': str(root),
        'page': page,
        'page_size': page_size,
        'total': total,
        'has_next': end < total,
        'items': page_items,
    }
