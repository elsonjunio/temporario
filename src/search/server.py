@mcp.tool()
def search_files(
    pattern: str,
    path: str = '.',
    page: int = 1,
    page_size: int = 50,
    recursive: bool = True,
    case_sensitive: bool = False,
) -> dict:
    """
    Search files and directories by name pattern.

    Supports shell-style wildcards:
    *.py
    *agent*
    config.*
    """
    return handle_search_files(
        pattern,
        path,
        page,
        page_size,
        recursive,
        case_sensitive,
    )


@mcp.tool()
def list_files(
    path: str = '.',
    page: int = 1,
    page_size: int = 100,
    recursive: bool = False,
    max_depth: int = 3,
    extension: str | None = None,
    tree: bool = False,
) -> dict:
    """
    List files and directories.

    Can return paginated results or a tree view.
    """
    return handle_list_files(
        path,
        page,
        page_size,
        recursive,
        max_depth,
        extension,
        tree,
    )
