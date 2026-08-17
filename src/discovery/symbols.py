from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterator

from src.orchestrator.languages import detect_language
from src.tools._fs import is_binary

MAX_RESULTS = 50
MAX_SAMPLE_LINES = 3
MAX_SCAN_BYTES = 1_000_000
MAX_SCAN_LINES = 4000
_SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".bike"}

_DEF_LINE_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"\b(?:def|async def|class)\s+(\w+)",
        r"^\s*(\w+)\s*(?::\s*[\w\[\], .]+)?\s*=",
    ],
    "typescript": [
        r"\b(?:function|class|interface|type|enum)\s+(\w+)",
        r"\b(?:const|let|var)\s+(\w+)\s*(?:=|:)",
        r"\bexport\s+default\s+(?:function|class)\s+(\w+)",
    ],
    "javascript": [
        r"\b(?:function|class)\s+(\w+)",
        r"\b(?:const|let|var)\s+(\w+)\s*(?:=|:)",
        r"\bexport\s+default\s+(?:function|class)\s+(\w+)",
    ],
    "go": [
        r"\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\b",
        r"\btype\s+(\w+)\s+(?:struct|interface)",
        r"\b(?:const|var)\s+(\w+)\b",
    ],
    "rust": [
        r"\bfn\s+(\w+)\b",
        r"\b(?:struct|enum|trait|type|mod|const|static)\s+(\w+)\b",
    ],
    "java": [
        r"\b(?:class|interface|enum)\s+(\w+)\b",
        r"\b(?:public|private|protected|static|final|abstract)?\s*[\w<>,.?\[\]]+\s+(\w+)\s*\(",
    ],
    "kotlin": [
        r"\bfun\s+(\w+)\b",
        r"\b(?:class|interface|object|enum)\s+(\w+)\b",
        r"\b(?:val|var)\s+(\w+)\s*:",
    ],
    "c": [
        r"\b(?:int|void|char|float|double|long|short|unsigned|signed|struct|static|extern|const)\s+(\w+)\s*(?:=|;|\()",
        r"^#define\s+(\w+)",
    ],
    "cpp": [
        r"\b(?:class|struct|enum|namespace)\s+(\w+)\b",
        r"\b(?:int|void|char|float|double|long|short|unsigned|signed|static|extern|const|std::string|auto)\s+(\w+)\s*(?:=|;|\()",
    ],
    "csharp": [
        r"\b(?:class|struct|interface|enum|record|namespace)\s+(\w+)\b",
        r"\b(?:public|private|protected|internal|static|async)\s+[\w<>,.?\[\]]+\s+(\w+)\s*\(",
    ],
    "swift": [
        r"\bfunc\s+(\w+)\b",
        r"\b(?:class|struct|enum|protocol|extension)\s+(\w+)\b",
        r"\b(?:var|let)\s+(\w+)\s*(?::|=)",
    ],
    "ruby": [
        r"\bdef\s+(?:self\.)?(\w+)\b",
        r"\b(?:class|module)\s+(\w+)\b",
    ],
    "php": [
        r"\bfunction\s+(\w+)\b",
        r"\b(?:class|interface|trait|enum)\s+(\w+)\b",
        r"\bconst\s+(\w+)\b",
    ],
}

# NAME is replaced with the escaped symbol name. Conservative "definition-like"
# lines used when the language family has no dedicated patterns.
_COMMON_DEF_PATTERNS = [
    r"\b(?:def|class|function|func|fn)\s+NAME\b",
    r"\b(?:const|let|var|struct|enum|trait|interface|type|namespace|module)\s+NAME\b",
]

# Language -> regexes to extract import statements. The first capture group is
# the imported module specifier.
_IMPORT_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^\s*import\s+([\w.]+)",
        r"^\s*from\s+([\w.]+)\s+import",
    ],
    "typescript": [
        r"import\s+(?:[^'\"\n]+?\s+from\s+)?['\"]([^'\"]+)['\"]",
        r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "javascript": [
        r"import\s+(?:[^'\"\n]+?\s+from\s+)?['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "go": [
        r'^\s*import\s+"([^"]+)"',
        r'^\s*"([\w./-]+)"\s*$',
        r'\bimport\s+alias\s+"([^"]+)"',
    ],
    "rust": [
        r"\buse\s+([\w:]+)(?:\s*::\s*\{[^}]*\})?;",
        r"\buse\s+([\w:]+)\s*;",
    ],
    "c": [
        r'^\s*#include\s*[<"]([^>"]+)[>"]',
    ],
    "cpp": [
        r'^\s*#include\s*[<"]([^>"]+)[>"]',
    ],
    "java": [
        r"^\s*import\s+(?:static\s+)?([\w.]+);",
    ],
    "kotlin": [
        r"^\s*import\s+([\w.]+)",
    ],
    "csharp": [
        r"^\s*using\s+([\w.]+);",
    ],
    "swift": [
        r"^\s*import\s+(\w+)",
    ],
    "ruby": [
        r"^\s*require(?:_relative)?\s+['\"]([^'\"]+)['\"]",
    ],
    "php": [
        r"^\s*use\s+([\w\\\\]+)",
        r"^\s*require(?:nce)?(?:_once)?\s+['\"]([^'\"]+)['\"]",
    ],
}

# Comment prefixes used to detect a leading purpose block per language.
_COMMENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "python": ("#",),
    "typescript": ("//", "/*", "*"),
    "javascript": ("//", "/*", "*"),
    "go": ("//", "/*", "*"),
    "rust": ("//", "/*", "*", "///", "//!"),
    "java": ("//", "/*", "*"),
    "kotlin": ("//", "/*", "*"),
    "c": ("//", "/*", "*"),
    "cpp": ("//", "/*", "*"),
    "csharp": ("//", "/*", "*"),
    "swift": ("//", "/*", "*", "///"),
    "ruby": ("#", "=begin"),
    "php": ("//", "/*", "*", "#"),
}

_DEF_KIND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "class",
        re.compile(
            r"\b(?:class|interface|struct|enum|trait|protocol|namespace|module)\s"
        ),
    ),
    ("function", re.compile(r"\b(?:def|function|func|fn)\s")),
]


def resolve(root: str | Path, path: str | None = None) -> Path:
    """Resolve ``path`` against ``root``; None or empty means the root."""
    root = Path(root).resolve()
    if not path:
        return root
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _read_text(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        if is_binary(path):
            return None
        if path.stat().st_size > MAX_SCAN_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def iter_source_files(
    root: Path,
    include: str | None = None,
) -> Iterator[Path]:
    """Yield source files under ``root`` (skipping dependency dirs)."""
    include_patterns = (
        [p.strip() for p in include.split(",") if p.strip()] if include else []
    )
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        try:
            rel = item.relative_to(root)
        except ValueError:
            continue
        if any(part in _SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if include_patterns and not any(
            item.match(pat) or item.name == pat for pat in include_patterns
        ):
            continue
        yield item


def _py_module_name(root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.stem
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts) if parts else path.stem


def _language_name(root: Path, path: Path) -> str | None:
    profile = detect_language(path)
    if profile is None:
        return None
    # JS files in a TS project are still analyzed with JS patterns; the
    # language name comes from the extension, which is what we want.
    return profile.name


def _def_kind(line: str) -> str:
    for kind, pattern in _DEF_KIND_PATTERNS:
        if pattern.search(line):
            return kind
    return "definition"


def _regex_defs(text: str, name: str, language: str | None) -> list[tuple[int, str]]:
    patterns = _COMMON_DEF_PATTERNS
    if language:
        patterns = patterns + [
            p.replace("NAME", re.escape(name))
            for p in _DEF_LINE_PATTERNS.get(language, [])
        ]
    patterns = [p.replace("NAME", re.escape(name)) for p in patterns]
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if lineno > MAX_SCAN_LINES:
            break
        for pattern in patterns:
            if re.search(pattern, line):
                out.append((lineno, _def_kind(line)))
                break
    return out


def _py_defs(text: str, name: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                out.append((node.lineno, "function"))
        elif isinstance(node, ast.ClassDef):
            if node.name == name:
                out.append((node.lineno, "class"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    out.append((node.lineno, "definition"))
    return out


def find_definition(
    root: str | Path,
    name: str,
    path: str | None = None,
    include: str | None = None,
) -> dict[str, Any]:
    if not name or not name.strip():
        return {"status": "error", "message": "name is required"}
    root_p = Path(root).resolve()
    target = resolve(root_p, path)
    name = name.strip()

    definitions: list[dict[str, Any]] = []
    files = [target] if target.is_file() else list(iter_source_files(target, include))

    for file_path in files:
        text = _read_text(file_path)
        if text is None:
            continue
        language = _language_name(root_p, file_path)
        if language == "python":
            hits = _py_defs(text, name)
        else:
            hits = _regex_defs(text, name, language)
        for lineno, kind in hits:
            definitions.append(
                {"file": str(file_path.resolve()), "line": lineno, "kind": kind}
            )
            if len(definitions) >= MAX_RESULTS:
                break
        if len(definitions) >= MAX_RESULTS:
            break

    return {
        "status": "success",
        "name": name,
        "path": str(target),
        "count": len(definitions),
        "truncated": len(definitions) >= MAX_RESULTS,
        "definitions": definitions,
    }


def find_references(
    root: str | Path,
    name: str,
    path: str | None = None,
    include: str | None = None,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    if not name or not name.strip():
        return {"status": "error", "message": "name is required"}
    root_p = Path(root).resolve()
    target = resolve(root_p, path)
    name = name.strip()
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(rf"\b{re.escape(name)}\b", flags)

    files: list[dict[str, Any]] = []
    total = 0
    truncated = False
    candidates = (
        [target] if target.is_file() else list(iter_source_files(target, include))
    )

    for file_path in candidates:
        text = _read_text(file_path)
        if text is None:
            continue
        sample: list[int] = []
        count = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            if lineno > MAX_SCAN_LINES:
                break
            if pattern.search(line):
                count += 1
                if len(sample) < MAX_SAMPLE_LINES:
                    sample.append(lineno)
        if count:
            total += 1
            files.append(
                {
                    "file": str(file_path.resolve()),
                    "count": count,
                    "lines": sample,
                }
            )
            if total >= MAX_RESULTS:
                truncated = True
                break

    return {
        "status": "success",
        "name": name,
        "path": str(target),
        "total_files": total,
        "truncated": truncated,
        "files": files,
    }


def find_symbol(
    root: str | Path,
    name: str,
    path: str | None = None,
    include: str | None = None,
) -> dict[str, Any]:
    if not name or not name.strip():
        return {"status": "error", "message": "name is required"}
    root_p = Path(root).resolve()
    target = resolve(root_p, path)
    name = name.strip()
    flags = re.IGNORECASE
    pattern = re.compile(rf"\b{re.escape(name)}\b", flags)

    definitions = find_definition(root_p, name, path=str(target), include=include)
    def_files = {(d["file"], d["line"]) for d in definitions.get("definitions", [])}

    usage_files: list[dict[str, Any]] = []
    truncated = False
    total = 0
    for file_path in (
        [target] if target.is_file() else list(iter_source_files(target, include))
    ):
        text = _read_text(file_path)
        if text is None:
            continue
        sample: list[int] = []
        count = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            if lineno > MAX_SCAN_LINES:
                break
            if pattern.search(line):
                count += 1
                if (str(file_path.resolve()), lineno) not in def_files and len(
                    sample
                ) < MAX_SAMPLE_LINES:
                    sample.append(lineno)
        if count:
            total += 1
            usage_files.append(
                {
                    "file": str(file_path.resolve()),
                    "count": count,
                    "sample_lines": sample,
                }
            )
            if total >= MAX_RESULTS:
                truncated = True
                break

    return {
        "status": "success",
        "name": name,
        "path": str(target),
        "definitions": definitions.get("definitions", []),
        "usage_files": usage_files,
        "total_usage_files": total,
        "truncated": truncated,
    }


def _py_imports(text: str) -> list[tuple[str, int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno, alias.asname or ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported = ",".join(a.name for a in node.names)
            out.append((module, node.lineno, imported))
    return out


def find_imports(root: str | Path, path: str) -> dict[str, Any]:
    root_p = Path(root).resolve()
    target = resolve(root_p, path)
    if not target.is_file():
        return {"status": "error", "message": "file not found", "path": str(target)}
    text = _read_text(target)
    if text is None:
        return {
            "status": "error",
            "message": "unreadable or binary file",
            "path": str(target),
        }

    language = _language_name(root_p, target)
    imports: list[dict[str, Any]] = []
    if language == "python":
        for module, lineno, alias in _py_imports(text):
            local = (root_p / (module.replace(".", "/") + ".py")).exists() or (
                root_p / module.replace(".", "/") / "__init__.py"
            ).exists()
            imports.append(
                {"module": module, "line": lineno, "alias": alias, "local": local}
            )
    elif language and language in _IMPORT_PATTERNS:
        seen: set[str] = set()
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in _IMPORT_PATTERNS[language]:
                for match in re.finditer(pattern, line):
                    module = match.group(1)
                    if module in seen:
                        continue
                    seen.add(module)
                    local = (
                        (root_p / module).exists()
                        or (root_p / f"{module}.py").exists()
                        or (root_p / module / "__init__.py").exists()
                    )
                    imports.append(
                        {"module": module, "line": lineno, "alias": "", "local": local}
                    )

    return {
        "status": "success",
        "path": str(target),
        "language": language,
        "count": len(imports),
        "truncated": len(imports) >= MAX_RESULTS,
        "imports": imports[:MAX_RESULTS],
    }


def find_importers(
    root: str | Path,
    path: str,
    include: str | None = None,
) -> dict[str, Any]:
    root_p = Path(root).resolve()
    target = resolve(root_p, path)
    if not target.is_file():
        return {"status": "error", "message": "file not found", "path": str(target)}

    module_key = _py_module_name(root_p, target)
    stem = target.stem
    pattern = re.compile(
        rf"(?:{re.escape(module_key)}|\b{re.escape(stem)}\b)",
        re.IGNORECASE,
    )

    importers: list[dict[str, Any]] = []
    truncated = False
    for file_path in iter_source_files(root_p, include):
        if file_path.resolve() == target.resolve():
            continue
        text = _read_text(file_path)
        if text is None:
            continue
        sample: list[int] = []
        count = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            if lineno > MAX_SCAN_LINES:
                break
            if pattern.search(line):
                count += 1
                if len(sample) < MAX_SAMPLE_LINES:
                    sample.append(lineno)
        if count:
            importers.append(
                {
                    "file": str(file_path.resolve()),
                    "count": count,
                    "lines": sample,
                }
            )
            if len(importers) >= MAX_RESULTS:
                truncated = True
                break

    return {
        "status": "success",
        "path": str(target),
        "module_key": module_key,
        "total_files": len(importers),
        "truncated": truncated,
        "files": importers,
    }


def _purpose(text: str, language: str | None) -> str:
    if language == "python":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            pass
        else:
            doc = ast.get_docstring(tree)
            if doc:
                return " ".join(doc.split())[:200]
    prefixes = _COMMENT_PREFIXES.get(language or "", ("#", "//"))
    lines: list[str] = []
    for line in text.splitlines()[:60]:
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if any(stripped.startswith(p) for p in prefixes):
            cleaned = stripped.lstrip("*#//- ")
            if cleaned:
                lines.append(cleaned)
            if len(lines) >= 3:
                break
        else:
            break
    return " ".join(lines)[:200]


def _top_level_symbols(
    path: Path, text: str, language: str | None
) -> list[dict[str, Any]]:
    if language == "python":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        out: list[dict[str, Any]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(
                    {"symbol": node.name, "line": node.lineno, "kind": "function"}
                )
            elif isinstance(node, ast.ClassDef):
                out.append({"symbol": node.name, "line": node.lineno, "kind": "class"})
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name):
                        out.append(
                            {"symbol": t.id, "line": node.lineno, "kind": "definition"}
                        )
        return out[:MAX_RESULTS]

    patterns = _DEF_LINE_PATTERNS.get(language or "", [])
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if lineno > MAX_SCAN_LINES:
            break
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                symbol = match.group(1)
                if symbol not in {o["symbol"] for o in out}:
                    out.append(
                        {"symbol": symbol, "line": lineno, "kind": _def_kind(line)}
                    )
                break
        if len(out) >= MAX_RESULTS:
            break
    return out


def inspect_file(root: str | Path, path: str) -> dict[str, Any]:
    root_p = Path(root).resolve()
    target = resolve(root_p, path)
    if not target.is_file():
        return {"status": "error", "message": "file not found", "path": str(target)}
    text = _read_text(target)
    if text is None:
        return {
            "status": "error",
            "message": "unreadable or binary file",
            "path": str(target),
        }

    language = _language_name(root_p, target)
    imports = find_imports(root_p, str(target))
    lines = len(text.splitlines())
    try:
        size = target.stat().st_size
    except OSError:
        size = None

    return {
        "status": "success",
        "path": str(target),
        "language": language,
        "lines": lines,
        "size": size,
        "purpose": _purpose(text, language),
        "symbols": _top_level_symbols(target, text, language),
        "imports": imports.get("imports", []),
    }


__all__ = [
    "find_definition",
    "find_importers",
    "find_imports",
    "find_references",
    "find_symbol",
    "inspect_file",
    "iter_source_files",
    "resolve",
]
