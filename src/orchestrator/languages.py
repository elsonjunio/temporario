from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class LanguageProfile:
    """Conventions for one programming language: how to detect it, where its
    tests live, how test files are named, and how to run them."""

    name: str
    extensions: tuple[str, ...]
    test_dirs: tuple[str, ...] = ()
    test_globs: tuple[str, ...] = ()
    # Patterns applied to the target stem to recognize a matching test file.
    # "{stem}" is replaced with the target filename stem (case preserved);
    # matching is done case-insensitively to accommodate Foo -> FooTest.
    test_patterns: tuple[str, ...] = ()
    # Marker files that indicate a configured test/build toolchain.
    tool_markers: tuple[str, ...] = ()
    # Framework shown in recommendations when no toolchain is configured.
    framework: str = ""
    # Substring expected in command output on success.
    success_marker: str = ""
    # Command template; {files} -> space-joined test files, {runner} -> the
    # detected runner executable, {test} -> a single proposed test path.
    command_template: str = ""
    # Command used when a toolchain is configured but nothing covers the target.
    suggested_template: str = ""
    # Extension family searched for cross-references (defaults to own extensions).
    reference_extensions: tuple[str, ...] = field(default_factory=tuple)


# Built-in runner detection, keyed by marker file basename.
_RUNNERS: dict[str, Callable[[Path], str | None]] = {}


def _register_runner(*markers: str) -> Callable[[Callable[[Path], str | None]], None]:
    def decorator(fn: Callable[[Path], str | None]) -> None:
        for marker in markers:
            _RUNNERS[marker] = fn
        return None

    return decorator


@_register_runner("package.json")
def _npm_runner(root: Path) -> str | None:
    pkg = root / "package.json"
    if not pkg.exists():
        return None
    text = pkg.read_text(encoding="utf-8", errors="replace")
    if '"vitest"' in text:
        return "npx vitest"
    if '"jest"' in text or '"jest-cli"' in text:
        return "npx jest"
    if '"mocha"' in text:
        return "npx mocha"
    if re.search(r'"test"\s*:', text):
        return "npm test"
    return None


@_register_runner(
    "pom.xml", "build.gradle", "build.gradle.kts", "gradlew", "gradle.properties"
)
def _jvm_runner(root: Path) -> str | None:
    if (root / "gradlew").exists():
        return "./gradlew test"
    if (root / "pom.xml").exists():
        return "mvn test"
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        return "gradle test"
    return None


@_register_runner("go.mod")
def _go_runner(root: Path) -> str:
    return "go test ./..."


@_register_runner("Cargo.toml")
def _rust_runner(root: Path) -> str:
    return "cargo test"


@_register_runner("Gemfile", "Rakefile")
def _ruby_runner(root: Path) -> str | None:
    if (root / "Gemfile").exists():
        gemfile = (root / "Gemfile").read_text(encoding="utf-8", errors="replace")
        if "rspec" in gemfile:
            return "bundle exec rspec"
        return "bundle exec rake test"
    if (root / "Rakefile").exists():
        return "rake test"
    return None


@_register_runner("composer.json")
def _php_runner(root: Path) -> str | None:
    if (root / "vendor" / "bin" / "phpunit").exists():
        return "vendor/bin/phpunit"
    composer = root / "composer.json"
    if composer.exists():
        text = composer.read_text(encoding="utf-8", errors="replace")
        if '"test"' in text:
            return "composer test"
    return "phpunit"


@_register_runner(
    "pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "requirements.txt"
)
def _python_runner(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        if "[tool.pytest" in text or "pytest" in text:
            return "python -m pytest"
    for marker in ("pytest.ini", "setup.cfg", "tox.ini"):
        if (root / marker).exists():
            return "python -m pytest"
    for req in [root / "requirements.txt", root / "requirements-dev.txt"] + list(
        root.rglob("requirements*.txt")
    ):
        if not req.exists():
            continue
        text = req.read_text(encoding="utf-8", errors="replace")
        if re.search(r"(?m)^\s*pytest", text):
            return "python -m pytest"
    return None


def _find_marker(root: Path, name: str) -> Path | None:
    marker = root / name
    if marker.exists():
        return marker
    for match in root.rglob(name):
        return match
    return None


def _has_sln_or_csproj(root: Path) -> bool:
    for pattern in ("*.sln", "*.csproj", "Directory.Build.props"):
        if next(root.rglob(pattern), None) is not None:
            return True
    return False


@_register_runner("CMakeLists.txt", "Makefile")
def _c_runner(root: Path) -> str | None:
    if (root / "CMakeLists.txt").exists():
        return "cmake --build build && ctest --test-dir build"
    if (root / "Makefile").exists():
        return "make test"
    return None


def _runner_command(profile: LanguageProfile, root: Path) -> str | None:
    for marker in profile.tool_markers:
        fn = _RUNNERS.get(marker)
        if fn is None:
            continue
        if marker == "*.sln" and not _has_sln_or_csproj(root):
            continue
        if _find_marker(root, marker) is not None:
            cmd = fn(root)
            if cmd:
                return cmd
    if profile is PYTHON:
        venv = root / ".venv" / "bin" / "python"
        python = str(venv) if venv.exists() else "python"
        return f"{python} -m unittest"
    return None


PYTHON = LanguageProfile(
    name="python",
    extensions=(".py",),
    test_dirs=("tests", "test"),
    test_globs=("test_*.py", "*_test.py"),
    test_patterns=("test_{stem}.py", "{stem}_test.py"),
    tool_markers=(
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
        "requirements.txt",
    ),
    framework="pytest",
    success_marker="passed",
    command_template="{runner} {files}",
    suggested_template="{runner} {test}",
)

TYPESCRIPT = LanguageProfile(
    name="typescript",
    extensions=(".ts", ".tsx", ".mts", ".cts"),
    test_dirs=("__tests__", "test", "tests"),
    test_globs=("*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx"),
    test_patterns=(
        "{stem}.test.ts",
        "{stem}.spec.ts",
        "{stem}.test.tsx",
        "{stem}.spec.tsx",
    ),
    tool_markers=("package.json",),
    framework="jest",
    success_marker="PASS",
    command_template="{runner} {files}",
    suggested_template="{runner} {test}",
    reference_extensions=(".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"),
)

JAVASCRIPT = LanguageProfile(
    name="javascript",
    extensions=(".js", ".jsx", ".mjs", ".cjs"),
    test_dirs=("__tests__", "test", "tests"),
    test_globs=("*.test.js", "*.spec.js", "*.test.jsx", "*.spec.jsx", "*_test.js"),
    test_patterns=(
        "{stem}.test.js",
        "{stem}.spec.js",
        "{stem}.test.jsx",
        "{stem}.spec.jsx",
    ),
    tool_markers=("package.json",),
    framework="jest",
    success_marker="PASS",
    command_template="{runner} {files}",
    suggested_template="{runner} {test}",
    reference_extensions=(".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"),
)

JAVA = LanguageProfile(
    name="java",
    extensions=(".java",),
    test_dirs=("src/test/java", "test", "tests"),
    test_globs=("*Test.java", "Test*.java"),
    test_patterns=("{stem}Test.java", "Test{stem}.java"),
    tool_markers=("pom.xml", "build.gradle", "build.gradle.kts", "gradlew"),
    framework="JUnit (mvn test / gradle test)",
    success_marker="BUILD SUCCESS",
    command_template="{runner} -Dtest={class}",
    suggested_template="{runner} -Dtest={class}",
)

KOTLIN = LanguageProfile(
    name="kotlin",
    extensions=(".kt",),
    test_dirs=("src/test/kotlin", "test", "tests"),
    test_globs=("*Test.kt",),
    test_patterns=("{stem}Test.kt",),
    tool_markers=("pom.xml", "build.gradle", "build.gradle.kts", "gradlew"),
    framework="kotlin.test (mvn test / gradle test)",
    success_marker="BUILD SUCCESS",
    command_template="{runner} -Dtest={class}",
    suggested_template="{runner} -Dtest={class}",
    reference_extensions=(".kt", ".kts", ".java"),
)

C = LanguageProfile(
    name="c",
    extensions=(".c", ".h"),
    test_dirs=("tests", "test"),
    test_globs=("test_*.c", "*_test.c"),
    test_patterns=("test_{stem}.c", "{stem}_test.c"),
    tool_markers=("CMakeLists.txt", "Makefile"),
    framework="CTest / assert harness",
    success_marker="100% tests passed",
    command_template="{runner}",
    suggested_template="cc {test} {source} -o /tmp/{stem}_test && /tmp/{stem}_test",
    reference_extensions=(".c", ".h", ".cpp", ".cc", ".hpp"),
)

CPP = LanguageProfile(
    name="cpp",
    extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hxx"),
    test_dirs=("tests", "test"),
    test_globs=("test_*.cpp", "*_test.cpp", "test_*.cc", "*_test.cc"),
    test_patterns=(
        "test_{stem}.cpp",
        "{stem}_test.cpp",
        "test_{stem}.cc",
        "{stem}_test.cc",
    ),
    tool_markers=("CMakeLists.txt", "Makefile"),
    framework="CTest / GTest",
    success_marker="100% tests passed",
    command_template="{runner}",
    suggested_template="g++ {test} {source} -lgtest -pthread -o /tmp/{stem}_test && /tmp/{stem}_test",
    reference_extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".c", ".h"),
)

GO = LanguageProfile(
    name="go",
    extensions=(".go",),
    test_dirs=("",),
    test_globs=("*_test.go",),
    test_patterns=("{stem}_test.go",),
    tool_markers=("go.mod",),
    framework="go test",
    success_marker="ok",
    command_template="{runner} {dir}",
    suggested_template="go test ./{dir}",
)

RUST = LanguageProfile(
    name="rust",
    extensions=(".rs",),
    test_dirs=("tests",),
    test_globs=("*_test.rs", "*.rs"),
    test_patterns=("{stem}_test.rs", "test_{stem}.rs"),
    tool_markers=("Cargo.toml",),
    framework="cargo test",
    success_marker="test result: ok",
    command_template="{runner}",
    suggested_template="{runner}",
)

RUBY = LanguageProfile(
    name="ruby",
    extensions=(".rb",),
    test_dirs=("test", "spec"),
    test_globs=("*_test.rb", "*_spec.rb"),
    test_patterns=("{stem}_test.rb", "{stem}_spec.rb"),
    tool_markers=("Gemfile", "Rakefile"),
    framework="minitest / rspec",
    success_marker="0 failures",
    command_template="{runner} {files}",
    suggested_template="{runner} {test}",
)

PHP = LanguageProfile(
    name="php",
    extensions=(".php",),
    test_dirs=("tests", "test"),
    test_globs=("*Test.php",),
    test_patterns=("{stem}Test.php",),
    tool_markers=("composer.json",),
    framework="PHPUnit",
    success_marker="OK",
    command_template="{runner} {files}",
    suggested_template="{runner} {test}",
)

CSHARP = LanguageProfile(
    name="csharp",
    extensions=(".cs",),
    test_dirs=("",),
    test_globs=("*Tests.cs",),
    test_patterns=("{stem}Tests.cs",),
    tool_markers=("*.sln",),
    framework="xUnit / NUnit (dotnet test)",
    success_marker="Passed!",
    command_template="{runner}",
    suggested_template="{runner}",
)

SWIFT = LanguageProfile(
    name="swift",
    extensions=(".swift",),
    test_dirs=("",),
    test_globs=("*Tests.swift",),
    test_patterns=("{stem}Tests.swift",),
    tool_markers=("Package.swift",),
    framework="swift-testing / XCTest",
    success_marker="Executed",
    command_template="{runner}",
    suggested_template="{runner}",
)

_PROFILES = {
    profile.name: profile
    for profile in (
        PYTHON,
        TYPESCRIPT,
        JAVASCRIPT,
        JAVA,
        KOTLIN,
        C,
        CPP,
        GO,
        RUST,
        RUBY,
        PHP,
        CSHARP,
        SWIFT,
    )
}

_EXT_TO_PROFILE: dict[str, LanguageProfile] = {}
for _profile in _PROFILES.values():
    for ext in _profile.extensions:
        _EXT_TO_PROFILE[ext] = _profile


def detect_language(path: str | Path) -> LanguageProfile | None:
    """Return the LanguageProfile for a file path, or None when unknown."""
    return _EXT_TO_PROFILE.get(Path(path).suffix.lower())


def _matches_target(profile: LanguageProfile, target: Path, test_file: Path) -> bool:
    stem = target.stem
    for pattern in profile.test_patterns:
        wanted = re.escape(pattern).replace(r"\{stem\}", re.escape(stem))
        if re.fullmatch(wanted, test_file.name, flags=re.IGNORECASE):
            return True
    return False


def find_test_files(
    profile: LanguageProfile,
    target: Path,
    root: Path,
) -> list[Path]:
    """Locate existing test files that cover ``target`` using the language's
    conventions. Returns absolute paths, sorted."""
    root = root.resolve()
    target = target.resolve()
    matches: set[Path] = set()

    search_dirs: list[Path] = [target.parent]
    if profile.test_dirs and profile.test_dirs != ("",):
        for rel in profile.test_dirs:
            search_dirs.append(root / rel)

    candidates: list[Path] = []
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for glob in profile.test_globs:
            for candidate in sorted(search_dir.rglob(glob)):
                candidates.append(candidate)

    for candidate in candidates:
        if _matches_target(profile, target, candidate):
            matches.add(candidate.resolve())

    if not matches and candidates:
        # Fallback to the language-independent convention: a test file that
        # references the target's stem by name (e.g. test_tools.py covering
        # tools/list_dir.py).
        stem = target.stem
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(rf"\b{re.escape(stem)}\b", text):
                matches.add(candidate.resolve())

    if not matches and profile is RUST:
        text = target.read_text(encoding="utf-8", errors="replace")
        if "#[cfg(test)]" in text or "mod tests" in text:
            matches.add(target.resolve())

    return sorted(matches)


def runner_command(profile: LanguageProfile, root: Path) -> str | None:
    """Detect the configured test runner for the project, or None."""
    return _runner_command(profile, root)


def success_marker_for(profile: LanguageProfile, runner: str | None) -> str:
    """Expected output substring on success for the detected runner."""
    if runner is None:
        return profile.success_marker
    if profile is PYTHON:
        return "passed" if "pytest" in runner else "OK"
    if profile is TYPESCRIPT or profile is JAVASCRIPT:
        if "vitest" in runner:
            return "Test Files"
        if "mocha" in runner:
            return "passing"
    if profile is JAVA or profile is KOTLIN:
        return "BUILD SUCCESS"
    if profile is GO:
        return "ok"
    if profile is RUBY:
        return "0 failures"
    return profile.success_marker


def build_command(
    profile: LanguageProfile,
    root: Path,
    test_files: list[Path],
    target: Path,
) -> str | None:
    """Concrete command to run the mapped test files."""
    runner = runner_command(profile, root)
    if runner is None:
        return None
    files = " ".join(str(f) for f in test_files)
    template = profile.command_template
    if profile is JAVA or profile is KOTLIN:
        class_name = Path(test_files[0]).stem
        return template.format(runner=runner, **{"class": class_name})
    if profile is GO:
        return template.format(runner=runner, dir=str(target.parent.resolve()))
    return template.format(runner=runner, files=files)


def _rel_dir(target: Path, root: Path) -> Path:
    try:
        return target.parent.relative_to(root)
    except ValueError:
        return target.parent


def suggest_test(
    profile: LanguageProfile,
    target: Path,
    root: Path,
    runner: str,
) -> dict[str, str]:
    """Propose a concrete test file and command when a framework is configured
    but nothing covers ``target`` yet."""
    target = target.resolve()
    root = root.resolve()
    stem = target.stem

    if profile is GO:
        new_path = str(target.parent / f"{stem}_test.go")
        rel = _rel_dir(target, root)
        command = "go test ./" + ("." if str(rel) == "." else str(rel))
    elif profile is JAVA:
        src_root = root / "src/main/java"
        try:
            rel = target.parent.relative_to(src_root)
        except ValueError:
            rel = _rel_dir(target, root)
        new_path = str(root / "src/test/java" / rel / f"{stem}Test.java")
        command = runner + f" -Dtest={stem}Test"
    elif profile is KOTLIN:
        src_root = root / "src/main/kotlin"
        try:
            rel = target.parent.relative_to(src_root)
        except ValueError:
            rel = _rel_dir(target, root)
        new_path = str(root / "src/test/kotlin" / rel / f"{stem}Test.kt")
        command = runner + f" -Dtest={stem}Test"
    elif profile is TYPESCRIPT or profile is JAVASCRIPT:
        new_path = str(target.parent / f"{stem}.test{target.suffix}")
        command = runner + " " + new_path
    elif profile is PYTHON:
        new_path = str(root / "tests" / f"test_{stem}.py")
        command = runner + " " + new_path
    elif profile is RUBY:
        new_path = str(root / "test" / f"{stem}_test.rb")
        command = runner + " " + new_path
    elif profile is PHP:
        new_path = str(root / "tests" / f"{stem}Test.php")
        command = runner + " " + new_path
    elif profile is RUST:
        new_path = str(target)
        command = runner
    elif profile is CSHARP:
        new_path = str(root / "tests" / f"{stem}Tests.cs")
        command = runner
    elif profile is SWIFT:
        new_path = str(root / "Tests" / f"{stem}Tests.swift")
        command = runner
    elif profile is C or profile is CPP:
        test_name = f"test_{stem}{profile.extensions[0]}"
        new_path = str(root / "tests" / test_name)
        command = profile.suggested_template.format(
            test=test_name, source=target.name, stem=stem
        )
    else:
        new_path = ""
        command = profile.suggested_template.format(
            test="<test file>", source=target.name, stem=stem
        )

    return {
        "framework": profile.framework,
        "command": command,
        "new_test_path": new_path,
        "success_marker": success_marker_for(profile, runner),
    }


def recommend_framework(profile: LanguageProfile, root: Path) -> str:
    """Human-facing guidance when no test toolchain is configured."""
    return (
        f"No unit-test setup detected for {profile.name} in this project. "
        f"Recommended: configure {profile.framework} "
        f"({profile.success_marker!r} is the expected success marker) or implement "
        f"a test suggested by the user."
    )
