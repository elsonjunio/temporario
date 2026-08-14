import tempfile
import unittest
from pathlib import Path

from src.orchestrator.languages import (
    build_command,
    detect_language,
    find_test_files,
    runner_command,
    suggest_test,
    success_marker_for,
)
from src.orchestrator.languages import (
    C,
    CPP,
    GO,
    JAVA,
    PYTHON,
    RUBY,
    TYPESCRIPT,
)


class LanguageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class TestDetectLanguage(LanguageTestCase):
    def test_extensions(self):
        cases = {
            "a.py": "python",
            "b.ts": "typescript",
            "b.tsx": "typescript",
            "c.js": "javascript",
            "c.mjs": "javascript",
            "Main.java": "java",
            "Util.kt": "kotlin",
            "main.c": "c",
            "main.cpp": "cpp",
            "main.go": "go",
            "main.rs": "rust",
            "main.rb": "ruby",
            "index.php": "php",
            "Service.cs": "csharp",
            "App.swift": "swift",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(detect_language(name).name, expected)

    def test_unknown(self):
        self.assertIsNone(detect_language("notes.txt"))
        self.assertIsNone(detect_language("README.md"))


class TestFindTestFiles(LanguageTestCase):
    def test_typescript_sibling(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "calc.ts").write_text(
            "export const add = (a, b) => a + b;\n"
        )
        (self.root / "src" / "calc.test.ts").write_text(
            "import { add } from './calc';\n"
        )
        found = find_test_files(TYPESCRIPT, self.root / "src/calc.ts", self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "calc.test.ts")

    def test_java_unders_test(self):
        (self.root / "src" / "main" / "java").mkdir(parents=True)
        (self.root / "src" / "test" / "java").mkdir(parents=True)
        (self.root / "src/main/java/Counter.java").write_text("class Counter {}\n")
        (self.root / "src/test/java/CounterTest.java").write_text(
            "class CounterTest {}\n"
        )
        found = find_test_files(
            JAVA, self.root / "src/main/java/Counter.java", self.root
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "CounterTest.java")

    def test_go_sibling(self):
        (self.root / "sum.go").write_text("package main\n")
        (self.root / "sum_test.go").write_text("package main\n")
        found = find_test_files(GO, self.root / "sum.go", self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "sum_test.go")

    def test_c_test_dir(self):
        (self.root / "tests").mkdir()
        (self.root / "stack.c").write_text("int push() { return 0; }\n")
        (self.root / "tests" / "test_stack.c").write_text("#include <assert.h>\n")
        found = find_test_files(C, self.root / "stack.c", self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "test_stack.c")

    def test_python_content_fallback(self):
        (self.root / "src" / "tools").mkdir(parents=True)
        (self.root / "src/tools/list_dir.py").write_text("def handle_list_dir(): ...\n")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_tools.py").write_text(
            "from src.tools.list_dir import handle_list_dir\n"
        )
        found = find_test_files(PYTHON, self.root / "src/tools/list_dir.py", self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "test_tools.py")

    def test_ruby_spec(self):
        (self.root / "lib").mkdir()
        (self.root / "lib" / "greeter.rb").write_text("class Greeter; end\n")
        (self.root / "spec").mkdir()
        (self.root / "spec" / "greeter_spec.rb").write_text("describe Greeter do end\n")
        found = find_test_files(RUBY, self.root / "lib/greeter.rb", self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "greeter_spec.rb")


class TestRunnerCommand(LanguageTestCase):
    def test_typescript_jest(self):
        (self.root / "package.json").write_text('{"dependencies": {"jest": "29"}}\n')
        self.assertEqual(runner_command(TYPESCRIPT, self.root), "npx jest")

    def test_java_maven(self):
        (self.root / "pom.xml").write_text("<project/>\n")
        self.assertEqual(runner_command(JAVA, self.root), "mvn test")

    def test_java_gradle_wrapper(self):
        (self.root / "gradlew").write_text("#!/bin/sh\n")
        self.assertEqual(runner_command(JAVA, self.root), "./gradlew test")

    def test_go_mod(self):
        (self.root / "go.mod").write_text("module example\n")
        self.assertEqual(runner_command(GO, self.root), "go test ./...")

    def test_foreign_markers_do_not_trigger(self):
        (self.root / "Cargo.toml").write_text("[package]\n")
        self.assertEqual(runner_command(RUBY, self.root), None)
        self.assertEqual(runner_command(GO, self.root), None)

    def test_python_unittest_fallback(self):
        self.assertEqual(runner_command(PYTHON, self.root), "python -m unittest")

    def test_python_pytest_when_configured(self):
        (self.root / "pyproject.toml").write_text("[tool.pytest]\n")
        self.assertEqual(runner_command(PYTHON, self.root), "python -m pytest")

    def test_python_pytest_via_requirements(self):
        (self.root / "requirements.txt").write_text("pytest\nhttpx\n")
        self.assertEqual(runner_command(PYTHON, self.root), "python -m pytest")

    def test_python_pytest_only_when_declared(self):
        (self.root / "requirements.txt").write_text("fastapi\nuvicorn\n")
        self.assertNotEqual(runner_command(PYTHON, self.root), "python -m pytest")

    def test_no_runner(self):
        self.assertIsNone(runner_command(CPP, self.root))


class TestBuildCommand(LanguageTestCase):
    def test_typescript(self):
        (self.root / "package.json").write_text('{"dependencies": {"jest": "29"}}\n')
        test = self.root / "src/calc.test.ts"
        cmd = build_command(TYPESCRIPT, self.root, [test], self.root / "src/calc.ts")
        self.assertIn("calc.test.ts", cmd)

    def test_java_class_name(self):
        (self.root / "pom.xml").write_text("<project/>\n")
        test = self.root / "src/test/java/CounterTest.java"
        cmd = build_command(
            JAVA, self.root, [test], self.root / "src/main/java/Counter.java"
        )
        self.assertIn("mvn test", cmd)
        self.assertIn("-Dtest=CounterTest", cmd)


class TestSuggestTest(LanguageTestCase):
    def test_typescript(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "calc.ts").write_text(
            "export const add = (a, b) => a + b;\n"
        )
        (self.root / "package.json").write_text('{"dependencies": {"jest": "29"}}\n')
        suggestion = suggest_test(
            TYPESCRIPT, self.root / "src/calc.ts", self.root, "npx jest"
        )
        self.assertEqual(
            suggestion["new_test_path"], str(self.root / "src/calc.test.ts")
        )
        self.assertIn("calc.test.ts", suggestion["command"])
        self.assertEqual(suggestion["success_marker"], "PASS")

    def test_java(self):
        (self.root / "src" / "main" / "java").mkdir(parents=True)
        (self.root / "src/main/java/Counter.java").write_text("class Counter {}\n")
        (self.root / "pom.xml").write_text("<project/>\n")
        suggestion = suggest_test(
            JAVA, self.root / "src/main/java/Counter.java", self.root, "mvn test"
        )
        self.assertEqual(
            suggestion["new_test_path"],
            str(self.root / "src/test/java/CounterTest.java"),
        )
        self.assertIn("CounterTest", suggestion["command"])

    def test_go(self):
        (self.root / "sum.go").write_text("package main\n")
        suggestion = suggest_test(GO, self.root / "sum.go", self.root, "go test ./...")
        self.assertEqual(suggestion["new_test_path"], str(self.root / "sum_test.go"))
        self.assertIn("ok", suggestion["success_marker"])

    def test_python_marker_ok_for_unittest(self):
        suggestion = suggest_test(
            PYTHON, self.root / "mod.py", self.root, "python3 -m unittest"
        )
        self.assertEqual(suggestion["success_marker"], "OK")
        self.assertEqual(
            suggestion["new_test_path"], str(self.root / "tests/test_mod.py")
        )


class TestSuccessMarker(LanguageTestCase):
    def test_unittest_vs_pytest(self):
        self.assertEqual(success_marker_for(PYTHON, "python3 -m unittest"), "OK")
        self.assertEqual(success_marker_for(PYTHON, "pytest"), "passed")

    def test_java(self):
        self.assertEqual(success_marker_for(JAVA, "mvn test"), "BUILD SUCCESS")


if __name__ == "__main__":
    unittest.main()
