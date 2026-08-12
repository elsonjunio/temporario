import tempfile
import unittest
from pathlib import Path

from src.tools import facade
from src.tools import (
    delete_file,
    grep_files,
    list_dir,
    move_file,
    patch_file,
    read_file,
    run_command,
    search_files,
    write_file,
)


class TestToolManuals(unittest.TestCase):
    def test_each_tool_exposes_manual(self):
        for module in (
            delete_file,
            grep_files,
            list_dir,
            move_file,
            patch_file,
            read_file,
            run_command,
            search_files,
            write_file,
        ):
            manual = module.get_manual()
            self.assertIsInstance(manual, str)
            self.assertTrue(manual.strip())
            self.assertIn("Actions:", manual)

    def test_facade_manual_aggregates(self):
        manual = facade.get_manual()
        for tool in (
            "read_file",
            "list_dir",
            "search_files",
            "grep_files",
            "write_file",
            "patch_file",
            "delete_file",
            "move_file",
            "run_command",
        ):
            self.assertIn(tool, manual)


class TestRunCommandDispatch(unittest.TestCase):
    def test_run_success(self):
        result = run_command.dispatch("run", command="echo hello")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("hello", result["stdout"])

    def test_run_nonzero_exit(self):
        result = run_command.dispatch("run", command="exit 3")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 3)

    def test_stderr_captured(self):
        result = run_command.dispatch("run", command="echo oops 1>&2")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("oops", result["stderr"])

    def test_stdin_fed(self):
        result = run_command.dispatch("run", command="cat", input="hello from stdin")
        self.assertIn("hello from stdin", result["stdout"])

    def test_cwd(self):
        result = run_command.dispatch("run", command="pwd", cwd="src/tools")
        self.assertIn("src/tools", result["stdout"])

    def test_timeout(self):
        result = run_command.dispatch("run", command="sleep 5", timeout=0.2)
        self.assertEqual(result["status"], "timeout")

    def test_output_capped(self):
        result = run_command.dispatch("run", command="echo x", max_output=1)
        self.assertTrue(result["stdout_truncated"])
        self.assertLessEqual(len(result["stdout"]), 100)

    def test_empty_command(self):
        result = run_command.dispatch("run", command="   ")
        self.assertEqual(result["error"], "invalid_arguments")

    def test_unknown_action(self):
        result = run_command.dispatch("nope", command="echo x")
        self.assertEqual(result["error"], "unknown_action")

    def test_invalid_arguments(self):
        result = run_command.dispatch("run")
        self.assertEqual(result["error"], "invalid_arguments")


class TestFileReaderDispatch(unittest.TestCase):
    def test_unknown_action(self):
        result = read_file.dispatch("nope", file_path="x.txt")
        self.assertEqual(result["error"], "unknown_action")

    def test_missing_file(self):
        result = read_file.dispatch("read", file_path="/nonexistent/xyz.txt")
        self.assertEqual(result["error"], "File not found")

    def test_invalid_arguments(self):
        result = read_file.dispatch("read")
        self.assertEqual(result["error"], "invalid_arguments")


class TestListDirDispatch(unittest.TestCase):
    def test_list(self):
        result = list_dir.dispatch("list", path="src/tools")
        self.assertGreater(result["total"], 0)
        names = {item["name"] for item in result["items"]}
        self.assertIn("facade.py", names)

    def test_unknown_tool_in_facade(self):
        result = facade.dispatch("does_not_exist", "list")
        self.assertEqual(result["error"], "unknown_tool")


class TestSearchFilesDispatch(unittest.TestCase):
    def test_search(self):
        result = search_files.dispatch("search", pattern="*.py", path="src/tools")
        self.assertGreater(result["total"], 0)

    def test_extension_filter(self):
        result = list_dir.dispatch("list", path="src", extension=".py")
        self.assertGreater(result["total"], 0)
        for item in result["items"]:
            if item["type"] == "file":
                self.assertTrue(item["name"].endswith(".py"))


class TestGrepFilesDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        (self.base / "app.py").write_text(
            "import os\n\ndef hello(name):\n    return f'hi {name}'\n",
            encoding="utf-8",
        )
        (self.base / "readme.txt").write_text(
            "Welcome to the app\nsee docs for more\n", encoding="utf-8"
        )

    def _grep(self, **params):
        return grep_files.dispatch("search", path=str(self.base), **params)

    def test_match_simple(self):
        result = self._grep(pattern="hello")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["items"][0]["file"], "app.py")
        self.assertEqual(result["items"][0]["matches"][0]["line"], 3)
        self.assertIn("hello", result["items"][0]["matches"][0]["content"])

    def test_case_insensitive_by_default(self):
        result = self._grep(pattern="WELCOME")
        self.assertEqual(result["total_files"], 1)

    def test_case_sensitive(self):
        result = self._grep(pattern="welcome", case_sensitive=True)
        self.assertEqual(result["total_files"], 0)

    def test_regex(self):
        result = self._grep(pattern=r"hi \S+")
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["total_matches"], 1)

    def test_include_filter(self):
        result = self._grep(pattern="the", include="*.py")
        self.assertEqual(result["total_files"], 0)
        result = self._grep(pattern="the", include="*.txt")
        self.assertEqual(result["total_files"], 1)

    def test_excluded_dir(self):
        (self.base / ".venv" / "lib").mkdir(parents=True)
        (self.base / ".venv" / "lib" / "x.py").write_text("hello world\n")
        result = self._grep(pattern="hello")
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["items"][0]["file"], "app.py")

    def test_binary_skipped(self):
        (self.base / "data.bin").write_bytes(b"\x00\x01hello\x00")
        result = self._grep(pattern="hello")
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["items"][0]["file"], "app.py")

    def test_context_lines(self):
        result = self._grep(pattern="hello", context_lines=2)
        context = result["items"][0]["matches"][0]["context"]
        self.assertIn("import os", context)

    def test_pagination(self):
        for i in range(5):
            (self.base / f"f{i}.py").write_text(f"needle {i}\n")
        result = self._grep(pattern="needle", page=1, page_size=2)
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(result["has_next"])

    def test_missing_path(self):
        result = grep_files.dispatch("search", pattern="x", path="/nonexistent/xyz")
        self.assertEqual(result["error"], "path_not_found")

    def test_invalid_regex(self):
        result = self._grep(pattern="[unclosed")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid regex", result["message"])

    def test_empty_pattern(self):
        result = self._grep(pattern="")
        self.assertEqual(result["error"], "invalid_arguments")

    def test_unknown_action(self):
        result = grep_files.dispatch("nope", pattern="x")
        self.assertEqual(result["error"], "unknown_action")

    def test_invalid_arguments(self):
        result = grep_files.dispatch("search")
        self.assertEqual(result["error"], "invalid_arguments")


class TestWriteFileDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_create_file(self):
        target = self.base / "hello.txt"
        result = write_file.dispatch("write", file_path=str(target), content="hi")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["mode"], "created")
        self.assertEqual(target.read_text(), "hi")

    def test_overwrite_file(self):
        target = self.base / "hello.txt"
        target.write_text("old")
        result = write_file.dispatch("write", file_path=str(target), content="new")
        self.assertEqual(result["mode"], "overwritten")
        self.assertEqual(target.read_text(), "new")

    def test_unchanged_content(self):
        target = self.base / "hello.txt"
        target.write_text("same")
        result = write_file.dispatch("write", file_path=str(target), content="same")
        self.assertEqual(result["mode"], "unchanged")
        self.assertEqual(target.read_text(), "same")

    def test_creates_parent_dirs(self):
        target = self.base / "a" / "b" / "c.txt"
        result = write_file.dispatch("write", file_path=str(target), content="x")
        self.assertEqual(result["status"], "success")
        self.assertTrue(target.exists())

    def test_append(self):
        target = self.base / "log.txt"
        write_file.dispatch("append", file_path=str(target), content="one\n")
        result = write_file.dispatch("append", file_path=str(target), content="two\n")
        self.assertEqual(result["mode"], "appended")
        self.assertEqual(target.read_text(), "one\ntwo\n")

    def test_unknown_action(self):
        result = write_file.dispatch("nope", file_path="x.txt", content="")
        self.assertEqual(result["error"], "unknown_action")

    def test_missing_content_argument(self):
        result = write_file.dispatch("write", file_path="x.txt")
        self.assertEqual(result["error"], "invalid_arguments")


class TestReadFileFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def _make(self, name: str, content: str) -> Path:
        target = self.base / name
        target.write_text(content, encoding="utf-8")
        return target

    def test_binary_file(self):
        target = self.base / "data.bin"
        target.write_bytes(b"\x00\x01\x02hello")
        result = read_file.dispatch("read", file_path=str(target))
        self.assertEqual(result["status"], "error")
        self.assertIn("Binary", result["message"])

    def test_not_a_file(self):
        result = read_file.dispatch("read", file_path=str(self.base))
        self.assertEqual(result["error"], "Not a file")

    def test_metadata_present(self):
        target = self._make("a.txt", "hello world")
        result = read_file.dispatch("read", file_path=str(target))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["encoding"], "utf-8")
        self.assertEqual(result["byte_size"], 11)
        self.assertEqual(result["total_lines"], 1)
        self.assertEqual(result["total_words"], 2)

    def test_latin1_fallback(self):
        target = self.base / "latin.txt"
        target.write_bytes("caf\xe9".encode("latin-1"))
        result = read_file.dispatch("read", file_path=str(target))
        self.assertEqual(result["encoding"], "latin-1")
        self.assertIn("caf", result["current_page_content"])

    def test_line_range(self):
        target = self._make("lines.txt", "one\ntwo\nthree\nfour\n")
        result = read_file.dispatch(
            "read", file_path=str(target), start_line=2, end_line=3
        )
        self.assertEqual(result["mode"], "lines")
        self.assertEqual(result["content"], "two\nthree")

    def test_line_range_single_line(self):
        target = self._make("lines.txt", "one\ntwo\n")
        result = read_file.dispatch("read", file_path=str(target), start_line=2)
        self.assertEqual(result["content"], "two")

    def test_line_range_truncated(self):
        target = self._make("lines.txt", "one\ntwo\nthree\n")
        result = read_file.dispatch(
            "read",
            file_path=str(target),
            start_line=1,
            end_line=3,
            max_lines=2,
        )
        self.assertTrue(result["truncated"])
        self.assertEqual(result["content"], "one\ntwo")

    def test_line_range_invalid(self):
        target = self._make("lines.txt", "one\n")
        result = read_file.dispatch("read", file_path=str(target), start_line=0)
        self.assertEqual(result["status"], "error")


class TestPatchFileDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def _file(self, name: str, content: str) -> Path:
        target = self.base / name
        target.write_text(content, encoding="utf-8")
        return target

    def test_apply_diff(self):
        target = self._file("a.txt", "line1\nline2\nline3\n")
        diff = (
            "--- a/a.txt\n"
            "+++ b/a.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+LINE2\n"
            " line3\n"
        )
        result = patch_file.dispatch("apply", file_path=str(target), diff=diff)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["hunks_applied"], 1)
        self.assertEqual(target.read_text(), "line1\nLINE2\nline3\n")

    def test_apply_hunk_with_offset_drift(self):
        target = self._file("a.txt", "keep\nkeep\nline2\nline3\n")
        diff = "@@ -3,3 +3,3 @@\n" " line2\n" "-line3\n" "+LINE3\n"
        result = patch_file.dispatch("apply", file_path=str(target), diff=diff)
        self.assertEqual(result["status"], "success")
        self.assertEqual(target.read_text(), "keep\nkeep\nline2\nLINE3\n")

    def test_apply_no_hunks(self):
        target = self._file("a.txt", "line1\n")
        result = patch_file.dispatch("apply", file_path=str(target), diff="--- a\na\n")
        self.assertEqual(result["status"], "error")
        self.assertIn("No @@ hunks", result["message"])

    def test_apply_hunk_not_found(self):
        target = self._file("a.txt", "line1\n")
        diff = "@@ -1,1 +1,1 @@\n-nope\n+yes\n"
        result = patch_file.dispatch("apply", file_path=str(target), diff=diff)
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_apply_missing_file(self):
        result = patch_file.dispatch(
            "apply", file_path="/nonexistent/x.txt", diff="@@ -1,1 +1,1 @@\n"
        )
        self.assertEqual(result["error"], "File not found")

    def test_replace(self):
        target = self._file("a.txt", "hello world\n")
        result = patch_file.dispatch(
            "replace", file_path=str(target), old="world", new="there"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["replaced"], 1)
        self.assertEqual(target.read_text(), "hello there\n")

    def test_replace_not_found(self):
        target = self._file("a.txt", "hello\n")
        result = patch_file.dispatch("replace", file_path=str(target), old="bye")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_replace_ambiguous(self):
        target = self._file("a.txt", "x y x\n")
        result = patch_file.dispatch("replace", file_path=str(target), old="x", new="z")
        self.assertEqual(result["status"], "error")
        self.assertIn("replace_all", result["message"])
        self.assertEqual(target.read_text(), "x y x\n")

    def test_replace_all(self):
        target = self._file("a.txt", "x y x\n")
        result = patch_file.dispatch(
            "replace", file_path=str(target), old="x", new="z", replace_all=True
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["replaced"], 2)
        self.assertEqual(target.read_text(), "z y z\n")

    def test_unknown_action(self):
        result = patch_file.dispatch("nope", file_path="x.txt")
        self.assertEqual(result["error"], "unknown_action")


class TestDeleteFileDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_delete_file(self):
        target = self.base / "a.txt"
        target.write_text("x")
        result = delete_file.dispatch("delete", path=str(target))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["mode"], "deleted_file")
        self.assertFalse(target.exists())

    def test_delete_missing(self):
        result = delete_file.dispatch("delete", path=str(self.base / "nope.txt"))
        self.assertEqual(result["error"], "Path not found")

    def test_directory_requires_recursive(self):
        target = self.base / "d"
        target.mkdir()
        result = delete_file.dispatch("delete", path=str(target))
        self.assertEqual(result["status"], "error")
        self.assertIn("recursive", result["message"])
        self.assertTrue(target.exists())

    def test_delete_directory_recursive(self):
        target = self.base / "d"
        (target / "sub").mkdir(parents=True)
        (target / "sub" / "f.txt").write_text("x")
        result = delete_file.dispatch("delete", path=str(target), recursive=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["mode"], "deleted_directory")
        self.assertFalse(target.exists())

    def test_unknown_action(self):
        result = delete_file.dispatch("nope", path="x.txt")
        self.assertEqual(result["error"], "unknown_action")


class TestMoveFileDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_move_file(self):
        src = self.base / "a.txt"
        dst = self.base / "b.txt"
        src.write_text("hello")
        result = move_file.dispatch("move", source=str(src), destination=str(dst))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["mode"], "moved")
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_text(), "hello")

    def test_move_missing_source(self):
        result = move_file.dispatch(
            "move",
            source=str(self.base / "nope.txt"),
            destination=str(self.base / "b.txt"),
        )
        self.assertEqual(result["error"], "Source not found")

    def test_move_creates_parents(self):
        src = self.base / "a.txt"
        dst = self.base / "x" / "y" / "b.txt"
        src.write_text("hello")
        result = move_file.dispatch("move", source=str(src), destination=str(dst))
        self.assertEqual(result["status"], "success")
        self.assertTrue(dst.exists())

    def test_move_existing_destination(self):
        src = self.base / "a.txt"
        dst = self.base / "b.txt"
        src.write_text("new")
        dst.write_text("old")
        result = move_file.dispatch("move", source=str(src), destination=str(dst))
        self.assertEqual(result["status"], "error")
        self.assertIn("overwrite", result["message"])
        self.assertEqual(dst.read_text(), "old")

    def test_move_overwrite(self):
        src = self.base / "a.txt"
        dst = self.base / "b.txt"
        src.write_text("new")
        dst.write_text("old")
        result = move_file.dispatch(
            "move", source=str(src), destination=str(dst), overwrite=True
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(dst.read_text(), "new")

    def test_move_same_path(self):
        src = self.base / "a.txt"
        src.write_text("x")
        result = move_file.dispatch("move", source=str(src), destination=str(src))
        self.assertEqual(result["mode"], "unchanged")

    def test_unknown_action(self):
        result = move_file.dispatch("nope", source="x", destination="y")
        self.assertEqual(result["error"], "unknown_action")


class TestFacadeRouting(unittest.TestCase):
    def test_routes_to_correct_tool(self):
        result = facade.dispatch("list_dir", "list", path="src")
        self.assertIn("items", result)
        self.assertNotIn("error", result)

    def test_new_tools_registered(self):
        self.assertEqual(
            set(facade.list_tools()),
            {
                "read_file",
                "list_dir",
                "search_files",
                "grep_files",
                "write_file",
                "patch_file",
                "delete_file",
                "move_file",
                "run_command",
            },
        )


if __name__ == "__main__":
    unittest.main()
