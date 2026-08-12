import unittest
from pathlib import Path
from src.tools.read_file import handle_read_file


class TestFileReader(unittest.TestCase):
    def setUp(self):
        self.test_file = Path("test_buggy_file.txt")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("word1 word2 word3\nline2 with word1\nthird line end")

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_attribute_error_repro(self):
        try:
            result = handle_read_file(str(self.test_file), page=1, page_size=1)
            self.assertEqual(result["status"], "success")
        except AttributeError as e:
            self.fail(f"AttributeError raised: {e}")

    def test_pagination_bounds(self):
        result = handle_read_file(str(self.test_file), page=10, page_size=5)
        self.assertEqual(result["status"], "error")

    def test_middle_of_line_pagination(self):
        test_file = Path("test_mid_line.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("word1 word2\nword3 word4")
        try:
            result = handle_read_file(str(test_file), page=2, page_size=2)
            self.assertEqual(result["status"], "success")
            self.assertIn("word3", result["current_page_content"])
            self.assertIn("word4", result["current_page_content"])
            self.assertNotIn("word1", result["current_page_content"])
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_empty_line_handling(self):
        test_file = Path("test_empty_lines.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("word1\n\nword2")
        try:
            result = handle_read_file(str(test_file), page=1, page_size=10)
            self.assertEqual(result["status"], "success")
            self.assertIn("word1", result["current_page_content"])
            self.assertIn("word2", result["current_page_content"])
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_leading_whitespace_preservation(self):
        test_file = Path("test_whitespace.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\t  word1 word2\n  word3 word4")
        try:
            # Page 1 should include the leading tabs/spaces of the first line
            result = handle_read_file(str(test_file), page=1, page_size=2)
            self.assertIn("\t  word1", result["current_page_content"])
            # Page 2 should start at word3 (no leading spaces from previous line)
            result = handle_read_file(str(test_file), page=2, page_size=2)
            self.assertIn("word3", result["current_page_content"])
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_empty_line_preservation(self):
        test_file = Path("test_empty_lines_preservation.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("word1\n\nword2")
        try:
            # Page 1 should include the empty line between word1 and word2
            result = handle_read_file(str(test_file), page=1, page_size=10)
            self.assertIn("word1\n\nword2", result["current_page_content"])
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    unittest.main()
