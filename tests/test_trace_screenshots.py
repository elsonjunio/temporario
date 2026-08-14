"""Unit tests for extracting screenshots from a Playwright trace zip.

Fabricates a trace with ``screencast-frame`` events referencing image
resources, mirroring the layout Playwright produces (``resources/`` with
``page@<page>-<ts>.jpeg`` entries and a ``trace.trace`` JSON-lines file).
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import zipfile

from src.navigation.trace import extract_trace_screenshots

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 32
_TEXT = b"<html><body>not an image</body></html>"


def _make_trace(dirname: str, events: list[dict], resources: dict[str, bytes]) -> str:
    zip_path = os.path.join(dirname, "trace.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        lines = "".join(f"{e!r}\n" for e in events) if events else ""
        zf.writestr("trace.trace", lines)
        for name, data in resources.items():
            zf.writestr(f"resources/{name}", data)
    return zip_path


class TestExtractTraceScreenshots(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_extracts_screencast_frames_oldest_first(self):
        events = [
            {
                "type": "screencast-frame",
                "pageId": "page@abc",
                "sha1": "page@abc-100.jpeg",
                "timestamp": 100.0,
            },
            {
                "type": "screencast-frame",
                "pageId": "page@abc",
                "sha1": "page@abc-200.jpeg",
                "timestamp": 200.0,
            },
        ]
        zip_path = _make_trace(
            self.tmp,
            events,
            {"page@abc-100.jpeg": _JPEG, "page@abc-200.jpeg": _JPEG},
        )
        out = extract_trace_screenshots(zip_path)
        self.assertEqual(len(out), 2)
        names = [os.path.basename(p) for p in out]
        self.assertEqual(names, ["screenshot-001.jpeg", "screenshot-002.jpeg"])
        for path in out:
            with open(path, "rb") as fh:
                self.assertTrue(fh.read().startswith(b"\xff\xd8\xff"))

    def test_skips_non_image_resources_and_deduplicates(self):
        events = [
            {
                "type": "screencast-frame",
                "pageId": "page@abc",
                "sha1": "page@abc-100.jpeg",
                "timestamp": 100.0,
            },
            {
                "type": "screencast-frame",
                "pageId": "page@abc",
                "sha1": "page@abc-100.jpeg",
                "timestamp": 100.0,
            },
        ]
        zip_path = _make_trace(
            self.tmp,
            events,
            {"page@abc-100.jpeg": _JPEG, "other.html": _TEXT},
        )
        out = extract_trace_screenshots(zip_path)
        self.assertEqual(len(out), 1)

    def test_supports_png_magic(self):
        zip_path = _make_trace(
            self.tmp,
            [{"type": "screencast-frame", "sha1": "page@abc-1.png", "timestamp": 1.0}],
            {"page@abc-1.png": _PNG},
        )
        out = extract_trace_screenshots(zip_path)
        self.assertEqual(len(out), 1)
        self.assertEqual(os.path.basename(out[0]), "screenshot-001.png")

    def test_fallback_by_page_naming_when_events_unparseable(self):
        zip_path = os.path.join(self.tmp, "trace.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("trace.trace", "this is { not valid json\n")
            zf.writestr("resources/page@abc-100.jpeg", _JPEG)
            zf.writestr("resources/page@abc-200.jpeg", _JPEG)
        out = extract_trace_screenshots(zip_path)
        self.assertEqual(len(out), 2)

    def test_writes_into_explicit_out_dir(self):
        zip_path = _make_trace(
            self.tmp,
            [{"type": "screencast-frame", "sha1": "page@abc-1.jpeg", "timestamp": 1.0}],
            {"page@abc-1.jpeg": _JPEG},
        )
        out_dir = os.path.join(self.tmp, "extracted")
        out = extract_trace_screenshots(zip_path, out_dir)
        self.assertTrue(os.path.isdir(out_dir))
        self.assertEqual(len(out), 1)
        self.assertEqual(os.path.dirname(out[0]), out_dir)

    def test_missing_trace_file_yields_empty(self):
        zip_path = os.path.join(self.tmp, "trace.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("resources/page@abc-100.jpeg", _JPEG)
        self.assertEqual(extract_trace_screenshots(zip_path), [])


if __name__ == "__main__":
    unittest.main()
