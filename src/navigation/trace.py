from __future__ import annotations

"""Helpers for working with Playwright trace zips.

Playwright stores each trace as a zip with ``trace.trace`` (JSON-lines
events), ``trace.network``, ``trace.stacks`` and a ``resources/`` directory.
Action screenshots are recorded as ``screencast-frame`` events whose ``sha1``
references an image blob under ``resources/`` (named ``page@<page>-<ts>.jpeg``
in current Playwright versions). ``extract_trace_screenshots`` turns those
back into image files on disk.
"""

import json
import os
import zipfile
from typing import Any

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_SCREENSHOT_PREFIX = "page@"


def _is_image(data: bytes) -> bool:
    return data.startswith(_JPEG_MAGIC) or data.startswith(_PNG_MAGIC)


def _trace_events(zf: zipfile.ZipFile, names: list[str]) -> list[dict[str, Any]] | None:
    """Parse ``trace.trace`` into a list of events. Returns None when the trace
    file is missing entirely (so callers can tell "no trace" apart from "trace
    present but unreadable")."""
    trace_name = next((n for n in names if n.endswith("trace.trace")), None)
    if trace_name is None:
        return None
    try:
        raw = zf.read(trace_name).decode("utf-8", "replace")
    except Exception:
        return []
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _resolve_resource(
    zf: zipfile.ZipFile, names: list[str], resource: str
) -> str | None:
    candidates = [resource, f"resources/{resource}", os.path.basename(resource)]
    for candidate in candidates:
        if candidate in names:
            return candidate
    base = os.path.basename(resource)
    return next((n for n in names if os.path.basename(n) == base), None)


def extract_trace_screenshots(trace_zip: str, out_dir: str | None = None) -> list[str]:
    """Extract the action screenshots recorded by a Playwright trace.

    Returns the written file paths, oldest frame first. Screenshots are saved
    into ``out_dir`` (default: the directory containing the zip) as
    ``screenshot-<n>.<ext>``. Resources that are not images are skipped; if the
    trace events cannot be parsed, resources following the ``page@`` screenshot
    naming convention are used as a fallback.
    """
    out = out_dir or os.path.dirname(os.path.abspath(trace_zip)) or "."
    os.makedirs(out, exist_ok=True)

    refs: dict[str, float] = {}
    with zipfile.ZipFile(trace_zip) as zf:
        names = zf.namelist()
        events = _trace_events(zf, names)
        if events:
            for event in events:
                if event.get("type") != "screencast-frame":
                    continue
                sha1 = event.get("sha1")
                if sha1:
                    try:
                        ts = float(event.get("timestamp") or 0.0)
                    except (TypeError, ValueError):
                        ts = 0.0
                    refs.setdefault(str(sha1), ts)
        elif events is not None:
            # trace.trace exists but could not be parsed: fall back to the
            # page@... screenshot naming convention.
            for name in names:
                if not name.startswith("resources/"):
                    continue
                if os.path.basename(name).startswith(_SCREENSHOT_PREFIX):
                    refs.setdefault(name, 0.0)

        ordered = sorted(refs.items(), key=lambda kv: (kv[1], kv[0]))
        written: list[str] = []
        for index, (resource, _timestamp) in enumerate(ordered, start=1):
            resolved = _resolve_resource(zf, names, resource)
            if resolved is None:
                continue
            try:
                data = zf.read(resolved)
            except KeyError:
                continue
            if not _is_image(data):
                continue
            ext = os.path.splitext(resolved)[1] or ".jpeg"
            out_path = os.path.join(out, f"screenshot-{index:03d}{ext}")
            with open(out_path, "wb") as fh:
                fh.write(data)
            written.append(out_path)
    return written
