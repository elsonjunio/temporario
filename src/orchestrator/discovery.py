from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.tools.registry import ToolRegistry

DEFAULT_MAX_CANDIDATES = 8

_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "in",
    "of",
    "for",
    "with",
    "and",
    "or",
    "on",
    "into",
    "from",
    "by",
    "at",
    "this",
    "that",
    "please",
    "can",
    "you",
    "change",
    "alter",
    "edit",
    "modify",
    "create",
    "add",
    "remove",
    "delete",
    "make",
    "file",
    "files",
    "it",
    "there",
    "they",
    "them",
}


def extract_terms(request: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_.\-]+", request.lower())
    terms = [
        word
        for word in words
        if len(word) >= 3 and word not in _STOPWORDS and not word.isdigit()
    ]
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return seen[:limit]


class Discovery:
    """Locates candidate targets for a modification request using the
    registry's read-only tools (search_files, list_dir, read_file)."""

    def __init__(
        self,
        registry: ToolRegistry,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        root: str = ".",
    ) -> None:
        self.registry = registry
        self.max_candidates = max_candidates
        self.root = root

    def _snippet(self, path: str) -> str:
        result = self.registry.dispatch(
            "read_file",
            "read",
            file_path=path,
            start_line=1,
            end_line=20,
        )
        return result.get("content", "") if result.get("status") == "success" else ""

    def _seed_candidates(self, paths: list[str]) -> list[dict[str, Any]]:
        """Turn explicit target paths into new-file candidates without
        searching. Relative paths resolve against ``self.root``, so a
        greenfield project can be planned file by file."""
        seen: dict[str, dict[str, Any]] = {}
        for raw in paths:
            path = (Path(self.root) / raw).resolve()
            key = str(path)
            if key in seen:
                continue
            exists = path.exists()
            seen[key] = {
                "name": path.name,
                "path": key,
                "type": "file" if exists else "new_file",
                "snippet": "",
            }
        return list(seen.values())

    def discover(
        self,
        request: str,
        terms: list[str] | None = None,
        seed_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if seed_paths:
            seeded = self._seed_candidates(seed_paths)
            return {
                "status": "ok",
                "request": request,
                "terms": terms or [],
                "reason": (
                    "explicit target paths seeded (no keyword search); "
                    "existing paths are 'file', new ones are 'new_file'"
                ),
                "count": len(seeded),
                "candidates": seeded,
            }

        terms = terms or extract_terms(request)
        if not terms:
            return {
                "status": "poor",
                "request": request,
                "terms": [],
                "reason": "no usable search terms extracted from the request",
                "count": 0,
                "candidates": [],
            }

        candidates: dict[str, dict[str, Any]] = {}
        for term in terms:
            result = self.registry.dispatch(
                "search_files",
                "search",
                pattern=f"*{term}*",
                path=self.root,
                recursive=True,
            )
            for item in result.get("items", []):
                if item["path"] not in candidates:
                    candidates[item["path"]] = {
                        "name": item["name"],
                        "path": item["path"],
                        "type": item["type"],
                        "snippet": "",
                    }

        if not candidates:
            path_like = []
            for term in terms:
                if "/" in term or "\\" in term or "." in term:
                    candidate = (Path(self.root) / term).resolve()
                    if candidate.parent.exists():
                        path_like.append(
                            {
                                "name": Path(term).name,
                                "path": str(candidate),
                                "type": "new_file",
                                "snippet": "",
                            }
                        )
            if path_like:
                return {
                    "status": "ok",
                    "request": request,
                    "terms": terms,
                    "reason": "no existing match; treating path-like terms as new targets",
                    "count": len(path_like),
                    "candidates": path_like,
                }
            return {
                "status": "poor",
                "request": request,
                "terms": terms,
                "reason": "no candidates found for the given terms",
                "count": 0,
                "candidates": [],
            }

        if len(candidates) > self.max_candidates:
            return {
                "status": "poor",
                "request": request,
                "terms": terms,
                "reason": (
                    f"too many candidates ({len(candidates)} > "
                    f"{self.max_candidates}); refine the terms"
                ),
                "count": len(candidates),
                "candidates": [
                    {"name": c["name"], "path": c["path"], "type": c["type"]}
                    for c in list(candidates.values())[: self.max_candidates]
                ],
            }

        ordered = sorted(candidates.values(), key=lambda c: c["path"])
        for match in ordered:
            if match["type"] == "file":
                match["snippet"] = self._snippet(match["path"])
        return {
            "status": "ok",
            "request": request,
            "terms": terms,
            "count": len(ordered),
            "candidates": ordered,
        }
