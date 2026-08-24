from __future__ import annotations

import json
import os
import re
from typing import Any

from src.utils import extract_json_object

SPLIT_THRESHOLD_ENV = "ORCH_SPLIT_THRESHOLD"
MAX_PARTS_ENV = "ORCH_MAX_PARTS"
DEFAULT_SPLIT_THRESHOLD = 3
DEFAULT_MAX_PARTS = 6

#: Numbered ("1.", "2)", "3:") or bulleted ("- ", "* ") list items.
_ENUM_LINE_RE = re.compile(r"^\s*(?:\d{1,2}\s*[.)\]:]|[-*+•])\s+\S", re.MULTILINE)

#: Additive/sequential connectors that typically introduce a NEW point inside
#: a compound request (PT and EN). Matched case-insensitively.
_CONNECTORS = (
    "também",
    "tambem",
    "e também",
    "e tambem",
    "além disso",
    "alem disso",
    "adicionalmente",
    "outro ponto",
    "outra coisa",
    "da mesma forma",
    "e depois",
    "e por fim",
    "por fim",
    "juntamente com",
    "bem como",
    "and also",
    "additionally",
    "in addition",
    "furthermore",
    "moreover",
    "as well as",
    "besides that",
    "on top of that",
)

#: Change verbs — distinct verbs suggest distinct concerns. Matched on word
#: boundaries so inflections/prefixes don't double count ("cria" != "criar").
_CHANGE_VERBS = (
    "criar",
    "crie",
    "cria",
    "adicionar",
    "adicione",
    "adiciona",
    "remover",
    "remova",
    "remover",
    "excluir",
    "exclua",
    "deletar",
    "delete",
    "alterar",
    "altere",
    "mudar",
    "mude",
    "atualizar",
    "atualize",
    "update",
    "refatorar",
    "refatore",
    "refactor",
    "corrigir",
    "corrija",
    "fix",
    "implementar",
    "implemente",
    "implement",
    "configurar",
    "configura",
    "configure",
    "renomear",
    "renomeie",
    "rename",
    "mover",
    "mova",
    "escrever",
    "escreva",
    "write",
    "gerar",
    "gere",
    "generate",
)

#: Path-like tokens used to count explicitly named files (foo.py, src/x.ts...).
_PATHLIKE_RE = re.compile(
    r"\b[\w./\\-]+\.(?:py|ts|tsx|js|jsx|java|kt|go|rs|rb|php|cs|swift|md|json|yaml|yml|html|css|scss|vue|svelte|sql)\b"
)

_MAX_SIGNAL_CAP = 4


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def score_request(request: str) -> dict[str, Any]:
    """Deterministically estimate how many separate adjustment points a
    request contains. Pure heuristics — no LLM cost, no I/O."""
    lowered = request.lower()

    enum_items = len(_ENUM_LINE_RE.findall(request))

    connectors: list[str] = []
    for connector in _CONNECTORS:
        found = lowered.count(connector)
        if found:
            connectors.extend([connector] * min(found, _MAX_SIGNAL_CAP))

    paths = sorted({match.group(0) for match in _PATHLIKE_RE.finditer(request)})

    verbs: list[str] = []
    for verb in _CHANGE_VERBS:
        if verb in verbs:
            continue
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            verbs.append(verb)

    points = 0
    # Every enumerated item is an independent point of work.
    if enum_items >= 2:
        points += min(enum_items, _MAX_SIGNAL_CAP)
    # Every connector occurrence introduces another independent point.
    points += min(len(connectors), _MAX_SIGNAL_CAP)
    # Extra files beyond the first hint at multiple touch points.
    if len(paths) >= 2:
        points += min(len(paths) - 1, _MAX_SIGNAL_CAP // 2)
    # Several distinct change verbs suggest several concerns.
    if len(verbs) >= 3:
        points += min(len(verbs) - 2, 2)

    return {
        "points": points,
        "signals": {
            "enumeration_items": enum_items,
            "connectors": connectors,
            "paths": paths[:_MAX_SIGNAL_CAP],
            "change_verbs": verbs[:_MAX_SIGNAL_CAP],
        },
    }


class PreAssessor:
    """Cheap deterministic gate that decides whether a request should be
    split into parts before planning. Threshold comes from the constructor
    or ``ORCH_SPLIT_THRESHOLD`` (default 3)."""

    def __init__(self, threshold: int | None = None) -> None:
        if threshold is not None and threshold > 0:
            self.threshold = threshold
        else:
            self.threshold = _env_int(SPLIT_THRESHOLD_ENV, DEFAULT_SPLIT_THRESHOLD)

    def assess(self, request: str) -> dict[str, Any]:
        result = score_request(request)
        result["threshold"] = self.threshold
        result["needs_split"] = result["points"] >= self.threshold
        return result


DECOMPOSE_SYSTEM = (
    "You are a decomposition engine for a filesystem agent. You only produce "
    "JSON; you never execute anything."
)

_DECOMPOSE_PROMPT = (
    "Decompose the request below into ordered, independently plannable parts "
    "for a filesystem agent.\n\n"
    "Request:\n{request_body}\n\n"
    "Pre-assessment signals:\n{signals}\n\n"
    "Rules:\n"
    "- Split ONLY into genuinely separable concerns; do not invent work that "
    "is not requested.\n"
    "- Order parts by dependency: foundations/setup first, consumers/tests "
    "last. A later part may build upon earlier ones.\n"
    "- Each part's 'request' must be SELF-CONTAINED: rewrite pronouns and "
    "references ('o mesmo endpoint' -> name it explicitly) so any part can be "
    "planned without reading the others.\n"
    "- Optionally give each part a 'paths' list with the CONCRETE files/dirs "
    "it touches (existing or to-be-created); this steers evidence search.\n"
    "- Keep each part small enough to fit roughly {max_steps} tool steps.\n"
    "- Between 2 and {max_parts} parts; prefer fewer, cohesive parts.\n"
    'Return ONLY JSON: {{"parts": [{{"id": "PART-001", "request": "...", '
    '"rationale": "...", "paths": ["src/x.py"]}}]}}'
)


def _normalize_id(index: int) -> str:
    return f"PART-{index:03d}"


class Decomposer:
    """Turns a compound request into ordered self-contained parts using the
    injected provider, with a deterministic fallback splitter when the LLM
    fails or returns unusable output."""

    def __init__(
        self, provider: Any, max_parts: int | None = None, max_steps_per_part: int = 12
    ) -> None:
        self.provider = provider
        if max_parts is not None and max_parts > 0:
            self.max_parts = max_parts
        else:
            self.max_parts = _env_int(MAX_PARTS_ENV, DEFAULT_MAX_PARTS)
        self.max_steps_per_part = max_steps_per_part

    # -- public API ---------------------------------------------------------

    def decompose(
        self, request: str, assessment: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raw = self._ask_provider(request, assessment)
        parts: list[dict[str, Any]] | None = None
        if raw is not None:
            parts = self._parse(raw)
        strategy = "llm"
        if parts is None:
            parts = self._fallback_parts(request)
            strategy = "heuristic_fallback"
        if len(parts) < 2:
            return {"status": "single", "parts": [], "strategy": strategy}
        cleaned = self._normalize(parts)
        if len(cleaned) < 2:
            return {"status": "single", "parts": [], "strategy": strategy}
        return {"status": "ok", "parts": cleaned, "strategy": strategy}

    # -- LLM -----------------------------------------------------------------

    def _ask_provider(
        self, request: str, assessment: dict[str, Any] | None
    ) -> str | None:
        prompt = _DECOMPOSE_PROMPT.format(
            request_body=request,
            signals=json.dumps(assessment or {}, ensure_ascii=False),
            max_parts=self.max_parts,
            max_steps=self.max_steps_per_part,
        )
        try:
            return self.provider.infer(prompt, DECOMPOSE_SYSTEM)
        except Exception:  # noqa: BLE001 - fall back to the heuristic splitter
            return None

    @staticmethod
    def _parse(raw: str) -> list[dict[str, Any]] | None:
        data = extract_json_object(raw)
        if not isinstance(data, dict):
            return None
        parts = data.get("parts")
        if not isinstance(parts, list):
            return None
        cleaned: list[dict[str, Any]] = []
        for entry in parts:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("request", "")).strip()
            if not text:
                continue
            cleaned.append(
                {
                    "request": text,
                    "rationale": str(entry.get("rationale", "")),
                    "paths": entry.get("paths"),
                }
            )
        return cleaned or None

    def _normalize(self, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean, merge excess and renumber so ids are always PART-001…"""
        merged = self._merge_excess(parts)
        cleaned: list[dict[str, Any]] = []
        for part in merged:
            request_text = str(part.get("request", "")).strip()
            if not request_text:
                continue
            raw_paths = part.get("paths")
            paths: list[str] = []
            if isinstance(raw_paths, list):
                paths = [str(p) for p in raw_paths if str(p).strip()]
            cleaned.append(
                {
                    "id": _normalize_id(len(cleaned) + 1),
                    "request": request_text,
                    "rationale": str(part.get("rationale", "")),
                    "paths": paths,
                }
            )
        return cleaned

    # -- deterministic fallback ----------------------------------------------

    def _fallback_parts(self, request: str) -> list[dict[str, Any]]:
        """Split on enumerated list items when present; otherwise start a new
        part at each additive connector."""
        items: list[tuple[int, int]] = []
        for match in _ENUM_LINE_RE.finditer(request):
            items.append((match.start(), match.end()))
        if len(items) >= 2:
            preamble = request[: items[0][0]].strip()
            parts: list[dict[str, Any]] = []
            for i, (item_start, _item_end) in enumerate(items):
                stop = items[i + 1][0] if i + 1 < len(items) else len(request)
                body = request[item_start:stop].strip()
                if preamble and i == 0:
                    body = f"{preamble} {body}".strip()
                if body:
                    parts.append({"request": body, "rationale": "enumerated item"})
            if len(parts) >= 2:
                return parts

        spans = self._connector_spans(request)
        if len(spans) >= 1:
            chunks: list[str] = []
            first_end = spans[0][0]
            head = request[:first_end].strip()
            if head:
                chunks.append(head)
            for i, (span_start, _span_end) in enumerate(spans):
                stop = spans[i + 1][0] if i + 1 < len(spans) else len(request)
                chunk = request[span_start:stop].strip()
                if chunk:
                    chunks.append(chunk)
            parts = [
                {"request": chunk, "rationale": "split at additive connector"}
                for chunk in chunks
                if len(chunk.split()) >= 3
            ]
            if len(parts) >= 2:
                return parts
        return []

    @staticmethod
    def _connector_spans(request: str) -> list[tuple[int, int]]:
        lowered = request.lower()
        spans: list[tuple[int, int]] = []
        for connector in _CONNECTORS:
            search_from = 0
            while True:
                index = lowered.find(connector, search_from)
                if index == -1:
                    break
                spans.append((index, index + len(connector)))
                search_from = index + len(connector)
        spans.sort()
        return spans

    # -- helpers --------------------------------------------------------------

    def _merge_excess(self, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cap the number of parts by merging the tail into the last allowed
        part (order preserved), so nothing the user asked for is dropped."""
        merged = [dict(part) for part in parts]
        while len(merged) > self.max_parts:
            tail = merged.pop()
            last = merged[-1]
            head = str(last["request"]).rstrip()
            if not head.endswith((".", "!", "?")):
                head += "."
            last["request"] = f"{head} {tail['request']}"
            rationale = (
                f"{last.get('rationale', '')} + {tail.get('rationale', '')}".strip(" +")
            )
            last["rationale"] = rationale
        return merged
