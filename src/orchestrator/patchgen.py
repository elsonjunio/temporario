from __future__ import annotations

from src.patcher.generator import (
    DEFAULT_INSTRUCTION_MAX_CHARS,
    PATCH_SYSTEM,
    PatchGenerator,
    build_patchgen_prompt,
    instruction_max_chars,
)

#: The generator moved to ``src.patcher.generator`` so it can serve scenarios
#: without the orchestrator; this module keeps the historical import path
#: working until the orchestrator is retired.
__all__ = [
    "DEFAULT_INSTRUCTION_MAX_CHARS",
    "PATCH_SYSTEM",
    "PatchGenerator",
    "build_patchgen_prompt",
    "instruction_max_chars",
]
