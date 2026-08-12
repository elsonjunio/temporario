from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ToolHandler = Callable[..., dict]


@dataclass
class ToolSpec:
    name: str
    handlers: dict[str, ToolHandler]
    manual: str
    _dispatch_cache: dict[str, Callable[[str, dict], dict]] = field(
        default_factory=dict, init=False, repr=False
    )

    def get_manual(self) -> str:
        return self.manual

    def dispatch(self, action: str, **params: object) -> dict:
        handler = self.handlers.get(action)
        if handler is None:
            return {"error": "unknown_action", "tool": self.name, "action": action}
        try:
            return handler(**params)
        except TypeError as exc:
            return {
                "error": "invalid_arguments",
                "tool": self.name,
                "action": action,
                "message": str(exc),
            }
