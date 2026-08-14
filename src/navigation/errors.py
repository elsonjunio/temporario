from __future__ import annotations

from typing import Any


class BrowserError(RuntimeError):
    """Base error carrying a machine-readable category and a flag signalling
    whether the operation can be retried safely by the LLM."""

    category = "browser_error"

    def __init__(
        self,
        message: str = "",
        *,
        recoverable: bool = True,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.recoverable = recoverable
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.category,
            "message": str(self),
            "recoverable": self.recoverable,
            **self.detail,
        }


class NavigationError(BrowserError):
    category = "navigation_error"


class TimeoutError_(BrowserError):
    category = "timeout"


class ElementNotFound(BrowserError):
    category = "element_not_found"


class ElementNotVisible(BrowserError):
    category = "element_not_visible"


class InvalidReference(BrowserError):
    category = "invalid_reference"


class JavascriptError(BrowserError):
    category = "javascript_error"


class NetworkError(BrowserError):
    category = "network_error"


class AssertionFailed(BrowserError):
    category = "assertion_failed"


class PermissionDenied(BrowserError):
    category = "permission_denied"


class InvalidArguments(BrowserError):
    category = "invalid_arguments"


class SessionClosed(BrowserError):
    category = "session_closed"


class ConsentRequired(BrowserError):
    category = "consent_required"


def error_result(
    exc: BrowserError,
    *,
    operation: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "error"}
    payload = exc.to_dict()
    if operation is not None:
        payload["operation"] = operation
    result["error"] = payload
    return result
