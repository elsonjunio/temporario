from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.planning.plan import as_list


class RiskLevel:
    """Risk levels for executor decisions (confirmation gates, reporting).

    Ordering: LOW < MEDIUM < HIGH < CRITICAL. ``NEVER`` is a sentinel used by
    the confirmation configuration to disable the gate entirely.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    NEVER = "NEVER"

    _ORDER = (LOW, MEDIUM, HIGH, CRITICAL)
    _POSITION = {level: index for index, level in enumerate(_ORDER)}

    def __init__(self) -> None:
        raise TypeError("RiskLevel is a namespace of constants")

    @classmethod
    def all(cls) -> list[str]:
        return list(cls._ORDER)

    @classmethod
    def is_valid(cls, level: str) -> bool:
        return level in cls._POSITION

    @classmethod
    def from_name(cls, name: str | None) -> str | None:
        """Parse a level name (case-insensitive); ``None`` when unknown."""
        if not name:
            return None
        for level in cls._ORDER:
            if name.strip().upper() == level:
                return level
        return None

    @classmethod
    def at_least(cls, level: str, threshold: str) -> bool:
        """True when ``level`` is at or above ``threshold`` (CRITICAL highest).
        ``NEVER`` as threshold disables the gate."""
        if threshold == cls.NEVER:
            return False
        if level not in cls._POSITION or threshold not in cls._POSITION:
            return False
        return cls._POSITION[level] >= cls._POSITION[threshold]

    @classmethod
    def max(cls, *levels: str) -> str:
        return max(levels, key=lambda level: cls._POSITION.get(level, -1))


#: Markers that classify a task as CRITICAL when found in its text. Order in
#: ``_RULES``: the first rule whose markers hit decides the level.
_CRITICAL_MARKERS = (
    "delete",
    "deletar",
    "exclu",
    "remov",
    "drop",
    "truncate",
    "rm ",
    "rm -",
    "credential",
    "secret",
    "senha",
    "password",
    "api key",
    "apikey",
    "access token",
    "security",
    "seguranca",
    "segurança",
    "auth key",
    ".env",
    "id_rsa",
    "irrevers",
    "destructive",
    "destrutiv",
    "data loss",
    "perda de dados",
    "fora do workspace",
    "outside the workspace",
    "database",
    "dados sensiveis",
    "dados sensíveis",
)

_HIGH_MARKERS = (
    "banco de dados",
    "banco",
    "database",
    "db:",
    "migration",
    "migrate",
    "schema",
    "infra",
    "infraestrutura",
    "deploy",
    "dependencia",
    "dependência",
    "dependency",
    "requirements",
    "pip install",
    "npm install",
    "go get",
    "cargo add",
    "apt install",
    "abrangente",
    "comprehensive",
    "many files",
    "varios arquivos",
    "vários arquivos",
    "scalability",
    "auth",
    "autenticacao",
    "autenticação",
)

_MEDIUM_MARKERS = (
    "api",
    "behavior",
    "comportamento",
    "multiplos componentes",
    "múltiplos componentes",
    "multi-component",
    "configuracao",
    "configuração",
    "config",
    "integration",
    "integracao",
    "integração",
    "env var",
    "variavel de ambiente",
    "variável de ambiente",
)

_LOW_MARKERS = (
    "css",
    "stylesheet",
    "documentacao",
    "documentação",
    "docs",
    "readme",
    "testes",
    "test",
    "typography",
    "tipografia",
    "spacing",
    "visual",
    "ui",
)


def _text_of(task: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "objective",
        "context",
        "expected_changes",
        "evidence",
        "title",
        "constraints",
        "risks",
        "files",
        "acceptance_criteria",
    ):
        value = task.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts).lower()


def classify_task(task: dict[str, Any]) -> str:
    """Classify a task's overall risk.

    An explicit ``task["risk"]`` (when present and valid) wins; otherwise the
    classifier scans the task text against the keyword rules, returning the
    highest level matched (default LOW).
    """
    explicit = RiskLevel.from_name(str(task.get("risk") or ""))
    if explicit is not None:
        return explicit

    text = _text_of(task)
    rules = (
        (RiskLevel.CRITICAL, _CRITICAL_MARKERS),
        (RiskLevel.HIGH, _HIGH_MARKERS),
        (RiskLevel.MEDIUM, _MEDIUM_MARKERS),
        (RiskLevel.LOW, _LOW_MARKERS),
    )
    for level, markers in rules:
        if any(marker in text for marker in markers):
            return level
    return RiskLevel.LOW


#: Command substrings that make a run_command call destructive/critical.
_CRITICAL_COMMAND_MARKERS = (
    "rm -rf",
    "rm -fr",
    "rm -r",
    "drop database",
    "drop table",
    "drop schema",
    "truncate table",
    "git reset --hard",
    "git clean -fdx",
    "git clean -fd",
    "chmod -R",
    "chown -R",
    ":(){",
    "mkfs",
    "format ",
    "shutdown",
    "kill -9",
    "pip uninstall",
    "npm uninstall",
    "yarn remove",
    "del /s",
    "rd /s",
    "> /dev/sda",
)

#: Command substrings that mark a HIGH-risk command (deps, migrations, infra).
_HIGH_COMMAND_MARKERS = (
    "pip install",
    "npm install",
    "yarn add",
    "go get",
    "cargo add",
    "apt-get install",
    "apt install",
    "brew install",
    "pipenv install",
    "poetry add",
    "alembic",
    "django migrate",
    "migrate",
    "rake db",
    "prisma migrate",
    "docker-compose",
    "docker compose",
    "kubectl",
    "terraform",
    "ansible",
)

#: File name fragments treated as HIGH risk when patched/written.
_SECRET_PATH_MARKERS = (
    ".env",
    "secret",
    "credential",
    "password",
    "id_rsa",
    ".pem",
    ".key",
    "api-key",
    "api_key",
    "access-token",
    "auth",
)


def classify_action(tool: str, action: str, params: dict[str, Any]) -> str:
    """Classify a single tool call's risk, deterministically.

    Read-only tools are LOW. Destructive filesystem operations are CRITICAL.
    Writes outside the workspace root are CRITICAL. Dependency/migration
    commands and patches touching secret files are HIGH. Everything else that
    mutates is MEDIUM.
    """
    if tool in ("read_file", "list_dir", "search_files", "grep_files"):
        return RiskLevel.LOW
    if tool in ("write_file", "patch_file", "move_file", "delete_file"):
        if tool == "delete_file":
            return RiskLevel.CRITICAL
        path = str(
            params.get("file_path")
            or params.get("path")
            or params.get("destination")
            or params.get("old_path")
            or ""
        )
        lowered = path.lower()
        if any(marker in lowered for marker in _SECRET_PATH_MARKERS):
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM
    if tool == "run_command":
        command = str(params.get("command") or "")
        lowered = command.lower()
        if any(marker in lowered for marker in _CRITICAL_COMMAND_MARKERS):
            return RiskLevel.CRITICAL
        if any(marker in lowered for marker in _HIGH_COMMAND_MARKERS):
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM
    return RiskLevel.MEDIUM


def outside_workspace(path: str, root: str) -> bool:
    """True when a resolved path escapes the workspace root."""
    try:
        resolved = str(Path(path).expanduser().resolve())
        root_resolved = str(Path(root).resolve())
    except OSError:
        return True
    if root_resolved == ".":
        return False
    return not resolved.startswith(root_resolved)


def risk_description(level: str) -> str:
    """Human-readable one-liner for a risk level (used by presenters)."""
    descriptions = {
        RiskLevel.LOW: "CSS, documentação, testes, alterações locais.",
        RiskLevel.MEDIUM: "APIs, comportamento, múltiplos componentes, "
        "configurações.",
        RiskLevel.HIGH: "banco, infraestrutura, dependências, alterações "
        "abrangentes.",
        RiskLevel.CRITICAL: "exclusão, operações destrutivas, segurança, "
        "credenciais, dados, operações fora do workspace.",
    }
    return descriptions.get(level, "")


def top_risk(tasks: Iterable[dict[str, Any]]) -> str:
    """Highest risk across a collection of tasks (plan summary)."""
    levels = [classify_task(task) for task in tasks]
    if not levels:
        return RiskLevel.LOW
    return RiskLevel.max(*levels)


__all__ = [
    "RiskLevel",
    "classify_action",
    "classify_task",
    "outside_workspace",
    "risk_description",
    "top_risk",
]
