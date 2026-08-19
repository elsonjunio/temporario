from __future__ import annotations

import re

from src.toolparse import is_tool_attempt

# ---------------------------------------------------------------------------
# Pattern sets – bilingual (PT-BR + EN)
# ---------------------------------------------------------------------------

# Responses where the model *describes* what it will do without doing it.
_THINKING_PATTERNS: list[re.Pattern[str]] = [
    # PT-BR
    re.compile(
        r"\bvou\s+(verificar|ver|ler|analisar|checar|examinar|buscar|procurar"
        r"|inspecionar|revisar|consultar|abrir|fechar|listar"
        r"|mostrar|exibir|imprimir|escrever|atualizar|deletar|remover"
        r"|copiar|mover|renomear|criar|instalar|configurar|executar|rodar"
        r"|testar|validar|confirmar|aprovar|iniciar|começar|continuar)",
        re.IGNORECASE,
    ),
    re.compile(r"\bdeixe[\s-]me\b", re.IGNORECASE),
    re.compile(r"\bdeixa eu\b", re.IGNORECASE),
    re.compile(r"\bagora vou\b", re.IGNORECASE),
    re.compile(
        r"\bpreciso\s+(verificar|ver|ler|analisar|checar|examinar|buscar"
        r"|procurar|inspecionar|revisar|consultar)",
        re.IGNORECASE,
    ),
    re.compile(r"\bvou\s+começar\b", re.IGNORECASE),
    re.compile(r"\bprimeiro,?\s*vou\b", re.IGNORECASE),
    re.compile(
        r"\bantes,?\s*(de|do|da)\s+(fazer|executar|rodar|chamar)", re.IGNORECASE
    ),
    re.compile(r"\bvou\s+precisar\b", re.IGNORECASE),
    re.compile(
        r"\bserá?\s+necessário\s+(verificar|ver|ler|analisar|checar)", re.IGNORECASE
    ),
    # EN
    re.compile(
        r"\blet\s+me\s+(check|see|read|analyze|analyse|inspect|review|examine"
        r"|look|find|search|open|close|list|show|display|print|write|update"
        r"|delete|remove|copy|move|rename|create|install|configure|run|test"
        r"|validate|confirm|start|begin|continue|verify)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi('ll| will|'m going to)\s+(check|see|read|analyze|analyse|inspect"
        r"|review|examine|look|find|search|open|close|list|show|display|print"
        r"|write|update|delete|remove|copy|move|rename|create|install|configure"
        r"|run|test|validate|confirm|start|begin|continue|verify)",
        re.IGNORECASE,
    ),
    re.compile(r"\bnow\s+i('ll| will|'m going to)\b", re.IGNORECASE),
    re.compile(
        r"\bi\s+need\s+to\s+(check|see|read|analyze|analyse|inspect|review"
        r"|examine|look|find|search|verify|validate|confirm)",
        re.IGNORECASE,
    ),
    re.compile(r"\bfirst,?\s*let\s+me\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+(that|doing|anything|I)\b", re.IGNORECASE),
    re.compile(r"\bI('ll| will)\s+first\b", re.IGNORECASE),
]

# Responses where the model describes a plan/approach without acting.
_PLANNING_PATTERNS: list[re.Pattern[str]] = [
    # PT-BR
    re.compile(r"\bmeu\s+plano\s+(é|seria|será)\b", re.IGNORECASE),
    re.compile(r"\ba\s+abordagem\s+(seria|será|é)\b", re.IGNORECASE),
    re.compile(r"\bpara\s+resolver\s+(isso|isto|o\s+problema)\b", re.IGNORECASE),
    re.compile(r"\bo\s+melhor\s+(seria|será|caminho|jeito)\b", re.IGNORECASE),
    re.compile(r"\bseria?\s+necessário\b", re.IGNORECASE),
    re.compile(r"\bprecisamos\s+(de|fazer|verificar|criar)\b", re.IGNORECASE),
    re.compile(r"\ba\s+estratégia\s+(seria|será|é)\b", re.IGNORECASE),
    # EN
    re.compile(r"\bmy\s+plan\s+(is|would\s+be)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+approach\s+(would\s+be|is)\b", re.IGNORECASE),
    re.compile(r"\bto\s+solve\s+(this|that|the\s+problem)\b", re.IGNORECASE),
    re.compile(
        r"\bthe\s+best\s+(approach|way|strategy)\s+(would\s+be|is)\b", re.IGNORECASE
    ),
    re.compile(r"\bit\s+would\s+be\s+(necessary|needed|required)\b", re.IGNORECASE),
    re.compile(
        r"\bwe\s+need\s+to\s+(first|start|begin|create|install|configure)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe\s+strategy\s+(would\s+be|is)\b", re.IGNORECASE),
    re.compile(r"\bhere('s| is)\s+what\s+I('ll| will)\s+do\b", re.IGNORECASE),
]

# Responses where the model mentions a tool name in prose (not JSON).
_TOOL_HINT_PATTERNS: list[re.Pattern[str]] = [
    # PT-BR
    re.compile(
        r"\bvou\s+(usar|chamar|invocar|executar)\s+"
        r"(read_file|write_file|patch_file|search_files|grep_files|list_dir"
        r"|run_command|move_file|delete_file|orchestrator|discovery|planner"
        r"|executor|navigation)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdeveria\s+(usar|chamar|invocar)\s+"
        r"(read_file|write_file|patch_file|search_files|grep_files|list_dir"
        r"|run_command|move_file|delete_file|orchestrator|discovery|planner"
        r"|executor|navigation)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdevemos\s+(usar|chamar|invocar|executar)\s+"
        r"(read_file|write_file|patch_file|search_files|grep_files|list_dir"
        r"|run_command|move_file|delete_file|orchestrator|discovery|planner"
        r"|executor|navigation)",
        re.IGNORECASE,
    ),
    # EN
    re.compile(
        r"\bi('ll| will|'m going to)\s+(use|call|invoke|run)\s+"
        r"(read_file|write_file|patch_file|search_files|grep_files|list_dir"
        r"|run_command|move_file|delete_file|orchestrator|discovery|planner"
        r"|executor|navigation)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bshould\s+(use|call|invoke|run)\s+"
        r"(read_file|write_file|patch_file|search_files|grep_files|list_dir"
        r"|run_command|move_file|delete_file|orchestrator|discovery|planner"
        r"|executor|navigation)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+should\s+(use|call|invoke|run)\s+"
        r"(read_file|write_file|patch_file|search_files|grep_files|list_dir"
        r"|run_command|move_file|delete_file|orchestrator|discovery|planner"
        r"|executor|navigation)",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Phantom action patterns – the model claims to have DONE something (past
# tense, mutations only) without actually calling a tool.  Read-only verbs
# (check, read, analyze, etc.) are intentionally excluded.
# ---------------------------------------------------------------------------

# PT-BR: explicit past-tense mutation verbs (1st/2nd/3rd person).
# Infinitive forms (criar, atualizar, etc.) are intentionally excluded.
_PHANTOM_PT_VERBS = (
    # criação
    r"criei|criou|criamos"
    r"|adicionei|adicionou|adicionamos"
    r"|gerei|gerou|geramos"
    r"|produzi|produziu|produzimos"
    # atualização / modificação
    r"|atualizei|atualizou|atualizamos"
    r"|modifiquei|modificou|modificamos"
    r"|alterei|alterou|alteramos"
    r"|editei|editou|editamos"
    r"|refatorei|refatorou|refatoramos"
    r"|migrei|migrou|migramos"
    r"|corrigi|corrigiu|corrigimos"
    r"|reparei|reparou|reparamos"
    r"|renomeei|renomeou|renomeamos"
    r"|movi|moveu|movemos"
    r"|copiei|copiou|copiamos"
    # exclusão
    r"|removi|removeu|removemos"
    r"|delet[ei]|deletou|deletamos"
    r"|apaguei|apagou|apagamos"
    r"|exclu[ií]|excluiu|excluimos"
    # instalação / configuração / execução
    r"|instalei|instalou|instalamos"
    r"|configurei|configurou|configuramos"
    r"|execut[ei]|executou|executamos"
    r"|rod[ei]|rodou|rodamos"
    r"|compilei|compilou|compilamos"
    r"|testei|testou|testamos"
    r"|implantei|implantou|implantamos"
    r"|deployei|deployou|deployamos"
)

# EN: explicit past-tense mutation verbs.  Base forms (create, update, etc.)
# are intentionally excluded to avoid false positives with "we need to create".
_PHANTOM_EN_VERBS = (
    r"created|updated|deleted|removed|installed|configured"
    r"|fixed|refactored|migrated|renamed|moved|copied"
    r"|edited|modified|compiled|tested|deployed|executed"
    r"|cleared|set\s+up|put\s+together|written"
)

_PHANTOM_ACTION_PATTERNS: list[re.Pattern[str]] = [
    # PT-BR – "Criei o arquivo", "Pronto, já atualizei"
    re.compile(rf"\b({_PHANTOM_PT_VERBS})\b", re.IGNORECASE),
    # PT-BR – passive: "O arquivo foi criado", "A config foi atualizada"
    re.compile(
        r"\b[oa]\s+(arquivo|código|modulo|módulo|config|configuração"
        r"|script|função|classe|serviço|servico|endpoint|rota"
        r"|variável|dependência|dependencia|pacote|package|ambiente)"
        r"\s+(foi\s+)?(criad|atualiz|modific|alterad|editad|deletad"
        r"|removid|instalad|configurad|migrad|corrigid"
        r"|refatorad|renomead|movid|copiad|excluid|apagad)",
        re.IGNORECASE,
    ),
    # EN – "I created the file", "I've updated the config"
    re.compile(
        r"\bI\s+(" + _PHANTOM_EN_VERBS + r")\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI('ve|\s+have|\s+'s)\s+(" + _PHANTOM_EN_VERBS + r")\b",
        re.IGNORECASE,
    ),
    # EN – passive: "The file has been created", "The config was updated"
    re.compile(
        r"\b(the|a|an)\s+(file|code|module|config|configuration"
        r"|script|function|class|service|endpoint|route"
        r"|variable|dependency|package|environment)"
        r"\s+(has\s+been|was|were)\s+"
        r"(created|updated|deleted|removed|installed|configured"
        r"|fixed|refactored|migrated|renamed|moved|copied"
        r"|edited|modified|compiled|tested|deployed|executed"
        r"|cleared|set\s+up)",
        re.IGNORECASE,
    ),
]

# Negation before a mutation verb → legitimate negative answer, NOT phantom.
_PHANTOM_NEGATION_RE = re.compile(
    r"\b(não|nao|no|not|never|didn'?t|doesn'?t|hasn'?t|haven'?t" r"|nunca|jamais)\b",
    re.IGNORECASE,
)

# Questions are not phantom actions – the model is asking the user.
_PHANTOM_QUESTION_RE = re.compile(
    r"\?\s*$",
    re.MULTILINE,
)

# Code blocks / structured data → legitimate answer.
_PHANTOM_CODE_RE = re.compile(
    r"(?:^|\n)\s*```",
    re.DOTALL,
)

# Indecision / stalling filler that adds no value.
# NOTE: "ok"/"okay" are intentionally excluded because they can be legitimate
# short answers (e.g. acknowledging a question).
_STALLING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:hmm|hum|hm|bem|well|então|so|umm|uhh|ah|eh)\s*[.!?.]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:interessante|interesting|entendo|i\s+see|right|vejo)\s*[.!?.]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:deixe[\s-]me\s+pensar|let\s+me\s+think)\s*[.!?.]*$", re.IGNORECASE
    ),
]

# Signals that the response is a genuine final answer (low false-positive risk).
# NOTE: file_path/filePath are intentionally excluded because they also appear in
# tool-call parameters and would cause false positives.
_ANSWER_SIGNALS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|\n)\s*(?:```|\"|\')", re.DOTALL),  # code blocks or quoted data
    re.compile(r"\bresumo\b.*:", re.IGNORECASE),  # "Resumo: ..."
    re.compile(r"\bsummary\b.*:", re.IGNORECASE),  # "Summary: ..."
    re.compile(r"\bresultado\b.*:", re.IGNORECASE),  # "Resultado: ..."
    re.compile(r"\bresult\b.*:", re.IGNORECASE),  # "Result: ..."
]

# Re-prompt budget: max consecutive re-prompts per iteration before giving up.
_MAX_REPROMPTS = 3

# Responses longer than this are almost certainly real answers.
_LONG_RESPONSE_THRESHOLD = 500


def classify_response_intent(text: str) -> str:
    """Classify an LLM response that was *not* a tool call.

    Returns one of:
    - ``"thinking"``       – describes what it will do without acting
    - ``"planning"``       – describes a plan/approach without acting
    - ``"tool_hint"``      – mentions a tool name in prose (not JSON)
    - ``"phantom_action"`` – claims to have done something without calling a tool
    - ``"stalling"``       – filler / indecision with no substance
    - ``"answer"``         – legitimate final answer
    """
    stripped = text.strip()

    # Empty / whitespace-only → stalling.
    if not stripped:
        return "stalling"

    # Very long → almost certainly a real answer.
    if len(stripped) > _LONG_RESPONSE_THRESHOLD:
        return "answer"

    # Tool hint (check before answer signals because tool calls may contain
    # patterns like ``file_path=`` that would otherwise false-positive).
    for pat in _TOOL_HINT_PATTERNS:
        if pat.search(stripped):
            return "tool_hint"

    # Also delegate to the existing ``is_tool_attempt`` heuristic.
    if is_tool_attempt(stripped):
        return "tool_hint"

    # If the response contains structured data, code, or a clear answer marker
    # it is probably a real answer even if it also matches a pattern.
    for sig in _ANSWER_SIGNALS:
        if sig.search(stripped):
            return "answer"

    # Stalling filler.
    for pat in _STALLING_PATTERNS:
        if pat.fullmatch(stripped):
            return "stalling"

    # Phantom action – the model claims to have done something without calling
    # a tool.  Protected by negation and question checks so that legitimate
    # negative answers and follow-up questions are not flagged.
    if not _PHANTOM_NEGATION_RE.search(stripped) and not _PHANTOM_QUESTION_RE.search(
        stripped
    ):
        if _PHANTOM_CODE_RE.search(stripped):
            pass  # code blocks are real answers
        else:
            for pat in _PHANTOM_ACTION_PATTERNS:
                if pat.search(stripped):
                    return "phantom_action"

    # Thinking – the model described an action without executing it.
    for pat in _THINKING_PATTERNS:
        if pat.search(stripped):
            return "thinking"

    # Planning – the model described a plan/approach without acting.
    for pat in _PLANNING_PATTERNS:
        if pat.search(stripped):
            return "planning"

    return "answer"


def should_reprompt(text: str) -> tuple[bool, str | None]:
    """Determine whether the agent loop should re-prompt instead of returning.

    Returns ``(True, reason)`` when the model should be asked to try again,
    or ``(False, None)`` when the response is a legitimate final answer.

    ``reason`` is a short, actionable instruction that is injected into the
    next prompt so the model knows *why* it is being re-prompted.
    """
    stripped = text.strip()

    # Empty / whitespace-only.
    if not stripped:
        return True, "Your response was empty."

    intent = classify_response_intent(stripped)

    if intent == "answer":
        return False, None

    _REASONS: dict[str, str] = {
        "thinking": (
            "You described what you would do instead of doing it. "
            "Do NOT narrate your actions – call the tool now by emitting "
            "the JSON block."
        ),
        "planning": (
            "You described a plan instead of acting on it. "
            "Do NOT explain your approach – call the tool now by emitting "
            "the JSON block."
        ),
        "tool_hint": (
            "You mentioned a tool in prose instead of calling it properly. "
            "Emit a valid JSON tool-call block now."
        ),
        "phantom_action": (
            "You claimed to have done something but no tool was actually "
            "called. Do NOT claim to have completed a mutation – actually "
            "call the tool now by emitting the JSON block."
        ),
        "stalling": (
            "Your response was empty or contained only filler. "
            "Call the tool now with a valid JSON block, or give your final answer."
        ),
    }

    return True, _REASONS[intent]
