# Agente com rodinhas

Versão do agente adaptada para rodar com modelos de linguagem **muito pequenos**
(daí o apelido "bicicleta com rodinhas"). Enquanto a branch `main` assume um
modelo grande (default `big-pickle` do OpenCode Zen), esta branch prioriza:

- **economia de contexto** — prompts curtos, manuais enxutos, história
  compactada;
- **menor exigência cognitiva** — instruções e protocolos de chamada de
  ferramenta simples que um modelo pequeno consiga seguir com robustez;
- **comportamento determinístico** — sempre que possível, decisões saem do
  loop de inferência (ex.: `classify_followup` já resolve aprovação/cancelamento
  sem round-trip do modelo).

> Status: a adaptação está em andamento. Neste momento a branch ainda não
> diverge de `main`; os ajustes de prompt e de protocolo serão aplicados
> incrementalmente aqui.

## Como usar

```bash
cp .env.example .env   # configurar provedor/modelo (ver abaixo)
.venv/bin/python src/main.py   # exige OPENCODE_API_KEY para o demo interativo
```

## Configuração (`.env`)

O provedor e suas opções são ajustados por variáveis de ambiente. Veja
`.env.example` para o template completo.

| Variável | Padrão | Descrição |
| -------- | ------ | --------- |
| `AGENT_PROVIDER` | `opencode` | Provedor: `opencode` ou `lmstudio` |
| `OPENCODE_API_KEY` | — | Token do OpenCode Zen |
| `OPENCODE_MODEL` | `big-pickle` | Modelo do OpenCode Zen |
| `OPENCODE_BASE_URL` | `https://opencode.ai/zen/v1` | Base da API |
| `LMSTUDIO_MODEL` | — | Modelo carregado no LM Studio (obrigatório) |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | Base do servidor local |
| `LMSTUDIO_API_KEY` | — | Token bearer (opcional) |
| `LMSTUDIO_TIMEOUT` / `OPENCODE_TIMEOUT` | `60` | Timeout das chamadas |
| `AGENT_LOG` | off | Grava respostas cruas da LLM + resultado do parse em JSONL |

O parser de chamadas (`src/toolparse.py`) repara saídas fora do padrão (aspas
simples, chaves sem aspas, JSON truncado, `<|tool_call>call:<tool>` com JSON
solto); só em último caso faz um único re-prompt pedindo JSON estrito.

## Estrutura

- `src/tools/` — registro de ferramentas de filesystem e facades.
- `src/orchestrator/` — ferramenta especial `orchestrator` (descobrir → planejar
  → executar → validar), opcional.
- `src/agent.py` — loop agente: prompt de config → `provider.infer` →
  `parse_tool_call` → `registry.dispatch`.
- `src/providers/opencode.py` — cliente OpenAI-compatible para OpenCode Zen.
- `src/context.py` / `src/memory.py` — compressão de contexto e memória.
- `src/utils.py` — prompt de configuração, parsing de chamadas, `classify_followup`.
- `src/registry.py` — varre provedores MCP com `run.sh`/`run.bat`.

## Testes e ferramentas

```bash
.venv/bin/python -m unittest discover -s tests   # testes
.venv/bin/python -m mypy src tests               # tipos
.venv/bin/python -m black src tests              # formatação
```

## Versões do projeto

| Branch | Público-alvo | Estratégia |
| ------ | ------------ | ---------- |
| `main` | modelos grandes | prompt rico, ferramentas completas |
| `bicicleta-com-rodinhas` | modelos muito pequenos | prompts minimalistas, fluxos determinísticos |
