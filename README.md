# Código temporário — agent-mini

Repositório mínimo: [agent-mini](https://pypi.org/project/agent-mini/0.7.0/) com dois
providers customizados, mantidos do projeto anterior:

- **OpenCode Go** (`opencode`) — `https://opencode.ai/inference/go/v1`, modelo `deepseek-v4.1-flash`
- **LM Studio** (`lmstudio`) — `http://localhost:1234/v1`, qualquer modelo local

Todo o framework anterior (orchestrator, discovery, navigation, tools, testes) foi
removido. A integração é um patch em `agent_mini.providers.create_provider` e
`agent_mini.config.PROVIDERS`, então o CLI original do agent-mini continua intacto.

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh setup                 # escreve ~/.agent-mini/config.json
export OPENCODE_API_KEY="..."  # chave do OpenCode Go
./run.sh doctor                # valida provider, modelo e workspace
./run.sh chat
```

Config alternativo (sem tocar em `~/.agent-mini`):

```bash
AGENT_MINI_HOME=/caminho/para/.agent-mini ./run.sh setup
```

## Providers

O provider ativo fica em `~/.agent-mini/config.json` (`"provider": "opencode"` ou
`"lmstudio"`). Variáveis de ambiente têm prioridade sobre o arquivo:

| Provider | Variáveis |
|---|---|
| opencode | `OPENCODE_API_KEY`, `OPENCODE_MODEL`, `OPENCODE_BASE_URL` |
| lmstudio | `LMSTUDIO_MODEL`, `LMSTUDIO_BASE_URL`, `LMSTUDIO_API_KEY` |

Trocar de provider numa sessão: `/model opencode/deepseek-v4.1-flash` ou
`/model lmstudio/qwen3.5-9b` (o comando `/model` do agent-mini). Também é possível
usar `./run.sh chat --provider lmstudio`.

## Uso

```bash
./run.sh setup --provider lmstudio --workspace ~/projetos/app
./run.sh chat -m "Quantos TODOs existem em src/?"
./run.sh chat --provider opencode --model deepseek-v4.1-flash
./run.sh chat --help
```

## Estrutura

```
src/
  main.py              # entrypoint: registra os providers e roda o CLI do agent-mini
  providers/
    __init__.py        # create_provider + register()
    base.py            # OpenAIProvider com limite de resposta
    opencode.py        # OpenCode Go
    lmstudio.py        # LM Studio / OpenAI-compatible local
config.example.json    # modelo de configuração
run.sh                 # wrapper do CLI usando o .venv
```
