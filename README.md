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

O `run.sh` define o workspace como a raiz deste repositório por padrão. Para usar
outro projeto, sobrescreva:

```bash
AGENT_MINI_WORKSPACE=~/projetos/app ./run.sh chat
# ou: ./run.sh chat --workspace ~/projetos/app
```

No `config.json`, `"workspace": null` faz o agent-mini usar
`~/.agent-mini/workspace`; o `setup` agora grava o diretório atual (ou
`AGENT_MINI_WORKSPACE`) para evitar esse fallback.

## Providers

O provider ativo fica em `~/.agent-mini/config.json` (`"provider": "opencode"`,
`"lmstudio"` ou `"stackspot"`). Variáveis de ambiente têm prioridade sobre o arquivo:

| Provider | Variáveis |
|---|---|
| opencode | `OPENCODE_API_KEY`, `OPENCODE_MODEL`, `OPENCODE_BASE_URL`, `OPENCODE_MAX_TOKENS` |
| lmstudio | `LMSTUDIO_MODEL`, `LMSTUDIO_BASE_URL`, `LMSTUDIO_API_KEY`, `LMSTUDIO_MAX_TOKENS` |
| stackspot | `STACKSPOT_CLIENT_ID`, `STACKSPOT_CLIENT_SECRET`, `STACKSPOT_AGENT_ID`, `STACKSPOT_REALM`, `STACKSPOT_BASE_URL`, `STACKSPOT_AUTH_URL`, `STACKSPOT_KNOWLEDGE`, `STACKSPOT_RETURN_KS` |

O provider `opencode` lê `/v1/models` e usa o contexto real (~1M tokens) e as
capabilities do modelo. Modelos do Go são classificados como tier `cloud`
(orçamento de 32k tokens, 25 iterações, resultados de tool até 50k chars); um
`agent.tier` ou `agent.contextWindow` explícito no config sobrescreve isso.

O limite de resposta é 4096 tokens por padrão (proteção do agent-mini para modelos
pequenos). Para respostas longas:

```bash
export OPENCODE_MAX_TOKENS=16384
# ou, no config: "providers": { "opencode": { "maxTokens": 16384 } }
```

### StackSpot (agent chat)

O endpoint `POST /v1/agent/<AGENT_ID>/chat` não é OpenAI-compatible: recebe um
`user_prompt` único, não tem parâmetro `tools` e usa OAuth2 `client_credentials`
em `https://idm.stackspot.com/<REALM>/oidc/oauth/token`. O adapter
`src/providers/stackspot.py`:

- autentica e cacheia o token; renova antes de expirar e em `401`
  (`authenticate()` / `refresh_token()`, com fallback de `refresh_token` para
  `client_credentials`);
- achata o histórico do agent-mini em `user_prompt` e envia
  `use_conversation: false`;
- embute os schemas das tools e o protocolo `<tool_call>{...}</tool_call>`;
- converte a resposta em `ToolCall` (aceita `<tool_call>`, `[TOOL:...]`,
  JSON em bloco cerca e JSON puro).

```bash
export STACKSPOT_CLIENT_ID="..."
export STACKSPOT_CLIENT_SECRET="..."
export STACKSPOT_AGENT_ID="01K..."
export STACKSPOT_REALM="stackspot-freemium"   # opcional
./run.sh setup --provider stackspot --force
./run.sh doctor
./run.sh chat
```

`STACKSPOT_KNOWLEDGE` (padrão `true`) e `STACKSPOT_RETURN_KS` (padrão `false`)
controlam `stackspot_knowledge` e `return_ks_in_response`. Como as tools são
descritas no prompt, a qualidade depende de o agente StackSpot seguir o protocolo.

Trocar de provider numa sessão: `/model opencode/deepseek-v4.1-flash` ou
`/model lmstudio/qwen3.5-9b` (o comando `/model` do agent-mini). Também é possível
usar `./run.sh chat --provider lmstudio`.

## Aprovação: só fora do workspace

O agent-mini não tem confirmação por caminho: `tools.confirm` lista apenas nomes de
tools. Este repositório instala uma política em `src/policy.py`:

- `write_file`/`edit_file`: roda sem perguntar quando o `path` resolve dentro do workspace;
- `shell`: roda sem perguntar quando o comando não referencia caminho absoluto, `~`,
  `..`, link simbólico que escape, `$`/backtick ou `cd` sem destino;
- qualquer caso externo cai no prompt normal do agent-mini (y/N/always/diff).

Exemplos sem prompt: `cat notes.txt`, `echo x > out.txt`, `cd src && pytest`.
Exemplos com prompt: `cat /etc/passwd`, `cd ..`, `echo $HOME`,
`python -c "open('/etc/passwd')"`.

É uma heurística, não uma fronteira de segurança: comandos que escrevem fora sem
caminho explícito (por exemplo `pip install` global) não são detectados. As tools de
arquivo continuam presas ao sandbox `workspace`. Para desligar a política:

```bash
AGENT_MINI_CONFIRM_POLICY=off ./run.sh chat
```

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
    base.py            # OpenAIProvider com limite de resposta configurável
    opencode.py        # OpenCode Go
    lmstudio.py        # LM Studio / OpenAI-compatible local
    stackspot.py       # StackSpot agent chat (OAuth2 + prompt tools)
  policy.py            # aprovação só fora do workspace
config.example.json    # modelo de configuração
run.sh                 # wrapper do CLI usando o .venv
```
