# Subagente de Navegação Web (`browser`)

Subagente que navega na web com **Playwright**, fazendo *discovery* (snapshot da
página), *planning* (escolha das ações) e *navegação/interação*, devolvendo
informações estruturadas e executando ações solicitadas pelo agente principal.

## Arquitetura

O registry principal recebe **um único entry** — `browser`. As dezenas de
operações de navegação **não** entram no registry principal; elas vivem em um
`ToolRegistry` privado do subagente e no `BrowserController`.

```
registry principal (read_file, ..., orchestrator, BROWSER)
      │  browser run "tarefa" | browser snapshot | browser click e3 | ...
      ▼
src/navigation/browser (ToolSpec)
      │  run → NavigationSubAgent (loop LLM próprio, registry privado)
      │  ações diretas → BrowserController
      ▼
BrowserSession (browser · context · page · refs · console · network · trace)
      ▼
Playwright (Chromium headless)
```

Princípio de baixo acoplamento (igual ao orchestrator): remover o entry
(`registry.unregister("browser")`) mantém o agente funcionando com os tools
base.

### Múltiplas abas

O subagente e as ações diretas operam sobre a **aba ativa**. Ações:

- **`new_tab`** — abre uma nova aba e a torna ativa (opcionalmente navega para
  `url`). 
- **`switch_tab`** — seleciona por `index` (int), `url` (substring) ou
  `title` (substring); exatamente uma aba deve corresponder (ambíguo → erro
  `invalid_arguments`).
- **`close_tab`** — fecha a aba atual (ou a indicada); a próxima aba restante
  vira ativa; `all_closed: true` quando nenhuma resta.
- **`list_tabs`** — `index`, `url`, `title`, `active`, `ref_count` de cada aba.
- **`status`** — agora inclui `tabs` e `active_index`.

Os **refs de snapshot são por aba** (`refs`/`ref_epoch` seguem o `switch_tab`);
cada aba guarda seu próprio snapshot e referências. Eventos de console, erros
de página e requisições/respostas HTTP são **etiquetados com o índice da aba**
(`"page": 0/1/...`), o que permite diagnosticar qual aba gerou cada log.

## Modo headed + proxy/credenciais com consentimento

### Navegador visível (headed)

O Chromium roda headless por padrão. Para controle humano (janela visível +
inspector do Playwright), a precedência é: parâmetro `headless` de
`create_navigation_tool` > env `BROWSER_HEADED=1` > env `BROWSER_HEADLESS`
(default `1`). Com `BROWSER_HEADED=1` a janela abre no desktop e `pause`
pausa a execução abrindo o inspector para o operador depurar interativamente.

### Proxy e credenciais de site

`create_navigation_tool(..., proxy=ProxyConfig(...), site_credentials=...)`
configura um proxy e credenciais HTTP por origem. As fontes preferidas são
esses argumentos; na ausência, `credentials_store` (ou o arquivo
`BROWSER_CREDENTIALS_FILE`, default `.browser-credentials.json`) é consultado:

```json
{
  "proxy": {"server": "http://gate:3128", "username": "u", "password": "p"},
  "sites": {"https://api.example.com": {"username": "svc", "password": "s3cret"}}
}
```

**Consentimento explícito**: enquanto proxy/credenciais estão configurados,
toda ação de navegação falha com `consent_required` e **nenhum navegador é
lançado** — nada sai pela rede até o operador humano aprovar. A aprovação é

- `consent` exige `confirm=true` (um `consent` sem confirmação é rejeitado);
  só deve ser usado após o operador verificar `list_credentials` (que nunca
  expõe senhas — só presença, username mascarado e host do proxy sem userinfo);
- `revoke_consent` re-locka e **fecha a sessão** (o contexto com credenciais
  deixa de existir; o próximo `open` exige consentimento novamente).

Após o consentimento, o proxy e `http_credentials` (com `origin` e
`send: "unauthorized"` — credenciais só enviadas após desafio 401 do servidor
daquela origem) são aplicados na criação do `context`. O Playwright suporta
**um** objeto `http_credentials` por context, então a primeira origem
configurada é aplicada; as demais são reportadas mas não aplicadas.

`status` inclui `consent: {required, granted, proxy, credentials}` e
`list_credentials` devolve os detalhes mascarados. Tudo funciona igual em modo
isolado (o worker recebe proxy/credenciais no `__init__`).

## Isolamento em subprocesso (IPC)

Por padrão o `browser` roda no processo do agente. Com `BROWSER_ISOLATED=1`
(ou `create_navigation_tool(..., isolated=True)`), o navegador (Chromium +
Playwright) roda em um **processo filho separado** — `src/navigation/worker.py`
— comunicando por **JSON-lines sobre stdio**:

```
Agente (BrowserClient)  ── {"id":1,"method":"open","params":{...}} ──►  Worker
                         ◄─ {"id":1,"result":{...}} ───────────────────  (Chromium)
```

- O client (`src/navigation/ipc.py::BrowserClient`) espelha a superfície do
  `BrowserController`: qualquer atributo é encaminhado como chamada RPC, e
  toda resposta (incluindo erros estruturados) volta como dict — o agente não
  distingue os dois modos.
- Um crash ou um navegador travado **não derruba o agente**: o worker é
  reenviado/reiniciado (as chamadas pós-crash retornam `session_closed`).
- Configuração é passada por `__init__` (timeouts, viewport, headless, policy
  allow/block hosts) e o worker constrói o controller sob demanda.
- `close` encerra o navegador e **recolhe o processo filho** (`wait`).
- Em modo isolado, após `close` o worker não existe mais: qualquer operação
  subsequente responde `session_closed` (equivalente ao modo in-process).

## Arquivos

```
src/navigation/
    __init__.py       create_navigation_tool(provider, isolated=..., proxy=..., ...) → ToolSpec
    errors.py         taxonomia de erros estruturados
    credentials.py    ProxyConfig / SiteCredentials / CredentialStore (arquivo local)
    security.py       validação de URL/hosts + mascaramento
    session.py        BrowserSession (estado da sessão, abas, refs por aba)
    snapshot.py       snapshot a11y/DOM + refs estáveis (e1, e2, ...)
    trace.py          extração de screenshots do trace (screencast-frame)
    controller.py     BrowserController (operações de baixo nível + consentimento)
    tools.py          ToolSpec `browser` + MANUAL (superfície LLM)
    subagent.py       NavigationSubAgent (loop LLM sobre o toolset)
    ipc.py            BrowserClient (RPC JSON-lines para o worker)
    worker.py         processo filho que hospeda o BrowserController
```

## Instalação

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
```

- Dependência adicionada ao `requirements.txt`: `playwright`.
- Funciona no Linux e no Windows (API sync do Playwright, sem dependência de
  subprocessos adicionais).

## Registro

```python
from src.navigation import create_navigation_tool

navigation = create_navigation_tool(provider, max_steps=12)
registry.register(navigation.name, navigation)
```

Em `src/main.py` o tool é registrado junto ao orchestrator automaticamente.

## Interface exposta ao agente

O entry `browser` expõe:

- **`run`** — delega uma tarefa em linguagem natural ao subagente
  (discovery + planning + execução). Params: `request` (str, obrigatório),
  `url` (str, opcional — abre primeiro), `max_steps` (int, opcional).
- **Ações diretas** — executam imediatamente na sessão viva, sem round-trip de
  LLM: `open`, `goto`, `reload`, `back`, `forward`, `wait`, `snapshot`,
  `snapshot_map`, `inspect`, `click`, `double_click`, `fill`, `type`, `press`, `select`,
  `check`, `uncheck`, `hover`, `focus`, `scroll`, `drag`, `upload`,
  `screenshot`, `screenshot_element`, `evaluate`, `set_html`, `set_text`,
  `set_attribute`, `remove_attribute`, `set_style`, `add_class`,
  `remove_class`, `insert_html`, `remove_element`, `console`, `clear_console`,
  `page_errors`, `network`, `get_request`, `get_response`, `clear_network`,
  `get_source`, `list_scripts`, `get_script`, `list_stylesheets`, `assert`,
  `set_viewport`, `cookies`, `local_storage`, `session_storage`,
  `clear_storage`, `trace_start`, `trace_stop`, `new_tab`, `switch_tab`,
  `close_tab`, `list_tabs`, `consent`, `revoke_consent`, `list_credentials`,
  `video`, `video_save`, `pause`, `status`, `close`.

### Referências (`ref`)

`snapshot` produz uma representação compacta da árvore de acessibilidade com
referências **estáveis dentro do snapshot**:

```
url: http://localhost:8000/dashboard   title: Dashboard
heading "Dashboard" [ref=e1]
textbox "Pesquisar" [ref=e2]
button "Novo relatório" [ref=e3]
```

O agente interage com `click e3` / `fill e2`, sem CSS selectors nem XPath.
Quando a página muda significativamente, um novo `snapshot` gera novas
referências (e incrementa a época). Ref ausente → erro `invalid_reference`.

### Invalidação automática de refs (MutationObserver)

Um `MutationObserver` é instalado em cada documento via `add_init_script`
(sobrevive a navegações). Ele marca o estado da página quando o DOM muda
**estruturalmente** (`childList`, subtree) — refs são selectors estruturais, por
isso digitação/`fill` e mudanças de atributos/texto **não** invalidam.

- Ao usar um ref (em `click`, `fill`, `wait`, `assert`, ...), se o DOM mudou
  desde o último snapshot **ou** a URL atual difere da URL do snapshot, a ação
  falha com `invalid_reference` ("stale … take a new snapshot"). Selectors CSS
  não são bloqueados.
- Um novo `snapshot` (re-crawl) zera o estado. `status` e `snapshot_map`
  reportam `mutated: true/false`.
- Navegação full-document reinicia o observer "limpo", mas a guarda de URL já
  invalida os refs do documento anterior.
- Páginas que re-renderizam continuamente (feeds, tickers) podem forçar
  re-snapshots: desligue o bloqueio com `BROWSER_MUTATION_INVALIDATE=0`
  (a flag `mutated` continua reportada).

### Mapa da página reutilizável (`snapshot_map`)

O `snapshot` é carregado **sem re-varrer o DOM** entre turnos: a sessão guarda
o último snapshot por aba (texto + refs + URL + época), e `snapshot_map`
devolve esse mapa imediatamente — economia de tokens e de tempo de navegação.

```json
{"tool": "browser", "action": "snapshot_map", "params": {}}
```

- `url` (opcional; default = URL da aba ativa) seleciona o mapa de uma aba ou
  do disco; `refresh: true` delega a `snapshot` (re-crawl completo).
- `fresh: true` quando o mapa corresponde à URL atual e a página não mudou;
  `stale: true` (com a `url` do mapa original e `mutated`) quando a página
  mudou desde o último snapshot — aí o agente deve chamar `snapshot`.
- `found: false` quando não há mapa nenhum (live nem disco).
- `source: "live" | "disk"`. A persistência em disco é **opt-in** via
  `BROWSER_SNAPSHOT_DIR` (ou `create_navigation_tool(snapshot_dir=...)`):
  a cada `snapshot` o mapa é gravado como `map-<hash(url)>.json` e sobrevive a
  `close`/reinício — uma nova execução pode reusar o mapa da URL **sem lançar
  navegador**. Sem `BROWSER_SNAPSHOT_DIR`, o mapa vive apenas em memória.
- O subagente e o MANUAL preferem `snapshot_map` quando a página não mudou.

### Vídeo e screenshots do trace (integrados ao `inspect`)

A gravação de vídeo é **opt-in**: defina `BROWSER_VIDEO_DIR` (ou passe
`video_dir` a `create_navigation_tool(..., video_dir=...)`) **antes** de abrir o
navegador; o tamanho do vídeo é o viewport da sessão (`video_size`, default
1440×900). Playwright grava um vídeo `.webm` **por página**, mas o arquivo só
é finalizado quando a página ou o contexto fecha — durante a gravação o
arquivo pode não existir (ou ter tamanho 0). Por isso:

- `video` → `{recording, path, finalized}`: `recording` indica se a gravação
  está ativa (videodir configurado), `path` é o alvo da aba ativa e
  `finalized` lista as gravações já escritas em disco (abas fechadas). `status`
  também expõe `video`.
- `video_save` → copia as gravações finalizadas para o `video_dir` (ou para o
  `path` passado, usado na primeira). Retorna `{saved: [{path, bytes, source}],
  pending: [{path, page}]}`; abas ainda abertas ficam em `pending` (gravadas
  automaticamente no `close`). Chamada repetida é idempotente (saved vazio).
  Erro `recoverable` quando não há gravação finalizada ainda.

```json
{"tool": "browser", "action": "video", "params": {}}
{"tool": "browser", "action": "video_save", "params": {"path": "/tmp/demo.webm"}}
```

`inspect` aceita dois includes extras que integram trace e vídeo ao diagnóstico:

```json
{"tool": "browser", "action": "inspect", "params": {"include": ["trace"]}}
{"tool": "browser", "action": "inspect", "params": {"include": ["video"]}}
```

- `include: ["trace"]` — garante tracing ativo (inicia se inativo), força um
  frame via `page.screenshot()`, para o trace e extrai os **screenshots** do
  `.zip` (eventos `screencast-frame` do Playwright 1.62). Devolve
  `{path, screenshots, frames}` com os PNGs/JPEGs em `screenshot-<n>.<ext>`
  na pasta do trace (`BROWSER_TRACE_DIR`, default `browser-traces`) — o agente
  pode inspecioná-los como evidência visual do estado.
- `include: ["video"]` — mesmo resultado da ação `video` (status da gravação).

### Formatos de retorno

Sucesso (padrão): `{"status": "success", "operation": "<ação>", ...}`.

Erro (padrão):

```json
{
  "status": "error",
  "error": {
    "type": "element_not_found",
    "message": "waiting for locator ...",
    "recoverable": true,
    "operation": "click"
  }
}
```

Tipos de erro: `navigation_error`, `timeout`, `element_not_found`,
`element_not_visible`, `invalid_reference`, `javascript_error`,
`network_error`, `assertion_failed`, `browser_error`, `permission_denied`,
`invalid_arguments`, `session_closed`, `consent_required`.

## Exemplos de uso

### Delegação (subagente)

```json
{"tool": "browser", "action": "run", "params": {
  "request": "abra o site, preencha o formulário de login e verifique se aparece o dashboard",
  "url": "http://localhost:8000/login"
}}
```

### Dirigido passo a passo

```json
{"tool": "browser", "action": "open", "params": {"url": "http://localhost:8000/index.html"}}
{"tool": "browser", "action": "snapshot", "params": {}}
{"tool": "browser", "action": "fill", "params": {"ref": "e5", "value": "Maria Souza"}}
{"tool": "browser", "action": "click", "params": {"ref": "e8"}}
{"tool": "browser", "action": "assert", "params": {"kind": "text", "ref": "e8", "expected": "Usuário criado"}}
{"tool": "browser", "action": "screenshot", "params": {"kind": "viewport"}}
{"tool": "browser", "action": "inspect", "params": {"include": ["dom", "console", "network", "errors"]}}
{"tool": "browser", "action": "set_viewport", "params": {"width": 375, "height": 812}}
{"tool": "browser", "action": "close", "params": {}}
```

### Diagnóstico

```json
{"tool": "browser", "action": "network", "params": {"errors_only": true}}
{"tool": "browser", "action": "get_response", "params": {"id": "r6", "full_body": true}}
{"tool": "browser", "action": "page_errors", "params": {"clear_after": true}}
```

## Segurança

- Apenas URLs `http`/`https` (bloqueia `file:` e `javascript:`).
- `BROWSER_ALLOW_HOSTS` / `BROWSER_BLOCK_HOSTS` (suporta `*.dominio`);
  block vence allow.
- Timeouts em todas as operações (`BROWSER_ACTION_TIMEOUT`, default 10s;
  `BROWSER_NAV_TIMEOUT`, default 30s).
- Limites de tamanho: snapshot, HTML/texto, retorno de `evaluate`, body de
  requisições/respostas, screenshots.
- `evaluate` é operação de privilégio maior: só executa JS explícito e pode ser
  desligado com `BROWSER_ALLOW_EVALUATE=0` (retorna `permission_denied`).
- Cookies e storage têm valores **mascarados** para chaves sensíveis
  (token/session/auth/jwt/password/...).
- Proxy e credenciais de site exigem **consentimento explícito**
  (`consent confirm=true`); nunca são ecoados ao LLM (apenas mascarados em
  `list_credentials`/`status`), e o arquivo de credenciais
  (`.browser-credentials.json`) deve ser gitignorado.
- O mapa da página é **opcionalmente persistido em disco** — somente quando
  `BROWSER_SNAPSHOT_DIR` for definido (conteúdo de páginas não é gravado por
  padrão).
- Vídeo: somente grava em disco quando `BROWSER_VIDEO_DIR` for definido; sem
  isso a sessão roda sem gravação (`.webm` não é gerado).
- Uploads restritos a roots permitidos (workspace/temp); downloads para
  diretório temporário.
- Contexto de navegador isolado por sessão; sem sleeps fixos (waits
  observáveis).

## Estado da sessão

```
browser_session
    ├── browser        (Chromium; headless por padrão, headed com BROWSER_HEADED=1)
    ├── context        (isolado por sessão; proxy/http_credentials após consentimento)
    ├── pages          (abas abertas; uma é a ativa)
    ├── refs           (referências do snapshot da aba ativa + época, por aba)
    ├── console logs   (log/info/debug/warning/error, etiquetados por aba)
    ├── page errors    (exceções não tratadas, etiquetadas por aba)
    ├── network        (requisições/respostas, etiquetadas por aba)
    ├── mutated        (DOM mudou estruturalmente desde o último snapshot)
    ├── consent        (required/granted quando proxy/credenciais configurados)
    ├── trace          (tracing opcional do Playwright)
    └── video          (gravação .webm opcional; gravada no close)
```

A sessão persiste entre chamadas; `close` encerra tudo. `status` devolve um
resumo (incluindo `tabs`, `active_index` e `consent`).

## Responsividade

```json
{"tool": "browser", "action": "set_viewport", "params": {"width": 1440, "height": 900}}
{"tool": "browser", "action": "set_viewport", "params": {"width": 768, "height": 1024}}
{"tool": "browser", "action": "set_viewport", "params": {"width": 375, "height": 812, "device_scale_factor": 2}}
```

## Testes

Suite em `tests/test_browser.py` + `tests/test_navigation_subagent.py` +
`tests/test_browser_ipc.py` + `tests/test_browser_consent.py` +
`tests/test_browser_snapshot_map.py` + `tests/test_browser_mutation.py` +
`tests/test_trace_screenshots.py` + `tests/test_browser_video.py`,
servida por `tests/browser_fixtures/` (app estático + mock de `/api/*` via
`http.server`, rota Basic-auth `/api/secure`, e proxy HTTP mínimo em
`proxy.py`). Se Playwright/binário não estiver disponível, os testes de
navegação são pulados (`skipUnless`).

```bash
.venv/bin/python -m unittest discover -s tests
```

Cobertos: abrir página, snapshot com refs, clique por ref, preenchimento e
envio de formulário, JS (`evaluate`), console, erro de página, request/response
HTTP, modificação temporária do DOM, screenshot (viewport e elemento), viewport,
assertions (sucesso e falha), trace start/stop, close, refs inválidas, URL
inválida, cookies/storage mascarados, fonte carregada — e, da segunda etapa:
múltiplas abas (`new_tab`/`switch_tab`/`close_tab`/`list_tabs`, refs por aba,
rede etiquetada por aba) e o modo isolado (`BrowserClient` + worker, erros
estruturados pelo pipe, tool completo e delegação do subagente em modo
isolado). Da terceira etapa: consentimento (`consent`/`revoke_consent`/
`list_credentials`, gate `consent_required` sem navegador, proxy com e sem
autenticação registrando `Proxy-Authorization`, credenciais de site em
`/api/secure` com desafio 401) — também pelo pipe isolado. Da quarta etapa:
mapa reutilizável (`snapshot_map` sem re-crawl via mock, `stale` após
navegação, `refresh` forçando re-crawl, persistência em disco entre sessões
`source: "disk"`, determinismo do nome do arquivo, e `snapshot_map` sobre o
pipe isolado). Da quinta etapa: invalidação automática de refs por
MutationObserver (mutação estrutural → `invalid_reference` e `mutated`,
re-snapshot limpando o estado, mudança não-estrutural não invalidando, `fill`
sem falso positivo, edição estrutural nossa invalidando, guarda de URL via
`pushState` e navegação full, `BROWSER_MUTATION_INVALIDATE=0` desligando o
bloqueio, e o mesmo comportamento pelo pipe isolado). Da sexta etapa: extração
de screenshots do trace (`screencast-frame`, dedup, nomes `screenshot-N`, zip
sem `trace.trace` → vazio) e vídeo integrado ao `inspect` (gravação opt-in,
estado `recording`/`finalized`, `video_save` com rotação de abas e idempotência,
`close` finalizando o `.webm` no `video_dir`, e `inspect` com `include: ["trace"]`
capturando + extraindo, e `include: ["video"]`).

## Limitações conhecidas

- Refs são baseadas no main frame; conteúdo em iframes não recebe refs.
- O snapshot infere `role` a partir do DOM (não usa 100% da a11y tree nativa).
- Sites com proteção anti-bot/Cloudflare podem não carregar; headless por
  padrão.
- `browser.evaluate` não possui timeout próprio do Playwright nesta versão
  (operações longas dependem do timeout global).
- No modo isolado (`BROWSER_ISOLATED=1`) o worker mantém **uma** sessão de
  navegador; o `request_timeout` do client é configurável
  (`BrowserClient(request_timeout=...)`), mas por padrão aguarda (o worker
  impõe timeouts próprios por operação).
- Vídeo: o `.webm` só existe completo após a página/contexto fechar (durante a
  gravação o arquivo pode ter tamanho 0); um vídeo por página.
