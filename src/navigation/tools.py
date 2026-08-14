from __future__ import annotations

from typing import Any, Callable

from src.tools.base import ToolSpec

MANUAL = (
    "browser: browse live web pages with Playwright, inspect them and execute\n"
    "actions. The session stays open across calls until 'close'.\n\n"
    "Workflow: start with open, then snapshot to get element refs (e1, e2, ...).\n"
    "Refs are valid until the next snapshot AND are auto-invalidated when the\n"
    "page changes (structural DOM mutation or navigation to another URL): then\n"
    "a ref action returns invalid_reference and you must take a new snapshot.\n"
    "Interact with refs or CSS selectors (pass one of ref | selector per\n"
    "action).\n\n"
    "Actions:\n"
    "  - open\n"
    "      url (str, required); wait (load|domcontentloaded|networkidle, default\n"
    "      domcontentloaded); timeout (float seconds). Only http(s) URLs are\n"
    "      allowed; host policy (BROWSER_ALLOW_HOSTS/BROWSER_BLOCK_HOSTS) applies.\n"
    "  - snapshot\n"
    "      max_refs (int, default 60); include_hidden (bool, default false).\n"
    "      Returns a compact a11y-like listing, e.g.:\n"
    '        heading "Dashboard" [ref=e1]\n'
    '        textbox "Pesquisar" [ref=e2]\n'
    '        button "Novo relatório" [ref=e3]\n'
    "  - snapshot_map\n"
    "      url (str, optional; defaults to the active page); refresh (bool,\n"
    "      default false). Returns the LAST snapshot without re-crawling the\n"
    "      DOM (token saver). found=false when nothing is cached; fresh=true\n"
    "      when the cache matches the current URL, stale=true otherwise; source\n"
    "      is 'live' or 'disk' (disk only when BROWSER_SNAPSHOT_DIR is set, so\n"
    "      the map survives close/restarts). mutated=true means a structural DOM\n"
    "      mutation was detected since the snapshot. Call snapshot (not refresh)\n"
    "      when the page may have changed.\n"
    "  - inspect\n"
    "      include (list[str]): url, title, dom, console, errors, network,\n"
    "      scripts, styles, video, trace. Compact diagnostic report (never full\n"
    "      HTML). include=['trace'] captures a trace of the current page and\n"
    "      returns its zip path plus the extracted screenshots; include=['video']\n"
    "      returns video recording status/path.\n"
    "  - click / double_click\n"
    "      ref | selector. click also accepts wait_for_navigation (bool).\n"
    "  - fill\n"
    "      ref | selector, value (str).\n"
    "  - type\n"
    "      ref | selector, text (str), delay (float seconds per char).\n"
    "  - press\n"
    "      key (str, default Enter); ref | selector (optional, page-level if absent).\n"
    "  - select\n"
    "      ref | selector, value | label | index.\n"
    "  - check / uncheck / hover / focus\n"
    "      ref | selector.\n"
    "  - scroll\n"
    "      ref | selector (scrolls into view); or direction (up|down|left|right)\n"
    "      + amount (px, default 400) for page-level scrolling.\n"
    "  - drag\n"
    "      source (ref|selector), target (ref|selector).\n"
    "  - upload\n"
    "      ref | selector, paths (list[str]) inside allowed roots only.\n"
    "  - screenshot\n"
    "      kind (viewport|full_page, default viewport); format (png|jpeg|webp);\n"
    "      path (optional); quality (optional); embed (bool, default true) returns\n"
    "      base64 data for multimodal analysis. Returns path, bytes, data.\n"
    "  - screenshot_element\n"
    "      ref | selector, format, path.\n"
    "  - evaluate\n"
    "      expression (str): JavaScript in the page; 'return ...' bodies are\n"
    "      supported. timeout, max_chars (default 5000). PRIVILEGED OPERATION:\n"
    "      runs arbitrary JS on the page (disabled with BROWSER_ALLOW_EVALUATE=0).\n"
    "  - assert\n"
    "      kind (url|title|exists|count|visible|checked|text|attribute|style);\n"
    "      ref | selector; expected; attr (for attribute); prop (for style);\n"
    "      exact (bool). Returns {success, expected, actual}.\n"
    "  - set_viewport\n"
    "      width (int), height (int), device_scale_factor (float, optional).\n"
    "  - console\n"
    "      clear_after (bool, default true), limit (int). Logs since last query\n"
    "      (log/info/debug/warning/error).\n"
    "  - network\n"
    "      limit (int), errors_only (bool). Request summaries since last query:\n"
    "      method url status error duration_ms.\n"
    "  - get_request / get_response\n"
    "      id (str, from network); get_response accepts full_body (bool).\n"
    "  - get_html / get_text / get_attributes / get_styles / get_bounds\n"
    "      ref | selector (optional; without ref, get_text/get_html target page).\n"
    "  - set_html / set_text / set_attribute / remove_attribute / set_style /\n"
    "    add_class / remove_class / insert_html / remove_element\n"
    "      ref | selector plus params. TEMPORARY in-page DOM edits only; they\n"
    "      never modify project files.\n"
    "  - cookies / local_storage / session_storage\n"
    "      limit. Values of sensitive keys are masked.\n"
    "  - clear_storage / clear_console / clear_network\n"
    "  - get_source / list_scripts / get_script / list_stylesheets\n"
    "      Inspect resources loaded by the page (not the workspace source).\n"
    "  - wait\n"
    "      condition (element|visible|hidden|network_idle|url|text|timeout);\n"
    "      ref | selector, text, url, timeout.\n"
    "  - goto / reload / back / forward\n"
    "      Navigation helpers.\n"
    "  - new_tab\n"
    "      url (optional): opens a new tab and makes it active; navigates to\n"
    "      url when given. Snapshot refs are scoped per tab.\n"
    "  - switch_tab\n"
    "      index (int) or url (substring) or title (substring). Exactly one tab\n"
    "      must match (use list_tabs to see indices). All actions then target\n"
    "      the new active tab.\n"
    "  - close_tab\n"
    "      index | url | title (default: current tab). The next remaining tab\n"
    "      becomes active; all_closed=true when none remain.\n"
    "  - list_tabs\n"
    "      index, url, title, active, ref_count for every open tab.\n"
    "  - list_tabs\n"
    "      index, url, title, active, ref_count for every open tab.\n"
    "  - consent\n"
    "      confirm (bool, required). Explicit human approval to use a configured\n"
    "      proxy and/or site credentials. Until granted, every navigation action\n"
    "      fails with consent_required and NO browser is launched. A bare consent\n"
    "      (without confirm=true) is rejected. Only call it after the operator\n"
    "      approves the configuration reported by list_credentials.\n"
    "  - revoke_consent\n"
    "      Revokes approval; when a proxy/credentials are configured the session\n"
    "      is closed so no traffic flows with them again until a fresh consent.\n"
    "  - list_credentials\n"
    "      Shows whether a proxy/credentials are configured and their masked\n"
    "      details (server host, partial username, has_password). Never exposes\n"
    "      full passwords or tokens.\n"
    "  - pause\n"
    "      Pauses a headful session and opens the Playwright inspector for human\n"
    "      debugging (no-op/error in headless mode).\n"
    "  - trace_start / trace_stop\n"
    "      Start/stop Playwright tracing; trace_stop returns the zip path. The\n"
    "      trace includes screenshots; use inspect include=['trace'] to also\n"
    "      extract them as image files.\n"
    "  - video\n"
    "      Video recording status: {recording, path, finalized}. Recording only\n"
    "      happens when BROWSER_VIDEO_DIR (or video_dir) was set before the\n"
    "      session started; path is the active page's target file and finalized\n"
    "      lists the recordings already written (closed pages).\n"
    "  - video_save\n"
    "      path (optional, used for the first saved file). Copies the session's\n"
    "      finalized recordings to the video dir (deterministic .webm names).\n"
    "      Returns {saved: [{path, bytes, source}], pending: [{path, page}]}.\n"
    "      Recordings still in progress are written automatically on close.\n"
    "  - status\n"
    "      Session state (open, url, tabs, active_index, refs, mutated, video,\n"
    "      console, errors, network, consent).\n"
    "  - close\n"
    "      Closes the browser session and all tabs; state is lost.\n\n"
    "Errors:\n"
    '  {"status":"error","error":{"type":...,"message":...,"recoverable":...}}\n'
    "  types: navigation_error, timeout, element_not_found, element_not_visible,\n"
    "  invalid_reference, javascript_error, network_error, assertion_failed,\n"
    "  browser_error, permission_denied, invalid_arguments, session_closed,\n"
    "  consent_required.\n\n"
    "Security: only http(s) URLs; size limits on snapshots, screenshots, logs,\n"
    "bodies and evaluate output; sensitive cookie/storage values masked; proxy\n"
    "and site credentials require explicit human consent (confirm=true) and are\n"
    "never echoed (only masked); all operations have timeouts; the browser runs\n"
    "in an isolated context."
)


def _make_actions(controller: Any) -> dict[str, Callable[..., dict]]:
    return {
        "open": controller.open,
        "goto": controller.goto,
        "reload": controller.reload,
        "back": controller.back,
        "forward": controller.forward,
        "wait": controller.wait,
        "snapshot": controller.snapshot,
        "snapshot_map": controller.snapshot_map,
        "inspect": controller.inspect,
        "get_html": controller.get_html,
        "get_text": controller.get_text,
        "get_attributes": controller.get_attributes,
        "get_styles": controller.get_styles,
        "get_bounds": controller.get_bounds,
        "click": controller.click,
        "double_click": controller.double_click,
        "fill": controller.fill,
        "type": controller.type,
        "press": controller.press,
        "select": controller.select,
        "check": controller.check,
        "uncheck": controller.uncheck,
        "hover": controller.hover,
        "focus": controller.focus,
        "scroll": controller.scroll,
        "drag": controller.drag,
        "upload": controller.upload,
        "screenshot": controller.screenshot,
        "screenshot_element": controller.screenshot_element,
        "evaluate": controller.evaluate,
        "set_html": controller.set_html,
        "set_text": controller.set_text,
        "set_attribute": controller.set_attribute,
        "remove_attribute": controller.remove_attribute,
        "set_style": controller.set_style,
        "add_class": controller.add_class,
        "remove_class": controller.remove_class,
        "insert_html": controller.insert_html,
        "remove_element": controller.remove_element,
        "console": controller.console,
        "clear_console": controller.clear_console,
        "page_errors": controller.page_errors,
        "network": controller.network,
        "get_request": controller.get_request,
        "get_response": controller.get_response,
        "clear_network": controller.clear_network,
        "get_source": controller.get_source,
        "list_scripts": controller.list_scripts,
        "get_script": controller.get_script,
        "list_stylesheets": controller.list_stylesheets,
        "assert": controller.assert_,
        "set_viewport": controller.set_viewport,
        "cookies": controller.cookies,
        "local_storage": controller.local_storage,
        "session_storage": controller.session_storage,
        "clear_storage": controller.clear_storage,
        "trace_start": controller.trace_start,
        "trace_stop": controller.trace_stop,
        "video": controller.video,
        "video_save": controller.video_save,
        "new_tab": controller.new_tab,
        "switch_tab": controller.switch_tab,
        "close_tab": controller.close_tab,
        "list_tabs": controller.list_tabs,
        "consent": controller.consent,
        "revoke_consent": controller.revoke_consent,
        "list_credentials": controller.list_credentials,
        "pause": controller.pause,
        "status": controller.status,
        "close": controller.close,
    }


def build_browser_spec(
    controller: Any,
    *,
    name: str = "browser",
    run_handler: Callable[..., dict] | None = None,
) -> ToolSpec:
    """Build the ``browser`` ToolSpec.

    ``run_handler`` (the subagent delegate) is only included when the spec is
    registered in the main registry; the subagent's own registry uses the spec
    without it to avoid recursive delegation.
    """
    actions = _make_actions(controller)
    if run_handler is not None:
        actions["run"] = run_handler
    return ToolSpec(name=name, handlers=actions, manual=MANUAL)
