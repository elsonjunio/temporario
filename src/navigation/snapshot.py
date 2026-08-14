from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.navigation.errors import BrowserError

_MAP_PREFIX = "map-"
_MAP_SUFFIX = ".json"


def snapshot_map_file(snapshot_dir: str | Path, url: str) -> Path:
    """Return the on-disk path for a page map of ``url``.

    Deterministic per URL so a fresh process can reuse a previously stored
    page map without a browser round-trip (``BROWSER_SNAPSHOT_DIR``).
    """
    digest = hashlib.sha256((url or "").encode("utf-8")).hexdigest()[:16]
    return Path(snapshot_dir) / f"{_MAP_PREFIX}{digest}{_MAP_SUFFIX}"


def snapshot_to_dict(
    refs: dict[str, dict[str, Any]],
    snapshot_text: str,
    url: str,
    *,
    epoch: int,
    title: str = "",
) -> dict[str, Any]:
    """Serialize a snapshot into a JSON-friendly page map."""
    return {
        "url": url,
        "title": title,
        "epoch": epoch,
        "ref_count": len(refs),
        "content": snapshot_text,
        "refs": {
            ref: {
                "role": meta.get("role"),
                "name": meta.get("name"),
                "tag": meta.get("tag"),
                "locator": meta.get("locator"),
            }
            for ref, meta in refs.items()
        },
        "captured_at": time.time(),
    }


def write_map_file(snapshot_dir: str | Path, data: dict[str, Any]) -> Path | None:
    """Persist a page map to ``snapshot_dir`` (best effort; returns the path
    written or None)."""
    url = data.get("url") or ""
    if not url:
        return None
    path = snapshot_map_file(snapshot_dir, url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path
    except OSError:
        return None


def read_map_file(snapshot_dir: str | Path, url: str) -> dict[str, Any] | None:
    """Load a page map from disk; returns None when absent or unreadable."""
    path = snapshot_map_file(snapshot_dir, url)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("url"):
        return None
    return data


_SNAPSHOT_JS = """
function collectSnapshot(opts) {
  opts = opts || {};
  var includeHidden = !!opts.includeHidden;
  var maxRefs = opts.maxRefs || 200;
  var results = [];

  var idMap = {};
  var ids = document.querySelectorAll('[id]');
  for (var k = 0; k < ids.length; k++) { idMap[ids[k].id] = ids[k]; }

  function uniqueId(el) {
    var id = el.getAttribute && el.getAttribute('id');
    if (id && idMap[id] === el) return '#' + escapeCss(id);
    return null;
  }

  function escapeCss(str) {
    return String(str).replace(/[^a-zA-Z0-9_-]/g, function (c) { return '\\\\' + c; });
  }

  function isVisible(el) {
    if (el.hidden) return false;
    var style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return false;
    var rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function pathOf(el) {
    var id = uniqueId(el);
    if (id) return id;
    var testid = el.getAttribute && el.getAttribute('data-testid');
    if (testid) return '[data-testid="' + escapeCss(testid) + '"]';
    var parts = [];
    var node = el;
    while (node && node.nodeType === 1) {
      var parent = node.parentElement;
      var tag = node.tagName.toLowerCase();
      if (tag === 'html') { parts.unshift('html'); break; }
      var nth = 1;
      var sibling = node.previousElementSibling;
      while (sibling) {
        if (sibling.tagName.toLowerCase() === tag) nth++;
        sibling = sibling.previousElementSibling;
      }
      parts.unshift(tag + ':nth-of-type(' + nth + ')');
      node = parent;
    }
    return parts.join(' > ');
  }

  function textOf(el) {
    var t = (el.textContent || '').replace(/\\s+/g, ' ').trim();
    return t.length > 120 ? t.slice(0, 120) + '…' : t;
  }

  function labelFor(el) {
    var id = el.getAttribute && el.getAttribute('id');
    if (id && window.CSS && CSS.escape) {
      var lbl = document.querySelector('label[for=' + CSS.escape(id) + ']');
      if (lbl) return textOf(lbl);
    }
    var wrap = el.closest && el.closest('label');
    if (wrap) return textOf(wrap);
    return '';
  }

  function roleOf(el) {
    var aria = el.getAttribute && el.getAttribute('role');
    if (aria) return aria;
    var tag = el.tagName.toLowerCase();
    var type = ((el.getAttribute && el.getAttribute('type')) || '').toLowerCase();
    if (tag === 'a' && el.href) return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'h1' || tag === 'h2' || tag === 'h3' || tag === 'h4' || tag === 'h5' || tag === 'h6') return 'heading';
    if (tag === 'input') {
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'range') return 'slider';
      return 'textbox';
    }
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'combobox';
    if (tag === 'img') return 'img';
    if (tag === 'table') return 'table';
    if (tag === 'nav') return 'navigation';
    if (tag === 'option') return 'option';
    return '';
  }

  function nameOf(el, role) {
    var aria = el.getAttribute && el.getAttribute('aria-label');
    if (aria) return aria.replace(/\\s+/g, ' ').trim();
    var tag = el.tagName.toLowerCase();
    var type = ((el.getAttribute && el.getAttribute('type')) || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      var lbl = labelFor(el);
      if (lbl) return lbl;
      var ph = el.getAttribute && el.getAttribute('placeholder');
      if (ph) return ph;
      if (type === 'radio' && el.value) return el.value;
      if (type === 'submit' || type === 'button' || type === 'reset') {
        var v = el.value;
        if (v) return v;
      }
    }
    if (tag === 'img') {
      if (el.alt) return el.alt;
      var title = el.getAttribute && el.getAttribute('title');
      if (title) return title;
    }
    return textOf(el);
  }

  var interesting = {
    heading: 1, link: 1, button: 1, textbox: 1, checkbox: 1, radio: 1,
    combobox: 1, listbox: 1, option: 1, slider: 1, switch: 1, tab: 1,
    tabpanel: 1, menu: 1, menuitem: 1, navigation: 1, searchbox: 1, img: 1,
    table: 1, meter: 1, progressbar: 1, form: 1, main: 1, banner: 1,
    contentinfo: 1, region: 1, article: 1
  };

  var all = document.querySelectorAll('body *');
  for (var i = 0; i < all.length && results.length < maxRefs; i++) {
    var el = all[i];
    var vis = isVisible(el);
    if (!includeHidden && !vis) continue;
    var role = roleOf(el);
    if (!interesting[role]) continue;
    var name = nameOf(el, role);
    if (!name && role !== 'form' && role !== 'main' && role !== 'navigation' && role !== 'region') continue;
    var entry = {
      tag: el.tagName.toLowerCase(),
      role: role,
      name: name,
      locator: pathOf(el),
      hidden: !vis
    };
    var value = el.value;
    if (value !== undefined && value !== null && value !== '') entry.value = String(value);
    if (role === 'checkbox' || role === 'radio' || role === 'switch') entry.checked = !!el.checked;
    if (role === 'option') entry.selected = !!el.selected;
    var ph = el.getAttribute && el.getAttribute('placeholder');
    if (ph) entry.placeholder = ph;
    var href = el.getAttribute && el.getAttribute('href');
    if (href) entry.href = href;
    if (el.disabled) entry.disabled = true;
    results.push(entry);
  }
  return results;
}
"""

_SNAPSHOT_FN = "(opts) => { " + _SNAPSHOT_JS + " return collectSnapshot(opts); }"

_MUTATION_INIT_JS = """(function () {
  if (!window.MutationObserver || window.__browserMutationObserver) return;
  window.__browserMutation = { dirty: false, count: 0, structural: 0 };
  var obs = new MutationObserver(function (records) {
    for (var i = 0; i < records.length; i++) {
      var r = records[i];
      window.__browserMutation.count++;
      if (r.type === 'childList' && (r.addedNodes.length || r.removedNodes.length)) {
        window.__browserMutation.structural++;
        window.__browserMutation.dirty = true;
      }
    }
  });
  window.__browserMutationObserver = obs;
  obs.observe(document.documentElement || document, {
    childList: true, subtree: true, attributes: false, characterData: false
  });
})()"""

_MUTATION_FLAG_JS = "!!(window.__browserMutation && window.__browserMutation.dirty)"

_MUTATION_CLEAR_JS = (
    "(function(){ if (window.__browserMutation) {"
    " window.__browserMutation.dirty = false;"
    " window.__browserMutation.count = 0;"
    " window.__browserMutation.structural = 0; } })()"
)


def install_mutation_observer(context: Any) -> None:
    """Install the structural MutationObserver on every document of ``context``
    (via ``add_init_script``, so it also runs after each navigation)."""
    try:
        context.add_init_script(script=_MUTATION_INIT_JS)
    except Exception:
        pass


def page_is_mutated(page: Any) -> bool:
    """True when the page's DOM changed structurally since the last snapshot
    reset. Best-effort: any failure is reported as False."""
    try:
        return bool(page.evaluate(_MUTATION_FLAG_JS))
    except Exception:
        return False


def clear_mutation_flag(page: Any) -> None:
    """Reset the in-page mutation flag (called right after a fresh snapshot)."""
    try:
        page.evaluate(_MUTATION_CLEAR_JS)
    except Exception:
        pass


def _format_name(name: str) -> str:
    name = (name or "").strip().replace("\n", " ")
    return name


def _entry_line(ref: str, entry: dict[str, Any]) -> str:
    role = entry.get("role", "element")
    name = _format_name(entry.get("name", ""))
    parts = [f'{role} "{name}" [ref={ref}]']
    extra = []
    if entry.get("checked") is not None:
        extra.append("checked" if entry["checked"] else "unchecked")
    if entry.get("selected"):
        extra.append("selected")
    if entry.get("disabled"):
        extra.append("disabled")
    if entry.get("value"):
        extra.append(f'value="{entry["value"]}"')
    if entry.get("placeholder"):
        extra.append(f'placeholder="{entry["placeholder"]}"')
    if extra:
        parts.append(" ".join(extra))
    if entry.get("hidden"):
        parts.append("(hidden)")
    return " ".join(parts)


def build_snapshot(
    page: Any,
    *,
    max_refs: int = 60,
    include_hidden: bool = False,
) -> tuple[dict[str, dict[str, Any]], str, int]:
    """Run the DOM-walk, assign stable refs (``e1``, ``e2``, ...) and render a
    compact LLM-oriented snapshot.

    Returns ``(refs, snapshot_text, epoch)`` where ``refs`` maps each ref to its
    metadata (including a stable ``locator`` usable for later interactions).
    """
    try:
        entries = page.evaluate(
            _SNAPSHOT_FN,
            {"includeHidden": include_hidden, "maxRefs": max_refs},
        )
    except Exception as exc:
        raise BrowserError(f"snapshot failed: {exc}") from exc

    if not isinstance(entries, list):
        entries = []

    refs: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        ref = f"e{index}"
        meta = dict(entry)
        meta["ref"] = ref
        refs[ref] = meta
        lines.append(_entry_line(ref, entry))

    return refs, "\n".join(lines), len(entries)
