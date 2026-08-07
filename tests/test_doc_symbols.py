"""Drift guard for docs/api.md.

api.md is a hand-curated narrative index of the public API. It rots silently
when a documented symbol is renamed or removed (the exact class of bug that
produced the stale ``cumulative_fusion`` / ``grade_query`` entries). This test
parses the page and asserts that every documented symbol still exists and is
importable from ``dynafx``:

* Fully-qualified ``dynafx.module.Symbol`` references are resolved directly.
* Bare ``Symbol`` table cells are resolved against the ``dynafx.X`` module given
  by the nearest ``## `` section heading.

Run with: uv run python -m pytest tests/test_doc_symbols.py
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs" / "api.md"

FQ_REF = re.compile(r"`(dynafx\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)`")
HEADING = re.compile(r"^## `(dynafx\.[A-Za-z_]\w*)`\s*$")
# A `` `Symbol` `` as the first cell of a table row, or a bare code-span.
# After the closing backtick may come `(`, `|` (next table cell), or end.
BARE_SYMBOL = re.compile(r"^\|\s*`([A-Za-z_]\w*)`(?:\s*\(|(?:\s*\||\s*$))")
CODE_SYMBOL = re.compile(r"^`([A-Za-z_]\w*)`(?:\s*\(|(?:\s*\||\s*$))")

# Documented symbols that are intentionally callables/syntactic on instances,
# not module attributes — resolved by hand so the import-check stays honest.
KNOWN_ATTRIBUTES: dict[str, set[str]] = {
    "dynafx.dynamics": {
        # instance/method attributes described as result fields
        "values",
        "aux_values",
        "times",
        "stocks",
        "method",
        "steps",
        "model_name",
        "abm_engine",
        "des_engine",
    },
    "dynafx.bridge": {
        "params_from_kb",
        "params_for_class",
        "evidence_from_result",
        "evidence_for_stock",
        "run_with_kb",
        "full_roundtrip",
        "record_provenance",
        "compare_runs",
    },
    "dynafx.knowledge": {
        "add",
        "remove",
        "triples",
        "triples_in_graph",
        "graphs",
        "all_triples",
        "copy_graph",
        "remove_graph",
        "on_add",
        "on_remove",
        "rows_parsed",
        "triples_added",
        "errors",
        "is_subtype",
        "get_ancestors",
        "bindings",
        "cardinality",
    },
    "dynafx.core": set(),
    "dynafx.patterns": set(),
}


def _resolve(module_name: str, symbol: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, symbol)


def _collect() -> list[tuple[str, str]]:
    """Return (module_path, symbol) pairs documented on the page."""
    text = DOCS.read_text(encoding="utf-8")
    found: list[tuple[str, str]] = []
    current_module = ""

    for line in text.splitlines():
        h = HEADING.match(line)
        if h:
            current_module = h.group(1)
            continue

        if not current_module:
            continue

        m = BARE_SYMBOL.match(line.strip()) or CODE_SYMBOL.match(line.strip())
        if m:
            symbol = m.group(1)
            if current_module and symbol not in KNOWN_ATTRIBUTES.get(current_module, set()):
                found.append((current_module, symbol))

    for m in FQ_REF.finditer(text):
        ref = m.group(1)
        module_name, symbol = ref.rsplit(".", 1)
        found.append((module_name, symbol))

    return found


def test_api_doc_is_present():
    assert DOCS.exists(), "docs/api.md missing"


@pytest.mark.parametrize("module_name,symbol", _collect())
def test_symbol_importable(module_name, symbol):
    try:
        _resolve(module_name, symbol)
    except (ImportError, AttributeError) as exc:
        pytest.fail(
            f"docs/api.md references {module_name}.{symbol} which no longer exists: {exc}"
        )
