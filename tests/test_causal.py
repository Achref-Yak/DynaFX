"""Tests for causal tracing: causes_tree, effects_tree, causes_strip, causal_trace."""

import pytest
from dynafx.dynamics.dsl import parse_sysd
from dynafx.dynamics.causal import (
    causes_tree,
    effects_tree,
    causes_strip,
    causal_trace,
    CausalNode,
    CausalStrip,
)


# ── causes_tree Tests ──────────────────────────────────────────

class TestCausesTree:
    def test_causes_tree_simple(self):
        """causes_tree for a stock should show its flows."""
        m = parse_sysd("""
model 'Simple'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'Inflow': 10
    - 'Outflow': 5
""")
        tree = causes_tree(m, "S")
        assert tree is not None
        assert tree.name == "S"
        child_names = {c.name for c in tree.children}
        assert "Inflow" in child_names
        assert "Outflow" in child_names

    def test_causes_tree_upstream(self):
        """causes_tree should walk upstream through flow expressions."""
        m = parse_sysd("""
model 'Upstream'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'Inflow': Rate * S
    - 'Outflow': 5
  aux 'Rate': 0.1
""")
        tree = causes_tree(m, "S")
        assert tree is not None
        # S depends on Inflow and Outflow
        child_names = {c.name for c in tree.children}
        assert "Inflow" in child_names
        # Inflow depends on Rate and S
        inflow_node = next(c for c in tree.children if c.name == "Inflow")
        inflow_refs = {c.name for c in inflow_node.children}
        assert "Rate" in inflow_refs

    def test_causes_tree_nonexistent(self):
        """causes_tree for nonexistent variable returns None."""
        m = parse_sysd("""
model 'Empty'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
""")
        tree = causes_tree(m, "Nonexistent")
        assert tree is None

    def test_causes_tree_max_depth(self):
        """causes_tree respects max_depth."""
        m = parse_sysd("""
model 'Depth'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': A
  aux 'A': B
  aux 'B': C
  aux 'C': 1
""")
        tree = causes_tree(m, "S", max_depth=2)
        assert tree is not None
        assert tree.depth() <= 2

    def test_causes_tree_table_ref(self):
        """causes_tree should include table references."""
        m = parse_sysd("""
model 'Table'
  dt 1
  from 0 to 10
  table 'rate'
    x: [0, 5, 10]
    y: [1, 2, 1]
  stock 'S': 100
    + 'In': rate(t)
""")
        tree = causes_tree(m, "S")
        assert tree is not None
        child_names = {c.name for c in tree.children}
        assert "In" in child_names
        inflow = next(c for c in tree.children if c.name == "In")
        inflow_refs = {c.name for c in inflow.children}
        assert "rate" in inflow_refs


# ── effects_tree Tests ─────────────────────────────────────────

class TestEffectsTree:
    def test_effects_tree_simple(self):
        """effects_tree for a variable should show what depends on it."""
        m = parse_sysd("""
model 'Simple'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': Rate
    - 'Out': 5
  aux 'Rate': 0.1
""")
        tree = effects_tree(m, "Rate")
        assert tree is not None
        assert tree.name == "Rate"
        child_names = {c.name for c in tree.children}
        assert "In" in child_names

    def test_effects_tree_stock(self):
        """effects_tree for a stock should show flows and downstream stocks."""
        m = parse_sysd("""
model 'Chain'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': 10
    - 'a_out': A * 0.1
  stock 'B': 0
    + 'b_in': a_out
    - 'b_out': 5
""")
        tree = effects_tree(m, "A")
        assert tree is not None
        child_names = {c.name for c in tree.children}
        assert "a_in" in child_names or "a_out" in child_names

    def test_effects_tree_nonexistent(self):
        """effects_tree for nonexistent variable returns None."""
        m = parse_sysd("""
model 'Empty'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
""")
        tree = effects_tree(m, "Nonexistent")
        assert tree is None

    def test_effects_tree深远(self):
        """effects_tree should trace multiple levels downstream."""
        m = parse_sysd("""
model 'Deep'
  dt 1
  from 0 to 10
  aux 'X': 1
  aux 'Y': X
  aux 'Z': Y
  stock 'S': 0
    + 'In': Z
""")
        tree = effects_tree(m, "X")
        assert tree is not None
        # X -> Y -> Z -> In
        names = []
        def collect(node, depth=0):
            names.append((node.name, depth))
            for c in node.children:
                collect(c, depth+1)
        collect(tree)
        name_set = {n for n, _ in names}
        assert "Y" in name_set
        assert "Z" in name_set


# ── causes_strip Tests ─────────────────────────────────────────

class TestCausesStrip:
    def test_causes_strip_simple(self):
        """causes_strip should decompose a stock's value into flow contributions."""
        m = parse_sysd("""
model 'Strip'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'Inflow': 10
    - 'Outflow': 5
""")
        state = {"S": 100, "Inflow": 10, "Outflow": 5}
        strip = causes_strip(m, "S", state)
        assert strip is not None
        assert strip.variable == "S"
        assert strip.total_value == 100
        factor_names = {f["name"] for f in strip.factors}
        assert "Inflow" in factor_names
        assert "Outflow" in factor_names

    def test_causes_strip_with_refs(self):
        """causes_strip should include referenced variable values."""
        m = parse_sysd("""
model 'Refs'
  dt 1
  from 0 to 10
  aux 'Rate': 0.1
  stock 'S': 100
    + 'In': Rate * 100
""")
        state = {"S": 100, "In": 10, "Rate": 0.1}
        strip = causes_strip(m, "S", state)
        assert strip is not None
        factor_names = {f["name"] for f in strip.factors}
        assert "In" in factor_names

    def test_causes_strip_nonexistent(self):
        """causes_strip for nonexistent variable returns None."""
        m = parse_sysd("""
model 'Empty'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
""")
        strip = causes_strip(m, "Nonexistent", {"S": 100})
        assert strip is None


# ── causal_trace Tests ─────────────────────────────────────────

class TestCausalTrace:
    def test_causal_trace_combined(self):
        """causal_trace should return causes, effects, and strip."""
        m = parse_sysd("""
model 'Trace'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': Rate * S
    - 'Out': 5
  aux 'Rate': 0.1
""")
        state = {"S": 100, "In": 10, "Out": 5, "Rate": 0.1}
        trace = causal_trace(m, "S", state)
        assert "variable" in trace
        assert "causes" in trace
        assert "effects" in trace
        assert "strip" in trace
        assert trace["variable"] == "S"
        assert trace["causes"] is not None
        assert trace["strip"] is not None

    def test_causal_trace_to_dict(self):
        """CausalNode and CausalStrip should serialize to dict."""
        m = parse_sysd("""
model 'Dict'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
""")
        tree = causes_tree(m, "S")
        d = tree.to_dict()
        assert d["name"] == "S"
        assert isinstance(d["children"], list)

        state = {"S": 100, "In": 10}
        strip = causes_strip(m, "S", state)
        d = strip.to_dict()
        assert d["variable"] == "S"
        assert isinstance(d["factors"], list)

    def test_causal_trace_nonexistent(self):
        """causal_trace for nonexistent variable returns None parts."""
        m = parse_sysd("""
model 'Empty'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
""")
        trace = causal_trace(m, "Nonexistent", {"S": 100})
        assert trace["causes"] is None
        assert trace["effects"] is None
        assert trace["strip"] is None
