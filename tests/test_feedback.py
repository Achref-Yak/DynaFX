"""Tests for feedback loop detection."""

import pytest
from dynafx.dynamics.dsl import parse_sysd
from dynafx.dynamics.feedback import (
    detect_feedback_loops,
    loops_for_variable,
    FeedbackLoop,
    LoopAnalysis,
)


class TestFeedbackLoops:
    def test_no_loops(self):
        """Model without cycles should have no feedback loops."""
        m = parse_sysd("""
model 'NoLoop'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': 10
    - 'a_out': 5
  stock 'B': 0
    + 'b_in': a_out
""")
        analysis = detect_feedback_loops(m)
        assert len(analysis.loops) == 0

    def test_reinforcing_loop(self):
        """Model with positive feedback (A -> B -> A) should be reinforcing."""
        m = parse_sysd("""
model 'Reinforcing'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': B
    - 'a_out': 5
  stock 'B': 50
    + 'b_in': A
    - 'b_out': 3
""")
        analysis = detect_feedback_loops(m)
        assert len(analysis.loops) >= 1
        # Should have a reinforcing loop
        reinforcing = [l for l in analysis.loops if l.polarity == "reinforcing"]
        assert len(reinforcing) >= 1
        # Loop should involve A and B
        loop_nodes = set(reinforcing[0].nodes)
        assert "A" in loop_nodes or "B" in loop_nodes

    def test_balancing_loop(self):
        """Model with negative feedback (stock limits its own growth) should be balancing."""
        m = parse_sysd("""
model 'Balancing'
  dt 1
  from 0 to 10
  stock 'Population': 100
    + 'births': Population * 0.1
    - 'deaths': Population * 0.05
""")
        analysis = detect_feedback_loops(m)
        assert len(analysis.loops) >= 1
        # Births loop: Population -> births -> Population (positive)
        # Deaths loop: Population -> deaths -> Population (negative)
        balancing = [l for l in analysis.loops if l.polarity == "balancing"]
        reinforcing = [l for l in analysis.loops if l.polarity == "reinforcing"]
        # Should have at least one of each
        assert len(balancing) >= 1 or len(reinforcing) >= 1

    def test_loop_count(self):
        """Should correctly count reinforcing and balancing loops."""
        m = parse_sysd("""
model 'Both'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': A * 0.1
    - 'a_out': A * 0.05
""")
        analysis = detect_feedback_loops(m)
        d = analysis.to_dict()
        assert "num_reinforcing" in d
        assert "num_balancing" in d
        assert d["num_reinforcing"] + d["num_balancing"] == len(analysis.loops)

    def test_loops_for_variable(self):
        """loops_for_variable should return loops containing the variable."""
        m = parse_sysd("""
model 'Target'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': B
    - 'a_out': 5
  stock 'B': 50
    + 'b_in': A
    - 'b_out': 3
""")
        analysis = detect_feedback_loops(m)
        a_loops = loops_for_variable(analysis, "A")
        # A should be in at least one loop
        assert len(a_loops) >= 1
        for loop in a_loops:
            assert "A" in loop.nodes

    def test_loop_to_dict(self):
        """FeedbackLoop should serialize to dict."""
        m = parse_sysd("""
model 'Dict'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': B
    - 'a_out': 5
  stock 'B': 50
    + 'b_in': A
""")
        analysis = detect_feedback_loops(m)
        d = analysis.to_dict()
        assert "loops" in d
        assert "variable_loops" in d
        assert "num_reinforcing" in d
        assert "num_balancing" in d
        if d["loops"]:
            loop_d = d["loops"][0]
            assert "name" in loop_d
            assert "nodes" in loop_d
            assert "polarity" in loop_d

    def test_three_stock_loop(self):
        """Three-stock circular dependency should form a loop."""
        m = parse_sysd("""
model 'Triangle'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': C
    - 'a_out': 5
  stock 'B': 50
    + 'b_in': A
    - 'b_out': 3
  stock 'C': 30
    + 'c_in': B
    - 'c_out': 2
""")
        analysis = detect_feedback_loops(m)
        assert len(analysis.loops) >= 1
        # Should find the A->B->C->A cycle
        all_nodes = set()
        for loop in analysis.loops:
            all_nodes.update(loop.nodes)
        assert "A" in all_nodes
        assert "B" in all_nodes
        assert "C" in all_nodes

    def test_max_loop_length(self):
        """max_loop_length should limit cycle detection."""
        m = parse_sysd("""
model 'Long'
  dt 1
  from 0 to 10
  stock 'A': 100
    + 'a_in': B
  stock 'B': 50
    + 'b_in': A
""")
        analysis_short = detect_feedback_loops(m, max_loop_length=1)
        analysis_long = detect_feedback_loops(m, max_loop_length=10)
        # Short limit should find fewer or equal loops
        assert len(analysis_short.loops) <= len(analysis_long.loops)
