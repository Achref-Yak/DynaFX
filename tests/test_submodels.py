"""Tests for submodels — Phase 6."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from cognitive_engine.system.dsl import parse_sysd, SubmodelDef, IncludeDef


# ═══════════════════════════════════════════════════════════════
# Submodel parsing
# ═══════════════════════════════════════════════════════════════

class TestSubmodelParsing:
    def test_submodel_definition(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
    stock E: 0
    stock I: 1
    stock R: 0
  dt 1
'''
        m = parse_sysd(src)
        assert len(m.submodels) == 1
        assert m.submodels[0].name == "SEIR"
        assert len(m.submodels[0].stocks) == 4
        assert m.submodels[0].stocks[0].name == "S"
        assert m.submodels[0].stocks[0].initial == 1000

    def test_include_directive(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
    stock I: 1
  include SEIR as pop1
  dt 1
'''
        m = parse_sysd(src)
        assert len(m.includes) == 1
        assert m.includes[0].submodel_name == "SEIR"
        assert m.includes[0].instance_name == "pop1"

    def test_include_with_params(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
    stock I: 1
  include SEIR as pop1 params: S=500, I=10
  dt 1
'''
        m = parse_sysd(src)
        assert m.includes[0].params == {"S": 500, "I": 10}


# ═══════════════════════════════════════════════════════════════
# Include expansion
# ═══════════════════════════════════════════════════════════════

class TestIncludeExpansion:
    def test_single_include(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
    stock I: 1
  include SEIR as pop1
  dt 1
'''
        m = parse_sysd(src)
        # Should have expanded stocks
        stock_names = [s.name for s in m.stocks]
        assert "pop1_S" in stock_names
        assert "pop1_I" in stock_names

    def test_include_with_param_override(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
    stock I: 1
  include SEIR as pop1 params: S=500
  dt 1
'''
        m = parse_sysd(src)
        pop1_s = next(s for s in m.stocks if s.name == "pop1_S")
        assert pop1_s.initial == 500

    def test_multiple_includes(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
    stock I: 1
  include SEIR as pop1
  include SEIR as pop2 params: S=500
  dt 1
'''
        m = parse_sysd(src)
        stock_names = [s.name for s in m.stocks]
        assert "pop1_S" in stock_names
        assert "pop2_S" in stock_names
        pop1_s = next(s for s in m.stocks if s.name == "pop1_S")
        pop2_s = next(s for s in m.stocks if s.name == "pop2_S")
        assert pop1_s.initial == 1000
        assert pop2_s.initial == 500

    def test_include_no_prefix(self):
        src = '''
model 'Test'
  submodel 'SEIR'
    stock S: 1000
  include SEIR
  dt 1
'''
        m = parse_sysd(src)
        stock_names = [s.name for s in m.stocks]
        assert "S" in stock_names

    def test_include_preserves_flows(self):
        src = '''
model 'Test'
  submodel 'SIR'
    stock S: 1000
      + infection
      - infection
    stock I: 1
    aux infection: S * I * 0.001
  include SIR as pop1
  dt 1
'''
        m = parse_sysd(src)
        pop1_s = next(s for s in m.stocks if s.name == "pop1_S")
        assert len(pop1_s.flows) == 2
        assert pop1_s.flows[0].name == "pop1_infection"
        assert pop1_s.flows[1].name == "pop1_infection"

    def test_include_updates_flow_expressions(self):
        src = '''
model 'Test'
  submodel 'SIR'
    stock S: 1000
    stock I: 1
    aux beta: 0.001
    aux infection: S * I * beta
    + infection
    - infection
  include SIR as pop1
  dt 1
'''
        m = parse_sysd(src)
        pop1_infection = next(a for a in m.aux_vars if a.name == "pop1_infection")
        # Expression should reference prefixed names
        assert "pop1_S" in pop1_infection.expr
        assert "pop1_I" in pop1_infection.expr
        assert "pop1_beta" in pop1_infection.expr


# ═══════════════════════════════════════════════════════════════
# Simulation with includes
# ═══════════════════════════════════════════════════════════════

class TestIncludeSimulation:
    def test_simulate_with_include(self):
        src = '''
model 'TwoPop'
  submodel 'Simple'
    stock X: 100
      + growth
    aux growth: X * 0.1
  include Simple as pop1
  include Simple as pop2 params: X=200
  dt 1
  from 0 to 10
'''
        m = parse_sysd(src)
        r = m.simulate()
        # pop1 starts at 100, pop2 starts at 200
        assert r["values"]["pop1_X"][0] == 100
        assert r["values"]["pop2_X"][0] == 200
        # Both should grow
        assert r["values"]["pop1_X"][-1] > 100
        assert r["values"]["pop2_X"][-1] > 200
        # pop2 should be larger (started higher, same growth rate)
        assert r["values"]["pop2_X"][-1] > r["values"]["pop1_X"][-1]

    def test_simulate_independent_populations(self):
        src = '''
model 'TwoPop'
  submodel 'SIR'
    stock S: 999
      + infection
    stock I: 1
      + infection
      - recovery
    aux beta: 0.3
    aux gamma: 0.1
    aux infection: beta * S * I / 1000
    aux recovery: gamma * I
  include SIR as city1
  include SIR as city2 params: I=5
  dt 0.1
  from 0 to 50
'''
        m = parse_sysd(src)
        r = m.simulate()
        # Both cities should have dynamics
        assert r["values"]["city1_I"][0] == 1
        assert r["values"]["city2_I"][0] == 5
        # Infection should spread
        assert max(r["values"]["city1_I"]) > 1
        assert max(r["values"]["city2_I"]) > 5
