"""Tests for the .sysd DSL parser and simulation."""

from dynafx.system.dsl import parse_sysd, ExprParser


# ── Expression parser ─────────────────────────────────────────

def _parse_expr(source: str):
    return ExprParser(source).parse()


def test_expr_literal():
    n = _parse_expr("42")
    assert n.value == 42.0


def test_expr_ref():
    n = _parse_expr("Demand")
    assert n.name == "Demand"


def test_expr_binop():
    n = _parse_expr("10 + 20")
    assert n.op == "+"
    assert n.left.value == 10.0
    assert n.right.value == 20.0


def test_expr_binop_precedence():
    n = _parse_expr("10 + 20 * 3")
    assert n.op == "+"
    assert n.left.value == 10.0
    assert n.right.op == "*"
    assert n.right.left.value == 20.0
    assert n.right.right.value == 3.0


def test_expr_parens():
    n = _parse_expr("(10 + 20) * 3")
    assert n.op == "*"
    assert n.left.op == "+"
    assert n.right.value == 3.0


def test_expr_unary_minus():
    n = _parse_expr("-5")
    assert n.op == "*"
    assert isinstance(n.left, type(n)) or n.left.value == -1.0
    assert n.right.value == 5.0


def test_expr_func_call():
    n = _parse_expr("MIN(a, b)")
    assert n.name == "MIN"
    assert n.args[0].name == "a"
    assert n.args[1].name == "b"


def test_expr_nested_func():
    n = _parse_expr("MAX(MIN(x, y), z)")
    assert n.name == "MAX"
    assert n.args[0].name == "MIN"
    assert n.args[0].args[0].name == "x"
    assert n.args[0].args[1].name == "y"
    assert n.args[1].name == "z"


def test_expr_dt():
    n = _parse_expr("dt")
    assert n.name == "dt"


def test_sci_notation():
    assert _parse_expr("3e-7").value == 3e-7
    assert _parse_expr("1.5e10").value == 1.5e10
    assert _parse_expr("2E5").value == 2e5


def test_comparison():
    n = _parse_expr("a > b")
    assert n.op == ">"
    assert n.left.name == "a"
    assert n.right.name == "b"
    n2 = _parse_expr("x >= 5")
    assert n2.op == ">="
    assert n2.right.value == 5.0


def test_comparison_in_if():
    n = _parse_expr("IF(x > 10, a, b)")
    assert n.args[0].op == ">"
    assert n.args[0].left.name == "x"
    assert n.args[1].name == "a"
    assert n.args[2].name == "b"


# ── Sysd parser ────────────────────────────────────────────────


def test_parse_empty_model():
    src = """
model 'Empty'
  dt 1
  from 0 to 10
"""
    m = parse_sysd(src)
    assert m.name == "Empty"
    assert m.dt == 1.0
    assert m.t_span == (0.0, 10.0)


def test_parse_single_stock():
    src = '''
model 'Test'
  dt 0.5
  stock 'Widgets': 100
    + 'Production': 10
    - 'Demand': 8
'''
    m = parse_sysd(src)
    assert len(m.stocks) == 1
    assert m.stocks[0].name == "Widgets"
    assert m.stocks[0].initial == 100.0
    assert len(m.stocks[0].flows) == 2
    assert m.stocks[0].flows[0].direction == "+"
    assert m.stocks[0].flows[0].name == "Production"
    assert m.stocks[0].flows[0].expr == "10"
    assert m.stocks[0].flows[1].direction == "-"
    assert m.stocks[0].flows[1].name == "Demand"
    assert m.stocks[0].flows[1].expr == "8"


def test_parse_multiple_stocks():
    src = '''
model 'Multi'
  stock 'A': 10
    + 'In': 5
  stock 'B': 20
    - 'Out': 3
'''
    m = parse_sysd(src)
    assert len(m.stocks) == 2
    assert m.stocks[0].name == "A"
    assert m.stocks[1].name == "B"


def test_parse_table():
    src = '''
model 'T'
  table 'seasonal'
    x: [0, 6, 12]
    y: [8, 15, 8]
'''
    m = parse_sysd(src)
    assert len(m.tables) == 1
    assert m.tables[0].name == "seasonal"
    assert m.tables[0].x == [0.0, 6.0, 12.0]
    assert m.tables[0].y == [8.0, 15.0, 8.0]


def test_parse_comments():
    src = '''
model 'Comments'
  // this is a comment
  dt 1
  stock 'X': 0
    // inline comment
    + 'Y': 5
'''
    m = parse_sysd(src)
    assert m.name == "Comments"
    assert len(m.stocks) == 1


def test_parse_expressions():
    src = '''
model 'Expr'
  stock 'S': 100
    + 'Inflow': MIN(Demand, Capacity / dt)
    - 'Outflow': MAX(0, S * 0.1)
'''
    m = parse_sysd(src)
    assert m.stocks[0].flows[0].expr == "MIN(Demand, Capacity / dt)"
    assert m.stocks[0].flows[1].expr == "MAX(0, S * 0.1)"


# ── Simulation ─────────────────────────────────────────────────


def test_simulate_linear():
    m = parse_sysd('''
model 'Linear'
  dt 1
  from 0 to 5
  stock 'X': 0
    + 'Growth': 10
''')
    result = m.simulate()
    assert result["steps"] == 5
    assert result["stocks"] == ["X"]
    assert abs(result["final_state"][0] - 50.0) < 1e-9


def test_simulate_two_stocks():
    m = parse_sysd('''
model 'Two'
  dt 1
  from 0 to 10
  stock 'A': 0
    + 'In': 10
  stock 'B': 10
    + 'In': 5
    - 'Out': 1
''')
    result = m.simulate()
    assert abs(result["values"]["A"][-1] - 100.0) < 1e-9
    assert abs(result["values"]["B"][-1] - 50.0) < 1e-9


def test_simulate_smooth():
    m = parse_sysd('''
model 'Smooth'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
    - 'Out': SMOOTH(8, 2)
''')
    result = m.simulate()
    # SMOOTH(8, 2) ramps outflow from 0 toward 8 with time constant 2
    # After 10 steps: inflow = 100, outflow ≈ ∫8*(1-exp(-t/2)) ≈ 64.1
    # S ≈ 100 + 100 - 64.1 ≈ 135.9
    s = result["values"]["S"][-1]
    assert 130 < s < 140, f"Expected ~135.9, got {s}"


def test_simulate_table():
    m = parse_sysd('''
model 'Table'
  dt 1
  from 0 to 12
  table 'rate'
    x: [0, 6, 12]
    y: [8, 15, 8]
  stock 'B': 50
    + 'Supply': 10
    - 'Demand': rate(t)
''')
    result = m.simulate()
    b = result["values"]["B"][-1]
    # demand = 8→15→8 as t varies 0→12, total outflow ≈ 138 > inflow 120
    assert 30 < b < 35, f"Expected ~32, got {b}"


def test_simulate_table_t_varies():
    """Verify t is actually evolving inside table lookups."""
    m = parse_sysd('''
model 'Ramp'
  dt 1
  from 0 to 5
  table 'ramp'
    x: [0, 5]
    y: [0, 50]
  stock 'X': 0
    + 'In': ramp(t)
''')
    result = m.simulate()
    # ramp(t) = 10*t, ∫₀⁵ 10t dt = 125
    assert abs(result["final_state"][0] - 125.0) < 1e-9


def test_simulate_euler_method():
    m = parse_sysd('''
model 'Euler'
  dt 1
  from 0 to 5
  stock 'X': 0
    + 'In': 10
''')
    rk4 = m.simulate(method="rk4")
    euler = m.simulate(method="euler")
    # Both should agree on linear ODE
    assert abs(rk4["final_state"][0] - euler["final_state"][0]) < 1e-9


# ── to_decomposer ──────────────────────────────────────────────


def test_to_decomposer():
    m = parse_sysd('''
model 'Test'
  stock 'Widgets': 100
    + 'Production': 10
    - 'Demand': 8
''')
    d = m.to_decomposer()
    assert d.graph.source_text == "Test"
    assert len(d.graph.nodes) == 3
    assert len(d.graph.edges) == 2
    meta = d.graph.metadata.get("sysd_model", {})
    assert meta["name"] == "Test"


# ── Aux variables ───────────────────────────────────────────────


def test_parse_aux():
    src = """
model 'AuxTest'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': rate
  aux 'rate': S * 0.1
"""
    m = parse_sysd(src)
    assert len(m.aux_vars) == 1
    assert m.aux_vars[0].name == "rate"
    assert m.aux_vars[0].expr == "S * 0.1"


def test_parse_multiple_auxes():
    src = """
model 'MultiAux'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': rate * discount
  aux 'rate': S * 0.1
  aux 'discount': 0.95
"""
    m = parse_sysd(src)
    assert len(m.aux_vars) == 2
    assert m.aux_vars[0].name == "rate"
    assert m.aux_vars[1].name == "discount"


def test_simulate_aux():
    m = parse_sysd('''
model 'AuxSim'
  dt 1
  from 0 to 5
  stock 'S': 100
    - 'Out': rate
  aux 'rate': S * 0.1
''')
    result = m.simulate(method="rk4")
    # Exponential decay: dS/dt = -0.1*S → S(5) = 100*exp(-0.5) ≈ 60.65
    assert abs(result["final_state"][0] - 60.653) < 0.01


def test_aux_with_table():
    m = parse_sysd('''
model 'AuxTbl'
  dt 1
  from 0 to 5
  table 'ramp'
    x: [0, 5]
    y: [0, 50]
  stock 'X': 0
    + 'In': tbl_val
  aux 'tbl_val': ramp(t)
''')
    result = m.simulate()
    # ramp(t) = 10*t, ∫₀⁵ 10t dt = 125
    assert abs(result["final_state"][0] - 125.0) < 1e-9
