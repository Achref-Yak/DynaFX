"""Tests for the .sysd DSL parser and simulation."""

from dynafx.dynamics.dsl import SysdModel, parse_sysd, ExprParser, _compile_system


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


# ── Python-native DSL API ──────────────────────────────────────

def test_python_api_stock():
    model = SysdModel()
    with model.stock("x", 10.0) as s:
        s.inflow("dx", "2.0")
        s.outflow("leak", "0.1 * x")
    assert len(model.stocks) == 1
    assert model.stocks[0].name == "x"
    assert model.stocks[0].initial == 10.0
    assert len(model.stocks[0].flows) == 2
    assert model.stocks[0].flows[0].name == "dx"
    assert model.stocks[0].flows[0].direction == "+"
    assert model.stocks[0].flows[0].expr == "2.0"
    assert model.stocks[0].flows[1].name == "leak"
    assert model.stocks[0].flows[1].direction == "-"
    assert model.stocks[0].flows[1].expr == "0.1 * x"


def test_python_api_stock_unit():
    model = SysdModel()
    with model.stock("displacement", 0.0, unit="m") as s:
        s.inflow("velocity", unit="m/s")
    assert model.stocks[0].units == "m"
    assert model.stocks[0].flows[0].units == "m/s"


def test_python_api_aux():
    model = SysdModel()
    model.aux("damping", "-c * velocity")
    model.aux("energy", "0.5 * m * v**2", unit="J")
    assert len(model.aux_vars) == 2
    assert model.aux_vars[0].name == "damping"
    assert model.aux_vars[0].expr == "-c * velocity"
    assert model.aux_vars[1].name == "energy"
    assert model.aux_vars[1].units == "J"


def test_python_api_table():
    model = SysdModel()
    model.table("gain", [0, 1, 2], [0.0, 0.5, 1.0])
    assert len(model.tables) == 1
    assert model.tables[0].name == "gain"
    assert model.tables[0].x == [0, 1, 2]
    assert model.tables[0].y == [0.0, 0.5, 1.0]


def test_python_api_param():
    model = SysdModel()
    model.param("k", 2.0)
    model.param("c", 0.5)
    assert model.params == {"k": 2.0, "c": 0.5}


def test_python_api_params_merged_at_simulate():
    model = SysdModel()
    model.param("k", 2.0)
    with model.stock("x", 0.0) as s:
        s.inflow("dx", "k")
    result = model.simulate(dt=0.5)
    # k=2, dt=0.5, t_span=(0,100) → 200 steps, x += k*dt = 1 per step → 200
    assert abs(result["final_state"][0] - 200.0) < 1e-9


def test_python_api_param_override_at_simulate():
    model = SysdModel()
    model.param("k", 2.0)
    with model.stock("x", 0.0) as s:
        s.inflow("dx", "k")
    result = model.simulate(params={"k": 5.0}, dt=0.5)
    assert abs(result["final_state"][0] - 500.0) < 1e-9


def test_python_api_agent():
    model = SysdModel()
    with model.agent("customer", 50) as a:
        a.prop("satisfaction", 1.0, min_val=0, max_val=1)
        a.prop("risk", 0.0)
        a.rule("churn", "satisfaction < 0.3", effects=["risk += 0.1"], priority=1)
    assert len(model.agents) == 1
    assert model.agents[0].name == "customer"
    assert model.agents[0].count == 50
    assert len(model.agents[0].properties) == 2
    assert model.agents[0].properties[0].name == "satisfaction"
    assert model.agents[0].properties[0].min == 0
    assert model.agents[0].properties[0].max == 1
    assert len(model.agents[0].rules) == 1
    assert model.agents[0].rules[0].name == "churn"
    assert model.agents[0].rules[0].priority == 1


def test_python_api_des():
    model = SysdModel()
    model.queue("orders", capacity=50, service_time="5.0", arrival_rate="10", initial=5)
    model.resource("staff", capacity=3, cost_per_unit=25.0)
    model.event("rush", rate="STEP(10,8)", target_queue="orders")
    assert len(model.queues) == 1
    assert model.queues[0].name == "orders"
    assert model.queues[0].capacity == 50
    assert model.queues[0].service_time == "5.0"
    assert model.queues[0].arrival_rate == "10"
    assert model.queues[0].initial == 5
    assert len(model.resources) == 1
    assert model.resources[0].name == "staff"
    assert model.resources[0].capacity == 3
    assert model.resources[0].cost_per_unit == 25.0
    assert len(model.events) == 1
    assert model.events[0].name == "rush"


def test_python_api_submodel():
    model = SysdModel()
    with model.submodel("sector") as sm:
        with sm.stock("population", 1000) as s:
            s.inflow("births", "population * birth_rate")
        sm.aux("birth_rate", "0.02")
    assert len(model.submodels) == 1
    assert model.submodels[0].name == "sector"
    assert len(model.submodels[0].stocks) == 1
    assert model.submodels[0].stocks[0].name == "population"
    assert model.submodels[0].stocks[0].initial == 1000
    assert model.submodels[0].stocks[0].flows[0].name == "births"
    assert model.submodels[0].stocks[0].flows[0].expr == "population * birth_rate"
    assert len(model.submodels[0].aux_vars) == 1
    assert model.submodels[0].aux_vars[0].name == "birth_rate"


def test_python_api_include():
    model = SysdModel()
    model.include("sector", alias="urban", params={"birth_rate": 0.03})
    model.include("sector", params={"birth_rate": 0.02})
    assert len(model.includes) == 2
    assert model.includes[0].submodel_name == "sector"
    assert model.includes[0].instance_name == "urban"
    assert model.includes[0].params == {"birth_rate": 0.03}
    assert model.includes[1].submodel_name == "sector"
    assert model.includes[1].instance_name == "sector_inst"


def test_python_api_construct_entire_model():
    """Build a model entirely via Python API and verify simulation works."""
    model = SysdModel("test_model")
    model.dt = 0.25
    model.t_span = (0.0, 10.0)

    with model.stock("x", 0.0) as s:
        s.inflow("dx", "y")
    with model.stock("y", 1.0) as s:
        s.inflow("dy", "-k * x - c * y")

    model.aux("k", "2.0")
    model.aux("c", "0.5")
    model.table("forcing", [0, 10], [0, 0])

    model.param("extra", 0.0)

    result = model.simulate()
    assert len(result.times) == 41
    assert result.values["x"][0] == 0.0
    assert result.values["y"][0] == 1.0


def test_python_api_matches_parse():
    """Python API model produces same results as .sysd equivalent."""
    sysd = """
    model "test"
    dt 0.25
    from 0 to 10
    stock x: 0
        + dx: y
    stock y: 1
        + dy: -k * x - c * y
    aux k: 2.0
    aux c: 0.5
    table forcing: (0,0),(10,0)
    """
    parsed = parse_sysd(sysd)

    built = SysdModel("test")
    built.dt = 0.25
    built.t_span = (0.0, 10.0)
    with built.stock("x", 0.0) as s:
        s.inflow("dx", "y")
    with built.stock("y", 1.0) as s:
        s.inflow("dy", "-k * x - c * y")
    built.aux("k", "2.0")
    built.aux("c", "0.5")
    built.table("forcing", [0, 10], [0, 0])

    r1 = parsed.simulate()
    r2 = built.simulate()
    assert r1.values["x"] == r2.values["x"]
    assert r1.values["y"] == r2.values["y"]


def test_python_api_method_chaining():
    model = SysdModel()
    model.aux("a", "1.0").aux("b", "2.0").table("t", [0], [0])
    assert len(model.aux_vars) == 2
    assert len(model.tables) == 1


def test_python_api_empty_model():
    model = SysdModel()
    result = model.simulate()
    assert len(result.times) == 101


# ── Multi-outflow auto-allocation ──────────────────────────────────

def test_auto_allocation_two_outflows():
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 100
      - O1: MIN(MAX(0, S) / dt, 10)
      - O2: MIN(MAX(0, S) / dt, 20)
    """
    model = parse_sysd(sysd)
    cache = model._compiled_cache or _compile_system(model)
    s_idx = cache.stock_names.index("S")
    assert "ALLOCATE_FRACTION" in cache.outflow_strs[s_idx], \
        "2-outflow MIN(…/dt) should auto-allocate"
    r = model.simulate(t_span=(0, 10))
    assert min(r.values["S"]) >= -0.01, "Stock went negative"


def test_auto_allocation_three_outflows():
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 100
      - O1: MIN(MAX(0, S) / dt, 5)
      - O2: MIN(MAX(0, S) / dt, 10)
      - O3: MIN(MAX(0, S) / dt, 15)
    """
    model = parse_sysd(sysd)
    r = model.simulate(t_span=(0, 10))
    assert min(r.values["S"]) >= -0.01


def test_auto_allocation_single_outflow_unchanged():
    """Single-outflow stock should not get ALLOCATE_FRACTION."""
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 100
      - O1: MIN(MAX(0, S) / dt, 10)
    """
    model = parse_sysd(sysd)
    cache = _compile_system(model)
    s_idx = cache.stock_names.index("S")
    assert "ALLOCATE_FRACTION" not in cache.outflow_strs[s_idx], \
        "Single outflow should not get ALLOCATE_FRACTION"


def test_auto_allocation_mixed_pattern_skipped():
    """Mixed MIN and non-MIN outflows should NOT auto-allocate."""
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 100
      - O1: MIN(MAX(0, S) / dt, 10)
      - O2: S * 0.1
    """
    model = parse_sysd(sysd)
    cache = _compile_system(model)
    s_idx = cache.stock_names.index("S")
    assert "ALLOCATE_FRACTION" not in cache.outflow_strs[s_idx]


def test_auto_allocation_validation_info_for_min_pattern():
    """Stock with all MIN(…/dt) outflows gets an info message at validation."""
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 100
      - O1: MIN(MAX(0, S) / dt, 10)
      - O2: MIN(MAX(0, S) / dt, 20)
    """
    model = parse_sysd(sysd)
    v = model.validate()
    info_msgs = [i for i in v.infos if "auto-apply" in i.message]
    assert len(info_msgs) == 1, f"Expected 1 info, got {len(info_msgs)}: {v.infos}"


def test_auto_allocation_validation_warn_for_non_min():
    """Stock with non-MIN outflows gets a warning at validation."""
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 100
      - O1: MIN(MAX(0, S) / dt, 10)
      - O2: S * 0.1
    """
    model = parse_sysd(sysd)
    v = model.validate()
    warn_msgs = [w for w in v.warnings if "cannot auto-allocate" in w.message]
    assert len(warn_msgs) == 1


def test_auto_allocation_already_allocated_no_warning():
    """Stock already using ALLOCATE_FRACTION gets no auto-allocation warning."""
    sysd = """
    T
    dt 0.25
    from 0 to 10
    aux avail: MAX(0, S) / dt
    aux total_d: 10 + 20
    stock S: 100
      - O1: ALLOCATE_FRACTION(avail, 10, total_d)
      - O2: ALLOCATE_FRACTION(avail, 20, total_d)
    """
    model = parse_sysd(sysd)
    v = model.validate()
    multi_warn = [w for w in v.warnings if "outflows" in w.message and "cannot auto-allocate" in w.message]
    assert len(multi_warn) == 0


def test_auto_allocation_outflow_capped_by_demand():
    """With available >> demand, each outflow gets its full demand."""
    sysd = """
    T
    dt 0.25
    from 0 to 10
    stock S: 1000
      - O1: MIN(MAX(0, S) / dt, 10)
      - O2: MIN(MAX(0, S) / dt, 20)
    """
    model = parse_sysd(sysd)
    r = model.simulate(t_span=(0, 10))
    s_vals = r.values["S"]
    # O1+O2 = 30/day drain rate, over 10 days: 1000 - 30*10 = 700
    expected = 1000 - 30 * 10
    assert abs(s_vals[-1] - expected) < 1.0, \
        f"Expected ~{expected}, got {s_vals[-1]}"


def test_auto_allocation_outflow_capped_by_available():
    """When total demand > available, outflows split proportionally."""
    sysd = """
    T
    dt 1.0
    from 0 to 5
    stock S: 10
      - O1: MIN(MAX(0, S) / dt, 100)
      - O2: MIN(MAX(0, S) / dt, 200)
    """
    model = parse_sysd(sysd)
    r = model.simulate(t_span=(0, 5))
    s_vals = r.values["S"]
    # At dt=1, S drains from 10. O1 gets 100 * 10/300 = 3.33, O2 gets 200 * 10/300 = 6.67
    # Per step: total = 10 (all of available). S → 0 after ~1 step.
    assert s_vals[-1] < 0.5, f"S should be near 0, got {s_vals[-1]}"
    assert min(s_vals) >= -0.01, "S should not go negative"
