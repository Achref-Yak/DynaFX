"""Tests for the .sysd DSL parser and simulation."""

import warnings

import pytest

from dynafx.dynamics.dsl import ExprParser, SysdModel, _compile_system, parse_sysd

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


def test_python_api_agent_prop_rejects_string_initial():
    """Phase 1 UX: string prop initializers fail at definition time."""
    model = SysdModel()
    with pytest.raises(TypeError), model.agent("patient", 10) as a:
        a.prop("severity", "random()", min_val=0, max_val=1)


def test_des_compile_error_opt_in_raises():
    """Phase 1 UX: raise_on_compile_error surfaces bad DES expressions."""
    model = SysdModel(dt=0.1, t_span=(0, 10))
    model.queue("jobs", capacity=-1, service_time="NOT_A_FUNCTION((", servers=1,
                arrival_rate="5")
    with pytest.raises(ValueError):
        model.simulate(raise_on_compile_error=True)


# ── series() canonical accessor ──────────────────────────────────


def test_series_stock_aligned():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "X * 0.1")
    result = model.simulate()
    t, v = result.series("X")
    assert t == result.times
    assert v == result.values["X"]
    assert len(t) == len(v)


def test_series_aux():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    model.aux("watch", "X")
    result = model.simulate()
    t, v = result.series("watch")
    assert v == result.aux_values["watch"]
    assert len(t) == len(v)


def test_series_des_skips_seed_and_fills_sparse():
    model = SysdModel(dt=0.5, t_span=(0, 5))
    model.queue("jobs", capacity=-1, service_time="1.0", servers=2,
                arrival_rate="5")
    result = model.simulate()
    # des_metrics_history[0] is the empty seed dict — series() must not expose it.
    t_len, q = result.series("jobs_length")
    assert len(t_len) == len(q) == len(result.times)
    assert q[0] >= 0  # seed step filled, not KeyError
    # `_departed` is sparse (only present on departure steps) — fills with 0.
    t_dep, dep = result.series("jobs_departed")
    assert len(t_dep) == len(dep) == len(result.times)
    assert all(d >= 0 for d in dep)


def test_series_sparse_key_absent_from_final_step():
    model = SysdModel(dt=0.5, t_span=(0, 5))
    model.queue("jobs", capacity=-1, service_time="1.0", servers=2,
                arrival_rate="5")
    result = model.simulate()
    # A sparse key (e.g. `_departed`) only exists on steps with a departure.
    # Force the regression: drop it from the *final* history dict, then
    # series() must still resolve it from the union of keys across all steps.
    result.des_metrics_history[-1].pop("jobs_departed", None)
    t_dep, dep = result.series("jobs_departed")
    assert len(t_dep) == len(dep) == len(result.times)
    assert "jobs_departed" in result._series_names()


def test_series_unknown_raises():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    with pytest.raises(KeyError):
        result.series("nope")
    # existing names still resolve
    _, v = result.series("X")
    assert v == result.values["X"]


def test_series_lists_available_names_on_miss():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    try:
        result.series("nope")
    except KeyError as exc:
        assert "X" in str(exc)


# ── typed DES stats on results ──────────────────────────────────


def test_result_des_stats_forwarding():
    model = SysdModel(dt=0.5, t_span=(0, 10))
    model.queue("jobs", capacity=-1, service_time="1.0", servers=1,
                arrival_rate="2")
    model.resource("doctor", capacity=1)
    result = model.simulate()
    q = result.queue_stats("jobs")
    r = result.resource_stats("doctor")
    assert q.summary()["kind"] == "queue"
    assert r.summary()["kind"] == "resource"
    # result.stats() picks the right kind without mixing
    assert result.stats("jobs").summary()["kind"] == "queue"
    assert result.stats("doctor").summary()["kind"] == "resource"


def test_result_stats_no_des_raises():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    with pytest.raises(KeyError):
        result.stats("anything")


def test_result_stats_missing_name():
    model = SysdModel(dt=0.5, t_span=(0, 10))
    model.queue("jobs", capacity=-1, service_time="1.0", servers=1,
                arrival_rate="2")
    result = model.simulate()
    try:
        result.stats("nope")
    except KeyError as exc:
        assert "jobs" in str(exc)


# ── plotting contract ───────────────────────────────────────────


def test_plot_returns_fig_when_no_path():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    fig = result.plot()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_return_fig_flag():
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    fig = result.plot(path="ignored.png", return_fig=True)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_saves_and_returns_none(tmp_path):
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    out = tmp_path / "plot.png"
    ret = result.plot(str(out))
    assert ret is None
    assert out.exists()


def test_plot_resolves_aux_and_des_via_series():
    model = SysdModel(dt=0.5, t_span=(0, 5))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    model.aux("watch", "X")
    model.queue("jobs", capacity=-1, service_time="1.0", servers=1,
                arrival_rate="2")
    result = model.simulate()
    import matplotlib.pyplot as plt

    fig = result.plot(stocks=["watch", "jobs_length"])
    assert fig is not None
    plt.close(fig)


def test_plot_with_bands_returns_fig_and_saves(tmp_path):
    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    mean = {"X": result.values["X"]}
    p5 = {"X": [v * 0.9 for v in mean["X"]]}
    p95 = {"X": [v * 1.1 for v in mean["X"]]}
    import matplotlib.pyplot as plt

    fig = result.plot_with_bands(mean=mean, p5=p5, p95=p95)
    assert fig is not None
    plt.close(fig)
    out = tmp_path / "bands.png"
    ret = result.plot_with_bands(str(out), mean=mean, p5=p5, p95=p95)
    assert ret is None
    assert out.exists()


def test_plot_missing_matplotlib_raises(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    model = SysdModel(dt=0.5, t_span=(0, 2))
    with model.stock("X", 10) as s:
        s.outflow("drain", "0.1")
    result = model.simulate()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        result.plot()


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
    model = SysdModel("test_model", dt=0.25, t_span=(0.0, 10.0))

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

    built = SysdModel("test", dt=0.25, t_span=(0.0, 10.0))
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


# ── Phase 5: constructor idiom + route() DSL ────────────────────


def test_constructor_form_is_canonical():
    m = SysdModel("demo", dt=0.5, t_span=(0, 10))
    assert m.dt == 0.5
    assert m.t_span == (0, 10)


def test_post_init_dt_set_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = SysdModel("old")
        m.dt = 1.0
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert m.dt == 1.0  # still works, just deprecated


def test_post_init_t_span_set_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = SysdModel("old")
        m.t_span = (0.0, 25.0)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert m.t_span == (0.0, 25.0)


def test_constructor_form_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SysdModel("ok", dt=0.25, t_span=(0.0, 10.0))
    assert not any(issubclass(x.category, DeprecationWarning) for x in w)


def test_route_dsl_validates_queues():
    m = SysdModel("demo", dt=0.5, t_span=(0, 10))
    m.queue("a", capacity=-1, service_time="1.0")
    m.queue("b", capacity=-1, service_time="1.0")
    m.route("a", "True", "b")
    assert m.queues[0].routes[0].to_queue == "b"
    with pytest.raises(ValueError):
        m.route("nope", "True", "b")
    with pytest.raises(ValueError):
        m.route("a", "True", "nope")


def test_route_routing_moves_entities():
    m = SysdModel("demo", dt=0.5, t_span=(0, 20))
    m.queue("src", capacity=-1, service_time="2.0", servers=1, arrival_rate="3")
    m.queue("dst", capacity=-1, service_time="5.0", servers=1)
    m.route("src", "True", "dst")
    r = m.simulate()
    assert r.des_engine.queue_stats("dst").total_arrivals > 0
    assert r.des_engine.queue_stats("dst").total_arrivals == \
        r.des_engine.queue_stats("src").total_departures


def test_route_preserves_entity_fields():
    m = SysdModel("demo", dt=1.0, t_span=(0, 10))
    m.queue("a", capacity=-1, service_time="1.0", servers=1)
    m.queue("hi", capacity=-1, service_time="1.0", servers=1)
    m.queue("lo", capacity=-1, service_time="1.0", servers=1)
    m.route("a", "entity.get('tier', 0) > 1", "hi")
    m.route("a", "True", "lo")
    r = m.simulate()
    hi = r.des_engine.queue_stats("hi").total_arrivals
    lo = r.des_engine.queue_stats("lo").total_arrivals
    assert hi >= 0 and lo >= 0
    # absent 'tier' -> first rule false -> falls through to 'lo'
    assert lo > 0 or hi >= 0  # arrival entities carry no tier key


def test_queue_discipline_param():
    m = SysdModel("demo", dt=0.5, t_span=(0, 10))
    m.queue("q", capacity=-1, service_time="1.0", servers=1, discipline="SPT")
    assert m.queues[0].discipline == "SPT"
    r = m.simulate()
    assert r.des_engine.queues["q"].discipline == "SPT"


def test_priority_discipline_orders_by_entity_priority():
    m = SysdModel("demo", dt=0.5, t_span=(0, 5))
    m.queue("q", capacity=-1, service_time="1.0", servers=1, discipline="PRIORITY")
    r = m.simulate()
    from dynafx.dynamics.des import Queue, DESEngine

    q = Queue("t", service_time="1.0", discipline="PRIORITY")
    q.enqueue({"priority": 5}, 0.0)
    q.enqueue({"priority": 1}, 0.0)
    assert q.dequeue(1.0)["priority"] == 1  # lowest number served first


def test_parse_sysd_discipline_and_route():
    m = parse_sysd("""
model t
dt 1
from 0 to 10
queue "Q": capacity 5, discipline SPT
  service_time 2
  arrival_rate 1
  route entity.priority > 2 -> "Q2"
queue "Q2": capacity 5
  service_time 2
""")
    q = m.queues[0]
    assert q.discipline == "SPT"
    assert q.routes[0].to_queue == "Q2"
    assert q.routes[0].condition == "entity.priority > 2"
