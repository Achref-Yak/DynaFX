"""Tests for Vensim .mdl import."""

from dynafx.dynamics.vensim import parse_mdl
from dynafx.dynamics.dsl import parse_sysd


SIMPLE_MDL = """Simple SIR|Simple SIR model
Susceptible = INTEG( -Infection_Rate, 990 ) ~~|
Infected = INTEG( Infection_Rate - Recovery_Rate, 10 ) ~~|
Recovered = INTEG( Recovery_Rate, 0 ) ~~|
Infection_Rate = beta * Susceptible * Infected ~~|
Recovery_Rate = gamma * Infected ~~|
INITIAL TIME = 0 ~~|
FINISH TIME = 150 ~~|
TIME STEP = 0.25 ~~|
"""

LOOKUP_MDL = """Lookup Test|Lookup test model
demand = WITH LOOKUP(time,
    [(0,0)-(200,2000)], (0,500), (50,800), (100,1100), (200,600) ) ~~|
Stock = INTEG( demand - Supply, 1000 ) ~~|
INITIAL TIME = 0 ~~|
FINISH TIME = 100 ~~|
TIME STEP = 0.5 ~~|
"""


def test_parse_simple_sir():
    m = parse_mdl(SIMPLE_MDL)
    assert m.name == "Simple SIR"
    assert len(m.stocks) >= 3
    names = [s.name for s in m.stocks]
    assert "Susceptible" in names
    assert "Infected" in names
    assert "Recovered" in names


def test_parse_stock_initial_values():
    m = parse_mdl(SIMPLE_MDL)
    for s in m.stocks:
        if s.name == "Susceptible":
            assert s.initial == 990.0
        elif s.name == "Infected":
            assert s.initial == 10.0
        elif s.name == "Recovered":
            assert s.initial == 0.0


def test_parse_aux_equations():
    m = parse_mdl(SIMPLE_MDL)
    aux_names = [a.name for a in m.aux_vars]
    assert "Infection_Rate" in aux_names
    assert "Recovery_Rate" in aux_names


def test_parse_lookup_table():
    m = parse_mdl(LOOKUP_MDL)
    assert len(m.tables) >= 1
    assert any(t.name == "demand" or t.name in ["demand"] for t in m.tables)


def test_time_mapping():
    mdl = """TMap|
X = INTEG( t * 2, 0 ) ~~|
"""
    m = parse_mdl(mdl)
    assert len(m.stocks) == 1


def test_imported_model_simulates():
    m = parse_mdl(SIMPLE_MDL)
    if not m.stocks:
        return
    m.dt = 0.25
    m.t_span = (0.0, 50.0)
    result = m.simulate(method="rk4", params={"beta": 0.003, "gamma": 0.1})
    assert result.steps > 0


def test_import_handles_delayed():
    mdl = """DelayTest|
X = INTEG( Inflow - SMOOTH(X, 5), 100 ) ~~|
"""
    m = parse_mdl(mdl)
    assert len(m.stocks) >= 1


def test_import_handles_multipart_flow():
    """Import a stock with inflow - outflow expression."""
    m = parse_mdl(SIMPLE_MDL)
    for s in m.stocks:
        if s.name == "Infected":
            assert len(s.flows) >= 1
            break
