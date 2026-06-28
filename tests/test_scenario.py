"""Tests for system/scenario.py — scenario comparison."""

import pytest
import math

from dynafx.system.dsl import SysdModel, parse_sysd, SysdModelResult
from dynafx.system.scenario import (
    ScenarioComparison,
    ScenarioDef,
    ScenarioResult,
)


SIMPLE_MODEL = """
Test Model
dt 0.1
from 0 to 5

stock "Population" = 100
  + "births": pop_growth * Population
  - "deaths": Population * mortality

aux "pop_growth" = 0.02
aux "mortality" = 0.01
"""


@pytest.fixture
def model():
    return parse_sysd(SIMPLE_MODEL)


# ── ScenarioDef ───────────────────────────────────────────────────


class TestScenarioDef:
    def test_construction(self):
        sd = ScenarioDef("test", {"a": 1, "b": 2})
        assert sd.name == "test"
        assert sd.params == {"a": 1, "b": 2}


# ── ScenarioResult ────────────────────────────────────────────────


class TestScenarioResult:
    def test_getitem(self, model):
        result = model.simulate(params={})
        sr = ScenarioResult("base", result, {})
        vals = sr["Population"]
        assert len(vals) > 0
        assert vals[0] == 100.0

    def test_attributes(self, model):
        result = model.simulate(params={})
        sr = ScenarioResult("base", result, {"a": 1})
        assert sr.name == "base"
        assert sr.params == {"a": 1}


# ── ScenarioComparison construction ──────────────────────────────


class TestScenarioComparisonConstruction:
    def test_basic(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Baseline", {}),
            ScenarioDef("High growth", {"pop_growth": 0.05}),
        ])
        assert len(comp.scenarios) == 2
        assert comp.names == ["Baseline", "High growth"]

    def test_times(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
        ])
        assert len(comp.times) > 0
        assert comp.times[0] == 0.0

    def test_get(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("A", {}),
            ScenarioDef("B", {"pop_growth": 0.05}),
        ])
        a = comp.get("A")
        assert a is not None
        assert a.name == "A"
        assert comp.get("Nonexistent") is None


# ── Summary ──────────────────────────────────────────────────────


class TestSummary:
    def test_summary(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        s = comp.summary()
        assert "Base" in s
        assert "High" in s
        assert "Population" in s["Base"]
        assert s["Base"]["Population"] is not None

    def test_summary_different_values(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        s = comp.summary()
        # High growth should give larger population
        assert s["High"]["Population"] > s["Base"]["Population"]

    def test_deviation_table_absolute(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        d = comp.deviation_table(baseline=0, mode="absolute")
        assert "High" in d
        assert "Base" in d

    def test_deviation_table_relative(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        d = comp.deviation_table(baseline=0, mode="relative")
        # High growth deviation should be positive
        assert d["High"]["Population"] > 0


# ── Tornado ──────────────────────────────────────────────────────


class TestTornado:
    def test_tornado_impacts(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
        ])
        # Can't test plotting without matplotlib, but we can test that
        # the simulation runs for each param extreme
        t = comp.times[-1]
        pop_at_t = comp.scenarios[0].result.values["Population"][-1]
        assert pop_at_t > 0

    def test_interp_before_start(self, model):
        comp = ScenarioComparison(model, [ScenarioDef("B", {})])
        # Can't test plotting, but we can test the interp helper
        r = comp.scenarios[0].result
        val = ScenarioComparison._interp_at(r, "Population", -1)
        assert val == r.values["Population"][0]

    def test_interp_after_end(self, model):
        comp = ScenarioComparison(model, [ScenarioDef("B", {})])
        r = comp.scenarios[0].result
        val = ScenarioComparison._interp_at(r, "Population", 999)
        assert val == r.values["Population"][-1]


# ── Plots (mock test — no matplotlib check) ──────────────────────


class TestPlots:
    def test_plot_comparison_no_error(self, model, tmp_path):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        out = tmp_path / "comp.png"
        comp.plot_comparison(str(out), stocks=["Population"])
        # matplotlib may or may not be installed — just test no crash
        assert True

    def test_plot_deviation_no_error(self, model, tmp_path):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        out = tmp_path / "dev.png"
        comp.plot_deviation(str(out), stocks=["Population"],
                            mode="absolute")
        assert True

    def test_plot_deviation_relative(self, model, tmp_path):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        out = tmp_path / "dev_rel.png"
        comp.plot_deviation(str(out), stocks=["Population"],
                            mode="relative")
        assert True

    def test_tornado_no_error(self, model, tmp_path):
        comp = ScenarioComparison(model, [
            ScenarioDef("Base", {}),
        ])
        out = tmp_path / "tornado.png"
        comp.tornado(str(out), {"pop_growth": (0.01, 0.05)},
                     output_stock="Population")
        assert True


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_scenarios(self, model):
        comp = ScenarioComparison(model, [])
        s = comp.summary()
        assert s == {}

    def test_single_scenario(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Only", {}),
        ])
        assert len(comp.scenarios) == 1
        s = comp.summary()
        assert "Only" in s

    def test_three_scenarios(self, model):
        comp = ScenarioComparison(model, [
            ScenarioDef("Low", {"pop_growth": 0.01}),
            ScenarioDef("Mid", {"pop_growth": 0.02}),
            ScenarioDef("High", {"pop_growth": 0.05}),
        ])
        s = comp.summary()
        assert s["Low"]["Population"] < s["High"]["Population"]
