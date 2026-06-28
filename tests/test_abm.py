"""Tests for ABM (Agent-Based Modeling) engine and DSL integration."""

from dynafx.system.dsl import (
    parse_sysd, AgentDef, AgentPropDef, AgentRuleDef,
)
from dynafx.system.agent import (
    AgentInstance, ABMEngine, _eval_condition, _eval_effect,
)


# ── Parser Tests ───────────────────────────────────────────────

class TestAgentParser:
    def test_parse_single_agent(self):
        m = parse_sysd('A\ndt 0.1\nagent "Buyer": 5\n')
        assert len(m.agents) == 1
        assert m.agents[0].name == "Buyer"
        assert m.agents[0].count == 5

    def test_parse_multiple_agents(self):
        m = parse_sysd('A\ndt 0.1\nagent "Buyer": 3\nagent "Seller": 2\n')
        assert len(m.agents) == 2
        assert m.agents[0].name == "Buyer"
        assert m.agents[1].name == "Seller"

    def test_parse_agent_properties(self):
        m = parse_sysd('A\ndt 0.1\nagent "X": 1\n  property "hp": 100, min=0, max=200\n')
        a = m.agents[0]
        assert len(a.properties) == 1
        p = a.properties[0]
        assert p.name == "hp"
        assert p.initial == 100.0
        assert p.min == 0.0
        assert p.max == 200.0

    def test_parse_agent_rules(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  property "hp": 100, min=0\n'
            '  rule heal: when hp < 50\n'
            '    hp += 10\n'
        )
        a = m.agents[0]
        assert len(a.rules) == 1
        r = a.rules[0]
        assert r.name == "heal"
        assert r.condition == "hp < 50"
        assert r.effects == ["hp += 10"]

    def test_parse_rule_multiple_effects(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  property "a": 0\n'
            '  property "b": 0\n'
            '  rule combo: when a < 10\n'
            '    a += 1\n'
            '    b += 2\n'
        )
        r = m.agents[0].rules[0]
        assert len(r.effects) == 2
        assert "a += 1" in r.effects
        assert "b += 2" in r.effects

    def test_parse_no_agents(self):
        m = parse_sysd('S\ndt 0.1\nstock "X": 10\n')
        assert len(m.agents) == 0

    def test_parse_agent_default_count(self):
        m = parse_sysd('A\ndt 0.1\nagent "X"\n')
        assert m.agents[0].count == 1

    def test_parse_property_defaults(self):
        m = parse_sysd('A\ndt 0.1\nagent "X": 1\n  property "v": 5\n')
        p = m.agents[0].properties[0]
        assert p.min == 0.0
        assert p.max == 1e18


# ── Condition Evaluation ───────────────────────────────────────

class TestEvalCondition:
    def test_simple_gt(self):
        assert _eval_condition("x > 5", {"x": 10.0}) is True
        assert _eval_condition("x > 5", {"x": 3.0}) is False

    def test_simple_lt(self):
        assert _eval_condition("x < 5", {"x": 3.0}) is True

    def test_simple_eq(self):
        assert _eval_condition("x == 5", {"x": 5.0}) is True
        assert _eval_condition("x == 5", {"x": 6.0}) is False

    def test_and_condition(self):
        assert _eval_condition("x > 0 and y > 0", {"x": 1.0, "y": 1.0}) is True
        assert _eval_condition("x > 0 and y > 0", {"x": 1.0, "y": 0.0}) is False

    def test_or_condition(self):
        assert _eval_condition("x > 5 or y > 5", {"x": 0.0, "y": 10.0}) is True
        assert _eval_condition("x > 5 or y > 5", {"x": 0.0, "y": 0.0}) is False

    def test_ge_and_le(self):
        assert _eval_condition("x >= 5", {"x": 5.0}) is True
        assert _eval_condition("x <= 5", {"x": 5.0}) is True

    def test_ne(self):
        assert _eval_condition("x != 0", {"x": 1.0}) is True
        assert _eval_condition("x != 0", {"x": 0.0}) is False

    def test_missing_variable(self):
        assert _eval_condition("x > 5", {}) is False

    def test_invalid_expr(self):
        assert _eval_condition(">>> bad", {"x": 1.0}) is False


# ── Effect Evaluation ──────────────────────────────────────────

class TestEvalEffect:
    def test_plus_equals(self):
        prop, delta = _eval_effect("budget -= 10", {"budget": 100.0})
        assert prop == "budget"
        assert delta == -10.0

    def test_minus_equals(self):
        prop, delta = _eval_effect("hp -= 5", {"hp": 80.0})
        assert prop == "hp"
        assert delta == -5.0

    def test_times_equals(self):
        prop, delta = _eval_effect("speed *= 2", {"speed": 5.0})
        assert prop == "speed"
        assert delta == 5.0  # 5*2 - 5 = 5

    def test_div_equals(self):
        prop, delta = _eval_effect("count /= 2", {"count": 10.0})
        assert prop == "count"
        assert delta == -5.0  # 10/2 - 10 = -5

    def test_effect_with_variable_rhs(self):
        prop, delta = _eval_effect("budget -= price", {"budget": 100.0, "price": 25.0})
        assert prop == "budget"
        assert delta == -25.0


# ── AgentInstance ──────────────────────────────────────────────

class TestAgentInstance:
    def _make_instance(self):
        ad = AgentDef(
            name="Buyer",
            count=1,
            properties=[
                AgentPropDef("budget", 100.0, min=0.0),
                AgentPropDef("satisfaction", 0.0, min=0.0, max=10.0),
            ],
            rules=[
                AgentRuleDef("buy", "budget > 10", ["budget -= 10", "satisfaction += 1"]),
            ],
        )
        return AgentInstance(ad, 0, {"budget": 100.0, "satisfaction": 0.0})

    def test_perceive_merges_env(self):
        inst = self._make_instance()
        env = {"Price": 10.0, "Other": 42.0}
        perceive = inst.perceive(env)
        assert perceive["budget"] == 100.0
        assert perceive["Price"] == 10.0
        assert perceive["Buyer.budget"] == 100.0

    def test_decide_fires_when_condition_met(self):
        inst = self._make_instance()
        perceive = inst.perceive({})
        effects = inst.decide(perceive)
        assert len(effects) == 2

    def test_decide_skips_when_condition_not_met(self):
        inst = self._make_instance()
        inst.state["budget"] = 5.0  # below threshold
        perceive = inst.perceive({})
        effects = inst.decide(perceive)
        assert len(effects) == 0

    def test_act_applies_effects(self):
        inst = self._make_instance()
        inst.act([("budget", -10.0), ("satisfaction", 1.0)])
        assert inst.state["budget"] == 90.0
        assert inst.state["satisfaction"] == 1.0

    def test_act_clamps_min(self):
        inst = self._make_instance()
        inst.act([("budget", -200.0)])
        assert inst.state["budget"] == 0.0

    def test_act_clamps_max(self):
        inst = self._make_instance()
        inst.act([("satisfaction", 100.0)])
        assert inst.state["satisfaction"] == 10.0

    def test_full_cycle(self):
        inst = self._make_instance()
        perceive = inst.perceive({})
        effects = inst.decide(perceive)
        inst.act(effects)
        assert inst.state["budget"] == 90.0
        assert inst.state["satisfaction"] == 1.0


# ── ABMEngine ──────────────────────────────────────────────────

class TestABMEngine:
    def _make_engine(self, n_buyers=3):
        ad = AgentDef(
            name="Buyer",
            count=n_buyers,
            properties=[
                AgentPropDef("budget", 100.0, min=0.0),
            ],
            rules=[
                AgentRuleDef("buy", "budget > 10", ["budget -= 10"]),
            ],
        )
        engine = ABMEngine([ad])
        engine.initialize()
        return engine

    def test_initialize_creates_instances(self):
        engine = self._make_engine(n_buyers=5)
        assert len(engine.instances) == 5

    def test_all_instances_start_with_same_state(self):
        engine = self._make_engine(n_buyers=3)
        for inst in engine.instances:
            assert inst.state["budget"] == 100.0

    def test_step_returns_metrics(self):
        engine = self._make_engine(n_buyers=3)
        metrics = engine.step(0.0, 0.1, {})
        assert "Buyer_budget_avg" in metrics
        assert "Buyer_budget_sum" in metrics
        assert "Buyer_budget_min" in metrics
        assert "Buyer_budget_max" in metrics
        assert "Buyer_count" in metrics

    def test_step_executes_rules(self):
        engine = self._make_engine(n_buyers=3)
        engine.step(0.0, 0.1, {})
        # All 3 buyers had budget=100 > 10, so each spent 10
        for inst in engine.instances:
            assert inst.state["budget"] == 90.0

    def test_metrics_correct_after_step(self):
        engine = self._make_engine(n_buyers=3)
        metrics = engine.step(0.0, 0.1, {})
        assert metrics["Buyer_budget_avg"] == 90.0
        assert metrics["Buyer_budget_sum"] == 270.0
        assert metrics["Buyer_count"] == 3.0

    def test_variance_computed_for_multiple_agents(self):
        engine = self._make_engine(n_buyers=2)
        # Give different initial states
        engine.instances[0].state["budget"] = 50.0
        engine.instances[1].state["budget"] = 150.0
        metrics = engine.get_metrics()
        assert "Buyer_budget_var" in metrics
        assert metrics["Buyer_budget_var"] > 0

    def test_agents_stop_spending_at_threshold(self):
        ad = AgentDef(
            name="Saver",
            count=1,
            properties=[AgentPropDef("budget", 5.0, min=0.0)],
            rules=[AgentRuleDef("buy", "budget > 10", ["budget -= 10"])],
        )
        engine = ABMEngine([ad])
        engine.initialize()
        metrics = engine.step(0.0, 0.1, {})
        # budget=5, condition budget>10 is False, no spending
        assert metrics["Saver_budget_avg"] == 5.0


# ── DSL Integration ────────────────────────────────────────────

class TestABMDslIntegration:
    def test_agents_field_on_model(self):
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 1\n'
            'agent "X": 2\n'
            '  property "v": 10\n'
        )
        assert len(m.agents) == 1
        assert m.agents[0].count == 2

    def test_simulate_with_agents(self):
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 1\n'
            'agent "X": 3\n'
            '  property "v": 10\n'
            '  rule grow: when v < 20\n'
            '    v += 1\n'
        )
        r = m.simulate()
        assert r.steps > 0
        assert r.abm_engine is not None

    def test_abm_engine_accessible_from_result(self):
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 1\n'
            'agent "X": 2\n'
            '  property "v": 0\n'
            '  rule inc: when v < 100\n'
            '    v += 1\n'
        )
        r = m.simulate()
        engine = r.abm_engine
        assert engine is not None
        metrics = engine.get_metrics()
        assert "X_v_avg" in metrics

    def test_agents_independent_of_sd(self):
        """Agents run even with no SD stocks."""
        m = parse_sysd(
            'A\ndt 1\nfrom 0 to 5\n'
            'agent "Counter": 1\n'
            '  property "x": 0\n'
            '  rule count: when x < 100\n'
            '    x += 1\n'
        )
        r = m.simulate()
        assert r.abm_engine.instances[0].state["x"] == 5.0

    def test_mixed_sd_agents(self):
        """SD stock can coexist with agents in same model."""
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 10\n'
            'stock "Resource": 100\n'
            '  - "Consumption": consumption_rate\n'
            'aux "consumption_rate": Buyer_count * 0.5\n'
            'agent "Buyer": 4\n'
            '  property "purchased": 0\n'
            '  rule buy: when Resource > 10\n'
            '    purchased += 1\n'
        )
        r = m.simulate()
        # Resource should have decreased due to consumption
        assert r.values["Resource"][-1] < 100.0
        # Buyers should have purchased something
        engine = r.abm_engine
        total_purchased = sum(
            inst.state["purchased"] for inst in engine.instances
        )
        assert total_purchased > 0

    def test_no_agents_no_abm_engine(self):
        m = parse_sysd('S\ndt 0.1\nfrom 0 to 1\nstock "X": 10\n')
        r = m.simulate()
        assert r.abm_engine is None
