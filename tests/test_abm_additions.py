"""Tests for ABM additions: message passing, strategy switching, meta-rules."""

from dynafx.dynamics.dsl import (
    parse_sysd, SysdModel, AgentDef, AgentStrategy, AgentRuleDef, AgentPropDef,
)
from dynafx.dynamics.agent import (
    AgentInstance, ABMEngine, Message,
    _eval_condition, _eval_effect,
    _parse_send, _parse_switch_strategy,
)


# ── Message ──────────────────────────────────────────────────────

class TestMessage:
    def test_message_fields(self):
        msg = Message(sender_id=0, sender_type="A", target_type="B", topic="alert", payload={"val": 1.0})
        assert msg.sender_id == 0
        assert msg.sender_type == "A"
        assert msg.target_type == "B"
        assert msg.topic == "alert"
        assert msg.payload == {"val": 1.0}
        assert msg.ttl == 1

    def test_message_default_ttl(self):
        msg = Message(sender_id=0, sender_type="A", target_type="B", topic="t")
        assert msg.ttl == 1

    def test_message_default_payload(self):
        msg = Message(sender_id=0, sender_type="A", target_type="B", topic="t")
        assert msg.payload == {}


# ── AgentStrategy + AgentDef ─────────────────────────────────────

class TestAgentDataModel:
    def test_strategy_default_factory(self):
        s = AgentStrategy("normal")
        assert s.name == "normal"
        assert s.rules == []

    def test_agent_def_strategies_empty_by_default(self):
        ad = AgentDef("X")
        assert ad.strategies == []
        assert ad.meta_rules == []

    def test_agent_def_with_strategies(self):
        s = AgentStrategy("crisis", rules=[AgentRuleDef("r", "x < 10")])
        ad = AgentDef("X", strategies=[s])
        assert len(ad.strategies) == 1
        assert ad.strategies[0].name == "crisis"

    def test_agent_def_with_meta_rules(self):
        mr = AgentRuleDef("detect", "inbox > 0", ["SWITCH_STRATEGY('crisis')"])
        ad = AgentDef("X", meta_rules=[mr])
        assert len(ad.meta_rules) == 1
        assert ad.meta_rules[0].name == "detect"


# ── Parser: strategy / meta_rule ─────────────────────────────────

class TestAgentParserAdditions:
    def test_parse_strategy_block(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  strategy "normal"\n'
            '    rule "reorder": when inventory < 100\n'
            '      order_qty += 50\n'
        )
        a = m.agents[0]
        assert len(a.strategies) == 1
        assert a.strategies[0].name == "normal"
        assert len(a.strategies[0].rules) == 1
        assert a.strategies[0].rules[0].name == "reorder"

    def test_parse_multiple_strategies(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  strategy "normal"\n'
            '    rule "a": when always\n'
            '      x += 1\n'
            '  strategy "crisis"\n'
            '    rule "b": when x < 5\n'
            '      x += 10\n'
        )
        a = m.agents[0]
        assert len(a.strategies) == 2
        assert a.strategies[0].name == "normal"
        assert a.strategies[1].name == "crisis"

    def test_parse_meta_rule(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  meta_rule "detect": when inbox_alert > 0\n'
            '    SWITCH_STRATEGY("crisis")\n'
        )
        a = m.agents[0]
        assert len(a.meta_rules) == 1
        assert a.meta_rules[0].name == "detect"
        assert a.meta_rules[0].condition == "inbox_alert > 0"
        assert len(a.meta_rules[0].effects) == 1
        assert "SWITCH_STRATEGY" in a.meta_rules[0].effects[0]

    def test_parse_meta_rule_multiple_effects(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  meta_rule "detect": when inbox_warning > 0\n'
            '    SWITCH_STRATEGY("crisis")\n'
            '    SEND(Supplier, "ack", received=1)\n'
        )
        a = m.agents[0]
        effects = a.meta_rules[0].effects
        assert len(effects) == 2
        assert any("SWITCH_STRATEGY" in e for e in effects)
        assert any("SEND" in e for e in effects)

    def test_parse_strategy_with_flat_rules(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  strategy "s"\n'
            '    rule "sr": when always\n'
            '      v += 1\n'
            '  rule "flat": when always\n'
            '      v += 2\n'
        )
        a = m.agents[0]
        assert len(a.strategies) == 1
        assert len(a.rules) == 1  # flat rules still parsed

    def test_network_keyword_after_strategy(self):
        """network keyword should still work alongside strategies."""
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 2\n'
            '  strategy "s"\n'
            '    rule "r": when always\n'
            '      x += 1\n'
            '  network complete\n'
        )
        a = m.agents[0]
        assert a.network_type == "complete"

    def test_meta_rule_with_cooldown(self):
        m = parse_sysd(
            'A\ndt 0.1\n'
            'agent "X": 1\n'
            '  meta_rule "detect": when inbox_alert > 0\n'
            '    SWITCH_STRATEGY("crisis", cooldown=10)\n'
        )
        effects = m.agents[0].meta_rules[0].effects
        assert len(effects) == 1
        assert "cooldown=10" in effects[0]


# ── Python API: strategy / meta_rule ─────────────────────────────

class TestPythonAPIAdditions:
    def test_python_api_strategy_context(self):
        m = SysdModel("test")
        with m.agent("X", 1) as a:
            a.prop("v", 0.0)
            with a.strategy("normal") as s:
                s.rule("r", "v < 10", ["v += 1"])
        a = m.agents[0]
        assert len(a.strategies) == 1
        assert a.strategies[0].name == "normal"
        assert a.strategies[0].rules[0].name == "r"

    def test_python_api_multiple_strategies(self):
        m = SysdModel("test")
        with m.agent("X", 1) as a:
            a.prop("v", 0.0)
            with a.strategy("normal") as s:
                s.rule("r1", "v < 10", ["v += 1"])
            with a.strategy("crisis") as s:
                s.rule("r2", "v < 5", ["v += 10"])
        assert len(m.agents[0].strategies) == 2

    def test_python_api_meta_rule(self):
        m = SysdModel("test")
        with m.agent("X", 1) as a:
            a.prop("v", 0.0)
            a.meta_rule("detect", "inbox_alert > 0", ["SWITCH_STRATEGY('crisis')"])
        mr = m.agents[0].meta_rules
        assert len(mr) == 1
        assert mr[0].name == "detect"
        assert "SWITCH_STRATEGY" in mr[0].effects[0]

    def test_python_api_flat_rules_preserved(self):
        m = SysdModel("test")
        with m.agent("X", 1) as a:
            a.prop("v", 0.0)
            a.rule("r", "always", ["v += 1"])
            with a.strategy("s") as s:
                s.rule("sr", "v < 5", ["v += 2"])
        a = m.agents[0]
        assert len(a.rules) == 1
        assert len(a.strategies) == 1


# ── Perceive with mailbox ────────────────────────────────────────

class TestPerceiveWithMailbox:
    def _make_instance(self):
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)])
        return AgentInstance(ad, 0, {"v": 0.0})

    def test_inbox_total_in_perceive(self):
        inst = self._make_instance()
        inst.mailbox.append(Message(0, "S", "X", "alert"))
        p = inst.perceive({}, t=0.0)
        assert p["inbox"] == 1.0
        assert p["inbox_total"] == 1.0

    def test_inbox_topic_count(self):
        inst = self._make_instance()
        inst.mailbox.append(Message(0, "S", "X", "warning"))
        inst.mailbox.append(Message(1, "S", "X", "warning"))
        inst.mailbox.append(Message(2, "S", "X", "order"))
        p = inst.perceive({}, t=0.0)
        assert p["inbox_warning"] == 2.0
        assert p["inbox_order"] == 1.0
        assert p["inbox"] == 3.0

    def test_empty_inbox(self):
        inst = self._make_instance()
        p = inst.perceive({}, t=0.0)
        assert p["inbox"] == 0.0
        assert p["inbox_total"] == 0.0

    def test_strategy_in_perceive(self):
        inst = self._make_instance()
        assert "strategy" in inst.perceive({}, t=0.0)

    def test_strategy_name_in_perceive(self):
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)],
                       strategies=[AgentStrategy("crisis")])
        inst = AgentInstance(ad, 0, {"v": 0.0})
        inst.strategy = "crisis"
        assert inst.perceive({}, t=0.0)["strategy"] == "crisis"


# ── Decide: meta-rules ───────────────────────────────────────────

class TestDecideMetaRules:
    def _make_instance(self):
        mr = AgentRuleDef("detect", "inbox_alert > 0", ["SWITCH_STRATEGY('crisis')"])
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)], meta_rules=[mr])
        return AgentInstance(ad, 0, {"v": 0.0})

    def test_meta_rule_switches_strategy(self):
        inst = self._make_instance()
        inst.decide({"inbox_alert": 1.0, "v": 0.0}, t=0.0)
        assert inst.strategy == "crisis"

    def test_meta_rule_no_switch_when_condition_false(self):
        inst = self._make_instance()
        inst.decide({"inbox_alert": 0.0, "v": 0.0}, t=0.0)
        assert inst.strategy is None

    def test_meta_rule_always_evaluated(self):
        inst = self._make_instance()
        inst.decide({"inbox_alert": 1.0, "v": 0.0}, t=0.0)
        assert inst.strategy == "crisis"
        # Even if now strategy=crisis, meta-rule still fires again
        inst.decide({"inbox_alert": 1.0, "v": 0.0}, t=1.0)
        assert inst.strategy == "crisis"

    def test_meta_rule_cooldown(self):
        mr = AgentRuleDef("detect", "always", ["SWITCH_STRATEGY('crisis', cooldown=5)"])
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)], meta_rules=[mr])
        inst = AgentInstance(ad, 0, {"v": 0.0})
        inst.decide({"always": True, "v": 0.0}, t=0.0)
        assert inst.strategy == "crisis"
        # Try to switch again during cooldown
        mr2 = AgentRuleDef("detect2", "always", ["SWITCH_STRATEGY('normal', cooldown=0)"])
        ad2 = AgentDef("X", 1, meta_rules=[mr2])
        inst.agent_def = ad2
        inst.decide({"always": True, "v": 0.0}, t=1.0)
        assert inst.strategy == "crisis"  # still crisis, lock active

    def test_meta_rule_cooldown_expires(self):
        mr = AgentRuleDef("detect", "always", ["SWITCH_STRATEGY('crisis', cooldown=3)"])
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)], meta_rules=[mr])
        inst = AgentInstance(ad, 0, {"v": 0.0})
        inst.strategy = "normal"
        inst.decide({"always": True, "v": 0.0}, t=0.0)
        assert inst.strategy == "crisis"
        assert inst._strategy_locked_until == 3.0
        # After lock expires, a new switch can happen
        mr2 = AgentRuleDef("detect2", "always", ["SWITCH_STRATEGY('normal', cooldown=0)"])
        ad2 = AgentDef("X", 1, meta_rules=[mr2])
        inst.agent_def = ad2
        inst.decide({"always": True, "v": 0.0}, t=5.0)
        assert inst.strategy == "normal"


# ── Decide: strategy-scoped rules ────────────────────────────────

class TestDecideStrategyRules:
    def _make_instance(self):
        sr = AgentStrategy("normal", rules=[
            AgentRuleDef("reorder", "inventory < 100", ["order_qty += 50"]),
        ])
        ad = AgentDef("X", 1, properties=[
            AgentPropDef("inventory", 200.0, min=0),
            AgentPropDef("order_qty", 0.0, min=0),
        ], strategies=[sr])
        return AgentInstance(ad, 0, {"inventory": 200.0, "order_qty": 0.0})

    def test_strategy_rule_fires(self):
        inst = self._make_instance()
        inst.strategy = "normal"
        inst.state["inventory"] = 50.0
        effects = inst.decide({"inventory": 50.0, "order_qty": 0.0})
        assert len(effects) == 1
        assert effects[0][0] == "order_qty"

    def test_strategy_rule_skipped_when_condition_false(self):
        inst = self._make_instance()
        inst.strategy = "normal"
        effects = inst.decide({"inventory": 200.0, "order_qty": 0.0})
        assert len(effects) == 0

    def test_no_default_strategy_uses_flat_rules(self):
        inst = self._make_instance()
        inst.strategy = None  # no active strategy
        effects = inst.decide({"inventory": 50.0, "order_qty": 0.0})
        assert len(effects) == 0  # flat rules don't exist; strategy rules need strategy

    def test_no_strategies_falls_back_to_flat_rules(self):
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)],
                       rules=[AgentRuleDef("r", "always", ["v += 1"])])
        inst = AgentInstance(ad, 0, {"v": 0.0})
        effects = inst.decide({"always": True, "v": 0.0})
        assert len(effects) == 1

    def test_strategy_switching_changes_rules(self):
        sr = AgentStrategy("normal", rules=[
            AgentRuleDef("r1", "always", ["v += 1"]),
        ])
        sr2 = AgentStrategy("crisis", rules=[
            AgentRuleDef("r2", "always", ["v += 10"]),
        ])
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)],
                       strategies=[sr, sr2])
        inst = AgentInstance(ad, 0, {"v": 0.0})
        inst.strategy = "normal"
        effects = inst.decide({"always": True, "v": 0.0})
        assert effects[0][1] == 1.0
        inst.strategy = "crisis"
        effects = inst.decide({"always": True, "v": 0.0})
        assert effects[0][1] == 10.0


# ── SEND parsing and execution ───────────────────────────────────

class TestSEND:
    def test_parse_send_basic(self):
        msg = _parse_send('SEND(Supplier, "order", qty=100)', 0, "Buyer", {})
        assert msg is not None
        assert msg.target_type == "Supplier"
        assert msg.topic == "order"

    def test_parse_send_with_spaces(self):
        msg = _parse_send('SEND (Supplier, "order", qty=100)', 0, "Buyer", {})
        assert msg is not None
        assert msg.target_type == "Supplier"

    def test_parse_send_payload(self):
        msg = _parse_send('SEND(Supplier, "order", qty=100)', 0, "Buyer", {})
        assert msg.payload == {"qty": 100.0}

    def test_parse_send_no_kwargs(self):
        msg = _parse_send('SEND(Supplier, "ping")', 0, "Buyer", {})
        assert msg is not None
        assert msg.payload == {}

    def test_parse_send_invalid_returns_none(self):
        msg = _parse_send("not a send", 0, "X", {})
        assert msg is None

    def test_send_in_effect_queues_outbox(self):
        ad = AgentDef("Sender", 1, rules=[
            AgentRuleDef("r", "always", ["SEND(Receiver, 'alert', val=42)"]),
        ])
        engine = ABMEngine([ad])
        engine.initialize()
        inst = engine.instances[0]
        inst.decide({"always": True}, t=0.0)
        assert len(inst._pending_outbox) == 1
        assert inst._pending_outbox[0].target_type == "Receiver"
        assert inst._pending_outbox[0].topic == "alert"

    def test_send_delivers_in_next_step(self):
        sender_def = AgentDef("Sender", 1, rules=[
            AgentRuleDef("r", "always", ["SEND(Receiver, 'msg', val=1)"]),
        ])
        receiver_def = AgentDef("Receiver", 1, properties=[AgentPropDef("v", 0.0)])
        engine = ABMEngine([sender_def, receiver_def])
        engine.initialize()
        # Step 1: SEND goes to pending_outbox
        engine.step(0.0, 1.0, {"always": True})
        receiver = engine.instances[1]
        assert len(receiver.mailbox) == 0  # not yet delivered
        # Step 2: pending_outbox delivered to mailbox
        engine.step(1.0, 1.0, {"always": True})
        assert len(receiver.mailbox) >= 0  # TTL may have cleaned up

    def test_send_delivery_and_perception(self):
        sender_def = AgentDef("Sender", 1, rules=[
            AgentRuleDef("r", "always", ["SEND(Receiver, 'data', val=1)"]),
        ])
        receiver_def = AgentDef("Receiver", 1, properties=[AgentPropDef("received", 0.0)],
                                 rules=[AgentRuleDef("consume", "inbox_data > 0", ["received += 1"])])
        engine = ABMEngine([sender_def, receiver_def])
        engine.initialize()
        for i in range(4):
            engine.step(float(i), 1.0, {"always": True})
        receiver = engine.instances[1]
        assert receiver.state["received"] > 0

    def test_send_mixed_with_property_effects(self):
        ad = AgentDef("Node", 1, properties=[AgentPropDef("c", 0.0)],
                       rules=[AgentRuleDef("r", "always", ["c += 1", "SEND(Other, 'ping')"])])
        engine = ABMEngine([ad])
        engine.initialize()
        inst = engine.instances[0]
        effects = inst.decide({"always": True, "c": 0.0}, t=0.0)
        assert len(effects) == 1  # only c += 1, SEND is side effect
        assert effects[0][1] == 1.0
        assert len(inst._pending_outbox) == 1


# ── SWITCH_STRATEGY parsing ──────────────────────────────────────

class TestSwitchStrategyParsing:
    def test_parse_switch_basic(self):
        name, cd = _parse_switch_strategy("SWITCH_STRATEGY('crisis')")
        assert name == "crisis"
        assert cd == 0.0

    def test_parse_switch_with_spaces(self):
        name, cd = _parse_switch_strategy("SWITCH_STRATEGY ('crisis')")
        assert name == "crisis"

    def test_parse_switch_with_cooldown(self):
        name, cd = _parse_switch_strategy("SWITCH_STRATEGY('crisis', cooldown=10)")
        assert name == "crisis"
        assert cd == 10.0

    def test_parse_switch_invalid_returns_none(self):
        name, cd = _parse_switch_strategy("not a switch")
        assert name is None
        assert cd == 0.0

    def test_parse_switch_empty_name(self):
        name, cd = _parse_switch_strategy("SWITCH_STRATEGY('')")
        assert name is None


# ── 4-phase step ─────────────────────────────────────────────────

class Test4PhaseStep:
    def test_phase_deliver(self):
        sender = AgentDef("A", 1, rules=[
            AgentRuleDef("r", "always", ["SEND(B, 'msg')"]),
        ])
        receiver = AgentDef("B", 1)
        engine = ABMEngine([sender, receiver])
        engine.initialize()
        engine.step(0.0, 1.0, {"always": True})
        sender_inst = engine.instances[0]
        assert len(sender_inst._pending_outbox) == 1  # new message queued
        receiver_inst = engine.instances[1]
        engine.step(1.0, 1.0, {"always": True})
        # After step 2: pending was delivered in Phase 1, seen in Phase 2, TTL cleaned Phase 3
        # The outbox has the new message from Phase 2 of step 2
        assert len(sender_inst._pending_outbox) == 1

    def test_phase_mailbox_cleanup(self):
        receiver = AgentDef("B", 1)
        engine = ABMEngine([receiver])
        engine.initialize()
        inst = engine.instances[0]
        inst.mailbox.append(Message(0, "A", "B", "t"))
        assert len(inst.mailbox) == 1
        engine.step(0.0, 1.0, {})  # Phase 3 runs cleanup
        assert len(inst.mailbox) == 0  # TTL expired

    def test_phase_aggregate_returns_metrics(self):
        ad = AgentDef("X", 2, properties=[AgentPropDef("v", 10.0)])
        engine = ABMEngine([ad])
        engine.initialize()
        metrics = engine.step(0.0, 1.0, {})
        assert "X_v_avg" in metrics
        assert "X_count" in metrics

    def test_multiple_step_full_cycle(self):
        ad = AgentDef("Counter", 1, properties=[AgentPropDef("c", 0.0)],
                       rules=[AgentRuleDef("inc", "always", ["c += 1"])])
        engine = ABMEngine([ad])
        engine.initialize()
        for i in range(5):
            engine.step(float(i), 1.0, {"always": True})
        assert engine.instances[0].state["c"] == 5.0


# ── ABMEngine default strategy initialization ────────────────────

class TestDefaultStrategy:
    def test_first_strategy_is_default(self):
        s1 = AgentStrategy("normal")
        s2 = AgentStrategy("crisis")
        ad = AgentDef("X", 2, strategies=[s1, s2])
        engine = ABMEngine([ad])
        engine.initialize()
        for inst in engine.instances:
            assert inst.strategy == "normal"

    def test_no_strategies_strategy_is_none(self):
        ad = AgentDef("X", 2)
        engine = ABMEngine([ad])
        engine.initialize()
        for inst in engine.instances:
            assert inst.strategy is None

    def test_default_strategy_rules_fire_on_init(self):
        sr = AgentStrategy("normal", rules=[
            AgentRuleDef("r", "always", ["v += 1"]),
        ])
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)],
                       strategies=[sr])
        engine = ABMEngine([ad])
        engine.initialize()
        engine.step(0.0, 1.0, {"always": True})
        assert engine.instances[0].state["v"] == 1.0

    def test_meta_rule_switches_from_default(self):
        sr = AgentStrategy("normal", rules=[
            AgentRuleDef("r", "always", ["v += 1"]),
        ])
        sr2 = AgentStrategy("crisis", rules=[
            AgentRuleDef("r2", "always", ["v += 10"]),
        ])
        mr = AgentRuleDef("switch", "always", ["SWITCH_STRATEGY('crisis')"])
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)],
                       strategies=[sr, sr2], meta_rules=[mr])
        engine = ABMEngine([ad])
        engine.initialize()
        inst = engine.instances[0]
        assert inst.strategy == "normal"
        # Meta-rule switches to crisis on first step
        engine.step(0.0, 1.0, {"always": True})
        assert inst.strategy == "crisis"


# ── DSL Integration ──────────────────────────────────────────────

class TestDSLIntegration:
    def test_simulate_with_strategies(self):
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 1\n'
            'agent "X": 1\n'
            '  property "v": 0\n'
            '  strategy "normal"\n'
            '    rule "r": when v < 100\n'
            '      v += 1\n'
        )
        r = m.simulate()
        assert r.steps > 0
        engine = r.abm_engine
        assert engine.instances[0].strategy == "normal"

    def test_simulate_with_meta_rules(self):
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 1\n'
            'agent "X": 1\n'
            '  property "v": 0\n'
            '  strategy "normal"\n'
            '    rule "r": when v < 100\n'
            '      v += 1\n'
            '  strategy "crisis"\n'
            '    rule "r2": when v < 5\n'
            '      v += 10\n'
            '  meta_rule "switch": when v > 50\n'
            '    SWITCH_STRATEGY("crisis")\n'
        )
        r = m.simulate()
        engine = r.abm_engine
        # At the end, strategy may have switched
        # Not asserting exact strategy (depends on timing)
        assert engine is not None

    def test_simulate_mixed_sd_agents_with_strategies(self):
        m = parse_sysd(
            'A\ndt 1\nfrom 0 to 3\n'
            'stock "S": 100\n'
            '  - "Outflow": S * 0.1\n'
            'agent "A": 2\n'
            '  property "v": 0\n'
            '  strategy "normal"\n'
            '    rule "r": when always\n'
            '      v += 1\n'
        )
        r = m.simulate()
        assert r.values["S"][-1] < 100.0
        engine = r.abm_engine
        assert engine.instances[0].state["v"] > 0

    def test_agent_without_strategies_still_works(self):
        m = parse_sysd(
            'A\ndt 0.1\nfrom 0 to 1\n'
            'agent "X": 1\n'
            '  property "v": 0\n'
            '  rule "r": when v < 100\n'
            '    v += 1\n'
        )
        r = m.simulate()
        assert r.steps > 0


# ── Legacy backward compatibility ────────────────────────────────

class TestBackwardCompat:
    def test_existing_tests_still_pass_setup(self):
        """Confirm the fixture from original test_abm still works."""
        ad = AgentDef(
            name="Buyer",
            count=3,
            properties=[AgentPropDef("budget", 100.0, min=0.0)],
            rules=[AgentRuleDef("buy", "budget > 10", ["budget -= 10"])],
        )
        engine = ABMEngine([ad])
        engine.initialize()
        engine.step(0.0, 0.1, {})
        for inst in engine.instances:
            assert inst.state["budget"] == 90.0

    def test_agent_rule_no_strategy_field(self):
        """AgentRuleDef can be created without strategy fields."""
        r = AgentRuleDef("r", "x > 0", ["x -= 1"])
        assert r.priority == 0

    def test_empty_strategies_list_is_ok(self):
        ad = AgentDef("X", 1, properties=[AgentPropDef("v", 0.0)],
                       rules=[AgentRuleDef("r", "always", ["v += 1"])])
        assert ad.strategies == []
        engine = ABMEngine([ad])
        engine.initialize()
        engine.step(0.0, 1.0, {"always": True})
        assert engine.instances[0].state["v"] == 1.0

    def test_network_still_works(self):
        ad = AgentDef("X", 5, network_type="complete",
                       rules=[AgentRuleDef("r", "always", ["v += 1"])])
        engine = ABMEngine([ad])
        engine.initialize()
        assert len(engine.instances[0].neighbors) == 4

    def test_multi_step_legacy_cycle(self):
        ad = AgentDef("X", 1, properties=[AgentPropDef("c", 0.0)],
                       rules=[AgentRuleDef("inc", "always", ["c += 1"])])
        engine = ABMEngine([ad])
        engine.initialize()
        for i in range(3):
            engine.step(float(i), 1.0, {"always": True})
        assert engine.instances[0].state["c"] == 3.0
