import pytest
from cognitive_engine.nlp.chunker import PropSpan
from cognitive_engine.core.models import NodeType, BfoCategory
from cognitive_engine.extract.types import assign_type, map_types, Relation, _char_span_relaxed

pytest.importorskip("spacy")
from cognitive_engine.nlp.preprocessor import load_spacy_pipeline

nlp = load_spacy_pipeline()


def _make_doc(text: str, chunk_start: int = 0) -> "spacy.tokens.Doc":
    doc = nlp(text)
    doc.user_data["chunk_start_char"] = chunk_start
    doc.user_data["chunk_end_char"] = chunk_start + len(text)
    return doc


def _make_span(start: int, end: int, text: str) -> PropSpan:
    return PropSpan(start_char=start, end_char=end, text=text, chunk_offsets=[0])


class TestAssignType:
    def test_axiom_with_modal(self):
        text = "The system must handle 10k requests."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.AXIOM

    def test_axiom_with_shall(self):
        text = "The system shall process all data."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.AXIOM

    def test_condition_with_if(self):
        text = "If load exceeds 10k"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CONDITION

    def test_condition_with_unless(self):
        text = "Unless we scale"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CONDITION

    def test_condition_requires_verb(self):
        text = "If it is okay with you"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CONDITION

    def test_justification_because(self):
        text = "because the query timed out"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.JUSTIFICATION

    def test_justification_since(self):
        text = "since traffic increased"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.JUSTIFICATION

    def test_fallacy_by_keyword(self):
        text = "That argument is misleading."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.FALLACY

    def test_fallacy_flawed(self):
        text = "This reasoning is flawed."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.FALLACY

    def test_counterclaim_adversative_however(self):
        text = "However, the latency increased."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.COUNTERCLAIM

    def test_counterclaim_adversative_but(self):
        text = "The system works but latency is high."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.COUNTERCLAIM

    def test_counterclaim_requires_lexical_marker(self):
        doc = _make_doc("Some claim.")
        span = _make_span(0, 10, "Some claim")
        attack_source = _make_span(20, 30, "attack text")
        rel = Relation(source_span=attack_source, target_span=span, label="Attack")
        assert assign_type(span, doc, relations=[rel]) == NodeType.CLAIM

    def test_claim_root_proposition(self):
        text = "The system provides a robust framework."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CLAIM

    def test_evidence_non_root(self):
        text = "The system scales because traffic is high."
        doc = _make_doc(text)
        claim_span = _make_span(0, 17, "The system scales")
        assert assign_type(claim_span, doc) == NodeType.CLAIM

        ev_span = _make_span(18, len(text), "because traffic is high.")
        ev_type = assign_type(ev_span, doc)
        assert ev_type in (NodeType.EVIDENCE, NodeType.JUSTIFICATION)

    def test_no_modal_no_axiom(self):
        text = "The system handles requests."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) != NodeType.AXIOM

    def test_empty_span_defaults_to_observation(self):
        doc = _make_doc("")
        span = _make_span(0, 0, "")
        assert assign_type(span, doc) == NodeType.OBSERVATION

    def test_char_span_out_of_bounds(self):
        doc = _make_doc("Short text.")
        span = _make_span(100, 200, "out of bounds")
        assert assign_type(span, doc) == NodeType.OBSERVATION


class TestMapTypes:
    def test_map_types_basic(self):
        text = "The system must handle 10k requests."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        results = map_types([span], [doc])
        assert len(results) == 1
        assert results[0][1] == NodeType.AXIOM

    def test_map_types_multiple_spans(self):
        doc = _make_doc("If load exceeds 10k, scale out.")
        if_span = _make_span(0, 21, "If load exceeds 10k")
        scale_span = _make_span(23, 33, "scale out")
        results = map_types([if_span, scale_span], [doc])
        types = {r[1] for r in results}
        assert NodeType.CONDITION in types

    def test_map_types_no_matching_doc(self):
        doc = _make_doc("Some text.")
        span = _make_span(999, 1005, "nowhere")
        results = map_types([span], [doc])
        assert results[0][1] == NodeType.EVIDENCE
        assert len(results[0]) == 3

    def test_map_types_empty_spans(self):
        doc = _make_doc("Some text.")
        results = map_types([], [doc])
        assert results == []

    def test_map_types_with_relations(self):
        text_a = "First claim."
        text_b = "But it fails."
        doc_a = _make_doc(text_a, chunk_start=0)
        doc_b = _make_doc(text_b, chunk_start=20)
        span_a = _make_span(0, 12, text_a)
        span_b = _make_span(20, 33, text_b)
        rel = Relation(source_span=span_b, target_span=span_a, label="Attack")
        results = map_types([span_a, span_b], [doc_a, doc_b], relations=[rel])
        types_by_span = {id(s): (t, c) for s, t, c in results}
        assert types_by_span[id(span_a)][0] == NodeType.CLAIM


class TestTypeRuleArchitecture:
    def test_rules_are_sorted_by_priority(self):
        from cognitive_engine.extract.types import TYPE_RULES
        priorities = [r.priority for r in TYPE_RULES]
        assert priorities == sorted(priorities, reverse=True)

    def test_all_rules_have_unique_priorities(self):
        from cognitive_engine.extract.types import TYPE_RULES
        priorities = [r.priority for r in TYPE_RULES]
        assert len(priorities) == len(set(priorities))

    def test_all_rules_have_matcher_callable(self):
        from cognitive_engine.extract.types import TYPE_RULES
        for rule in TYPE_RULES:
            assert callable(rule.matcher)

    def test_agent_rule_priority高于_process(self):
        from cognitive_engine.extract.types import TYPE_RULES
        by_name = {r.name: r for r in TYPE_RULES}
        assert by_name["agent"].priority > by_name["process"].priority

    def test_world_model_rules高于_argumentation(self):
        from cognitive_engine.extract.types import TYPE_RULES
        by_name = {r.name: r for r in TYPE_RULES}
        # Argumentation keyword rules have highest priority (most specific patterns)
        # Frame-based world-model rules follow (semantic precision)
        # Keyword world-model fallback rules have lowest priority
        assert by_name["agent"].priority > by_name["process"].priority
        assert by_name["condition"].priority > by_name["state"].priority


class TestClassificationContext:
    def test_context_is_frozen(self):
        from cognitive_engine.extract.types import ClassificationContext, DependencyFeatures
        deps = DependencyFeatures(verbs=(), modals=(), mark_relations=())
        ctx = ClassificationContext(span=_make_span(0, 5, "hello"), doc=None, text_lower="hello", deps=deps)
        with pytest.raises(AttributeError):
            ctx.text_lower = "changed"

    def test_dependency_features_is_frozen(self):
        from cognitive_engine.extract.types import DependencyFeatures
        deps = DependencyFeatures(verbs=(), modals=(), mark_relations=())
        with pytest.raises(AttributeError):
            deps.verbs = (1,)


class TestDeclarativeBfo:
    def test_nodetype_direct_map(self):
        from cognitive_engine.extract.types import assign_bfo
        assert assign_bfo(NodeType.EVENT) == BfoCategory.PROCESS
        assert assign_bfo(NodeType.ACTION) == BfoCategory.PROCESS

    def test_concept_direct_map(self):
        from cognitive_engine.extract.types import assign_bfo
        assert assign_bfo(NodeType.CLAIM, "TEMPERATURE") == BfoCategory.QUALITY
        assert assign_bfo(NodeType.CLAIM, "CONDITION") == BfoCategory.REALIZABLE_ENTITY
        assert assign_bfo(NodeType.CLAIM, "PRECEDENT") == BfoCategory.REALIZABLE_ENTITY

    def test_ner_override(self):
        from cognitive_engine.extract.types import assign_bfo
        assert assign_bfo(NodeType.ENTITY, "", entity_kind="PERSON") == BfoCategory.QUALITY
        assert assign_bfo(NodeType.ENTITY, "", entity_kind="ORG") == BfoCategory.IMMATERIAL_ENTITY
        assert assign_bfo(NodeType.ENTITY, "", entity_kind="GPE") == BfoCategory.MATERIAL_ENTITY

    def test_location_special_case(self):
        from cognitive_engine.extract.types import assign_bfo
        assert assign_bfo(NodeType.ENTITY, "LOCATION") == BfoCategory.MATERIAL_ENTITY
        assert assign_bfo(NodeType.CLAIM, "LOCATION") == BfoCategory.IMMATERIAL_ENTITY


class TestConceptModules:
    def test_identity_person_name(self):
        from cognitive_engine.extract.concepts.identity import match
        assert match("my name is alice", NodeType.CLAIM) == "PERSON_NAME"

    def test_identity_returns_none_on_no_match(self):
        from cognitive_engine.extract.concepts.identity import match
        assert match("the weather is nice", NodeType.CLAIM) is None

    def test_measurement_temperature(self):
        from cognitive_engine.extract.concepts.measurement import match
        assert match("the temperature is 30°c", NodeType.OBSERVATION) == "TEMPERATURE"

    def test_measurement_budget(self):
        from cognitive_engine.extract.concepts.measurement import match
        assert match("the cost is $500", NodeType.OBSERVATION) == "BUDGET"

    def test_measurement_date(self):
        from cognitive_engine.extract.concepts.measurement import match
        assert match("deadline is january 15", NodeType.OBSERVATION) == "DATE"

    def test_measurement_returns_none(self):
        from cognitive_engine.extract.concepts.measurement import match
        assert match("hello world", NodeType.OBSERVATION) is None

    def test_preference_match(self):
        from cognitive_engine.extract.concepts.preference import match
        assert match("i prefer dark mode", NodeType.OBSERVATION) == "PREFERENCE"

    def test_preference_style(self):
        from cognitive_engine.extract.concepts.preference import match
        assert match("dark mode looks better", NodeType.OBSERVATION) == "STYLE"

    def test_preference_location(self):
        from cognitive_engine.extract.concepts.preference import match
        assert match("i live in berlin", NodeType.OBSERVATION) == "LOCATION"

    def test_preference_returns_none(self):
        from cognitive_engine.extract.concepts.preference import match
        assert match("the system works", NodeType.OBSERVATION) is None

    def test_reasoning_hypothesis(self):
        from cognitive_engine.extract.concepts.reasoning import match
        assert match("we hypothesize x", NodeType.OBSERVATION) == "HYPOTHESIS"

    def test_reasoning_decision(self):
        from cognitive_engine.extract.concepts.reasoning import match
        assert match("we decided to proceed", NodeType.CLAIM) == "DECISION"

    def test_reasoning_observation(self):
        from cognitive_engine.extract.concepts.reasoning import match
        assert match("we observed a pattern", NodeType.OBSERVATION) == "OBSERVATION"

    def test_reasoning_returns_none(self):
        from cognitive_engine.extract.concepts.reasoning import match
        assert match("hello world", NodeType.OBSERVATION) is None
