"""Tests for FrameNet, VerbNet, and SRL integration."""

import pytest
from cognitive_engine.nlp.semantic_resources import SemanticResources


class TestSemanticResources:
    def test_singleton(self):
        SemanticResources.reset()
        res1 = SemanticResources.instance()
        res2 = SemanticResources.instance()
        assert res1 is res2

    def test_reset_creates_new_instance(self):
        res1 = SemanticResources.instance()
        SemanticResources.reset()
        res2 = SemanticResources.instance()
        assert res1 is not res2

    def test_framenet_loaded(self):
        res = SemanticResources.instance()
        frames = res.frames_for_lemma("fry")
        assert len(frames) > 0
        assert "Apply_heat" in frames or "Cooking" in frames

    def test_framenet_unknown_lemma(self):
        res = SemanticResources.instance()
        frames = res.frames_for_lemma("xyzzy_unknown_lemma")
        assert frames == []

    def test_frame_elements(self):
        res = SemanticResources.instance()
        fes = res.frame_elements("Apply_heat")
        assert len(fes) > 0
        assert "Heater" in fes or "Food" in fes

    def test_frame_object(self):
        res = SemanticResources.instance()
        obj = res.frame_object("Apply_heat")
        assert obj is not None
        assert obj.name == "Apply_heat"

    def test_has_frame(self):
        res = SemanticResources.instance()
        assert res.has_frame("Apply_heat")
        assert not res.has_frame("NonexistentFrameXYZ")

    def test_verbnet_loaded(self):
        res = SemanticResources.instance()
        classes = res.vn_classes_for_lemma("give")
        assert len(classes) > 0
        assert any("give" in c for c in classes)

    def test_verbnet_unknown_lemma(self):
        res = SemanticResources.instance()
        classes = res.vn_classes_for_lemma("xyzzy_unknown_lemma")
        assert classes == []

    def test_verbnet_themroles(self):
        res = SemanticResources.instance()
        classes = res.vn_classes_for_lemma("give")
        if classes:
            # Check any class has thematic roles
            all_roles = []
            for cid in classes[:3]:
                roles = res.vn_themroles(cid)
                all_roles.extend(r.get("type", "") for r in roles)
            assert len(all_roles) > 0

    def test_verbnet_all_roles(self):
        res = SemanticResources.instance()
        roles = res.vn_all_roles("give")
        assert len(roles) > 0

    def test_propbank_loaded(self):
        res = SemanticResources.instance()
        # Use a common roleset that's guaranteed to exist
        rs = res.pb_roleset("go.01")
        assert rs is not None

    def test_propbank_unknown(self):
        res = SemanticResources.instance()
        rs = res.pb_roleset("xyzzy_nonexistent.01")
        assert rs is None


class TestFrameRules:
    def test_classify_by_frame_agent(self):
        from cognitive_engine.extract.frame_rules import classify_by_frame, FRAMENodeType_MAP
        from cognitive_engine.core.models import NodeType
        assert FRAMENodeType_MAP.get("Personal_info") == NodeType.AGENT
        assert FRAMENodeType_MAP.get("Being_named") == NodeType.AGENT

    def test_classify_by_frame_process(self):
        from cognitive_engine.extract.frame_rules import FRAMENodeType_MAP
        from cognitive_engine.core.models import NodeType
        assert FRAMENodeType_MAP.get("Cause_change") == NodeType.PROCESS
        assert FRAMENodeType_MAP.get("Cooking") == NodeType.PROCESS

    def test_classify_by_frame_state(self):
        from cognitive_engine.extract.frame_rules import FRAMENodeType_MAP
        from cognitive_engine.core.models import NodeType
        assert FRAMENodeType_MAP.get("Being_located") == NodeType.STATE

    def test_frame_priority_world_model(self):
        from cognitive_engine.extract.frame_rules import get_frame_priority
        assert get_frame_priority("Personal_info") == 100
        assert get_frame_priority("Cause_change") == 100

    def test_frame_priority_argumentation(self):
        from cognitive_engine.extract.frame_rules import get_frame_priority
        assert get_frame_priority("Statement") == 80
        assert get_frame_priority("Concession") == 80

    def test_frame_priority_unknown(self):
        from cognitive_engine.extract.frame_rules import get_frame_priority
        assert get_frame_priority("NonexistentFrame") == 0


class TestVerbNetRoles:
    def test_vn_concept_for_lemma(self):
        from cognitive_engine.extract.verbnet_roles import vn_concept_for_lemma
        assert vn_concept_for_lemma("give") == "TRANSFER"
        assert vn_concept_for_lemma("say") == "COMMUNICATION"
        assert vn_concept_for_lemma("believe") == "BELIEF"

    def test_vn_concept_unknown(self):
        from cognitive_engine.extract.verbnet_roles import vn_concept_for_lemma
        assert vn_concept_for_lemma("xyzzy_unknown") is None

    def test_vn_edgetype_for_lemma(self):
        from cognitive_engine.extract.verbnet_roles import vn_edgetype_for_lemma
        assert vn_edgetype_for_lemma("cause") == "CAUSES"
        assert vn_edgetype_for_lemma("depend") == "DEPENDS"

    def test_vn_edgetype_unknown(self):
        from cognitive_engine.extract.verbnet_roles import vn_edgetype_for_lemma
        assert vn_edgetype_for_lemma("xyzzy_unknown") is None

    def test_vn_roles_for_span(self):
        from cognitive_engine.extract.verbnet_roles import vn_roles_for_span
        roles = vn_roles_for_span("give")
        assert len(roles) > 0

    def test_classify_relation_by_vn(self):
        from cognitive_engine.extract.verbnet_roles import classify_relation_by_vn
        from cognitive_engine.core.models import NodeType
        result = classify_relation_by_vn("cause", NodeType.AGENT, NodeType.PROCESS)
        assert result == "CAUSES"

    def test_classify_relation_by_vn_depend(self):
        from cognitive_engine.extract.verbnet_roles import classify_relation_by_vn
        from cognitive_engine.core.models import NodeType
        result = classify_relation_by_vn("depend", NodeType.PROCESS, NodeType.RESOURCE)
        assert result == "DEPENDS"

    def test_classify_relation_by_vn_unknown(self):
        from cognitive_engine.extract.verbnet_roles import classify_relation_by_vn
        from cognitive_engine.core.models import NodeType
        result = classify_relation_by_vn("xyzzy_unknown", NodeType.PROCESS, NodeType.PROCESS)
        assert result is None


class TestSRLRelations:
    def test_predict_srl_lightweight(self):
        pytest.importorskip("spacy")
        from cognitive_engine.nlp.preprocessor import load_spacy_pipeline
        from cognitive_engine.extract.srl_relations import predict_srl_lightweight

        nlp = load_spacy_pipeline()
        doc = nlp("The cat sat on the mat.")
        doc.user_data["chunk_start_char"] = 0
        frames = predict_srl_lightweight(doc)
        assert len(frames) > 0
        assert frames[0].verb_lemma == "sit"

    def test_srl_frame_has_arguments(self):
        pytest.importorskip("spacy")
        from cognitive_engine.nlp.preprocessor import load_spacy_pipeline
        from cognitive_engine.extract.srl_relations import predict_srl_lightweight

        nlp = load_spacy_pipeline()
        doc = nlp("Alice gave Bob a book.")
        doc.user_data["chunk_start_char"] = 0
        frames = predict_srl_lightweight(doc)
        assert len(frames) > 0
        arg_roles = {a.role for a in frames[0].arguments}
        assert "ARG0" in arg_roles  # Alice
        assert "ARG1" in arg_roles  # book

    def test_srl_arguments_expand_noun_chunks(self):
        pytest.importorskip("spacy")
        from cognitive_engine.nlp.preprocessor import load_spacy_pipeline
        from cognitive_engine.extract.srl_relations import predict_srl_lightweight

        nlp = load_spacy_pipeline()
        doc = nlp("The big red cat sat on the mat.")
        doc.user_data["chunk_start_char"] = 0
        frames = predict_srl_lightweight(doc)
        assert len(frames) > 0
        # ARG0 should be "The big red cat" (expanded noun chunk)
        arg0 = next((a for a in frames[0].arguments if a.role == "ARG0"), None)
        assert arg0 is not None
        assert "cat" in arg0.text


class TestFrameNetTypeRules:
    """Test that FrameNet rules fire correctly in assign_type."""

    def test_frame_rule_priority高于_keyword(self):
        from cognitive_engine.extract.types import TYPE_RULES
        by_name = {r.name: r for r in TYPE_RULES}
        # Agent keyword should have higher priority than process frame
        assert by_name["agent"].priority > by_name["process_frame"].priority

    def test_argumentation_keyword_highest_priority(self):
        from cognitive_engine.extract.types import TYPE_RULES
        by_name = {r.name: r for r in TYPE_RULES}
        # Argumentation keyword rules should be above all frame rules
        assert by_name["condition"].priority > by_name["agent_frame"].priority
        assert by_name["axiom"].priority > by_name["agent_frame"].priority
