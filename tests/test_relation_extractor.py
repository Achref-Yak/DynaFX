from uuid import UUID, uuid4

import pytest

from cognitive_engine.core.models import Entity, Graph, Span, WorldRelation
from cognitive_engine.extract.relations import (
    _PHRASE_MAP,
    _VERB_LEMMA_MAP,
    _build_relation_phrase,
    _classify_relation,
    _expand_to_noun_chunk,
    _extract_triple,
    extract_relations,
)


@pytest.fixture(scope="module")
def nlp():
    from cognitive_engine.nlp.preprocessor import load_spacy_pipeline
    return load_spacy_pipeline()


def test_expand_to_noun_chunk_finds_full_phrase(nlp):
    doc = nlp("The red car crashed into the wall.")
    car = None
    for token in doc:
        if token.text == "car":
            car = token
            break
    assert car is not None
    chunk = _expand_to_noun_chunk(doc, car)
    assert chunk is not None
    assert chunk.text == "The red car"


def test_expand_to_noun_chunk_returns_none_for_non_noun(nlp):
    doc = nlp("The system crashed.")
    for token in doc:
        if token.pos_ == "VERB":
            chunk = _expand_to_noun_chunk(doc, token)
            assert chunk is None


def test_build_relation_phrase_basic(nlp):
    doc = nlp("The system contains data.")
    verb = None
    for token in doc:
        if token.text == "contains":
            verb = token
            break
    assert verb is not None
    phrase = _build_relation_phrase(verb)
    assert phrase == "contain"


def test_build_relation_phrase_with_auxiliary(nlp):
    doc = nlp("The system must not prevent access.")
    verb = None
    for token in doc:
        if token.text == "prevent":
            verb = token
            break
    assert verb is not None
    phrase = _build_relation_phrase(verb)
    assert "must" in phrase
    assert "not" in phrase
    assert "prevent" in phrase


def test_build_relation_phrase_with_particle(nlp):
    doc = nlp("The system sets up a connection.")
    verb = None
    for token in doc:
        if token.text == "sets":
            verb = token
            break
    assert verb is not None
    phrase = _build_relation_phrase(verb)
    assert phrase == "set up"


def test_build_relation_phrase_with_preposition(nlp):
    doc = nlp("The report depends on the data.")
    verb = None
    for token in doc:
        if token.text == "depends":
            verb = token
            break
    assert verb is not None
    phrase = _build_relation_phrase(verb)
    assert phrase == "depend on"


def test_extract_triple_basic(nlp):
    doc = nlp("The expert holds a PhD")
    verb = None
    for token in doc:
        if token.text == "holds":
            verb = token
            break
    assert verb is not None
    triple = _extract_triple(doc, verb)
    assert triple is not None
    subj_text, obj_text, rel_text, *_ = triple
    assert subj_text == "The expert"
    assert obj_text == "a PhD"
    assert rel_text == "hold"


def test_extract_triple_no_subject(nlp):
    doc = nlp("Run the test.")
    verb = None
    for token in doc:
        if token.text == "Run":
            verb = token
            break
    assert verb is not None
    triple = _extract_triple(doc, verb)
    assert triple is None


def test_extract_triple_no_object(nlp):
    doc = nlp("The system runs.")
    verb = None
    for token in doc:
        if token.text == "runs":
            verb = token
            break
    assert verb is not None
    triple = _extract_triple(doc, verb)
    assert triple is None


def test_extract_triple_with_negation(nlp):
    doc = nlp("The company does not require approval.")
    verb = None
    for token in doc:
        if token.text == "require":
            verb = token
            break
    assert verb is not None
    triple = _extract_triple(doc, verb)
    assert triple is not None
    subj_text, obj_text, rel_text, *_ = triple
    assert subj_text == "The company"
    assert obj_text == "approval"
    assert "not" in rel_text
    assert "require" in rel_text


def test_classify_phrase_match(nlp):
    assert _classify_relation("consists of", "consist") == "CONSISTS_OF"
    assert _classify_relation("depends on", "depend") == "DEPENDS_ON"
    assert _classify_relation("leads to", "lead") == "CAUSES"


def test_classify_verb_lemma_fallback(nlp):
    assert _classify_relation("authenticate", "authenticate") == "AUTHENTICATES"
    assert _classify_relation("configure", "configure") == "CONFIGURES"
    assert _classify_relation("encrypt", "encrypt") == "ENCRYPTS"


def test_classify_related_to_fallback(nlp):
    assert _classify_relation("walks quickly", "walk") == "RELATED_TO"
    assert _classify_relation("eats food", "eat") == "RELATED_TO"


def test_extract_relations_adds_world_relations(nlp):
    doc = nlp("The system contains data")
    doc.user_data["chunk_start_char"] = 0
    doc.user_data["chunk_end_char"] = len(doc.text)

    subj_entity = Entity(
        id=uuid4(), kind="System", name="The system",
        spans=[Span(start=0, end=10, text="The system")],
    )
    obj_entity = Entity(
        id=uuid4(), kind="Data", name="data",
        spans=[Span(start=20, end=24, text="data")],
    )
    graph = Graph(entities={subj_entity.id: subj_entity, obj_entity.id: obj_entity})

    extract_relations(graph, [doc])

    assert len(graph.world_relations) == 1
    wr = graph.world_relations[0]
    assert wr.source_id == subj_entity.id
    assert wr.target_id == obj_entity.id
    assert wr.kind == "CONSISTS_OF"
    assert wr.metadata["relation_phrase"] == "contain"


def test_extract_relations_deduplicates(nlp):
    doc = nlp("The system contains data")
    doc.user_data["chunk_start_char"] = 0
    doc.user_data["chunk_end_char"] = len(doc.text)

    subj_entity = Entity(
        id=uuid4(), kind="System", name="The system",
        spans=[Span(start=0, end=10, text="The system")],
    )
    obj_entity = Entity(
        id=uuid4(), kind="Data", name="data",
        spans=[Span(start=20, end=24, text="data")],
    )
    graph = Graph(entities={subj_entity.id: subj_entity, obj_entity.id: obj_entity})
    graph.world_relations.append(WorldRelation(
        id=uuid4(),
        source_id=subj_entity.id,
        target_id=obj_entity.id,
        kind="CONSISTS_OF",
    ))

    extract_relations(graph, [doc])

    assert len(graph.world_relations) == 2


def test_extract_relations_skips_verbs_without_entities(nlp):
    doc = nlp("The expert holds a PhD")
    doc.user_data["chunk_start_char"] = 0
    doc.user_data["chunk_end_char"] = len(doc.text)

    graph = Graph()

    extract_relations(graph, [doc])
    assert len(graph.world_relations) == 0


def test_extract_relations_uses_classify_relation(nlp):
    doc = nlp("The system depends on data")
    doc.user_data["chunk_start_char"] = 0
    doc.user_data["chunk_end_char"] = len(doc.text)

    subj_entity = Entity(
        id=uuid4(), kind="System", name="The system",
        spans=[Span(start=0, end=10, text="The system")],
    )
    obj_entity = Entity(
        id=uuid4(), kind="Data", name="data",
        spans=[Span(start=25, end=29, text="data")],
    )
    graph = Graph(entities={subj_entity.id: subj_entity, obj_entity.id: obj_entity})

    extract_relations(graph, [doc])

    assert len(graph.world_relations) == 1
    assert graph.world_relations[0].kind == "DEPENDS_ON"


def test_extract_relations_multiple_verbs(nlp):
    doc = nlp("The system contains data and encrypts traffic")
    doc.user_data["chunk_start_char"] = 0
    doc.user_data["chunk_end_char"] = len(doc.text)

    system_eid = uuid4()
    system_entity = Entity(
        id=system_eid, kind="System", name="The system",
        spans=[Span(start=0, end=10, text="The system")],
    )
    data_entity = Entity(
        id=uuid4(), kind="Data", name="data",
        spans=[Span(start=20, end=24, text="data")],
    )
    traffic_entity = Entity(
        id=uuid4(), kind="Traffic", name="traffic",
        spans=[Span(start=38, end=45, text="traffic")],
    )
    graph = Graph(entities={
        system_eid: system_entity,
        data_entity.id: data_entity,
        traffic_entity.id: traffic_entity,
    })

    extract_relations(graph, [doc])

    assert len(graph.world_relations) == 2


def test_phrase_map_consistency():
    keys = set(_PHRASE_MAP.values())
    verb_keys = set(_VERB_LEMMA_MAP.values())
    extras = {"TRADE_OFF", "SCALES", "SATISFIES", "PRIORITIZES",
              "RECOMMENDS", "DEMONSTRATES", "NEGLECTS",
              "INTRODUCES", "MAINTAINS", "DEPLOYS",
              "CONFIGURES", "AUTHENTICATES", "EXPOSES",
              "ROUTES_TO", "SPECIFIES", "PROVIDES"}
    assert keys.issubset(verb_keys | extras)


def test_no_verb_lemma_collision():
    assert len(_VERB_LEMMA_MAP) == len(set(_VERB_LEMMA_MAP.keys()))
