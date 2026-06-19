from __future__ import annotations

import logging

from cognitive_engine.core.models import Graph
from cognitive_engine.core.embeddings import EmbeddingModel
from cognitive_engine.domain import domain

logger = logging.getLogger(__name__)

def resolve_entities(graph: Graph) -> None:
    """Soft wordlist / Entity Linking step.
    
    Uses Embedding Clusters to map raw extracted entity names to 
    canonical concepts from the domain ontology.
    """
    cfg = domain.active()
    concepts = list(cfg.canonical_concepts.keys())
    if not concepts:
        return
        
    model = EmbeddingModel.get_instance()
    
    # In a production system, these canonical concept embeddings 
    # would be cached rather than computed on every pass.
    # For now, we compute them or rely on internal sentence-transformers caching.
    concept_embeddings = model.encode_batch(concepts)
    
    entities = list(graph.entities.values())
    if not entities:
        return
        
    # We only want to resolve entities that don't already have a strong NER label like PERSON, GPE, etc.
    # We'll re-link those that have a fallback kind like "Entity" or ones that are just lowercased nouns.
    # Actually, we can link anything, but let's be careful. Let's just try linking them all if they match highly.
    entity_texts = [e.name for e in entities]
    entity_embeddings = model.encode_batch(entity_texts)
    
    linked_count = 0
    for entity, e_vec in zip(entities, entity_embeddings):
        best_score = 0.0
        best_concept = None
        
        for concept, c_vec in zip(concepts, concept_embeddings):
            score = model.cosine_similarity(e_vec, c_vec)
            if score > best_score:
                best_score = score
                best_concept = concept
                
        if best_concept and best_score >= cfg.entity_linking_threshold:
            mapped_kind = cfg.canonical_concepts[best_concept]
            logger.debug(
                "Linked entity '%s' to canonical concept '%s' (score %.2f)",
                entity.name, mapped_kind, best_score
            )
            entity.kind = mapped_kind
            # We can also store the mapping in metadata
            entity.metadata["linked_concept"] = mapped_kind
            entity.metadata["linking_score"] = best_score
            linked_count += 1

    if linked_count > 0:
        logger.info("Linked %d entities to canonical concepts.", linked_count)
