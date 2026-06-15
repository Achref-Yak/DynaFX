"""Schemas — domain-specific configurations."""

from cognitive_engine.schemas.legal import LEGAL_SCHEMA
from cognitive_engine.schemas.research import RESEARCH_SCHEMA
from cognitive_engine.schemas.debate import DEBATE_SCHEMA

SCHEMA_REGISTRY = {
    "legal": LEGAL_SCHEMA,
    "research": RESEARCH_SCHEMA,
    "debate": DEBATE_SCHEMA,
}


def get_schema(name: str):
    """Get a schema by name."""
    schema = SCHEMA_REGISTRY.get(name)
    if schema is None:
        raise ValueError(f"Unknown schema: {name}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return schema
