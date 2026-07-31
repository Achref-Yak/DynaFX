"""TBox loader — loads domain type hierarchies and axioms.

A TBox (terminological box) defines the type hierarchy, valid
relation types, and inference axioms for a domain. It is the
OWL2-style schema that the ABox assertions must conform to.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# Category level constants (moved from deleted core/math.py)
CATEGORY_LEVELS: dict[str, int] = {
    "NECESSITY": 1, "AXIOM": 1,
    "FACT": 2, "EVIDENCE": 2, "OBSERVATION": 2, "DOCUMENT": 2,
    "BELIEF": 3, "CLAIM": 3, "CONDITION": 3, "JUSTIFICATION": 3,
    "COUNTERCLAIM": 3, "HYPOTHESIS": 3, "RULE": 3, "DECISION": 3, "ACTION": 3,
    "CONCEPT": 4, "ENTITY": 4, "EVENT": 4, "FALLACY": 4,
}

logger = logging.getLogger(__name__)


@dataclass
class TBox:
    """Domain type hierarchy and axioms.

    Attributes:
        name: Domain name (e.g., "legal", "scientific").
        node_types: Allowed node types with their category levels.
        edge_types: Allowed edge types with their weights.
        axioms: SWRL-like inference rules as (antecedent, consequent) pairs.
        valid_edges: Valid (source_type, edge_type, target_type) triples.
    """
    name: str = "general"
    node_types: dict[str, int] = field(default_factory=dict)
    edge_types: dict[str, float] = field(default_factory=dict)
    axioms: list[dict] = field(default_factory=list)
    valid_edges: list[tuple[str, str, str]] = field(default_factory=list)


# ── General-purpose TBox ──────────────────────────────────────────

GENERAL_TBOX = TBox(
    name="general",
    node_types=dict(CATEGORY_LEVELS),
    edge_types={
        "INFERS": 0.9, "SUPPORTS": 0.85, "DIRECT": 0.95, "JUSTIFIES": 0.8,
        "CIRCUMSTANTIAL": 0.6, "QUALIFIES": 0.5, "REBUTS": 0.6, "HEARSAY": 0.4,
        "CONTRADICTS": 0.85, "ATTACKS": 0.8, "CAUSES": 0.8, "SUPPORT": 0.75,
        "ENABLES": 0.7, "DEPENDS": 0.6, "TEMPORAL": 0.5, "SIMILAR": 0.5,
        "EVIDENCE": 0.8, "PART_OF": 0.6, "CITES": 0.6, "FLOWS_TO": 0.6,
        "HAS_ATTRIBUTE": 0.7, "LOCATED_AT": 0.6, "EMPLOYED_BY": 0.6,
        "ASSOCIATED_WITH": 0.5, "CONTACT_OF": 0.5,
        "HAS_GOAL": 0.7, "INTENDS": 0.7, "KNOWS": 0.7, "COMMUNICATED": 0.6,
        "PREFERS": 0.6, "USES": 0.7, "PRODUCES": 0.7, "CONSUMES": 0.7,
        "TRANSFORMS": 0.7,
    },
    axioms=[
        {"antecedents": ["type_EVIDENCE", "edge_SUPPORTS"], "consequent": "belief_increase"},
        {"antecedents": ["type_COUNTERCLAIM", "edge_ATTACKS"], "consequent": "belief_decrease"},
        {"name": "agent_process_enablement", "when": "AGENT + ENABLES + PROCESS", "then": "belief_increase"},
        {"name": "process_causes_state", "when": "PROCESS + CAUSES + STATE", "then": "belief_increase"},
        {"name": "goal_action_link", "when": "GOAL + ENABLES + ACTION", "then": "belief_increase"},
        {"name": "resource_consumption", "when": "PROCESS + CONSUMES + RESOURCE", "then": "belief_increase"},
    ],
)

BUILTIN_TBOXES: dict[str, TBox] = {
    "general": GENERAL_TBOX,
}


def load_tbox(name: str = "general") -> TBox:
    """Load a TBox by name.

    Args:
        name: TBox name ("general", "legal", "scientific", or path to JSON).

    Returns:
        TBox instance.
    """
    if name in BUILTIN_TBOXES:
        return BUILTIN_TBOXES[name]

    path = Path(name)
    if path.exists():
        data = json.loads(path.read_text())
        return TBox(**data)

    logger.warning("TBox %r not found, falling back to general", name)
    return GENERAL_TBOX


def validate_against_tbox(node_type: str, edge_type: str, tbox: TBox) -> bool:
    """Check if a node type and edge type are valid in the TBox."""
    if node_type.upper() not in {k.upper() for k in tbox.node_types}:
        return False
    if edge_type.upper() not in {k.upper() for k in tbox.edge_types}:
        return False
    return True
