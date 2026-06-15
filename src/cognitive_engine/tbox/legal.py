"""Legal domain TBox — type hierarchy and axioms for legal argumentation.

Extends the general TBox with legal-specific node types, edge types,
and inference axioms for contract review, case analysis, and
legal argument mapping.
"""

from cognitive_engine.core.math import CATEGORY_LEVELS
from cognitive_engine.tbox.loader import TBox

LEGAL_TBOX = TBox(
    name="legal",
    node_types={
        **CATEGORY_LEVELS,
        "STATUTE": 2,
        "PRECEDENT": 2,
        "CONTRACT_CLAUSE": 2,
        "ARGUMENT": 3,
        "OBJECTION": 3,
        "RULING": 2,
    },
    edge_types={
        "CITES": 0.6,
        "OVERRULES": 0.8,
        "DISTINGUISHES": 0.5,
        "SUPPORTS": 0.85,
        "ATTACKS": 0.8,
        "CONTRADICTS": 0.85,
        "JUSTIFIES": 0.8,
        "EVIDENCE": 0.8,
        "QUALIFIES": 0.5,
        "REBUTS": 0.6,
    },
    axioms=[
        {"antecedents": ["type_PRECEDENT", "edge_CITES"], "consequent": "persuasive_authority"},
        {"antecedents": ["type_STATUTE", "edge_CITES"], "consequent": "binding_authority"},
        {"antecedents": ["type_OBJECTION", "edge_ATTACKS"], "consequent": "argument_weakened"},
        {"antecedents": ["type_RULING", "edge_OVERRULES"], "consequent": "precedent_overturned"},
    ],
    valid_edges=[
        ("EVIDENCE", "SUPPORTS", "CLAIM"),
        ("EVIDENCE", "SUPPORTS", "ARGUMENT"),
        ("COUNTERCLAIM", "ATTACKS", "CLAIM"),
        ("OBJECTION", "ATTACKS", "ARGUMENT"),
        ("PRECEDENT", "CITES", "STATUTE"),
        ("RULING", "OVERRULES", "PRECEDENT"),
        ("STATUTE", "CITES", "CONTRACT_CLAUSE"),
    ],
)
