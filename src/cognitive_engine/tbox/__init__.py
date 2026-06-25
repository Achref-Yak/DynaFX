"""Domain TBox — OWL2-style type hierarchy and SWRL-like axioms.
"""

from cognitive_engine.tbox.loader import TBox, load_tbox, GENERAL_TBOX
from cognitive_engine.tbox.hierarchy import TypeNode, TypeHierarchy, MDM_TYPE_HIERARCHY

__all__ = [
    "TBox",
    "load_tbox",
    "GENERAL_TBOX",
    "TypeNode",
    "TypeHierarchy",
    "MDM_TYPE_HIERARCHY",
]
