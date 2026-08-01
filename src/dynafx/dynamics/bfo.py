"""BFO (Basic Formal Ontology) alignment for system dynamics.

Aligns role taxonomy with BFO's continuant/occurrent distinction:
- Stock = continuant (something that persists and accumulates)
- Flow = occurrent (a process/occurrence)
- Input/Output = roles relative to system boundary

This is "aligned but not implemented" — naming convention now, full integration later.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class BfoContinuantCategory(Enum):
    """BFO top-level categories."""
    CONTINUANT = "continuant"
    OCCURRENT = "occurrent"


class ContinuantSubcategory(Enum):
    """BFO continuant subcategories relevant to system dynamics."""
    MATERIAL_ENTITY = "material_entity"
    IMMATERIAL_ENTITY = "immaterial_entity"
    QUALITY = "quality"
    REALIZABLE_ENTITY = "realizable_entity"
    SPATIAL_REGION = "spatial_region"
    TEMPORAL_REGION = "temporal_region"


class OccurrentSubcategory(Enum):
    """BFO occurrent subcategories relevant to system dynamics."""
    PROCESS = "process"
    PROCESS_BOUNDARY = "process_boundary"
    TEMPORAL_REGION = "temporal_region"
    SPATIAL_REGION = "spatial_region"


# ── Role → BFO mapping ─────────────────────────────────────────

ROLE_TO_BFO: dict[str, dict[str, str]] = {
    # Stock = Continuant (persists and accumulates)
    "stock": {
        "category": BfoContinuantCategory.CONTINUANT.value,
        "subcategory": ContinuantSubcategory.MATERIAL_ENTITY.value,
        "description": "Something that persists and accumulates (e.g., inventory, cash, people)",
    },
    # Flow = Occurrent (process/occurrence)
    "flow": {
        "category": BfoContinuantCategory.OCCURRENT.value,
        "subcategory": OccurrentSubcategory.PROCESS.value,
        "description": "A process that changes stock levels (e.g., production, sales, hiring)",
    },
    # Auxiliary = Continuant (supporting entity)
    "auxiliary": {
        "category": BfoContinuantCategory.CONTINUANT.value,
        "subcategory": ContinuantSubcategory.QUALITY.value,
        "description": "A supporting entity that influences flows (e.g., capacity, rate, threshold)",
    },
    # Input = Continuant (relative to system boundary)
    "input": {
        "category": BfoContinuantCategory.CONTINUANT.value,
        "subcategory": ContinuantSubcategory.MATERIAL_ENTITY.value,
        "description": "An entity entering the system (relative to boundary)",
    },
    # Output = Continuant (relative to system boundary)
    "output": {
        "category": BfoContinuantCategory.CONTINUANT.value,
        "subcategory": ContinuantSubcategory.MATERIAL_ENTITY.value,
        "description": "An entity leaving the system (relative to boundary)",
    },
    # Constant = Continuant (unchanging reference)
    "constant": {
        "category": BfoContinuantCategory.CONTINUANT.value,
        "subcategory": ContinuantSubcategory.QUALITY.value,
        "description": "A fixed reference value (e.g., safety stock threshold)",
    },
}


def get_bfo_alignment(role: str) -> dict[str, str] | None:
    """Get BFO alignment for a role.

    Args:
        role: The system dynamics role (stock, flow, auxiliary, etc.)

    Returns:
        Dictionary with BFO category and subcategory, or None if role not found
    """
    return ROLE_TO_BFO.get(role)


def get_all_bfo_alignments() -> dict[str, dict[str, str]]:
    """Get all role-to-BFO alignments."""
    return ROLE_TO_BFO.copy()


def validate_bfo_alignment(role: str) -> bool:
    """Check if a role has valid BFO alignment.

    Args:
        role: The system dynamics role to validate

    Returns:
        True if role has valid BFO alignment, False otherwise
    """
    return role in ROLE_TO_BFO


def get_bfo_description(role: str) -> str:
    """Get human-readable description of BFO alignment for a role.

    Args:
        role: The system dynamics role

    Returns:
        Description string, or empty string if role not found
    """
    alignment = ROLE_TO_BFO.get(role)
    if alignment:
        return alignment.get("description", "")
    return ""


def get_continuant_roles() -> list[str]:
    """Get all roles that map to BFO continuants."""
    return [
        role for role, alignment in ROLE_TO_BFO.items()
        if alignment.get("category") == BfoContinuantCategory.CONTINUANT.value
    ]


def get_occurrent_roles() -> list[str]:
    """Get all roles that map to BFO occurrents."""
    return [
        role for role, alignment in ROLE_TO_BFO.items()
        if alignment.get("category") == BfoContinuantCategory.OCCURRENT.value
    ]


def get_bfo_summary() -> dict[str, list[str]]:
    """Get summary of BFO alignments by category.

    Returns:
        Dictionary mapping BFO categories to list of roles
    """
    summary: dict[str, list[str]] = {
        BfoContinuantCategory.CONTINUANT.value: [],
        BfoContinuantCategory.OCCURRENT.value: [],
    }

    for role, alignment in ROLE_TO_BFO.items():
        category = alignment.get("category", "")
        if category in summary:
            summary[category].append(role)

    return summary
