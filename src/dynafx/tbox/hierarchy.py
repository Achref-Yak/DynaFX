"""TBox Type Hierarchy — OWL2-style type lattice for MDM.

Provides subtyping, property chains, and transitive properties
for the Higraph blob types, interaction types, and structural types.

Based on:
    - Harel (1985) "On Visual Formalisms"
    - Fogarty (2006) "System Modeling and Traceability Applications
      of the Higraph Formalism"
    - Behavioral-GST Ontology (Goal-Scenario-Task structure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TypeNode:
    """A node in the type hierarchy.

    Attributes:
        name: Type name (e.g., "AGENT", "PERSON").
        parent: Parent type name (None for root).
        properties: Type properties (transitive, symmetric, reflexive, etc.).
        description: Human-readable description.
    """
    name: str
    parent: Optional[str] = None
    properties: dict[str, bool] = field(default_factory=dict)
    description: str = ""


@dataclass
class TypeHierarchy:
    """OWL2-style type lattice with subtyping and property chains.

    Supports:
        - Subtype checking: is_subtype("PERSON", "AGENT") → True
        - Ancestor enumeration: get_ancestors("PERSON") → ["PERSON", "AGENT", "BLOB"]
        - Descendant enumeration: get_descendants("BLOB") → ["AGENT", "PERSON", ...]
        - Property chain reasoning: if CAUSES is transitive, then
          A CAUSES B and B CAUSES C implies A CAUSES C
    """
    types: dict[str, TypeNode] = field(default_factory=dict)

    def add_type(
        self,
        name: str,
        parent: Optional[str] = None,
        description: str = "",
        **properties: bool,
    ) -> None:
        """Add a type to the hierarchy.

        Args:
            name: Type name (must be unique).
            parent: Parent type name (None for root types).
            description: Human-readable description.
            **properties: Boolean properties (transitive, symmetric, reflexive).
        """
        self.types[name] = TypeNode(
            name=name,
            parent=parent,
            properties=properties,
            description=description,
        )

    def is_subtype(self, sub: str, parent: str) -> bool:
        """Check if 'sub' is a subtype of 'parent'.

        Traverses the hierarchy upward from 'sub' to find 'parent'.

        Example:
            is_subtype("PERSON", "AGENT") → True
            is_subtype("PERSON", "BLOB") → True
            is_subtype("AGENT", "PERSON") → False
        """
        if sub == parent:
            return True
        current = sub
        visited: set[str] = set()
        while current:
            if current == parent:
                return True
            if current in visited:
                break
            visited.add(current)
            node = self.types.get(current)
            if not node or not node.parent:
                break
            current = node.parent
        return False

    def get_ancestors(self, type_name: str) -> list[str]:
        """Get all ancestors of a type (including itself).

        Example:
            get_ancestors("PERSON") → ["PERSON", "AGENT", "BLOB"]
        """
        ancestors: list[str] = []
        current: Optional[str] = type_name
        visited: set[str] = set()
        while current:
            if current in visited:
                break
            visited.add(current)
            ancestors.append(current)
            node = self.types.get(current)
            if not node:
                break
            current = node.parent
        return ancestors

    def get_children(self, type_name: str) -> list[str]:
        """Get direct children of a type."""
        return [
            name for name, node in self.types.items()
            if node.parent == type_name
        ]

    def get_descendants(self, type_name: str) -> list[str]:
        """Get all descendants of a type (recursive children).

        Example:
            get_descendants("BLOB") → ["AGENT", "PERSON", "PROCESS", ...]
        """
        descendants: list[str] = []
        stack = self.get_children(type_name)
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)
            stack.extend(self.get_children(current))
        return descendants

    def get_siblings(self, type_name: str) -> list[str]:
        """Get all siblings (types with the same parent)."""
        node = self.types.get(type_name)
        if not node or not node.parent:
            return []
        return [
            name for name, n in self.types.items()
            if n.parent == node.parent and name != type_name
        ]

    def is_transitive(self, type_name: str) -> bool:
        """Check if a type has the transitive property."""
        node = self.types.get(type_name)
        return node.properties.get("transitive", False) if node else False

    def is_symmetric(self, type_name: str) -> bool:
        """Check if a type has the symmetric property."""
        node = self.types.get(type_name)
        return node.properties.get("symmetric", False) if node else False

    def is_reflexive(self, type_name: str) -> bool:
        """Check if a type has the reflexive property."""
        node = self.types.get(type_name)
        return node.properties.get("reflexive", False) if node else False

    def validate_edge(
        self,
        source_type: str,
        edge_type: str,
        target_type: str,
    ) -> bool:
        """Validate if an edge between two types is allowed.

        Checks:
            1. Source type exists in hierarchy
            2. Target type exists in hierarchy
            3. Edge type exists in hierarchy
            4. Source is a subtype of edge's domain (if defined)
            5. Target is a subtype of edge's range (if defined)
        """
        if source_type not in self.types:
            return False
        if target_type not in self.types:
            return False
        if edge_type not in self.types:
            return False

        edge_node = self.types.get(edge_type)
        if not edge_node:
            return False

        # Check domain constraint (stored as "domain" property)
        domain = edge_node.properties.get("domain")
        if domain and not self.is_subtype(source_type, domain):
            return False

        # Check range constraint (stored as "range" property)
        range_type = edge_node.properties.get("range")
        if range_type and not self.is_subtype(target_type, range_type):
            return False

        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            name: {
                "parent": node.parent,
                "properties": node.properties,
                "description": node.description,
            }
            for name, node in self.types.items()
        }


# ── Pre-built MDM Type Hierarchy ────────────────────────────────

def build_mdm_type_hierarchy() -> TypeHierarchy:
    """Build the standard MDM type hierarchy.

    Blob types form a lattice:
        BLOB
        ├── AGENT (PERSON, ORGANIZATION, SYSTEM)
        ├── PROCESS (ACTION, EVENT, TRANSFORMATION)
        ├── RESOURCE (INFORMATION, MATERIAL, ENERGY)
        ├── GOAL (INTENTION, REQUIREMENT)
        ├── CONSTRAINT (RULE, REGULATION, LIMITATION)
        ├── STATE (CONDITION, STATUS)
        └── PROPERTY (QUALITY, QUANTITY)

    Interaction types:
        INTERACTION
        ├── CAUSAL (CAUSES, ENABLES, DEPENDS, PREVENTS)
        ├── INFORMATIONAL (COMMUNICATED, SUPPORTS, ATTACKS, CONTRADICTS)
        └── MATERIAL (FLOWS_TO, PRODUCES, CONSUMES, TRANSFORMS)

    Structural types:
        STRUCTURAL
        ├── MERONYMIC (PART_OF, HAS_ATTRIBUTE)
        ├── SPATIAL (LOCATED_AT, EMPLOYED_BY)
        └── TAXONOMIC (IS_A, INSTANCE_OF)
    """
    h = TypeHierarchy()

    # ── Blob types ───────────────────────────────────────────────
    h.add_type("BLOB", description="Root of all blob types")
    h.add_type("AGENT", parent="BLOB", description="Entities that act")
    h.add_type("PERSON", parent="AGENT", description="Human agent")
    h.add_type("ORGANIZATION", parent="AGENT", description="Group of agents")
    h.add_type("SYSTEM", parent="AGENT", description="Technical system")

    h.add_type("PROCESS", parent="BLOB", description="Things that happen")
    h.add_type("ACTION", parent="PROCESS", description="Deliberate activity")
    h.add_type("EVENT", parent="PROCESS", description="Occurrence")
    h.add_type("TRANSFORMATION", parent="PROCESS", description="State change")

    h.add_type("RESOURCE", parent="BLOB", description="Consumable/producible")
    h.add_type("INFORMATION", parent="RESOURCE", description="Data/knowledge")
    h.add_type("MATERIAL", parent="RESOURCE", description="Physical stuff")
    h.add_type("ENERGY", parent="RESOURCE", description="Power/force")

    h.add_type("GOAL", parent="BLOB", description="Desired outcome")
    h.add_type("INTENTION", parent="GOAL", description="Planned goal")
    h.add_type("REQUIREMENT", parent="GOAL", description="Mandatory goal")

    h.add_type("CONSTRAINT", parent="BLOB", description="Limiting factor")
    h.add_type("RULE", parent="CONSTRAINT", description="Formal rule")
    h.add_type("REGULATION", parent="CONSTRAINT", description="External regulation")
    h.add_type("LIMITATION", parent="CONSTRAINT", description="Physical limitation")

    h.add_type("STATE", parent="BLOB", description="Condition/status")
    h.add_type("CONDITION", parent="STATE", description="Environmental state")
    h.add_type("STATUS", parent="STATE", description="Entity state")

    h.add_type("PROPERTY", parent="BLOB", description="Quality/quantity")
    h.add_type("QUALITY", parent="PROPERTY", description="Qualitative property")
    h.add_type("QUANTITY", parent="PROPERTY", description="Quantitative property")

    # ── Observation types (raw simulation output) ─────────────────
    h.add_type("OBSERVATION", description="Root of observation types")
    h.add_type("SIMULATION_OBSERVATION", parent="OBSERVATION",
               description="Metric observation from a simulation run")

    # ── Interaction types ────────────────────────────────────────
    h.add_type("INTERACTION", description="Root of all interaction types")

    h.add_type("CAUSAL", parent="INTERACTION", description="Cause-effect flows")
    h.add_type("CAUSES", parent="CAUSAL", transitive=True,
               description="Direct causation")
    h.add_type("ENABLES", parent="CAUSAL", description="Enabling condition")
    h.add_type("DEPENDS", parent="CAUSAL", transitive=True,
               description="Dependency")
    h.add_type("PREVENTS", parent="CAUSAL", description="Prevention")

    h.add_type("INFORMATIONAL", parent="INTERACTION", description="Info flows")
    h.add_type("COMMUNICATED", parent="INFORMATIONAL", description="Information transfer")
    h.add_type("SUPPORTS", parent="INFORMATIONAL", description="Argument support")
    h.add_type("ATTACKS", parent="INFORMATIONAL", description="Argument attack")
    h.add_type("CONTRADICTS", parent="INFORMATIONAL", symmetric=True,
               description="Contradiction")

    h.add_type("MATERIAL", parent="INTERACTION", description="Material flows")
    h.add_type("FLOWS_TO", parent="MATERIAL", description="Material flow")
    h.add_type("PRODUCES", parent="MATERIAL", description="Production")
    h.add_type("CONSUMES", parent="MATERIAL", description="Consumption")
    h.add_type("TRANSFORMS", parent="MATERIAL", description="Transformation")

    # ── Structural types ─────────────────────────────────────────
    h.add_type("STRUCTURAL", description="Root of all structural types")

    h.add_type("MERONYMIC", parent="STRUCTURAL", description="Part-whole")
    h.add_type("PART_OF", parent="MERONYMIC", transitive=True,
               description="Part of (meronymy)")
    h.add_type("HAS_ATTRIBUTE", parent="MERONYMIC", description="Has property")

    h.add_type("SPATIAL", parent="STRUCTURAL", description="Spatial relations")
    h.add_type("LOCATED_AT", parent="SPATIAL", description="Location")
    h.add_type("EMPLOYED_BY", parent="SPATIAL", description="Employment")

    h.add_type("TAXONOMIC", parent="STRUCTURAL", description="Classification")
    h.add_type("IS_A", parent="TAXONOMIC", transitive=True,
               description="Is a (hyponymy)")
    h.add_type("INSTANCE_OF", parent="TAXONOMIC", description="Instance of class")

    return h


# Singleton instance
MDM_TYPE_HIERARCHY = build_mdm_type_hierarchy()
