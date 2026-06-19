"""Higraph — core data model for Multiple Domain Matrix (MDM).

Based on Harel (1985) "On Visual Formalisms":
    A higraph extends standard graphs by adding notions of depth
    (hierarchy, ρ) and orthogonality (partition, Π).

    Higraph = (B, E, ρ, Π) where:
        B = set of blobs (system entities)
        E = set of edges (interactions + structural relationships)
        ρ: B → P(B) = hierarchy function (parent → children, multi-parent)
        Π: B → P = partition function (blob → partition)

Reference: Fogarty (2006) "System Modeling and Traceability Applications
of the Higraph Formalism", University of Maryland.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from uuid import UUID, uuid4


# ── Enumerations ────────────────────────────────────────────────

class BlobType(Enum):
    """Types of system entities (Blobs).

    Organized as a lattice:
        BLOB
        ├── AGENT (PERSON, ORGANIZATION, SYSTEM)
        ├── PROCESS (ACTION, EVENT, TRANSFORMATION)
        ├── RESOURCE (INFORMATION, MATERIAL, ENERGY)
        ├── GOAL (INTENTION, REQUIREMENT)
        ├── CONSTRAINT (RULE, REGULATION, LIMITATION)
        ├── STATE (CONDITION, STATUS)
        └── PROPERTY (QUALITY, QUANTITY)
    """
    BLOB = auto()
    AGENT = auto()
    PERSON = auto()
    ORGANIZATION = auto()
    SYSTEM = auto()
    PROCESS = auto()
    ACTION = auto()
    EVENT = auto()
    TRANSFORMATION = auto()
    RESOURCE = auto()
    INFORMATION = auto()
    MATERIAL = auto()
    ENERGY = auto()
    GOAL = auto()
    INTENTION = auto()
    REQUIREMENT = auto()
    CONSTRAINT = auto()
    RULE = auto()
    REGULATION = auto()
    LIMITATION = auto()
    STATE = auto()
    CONDITION = auto()
    STATUS = auto()
    PROPERTY = auto()
    QUALITY = auto()
    QUANTITY = auto()


class InteractionType(Enum):
    """Dynamic flows between Blobs.

    Organized by domain:
        CAUSAL: CAUSES, ENABLES, DEPENDS, PREVENTS
        INFORMATIONAL: COMMUNICATED, SUPPORTS, ATTACKS, CONTRADICTS
        MATERIAL: FLOWS_TO, PRODUCES, CONSUMES, TRANSFORMS
    """
    CAUSES = auto()
    ENABLES = auto()
    DEPENDS = auto()
    PREVENTS = auto()
    COMMUNICATED = auto()
    SUPPORTS = auto()
    ATTACKS = auto()
    CONTRADICTS = auto()
    FLOWS_TO = auto()
    PRODUCES = auto()
    CONSUMES = auto()
    TRANSFORMS = auto()


class StructuralType(Enum):
    """Static hierarchy between Blobs.

    Organized by domain:
        MERONYMIC: PART_OF, HAS_ATTRIBUTE
        SPATIAL: LOCATED_AT, EMPLOYED_BY
        TAXONOMIC: IS_A, INSTANCE_OF
    """
    PART_OF = auto()
    HAS_ATTRIBUTE = auto()
    LOCATED_AT = auto()
    EMPLOYED_BY = auto()
    IS_A = auto()
    INSTANCE_OF = auto()


# ── Data Classes ────────────────────────────────────────────────

@dataclass
class Blob:
    """A system entity (component, function, state, etc.).

    Harel (1985): "Blobs are the nodes representing system entities,
    such as components, functions, or states."

    Attributes:
        id: Unique identifier.
        name: Human-readable name.
        text: Source text that spawned this blob.
        blob_type: Type classification (from BlobType lattice).
        bfo_category: BFO ontological category (optional).
        opinion: Subjective Logic opinion as (belief, disbelief, uncertainty, prior).
        spans: Source text spans [{start, end, text}].
        attributes: Arbitrary key-value attributes.
        metadata: Extra metadata (demarcation, concept, etc.).
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    text: str = ""
    blob_type: BlobType = BlobType.STATE
    bfo_category: Optional[str] = None
    opinion: Optional[tuple[float, float, float, float]] = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id.hex,
            "name": self.name,
            "text": self.text,
            "blob_type": self.blob_type.name,
        }
        if self.bfo_category:
            d["bfo_category"] = self.bfo_category
        if self.opinion:
            d["opinion"] = list(self.opinion)
        if self.spans:
            d["spans"] = self.spans
        if self.attributes:
            d["attributes"] = self.attributes
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Interaction:
    """Dynamic flow between Blobs (causes, enables, depends, etc.).

    Harel (1985): "Edges represent the relationships or 'generic flows'
    of information, energy, or material between blobs."

    Attributes:
        id: Unique identifier.
        source: Source Blob ID.
        target: Target Blob ID.
        interaction_type: Type of interaction.
        weight: Strength of interaction (0.0 to 1.0).
        belief: Subjective Logic opinion as (b, d, u, prior).
        participants: List of participant roles [{role, blob_id}].
        metadata: Extra metadata.
    """
    id: UUID = field(default_factory=uuid4)
    source: UUID = field(default_factory=uuid4)
    target: UUID = field(default_factory=uuid4)
    interaction_type: InteractionType = InteractionType.CAUSES
    weight: float = 0.5
    belief: Optional[tuple[float, float, float, float]] = None
    participants: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id.hex,
            "source": self.source.hex,
            "target": self.target.hex,
            "interaction_type": self.interaction_type.name,
            "weight": self.weight,
        }
        if self.belief:
            d["belief"] = list(self.belief)
        if self.participants:
            d["participants"] = self.participants
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class StructuralRelationship:
    """Static hierarchy between Blobs (part-of, has-attribute, etc.).

    Attributes:
        id: Unique identifier.
        source: Source Blob ID.
        target: Target Blob ID.
        relationship_type: Type of structural relationship.
        weight: Strength (0.0 to 1.0).
        metadata: Extra metadata.
    """
    id: UUID = field(default_factory=uuid4)
    source: UUID = field(default_factory=uuid4)
    target: UUID = field(default_factory=uuid4)
    relationship_type: StructuralType = StructuralType.PART_OF
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id.hex,
            "source": self.source.hex,
            "target": self.target.hex,
            "relationship_type": self.relationship_type.name,
            "weight": self.weight,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ── Higraph ─────────────────────────────────────────────────────

@dataclass
class Higraph:
    """Higraph = (B, E, ρ, Π)

    The structural foundation for Multiple Domain Matrix (MDM) reasoning.

    - B = set of Blobs (system entities)
    - E = set of Edges (Interactions + StructuralRelationships)
    - ρ: B → P(B) = hierarchy function (multi-parent supported)
    - Π: B → P = partition function (blob → partition name)

    Harel (1985): "A higraph extends standard graphs by adding
    notions of depth (hierarchy) and orthogonality (partition)."

    Fogarty (2006): "With a higraph system model, all relationships
    between system requirements, components, and behaviors are formalized."
    """
    blobs: dict[UUID, Blob] = field(default_factory=dict)
    interactions: dict[UUID, Interaction] = field(default_factory=dict)
    structural: dict[UUID, StructuralRelationship] = field(default_factory=dict)

    # ρ: Hierarchy function (multi-parent supported)
    # Key = parent blob ID, Value = set of child blob IDs
    hierarchy: dict[UUID, set[UUID]] = field(default_factory=dict)
    # Reverse lookup: child → set of parents
    _parents: dict[UUID, set[UUID]] = field(default_factory=dict)

    # Π: Orthogonality partitions
    # Key = partition name, Value = set of blob IDs
    partitions: dict[str, set[UUID]] = field(default_factory=dict)

    # ── Blob operations ──────────────────────────────────────────

    def add_blob(self, blob: Blob) -> None:
        """Add a Blob to the higraph."""
        self.blobs[blob.id] = blob

    def remove_blob(self, blob_id: UUID) -> None:
        """Remove a Blob and all its edges."""
        self.blobs.pop(blob_id, None)
        # Remove from hierarchy
        self.hierarchy.pop(blob_id, None)
        self._parents.pop(blob_id, None)
        for children in self.hierarchy.values():
            children.discard(blob_id)
        for parents in self._parents.values():
            parents.discard(blob_id)
        # Remove from partitions
        for blobs in self.partitions.values():
            blobs.discard(blob_id)
        # Remove edges
        to_remove_inter = [
            iid for iid, inter in self.interactions.items()
            if inter.source == blob_id or inter.target == blob_id
        ]
        for iid in to_remove_inter:
            self.interactions.pop(iid, None)
        to_remove_struct = [
            sid for sid, s in self.structural.items()
            if s.source == blob_id or s.target == blob_id
        ]
        for sid in to_remove_struct:
            self.structural.pop(sid, None)

    def get_blob(self, blob_id: UUID) -> Optional[Blob]:
        """Get a Blob by ID."""
        return self.blobs.get(blob_id)

    def get_blobs_by_type(self, blob_type: BlobType) -> list[Blob]:
        """Get all Blobs of a given type."""
        return [b for b in self.blobs.values() if b.blob_type == blob_type]

    # ── Interaction operations ───────────────────────────────────

    def add_interaction(self, interaction: Interaction) -> None:
        """Add an Interaction to the higraph."""
        self.interactions[interaction.id] = interaction

    def remove_interaction(self, interaction_id: UUID) -> None:
        """Remove an Interaction."""
        self.interactions.pop(interaction_id, None)

    def get_interactions_from(self, blob_id: UUID) -> list[Interaction]:
        """Get all Interactions originating from a Blob."""
        return [i for i in self.interactions.values() if i.source == blob_id]

    def get_interactions_to(self, blob_id: UUID) -> list[Interaction]:
        """Get all Interactions targeting a Blob."""
        return [i for i in self.interactions.values() if i.target == blob_id]

    def get_interactions_between(
        self, source_id: UUID, target_id: UUID
    ) -> list[Interaction]:
        """Get all Interactions between two Blobs."""
        return [
            i for i in self.interactions.values()
            if i.source == source_id and i.target == target_id
        ]

    # ── Structural relationship operations ───────────────────────

    def add_structural(self, structural: StructuralRelationship) -> None:
        """Add a StructuralRelationship to the higraph."""
        self.structural[structural.id] = structural

    def remove_structural(self, structural_id: UUID) -> None:
        """Remove a StructuralRelationship."""
        self.structural.pop(structural_id, None)

    def get_structural_from(self, blob_id: UUID) -> list[StructuralRelationship]:
        """Get all StructuralRelationships originating from a Blob."""
        return [s for s in self.structural.values() if s.source == blob_id]

    def get_structural_to(self, blob_id: UUID) -> list[StructuralRelationship]:
        """Get all StructuralRelationships targeting a Blob."""
        return [s for s in self.structural.values() if s.target == blob_id]

    # ── Hierarchy operations (ρ) ─────────────────────────────────

    def set_parent(self, parent_id: UUID, child_id: UUID) -> None:
        """ρ: Set parent-child relationship (multi-parent supported).

        A Blob can have multiple parents, allowing it to appear in
        multiple hierarchies simultaneously (e.g., a function in both
        "Safety" and "Performance" hierarchies).
        """
        if parent_id not in self.hierarchy:
            self.hierarchy[parent_id] = set()
        self.hierarchy[parent_id].add(child_id)

        if child_id not in self._parents:
            self._parents[child_id] = set()
        self._parents[child_id].add(parent_id)

    def remove_parent(self, parent_id: UUID, child_id: UUID) -> None:
        """ρ: Remove a parent-child relationship."""
        if parent_id in self.hierarchy:
            self.hierarchy[parent_id].discard(child_id)
            if not self.hierarchy[parent_id]:
                del self.hierarchy[parent_id]
        if child_id in self._parents:
            self._parents[child_id].discard(parent_id)
            if not self._parents[child_id]:
                del self._parents[child_id]

    def get_children(self, blob_id: UUID) -> set[UUID]:
        """ρ: Get direct children of a Blob."""
        return self.hierarchy.get(blob_id, set()).copy()

    def get_parents(self, blob_id: UUID) -> set[UUID]:
        """ρ: Get all parents of a Blob (multi-parent)."""
        return self._parents.get(blob_id, set()).copy()

    def get_descendants(self, blob_id: UUID) -> set[UUID]:
        """ρ: Get all descendants (recursive children)."""
        descendants: set[UUID] = set()
        stack = list(self.get_children(blob_id))
        while stack:
            current = stack.pop()
            if current not in descendants:
                descendants.add(current)
                stack.extend(self.get_children(current))
        return descendants

    def get_ancestors(self, blob_id: UUID) -> set[UUID]:
        """ρ: Get all ancestors (recursive parents)."""
        ancestors: set[UUID] = set()
        stack = list(self.get_parents(blob_id))
        while stack:
            current = stack.pop()
            if current not in ancestors:
                ancestors.add(current)
                stack.extend(self.get_parents(current))
        return ancestors

    def get_roots(self) -> set[UUID]:
        """ρ: Get all root Blobs (no parents)."""
        return {
            bid for bid in self.blobs
            if bid not in self._parents or not self._parents[bid]
        }

    def get_leaves(self) -> set[UUID]:
        """ρ: Get all leaf Blobs (no children)."""
        return {
            bid for bid in self.blobs
            if bid not in self.hierarchy or not self.hierarchy[bid]
        }

    # ── Partition operations (Π) ─────────────────────────────────

    def set_partition(self, partition_name: str, blob_id: UUID) -> None:
        """Π: Assign a Blob to a partition."""
        if partition_name not in self.partitions:
            self.partitions[partition_name] = set()
        self.partitions[partition_name].add(blob_id)

    def remove_partition(self, partition_name: str, blob_id: UUID) -> None:
        """Π: Remove a Blob from a partition."""
        if partition_name in self.partitions:
            self.partitions[partition_name].discard(blob_id)
            if not self.partitions[partition_name]:
                del self.partitions[partition_name]

    def get_partition(self, partition_name: str) -> set[UUID]:
        """Π: Get all Blobs in a partition."""
        return self.partitions.get(partition_name, set()).copy()

    def get_blob_partitions(self, blob_id: UUID) -> list[str]:
        """Π: Get all partitions a Blob belongs to."""
        return [
            name for name, blobs in self.partitions.items()
            if blob_id in blobs
        ]

    def get_all_partitions(self) -> dict[str, set[UUID]]:
        """Π: Get all partitions."""
        return {name: blobs.copy() for name, blobs in self.partitions.items()}

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "blobs": {bid.hex: b.to_dict() for bid, b in self.blobs.items()},
            "interactions": {
                iid.hex: i.to_dict() for iid, i in self.interactions.items()
            },
            "structural": {
                sid.hex: s.to_dict() for sid, s in self.structural.items()
            },
            "hierarchy": {
                pid.hex: [cid.hex for cid in children]
                for pid, children in self.hierarchy.items()
            },
            "partitions": {
                name: [bid.hex for bid in blobs]
                for name, blobs in self.partitions.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_compact_str(self) -> str:
        """Human-readable compact string representation."""
        lines: list[str] = []
        lines.append(f"Higraph: {len(self.blobs)} blobs, "
                     f"{len(self.interactions)} interactions, "
                     f"{len(self.structural)} structural, "
                     f"{len(self.partitions)} partitions")
        for bid, blob in self.blobs.items():
            lines.append(f"  BLOB {bid.hex[:8]} {blob.blob_type.name} "
                        f"\"{blob.name[:40]}\"")
        for iid, inter in self.interactions.items():
            lines.append(f"  INTER {inter.source.hex[:8]} "
                        f"--{inter.interaction_type.name}--> "
                        f"{inter.target.hex[:8]}")
        for sid, s in self.structural.items():
            lines.append(f"  STRUCT {s.source.hex[:8]} "
                        f"--{s.relationship_type.name}--> "
                        f"{s.target.hex[:8]}")
        for name, blobs in self.partitions.items():
            lines.append(f"  PARTITION {name}: {len(blobs)} blobs")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Higraph:
        """Deserialize from dictionary."""
        h = cls()

        # Blobs
        for bid_hex, bdata in data.get("blobs", {}).items():
            blob = Blob(
                id=UUID(bid_hex),
                name=bdata.get("name", ""),
                text=bdata.get("text", ""),
                blob_type=BlobType[bdata.get("blob_type", "STATE")],
                bfo_category=bdata.get("bfo_category"),
                opinion=tuple(bdata["opinion"]) if "opinion" in bdata else None,
                spans=bdata.get("spans", []),
                attributes=bdata.get("attributes", {}),
                metadata=bdata.get("metadata", {}),
            )
            h.add_blob(blob)

        # Interactions
        for iid_hex, idata in data.get("interactions", {}).items():
            inter = Interaction(
                id=UUID(iid_hex),
                source=UUID(idata["source"]),
                target=UUID(idata["target"]),
                interaction_type=InteractionType[idata.get("interaction_type", "CAUSES")],
                weight=idata.get("weight", 0.5),
                belief=tuple(idata["belief"]) if "belief" in idata else None,
                participants=idata.get("participants", []),
                metadata=idata.get("metadata", {}),
            )
            h.add_interaction(inter)

        # Structural relationships
        for sid_hex, sdata in data.get("structural", {}).items():
            s = StructuralRelationship(
                id=UUID(sid_hex),
                source=UUID(sdata["source"]),
                target=UUID(sdata["target"]),
                relationship_type=StructuralType[sdata.get("relationship_type", "PART_OF")],
                weight=sdata.get("weight", 1.0),
                metadata=sdata.get("metadata", {}),
            )
            h.add_structural(s)

        # Hierarchy
        for pid_hex, children_hex in data.get("hierarchy", {}).items():
            parent_id = UUID(pid_hex)
            for cid_hex in children_hex:
                h.set_parent(parent_id, UUID(cid_hex))

        # Partitions
        for name, blob_hexes in data.get("partitions", {}).items():
            for bid_hex in blob_hexes:
                h.set_partition(name, UUID(bid_hex))

        return h
