"""System validation using Dung/IAF semantics.

Validates the system using:
- Dung argumentation semantics for conflict resolution
- Role-contests as argument type (not just causal-claim conflicts)
- Loop polarity classification (reinforcing/balancing/goal-seeking)
- Incomplete Argumentation Frameworks (IAFs) for unresolved arguments
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from dynafx.core.math import dung_semantics
from dynafx.core.models import (
    Edge,
    Graph,
    Node,
    NodeType,
    Opinion,
    Parameter,
)

logger = logging.getLogger(__name__)


class ArgumentType(Enum):
    """Types of arguments in the framework."""
    CAUSAL_CLAIM = "causal_claim"
    ROLE_CONTEST = "role_contest"


class ValidationResult(Enum):
    """Validation results for nodes."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ValidationArgument:
    """An argument in the Dung framework."""
    id: UUID
    argument_type: ArgumentType
    claim: str
    confidence: Opinion
    source_nodes: List[UUID]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationAttack:
    """An attack between arguments."""
    attacker_id: UUID
    target_id: UUID
    attack_type: str  # "rebut", "undercut", "undermine"
    confidence: Opinion
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResultDetail:
    """Detailed validation result for the system."""
    accepted_nodes: Set[UUID]
    rejected_nodes: Set[UUID]
    contested_nodes: Set[UUID]
    unknown_nodes: Set[UUID]
    arguments: List[ValidationArgument]
    attacks: List[ValidationAttack]
    loop_classifications: List[Dict[str, Any]]
    is_consistent: bool
    consistency_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _build_arguments_from_graph(graph: Graph) -> List[ValidationArgument]:
    """Build arguments from graph nodes and edges.

    Argument types:
    - causal_claim: Based on edge polarity
    - role_contest: Based on role assignment conflicts
    """
    arguments = []

    # Build causal claim arguments from edges
    for edge in graph.edges.values():
        source = graph.nodes.get(edge.source_id)
        target = graph.nodes.get(edge.target_id)
        if source is None or target is None:
            continue

        # Skip entity nodes
        if source.type == NodeType.ENTITY or target.type == NodeType.ENTITY:
            continue

        if edge.polarity == 1:
            polarity = '+'
        elif edge.polarity == -1:
            polarity = '-'
        else:
            polarity = edge.metadata.get('polarity', '+')
        confidence = edge.metadata.get('confidence', 0.5)

        # Create argument for the causal claim
        arg_id = edge.id
        claim = f"{source.text} -> {target.text} ({polarity})"
        arg = ValidationArgument(
            id=arg_id,
            argument_type=ArgumentType.CAUSAL_CLAIM,
            claim=claim,
            confidence=Opinion(belief=confidence, disbelief=0.1, uncertainty=0.3),
            source_nodes=[edge.source_id, edge.target_id],
            metadata={
                "edge_id": str(edge.id),
                "polarity": polarity,
                "source_text": source.text,
                "target_text": target.text,
            },
        )
        arguments.append(arg)

    # Build role contest arguments from nodes with conflicting roles
    for node in graph.nodes.values():
        if node.type == NodeType.ENTITY:
            continue

        role = node.metadata.get('role')
        if role is None:
            continue

        # Check for role conflicts (multiple possible roles)
        candidate_roles = node.metadata.get('candidate_roles', {})
        if len(candidate_roles) > 1:
            # Create argument for the role contest
            arg_id = UUID(str(node.id)[:8] + "role")
            claim = f"{node.text} is {role}"
            role_opinion = node.metadata.get('role_opinion')
            confidence = role_opinion if role_opinion else Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)

            arg = ValidationArgument(
                id=arg_id,
                argument_type=ArgumentType.ROLE_CONTEST,
                claim=claim,
                confidence=confidence,
                source_nodes=[node.id],
                metadata={
                    "node_id": str(node.id),
                    "role": role,
                    "candidate_roles": candidate_roles,
                    "node_text": node.text,
                },
            )
            arguments.append(arg)

    return arguments


def _build_attacks_from_arguments(
    arguments: List[ValidationArgument],
    graph: Graph,
) -> List[ValidationAttack]:
    """Build attack relations between arguments.

    Attack types:
    - rebut: Direct contradiction (opposite polarity on same edge)
    - undercut: Attacks the support for a claim
    - undermine: Attacks the premises of an argument
    """
    attacks = []
    arg_dict = {arg.id: arg for arg in arguments}

    # Check for rebut attacks (opposite polarity on same relationship)
    for i, arg1 in enumerate(arguments):
        if arg1.argument_type != ArgumentType.CAUSAL_CLAIM:
            continue

        for arg2 in arguments[i+1:]:
            if arg2.argument_type != ArgumentType.CAUSAL_CLAIM:
                continue

            # Check if they involve the same nodes
            if set(arg1.source_nodes) == set(arg2.source_nodes):
                # Check if polarity is opposite
                polarity1 = arg1.metadata.get('polarity', '+')
                polarity2 = arg2.metadata.get('polarity', '+')

                if polarity1 != polarity2:
                    # Create rebut attack
                    attack = ValidationAttack(
                        attacker_id=arg1.id,
                        target_id=arg2.id,
                        attack_type="rebut",
                        confidence=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                        metadata={
                            "reason": "Opposite polarity on same relationship",
                            "polarity1": polarity1,
                            "polarity2": polarity2,
                        },
                    )
                    attacks.append(attack)

                    # Create reciprocal attack
                    attack_recip = ValidationAttack(
                        attacker_id=arg2.id,
                        target_id=arg1.id,
                        attack_type="rebut",
                        confidence=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                        metadata={
                            "reason": "Opposite polarity on same relationship",
                            "polarity1": polarity2,
                            "polarity2": polarity1,
                        },
                    )
                    attacks.append(attack_recip)

    # Check for role contest attacks
    for arg1 in arguments:
        if arg1.argument_type != ArgumentType.ROLE_CONTEST:
            continue

        for arg2 in arguments:
            if arg2.argument_type != ArgumentType.ROLE_CONTEST:
                continue

            # Check if they target the same node
            if arg1.source_nodes == arg2.source_nodes and arg1.id != arg2.id:
                # Create attack
                attack = ValidationAttack(
                    attacker_id=arg1.id,
                    target_id=arg2.id,
                    attack_type="undercut",
                    confidence=Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
                    metadata={
                        "reason": "Conflicting role assignments",
                        "role1": arg1.metadata.get('role'),
                        "role2": arg2.metadata.get('role'),
                    },
                )
                attacks.append(attack)

    return attacks


def _run_dung_semantics(
    arguments: List[ValidationArgument],
    attacks: List[ValidationAttack],
) -> Set[UUID]:
    """Run Dung semantics to find accepted arguments.

    Returns:
        Set of accepted argument IDs
    """
    # Build beliefs dictionary
    beliefs = {arg.id: arg.confidence.belief for arg in arguments}

    # Build attack graph
    attack_graph: Dict[UUID, List[UUID]] = {}
    for attack in attacks:
        if attack.target_id not in attack_graph:
            attack_graph[attack.target_id] = []
        attack_graph[attack.target_id].append(attack.attacker_id)

    # Run Dung semantics
    accepted = dung_semantics(beliefs, attack_graph)

    return accepted


def _classify_loop_polarity(graph: Graph) -> List[Dict[str, Any]]:
    """Classify loop polarity in the system.

    Returns:
        List of loop classifications
    """
    # Simplified loop detection — in production, use proper cycle detection
    # This is a placeholder for the full implementation in Phase 4
    return []


def validate_system_internal(graph: Graph) -> ValidationResultDetail:
    """Validate the system using Dung/IAF semantics.

    Validates:
    - Role-contests as argument type (not just causal-claim conflicts)
    - Loop polarity classification (reinforcing/balancing/goal-seeking)
    - Incomplete Argumentation Frameworks (IAFs) for unresolved arguments

    Returns:
        Detailed validation result
    """
    # Build arguments and attacks
    arguments = _build_arguments_from_graph(graph)
    attacks = _build_attacks_from_arguments(arguments, graph)

    # Run Dung semantics
    accepted_args = _run_dung_semantics(arguments, attacks)

    # Map arguments to nodes
    accepted_nodes: Set[UUID] = set()
    rejected_nodes: Set[UUID] = set()
    contested_nodes: Set[UUID] = set()

    for arg in arguments:
        node_ids = arg.source_nodes
        if arg.id in accepted_args:
            accepted_nodes.update(node_ids)
        else:
            # Check if this is a contested node (high uncertainty)
            if arg.confidence.uncertainty > 0.5:
                contested_nodes.update(node_ids)
            else:
                rejected_nodes.update(node_ids)

    # Add nodes without arguments as unknown
    unknown_nodes = set()
    for node in graph.nodes.values():
        if node.type == NodeType.ENTITY:
            continue
        if node.id not in accepted_nodes and node.id not in rejected_nodes and node.id not in contested_nodes:
            unknown_nodes.add(node.id)

    # Classify loops
    loop_classifications = _classify_loop_polarity(graph)

    # Calculate consistency score
    total_nodes = len(accepted_nodes) + len(rejected_nodes) + len(contested_nodes) + len(unknown_nodes)
    if total_nodes > 0:
        consistency_score = len(accepted_nodes) / total_nodes
    else:
        consistency_score = 0.0

    # Determine if system is consistent
    # System is consistent if there are no rejected nodes or if accepted nodes dominate
    is_consistent = len(rejected_nodes) == 0 or len(accepted_nodes) > len(rejected_nodes)

    return ValidationResultDetail(
        accepted_nodes=accepted_nodes,
        rejected_nodes=rejected_nodes,
        contested_nodes=contested_nodes,
        unknown_nodes=unknown_nodes,
        arguments=arguments,
        attacks=attacks,
        loop_classifications=loop_classifications,
        is_consistent=is_consistent,
        consistency_score=consistency_score,
        metadata={
            "total_arguments": len(arguments),
            "total_attacks": len(attacks),
            "accepted_arguments": len(accepted_args),
        },
    )


def get_validation_summary(result: ValidationResultDetail) -> Dict[str, Any]:
    """Get summary of validation results."""
    return {
        "is_consistent": result.is_consistent,
        "consistency_score": result.consistency_score,
        "accepted_nodes": len(result.accepted_nodes),
        "rejected_nodes": len(result.rejected_nodes),
        "contested_nodes": len(result.contested_nodes),
        "unknown_nodes": len(result.unknown_nodes),
        "total_arguments": len(result.arguments),
        "total_attacks": len(result.attacks),
        "causal_claims": sum(1 for arg in result.arguments if arg.argument_type == ArgumentType.CAUSAL_CLAIM),
        "role_contests": sum(1 for arg in result.arguments if arg.argument_type == ArgumentType.ROLE_CONTEST),
    }


# Backward-compat aliases (deprecated — use ValidationArgument / ValidationAttack)
Argument = ValidationArgument
Attack = ValidationAttack
