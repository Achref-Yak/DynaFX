"""Parameter binding with SL opinions for system dynamics.

Every parameter gets an SL opinion:
- Explicit values → low uncertainty, b/d reflect confidence
- Inferred/absent → high uncertainty
- An unparameterized system isn't a failure state — it's a system with visible, first-class uncertainty
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from dynafx.core.models import (
    Graph,
    Node,
    NodeType,
    Opinion,
    Parameter,
)

logger = logging.getLogger(__name__)


def _extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text, handling various formats.

    Examples:
        "1200 units" → 1200.0
        "60%" → 60.0
        "5-7 day" → 5.0 (takes lower bound)
        "3x cost" → 3.0
        "200 units/day" → 200.0
    """
    if not text:
        return None

    # Remove common units and suffixes
    cleaned = text.lower().strip()
    cleaned = re.sub(r'\s*(units?|per\s+\w+|a\s+day|daily|monthly|yearly)\s*$', '', cleaned)

    # Handle percentage
    if '%' in cleaned:
        cleaned = cleaned.replace('%', '')
        try:
            return float(cleaned)
        except ValueError:
            return None

    # Handle multiplier (e.g., "3x")
    match = re.search(r'(\d+(?:\.\d+)?)\s*x', cleaned)
    if match:
        return float(match.group(1))

    # Handle range (e.g., "5-7 day")
    match = re.search(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', cleaned)
    if match:
        return float(match.group(1))  # Take lower bound

    # Handle plain number
    match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if match:
        return float(match.group(1))

    return None


def _extract_percentage(text: str) -> Optional[float]:
    """Extract percentage value from text.

    Examples:
        "60%" → 0.6
        "80% of inventory" → 0.8
        "15% increase" → 0.15
    """
    if not text:
        return None

    match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if match:
        return float(match.group(1)) / 100.0

    return None


def _infer_confidence_from_context(node: Node, graph: Graph) -> tuple[float, float, float]:
    """Infer confidence (belief, disbelief, uncertainty) from node context.

    Returns:
        (belief, disbelief, uncertainty) tuple
    """
    text = node.text.lower() if node.text else ""

    # High confidence for explicit statements
    if any(phrase in text for phrase in ["is", "are", "currently", "holds", "has"]):
        return (0.8, 0.1, 0.1)

    # Medium confidence for estimates
    if any(phrase in text for phrase in ["about", "approximately", "roughly", "around"]):
        return (0.6, 0.2, 0.2)

    # Low confidence for uncertainty markers
    if any(phrase in text for phrase in ["may", "might", "could", "possibly", "uncertain"]):
        return (0.4, 0.3, 0.3)

    # Very low confidence for questions or unknowns
    if any(phrase in text for phrase in ["?", "unknown", "unclear"]):
        return (0.2, 0.4, 0.4)

    # Default medium confidence
    return (0.5, 0.2, 0.3)


def bind_parameters(graph: Graph) -> None:
    """Bind SL opinions to all parameters in graph.

    Every parameter gets an SL opinion:
    - Explicit values → low uncertainty, b/d reflect confidence
    - Inferred/absent → high uncertainty
    - An unparameterized system isn't a failure state — it's a system with visible, first-class uncertainty

    This function modifies the graph in place.
    """
    nodes_processed = 0
    parameters_bound = 0

    for node in graph.nodes.values():
        # Skip entity nodes
        if node.type == NodeType.ENTITY:
            continue

        nodes_processed += 1

        # Check if node already has a parameter with explicit value
        existing_param = node.metadata.get('parameter')
        if existing_param is not None and existing_param.value is not None:
            # Already parameterized, just ensure opinion exists
            if existing_param.opinion is None:
                # Infer confidence from context
                b, d, u = _infer_confidence_from_context(node, graph)
                existing_param.opinion = Opinion(belief=b, disbelief=d, uncertainty=u)
                parameters_bound += 1
            continue

        # Try to extract numeric value from node text
        value = _extract_numeric_value(node.text)

        # If no numeric value, try to extract percentage
        if value is None:
            pct = _extract_percentage(node.text)
            if pct is not None:
                value = pct

        # Determine confidence based on whether value was explicit
        if value is not None:
            # Explicit value found — low uncertainty
            b, d, u = 0.8, 0.1, 0.1
        else:
            # No explicit value — high uncertainty
            b, d, u = 0.3, 0.3, 0.4

        # Create parameter
        parameter = Parameter(
            value=value,
            opinion=Opinion(belief=b, disbelief=d, uncertainty=u),
        )
        node.metadata['parameter'] = parameter
        parameters_bound += 1

    logger.info(
        "Bound parameters for %d nodes (%d parameters total)",
        nodes_processed, parameters_bound,
    )


def get_parameter_summary(graph: Graph) -> dict:
    """Get a summary of parameter binding status.

    Returns:
        Dictionary with parameter statistics
    """
    total_nodes = 0
    parameterized_nodes = 0
    explicit_values = 0
    inferred_values = 0
    no_values = 0
    avg_uncertainty = 0.0
    uncertainty_count = 0

    for node in graph.nodes.values():
        if node.type == NodeType.ENTITY:
            continue

        total_nodes += 1
        param = node.metadata.get('parameter')

        if param is not None:
            parameterized_nodes += 1
            if param.value is not None:
                explicit_values += 1
            else:
                inferred_values += 1
            if param.opinion is not None:
                avg_uncertainty += param.opinion.uncertainty
                uncertainty_count += 1
        else:
            no_values += 1

    if uncertainty_count > 0:
        avg_uncertainty /= uncertainty_count

    return {
        "total_nodes": total_nodes,
        "parameterized_nodes": parameterized_nodes,
        "explicit_values": explicit_values,
        "inferred_values": inferred_values,
        "no_values": no_values,
        "parameterization_rate": parameterized_nodes / total_nodes if total_nodes > 0 else 0.0,
        "average_uncertainty": avg_uncertainty,
    }
