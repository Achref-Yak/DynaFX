from __future__ import annotations

__doc__ = """Entity attribute extraction — scans proposition texts for structured entity properties.

Patterns matched:
- Currency amounts -> HAS_ATTRIBUTE
- Email addresses -> CONTACT_OF
- Phone numbers -> CONTACT_OF
- Age expressions -> HAS_ATTRIBUTE
- Employer mentions -> EMPLOYED_BY
- Address/location mentions -> LOCATED_AT
- Name declarations -> HAS_ATTRIBUTE
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from cognitive_engine.core.models import BfoCategory, Edge, EdgeType, Graph, Node, NodeType

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_SALARY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_AGE_RE = re.compile(r"(\d+)[-\s]year[-\s]old")
_NAMED_RE = re.compile(r"named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")
_EMPLOYER_RE = re.compile(
    r"(?:works?|working|employed)\s+(?:at|for)\s+([A-Z][\w\s]+?)(?:\.|,|and|for|as|with|\s+the\s)",
)
_ADDRESS_RE = re.compile(
    r"(?:live[sd]?|living|reside[sd]?|residing)\s+(?:at|in)\s+(.+?)(?:\.|,| and| \|)",
)


def extract_attributes(
    graph: Graph,
    spacy_docs: list | None = None,
    coref_data: dict | None = None,
) -> list[tuple[UUID, str, str, UUID]]:
    """Scan proposition nodes for entity attributes and record link records.

    Returns list of (entity_node_id, attr_name, attr_value, proposition_node_id)
    tuples so the caller can create edges.

    Does NOT modify the graph directly — returns records for the caller to apply.

    Also populates graph.entities[].attributes dicts for each matched entity.
    """
    entity_nodes: list[Node] = [n for n in graph.nodes.values() if n.type == NodeType.ENTITY]
    proposition_nodes: list[Node] = [
        n for n in graph.nodes.values() if n.type in (NodeType.CLAIM, NodeType.EVIDENCE)
    ]
    entity_name_map: dict[str, list[Node]] = {}
    for en in entity_nodes:
        key = en.text.lower().strip()
        entity_name_map.setdefault(key, []).append(en)

    results: list[tuple[UUID, str, str, UUID]] = []

    for pn in proposition_nodes:
        text = pn.text
        matched_entities = _find_entities_in_text(text, entity_name_map)

        for entity_node in matched_entities:
            _process_entity_attributes(text, entity_node, pn, results, graph, entity_name_map)

    return results


def _find_entities_in_text(
    text: str, entity_name_map: dict[str, list[Node]]
) -> list[Node]:
    """Find which entity nodes are mentioned in this proposition text."""
    text_lower = text.lower()
    matched: list[Node] = []
    seen: set[UUID] = set()
    for name, nodes in entity_name_map.items():
        if name and name in text_lower:
            for node in nodes:
                if node.id not in seen:
                    matched.append(node)
                    seen.add(node.id)
    return matched


def _process_entity_attributes(
    text: str,
    entity_node: Node,
    prop_node: Node,
    results: list[tuple[UUID, str, str, UUID]],
    graph: Graph,
    entity_name_map: dict[str, list[Node]],
) -> None:
    entity_kind = entity_node.metadata.get("ner_label", "")
    entity_id = entity_node.metadata.get("entity_id", entity_node.id)

    email_match = _EMAIL_RE.search(text)
    if email_match:
        _record_attr(results, entity_id, "email", email_match.group(), prop_node.id, EdgeType.CONTACT_OF)
        _ensure_extra_entity(graph, email_match.group(), "EMAIL", entity_name_map)

    phone_match = _PHONE_RE.search(text)
    if phone_match:
        _record_attr(results, entity_id, "phone", phone_match.group(), prop_node.id, EdgeType.CONTACT_OF)
        _ensure_extra_entity(graph, phone_match.group(), "PHONE", entity_name_map)

    salary_match = _SALARY_RE.search(text)
    if salary_match:
        raw = salary_match.group().replace("$", "").replace(",", "")
        _record_attr(results, entity_id, "salary", raw, prop_node.id, EdgeType.HAS_ATTRIBUTE)

    age_match = _AGE_RE.search(text)
    if age_match:
        _record_attr(results, entity_id, "age", age_match.group(1), prop_node.id, EdgeType.HAS_ATTRIBUTE)

    name_match = _NAMED_RE.search(text)
    if name_match:
        _record_attr(results, entity_id, "full_name", name_match.group(1), prop_node.id, EdgeType.HAS_ATTRIBUTE)

    if entity_kind == "PERSON" or entity_kind == "ORG":
        employer_match = _EMPLOYER_RE.search(text)
        if employer_match:
            employer_name = employer_match.group(1).strip().rstrip(",")
            _record_attr(results, entity_id, "employer", employer_name, prop_node.id, EdgeType.EMPLOYED_BY)

    address_match = _ADDRESS_RE.search(text)
    if address_match:
        addr = address_match.group(1).strip().rstrip(",")
        _record_attr(results, entity_id, "address", addr, prop_node.id, EdgeType.LOCATED_AT)


def _record_attr(
    results: list[tuple[UUID, str, str, UUID]],
    entity_id: UUID,
    attr_name: str,
    attr_value: str,
    prop_id: UUID,
    edge_type: EdgeType,
) -> None:
    results.append((entity_id, attr_name, attr_value, prop_id))


def _ensure_extra_entity(
    graph: Graph,
    value: str,
    kind: str,
    entity_name_map: dict[str, list[Node]],
) -> None:
    """Create an ENTITY node for extracted contact values (email, phone)."""
    key = value.lower().strip()
    if key in entity_name_map:
        return
    from uuid import uuid4
    node = Node(
        id=uuid4(),
        type=NodeType.ENTITY,
        text=value,
        metadata={"ner_label": kind, "entity_kind": kind},
        bfo_category=BfoCategory.QUALITY if kind == "PHONE" else BfoCategory.INFORMATION_CONTENT_ENTITY,
    )
    graph.nodes[node.id] = node
    entity_name_map[key] = [node]
