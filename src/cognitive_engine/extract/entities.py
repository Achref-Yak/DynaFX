from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from cognitive_engine.core.models import Entity, Graph, Span

logger = logging.getLogger(__name__)

_ADDITIONAL_KINDS: Dict[str, str] = {
    "access": "Access",
    "algorithm": "Algorithm",
    "analysis": "Analysis",
    "application": "Application",
    "approach": "Approach",
    "architecture": "Architecture",
    "argument": "Argument",
    "attack": "Attack",
    "audit": "Audit",
    "authentication": "Authentication",
    "client": "Client",
    "compatibility": "Compatibility",
    "compliance": "Compliance",
    "complexity": "Complexity",
    "component": "Component",
    "config": "Config",
    "configuration": "Configuration",
    "consistency": "Consistency",
    "data": "Data",
    "database": "Database",
    "deadline": "Deadline",
    "endpoint": "Endpoint",
    "environment": "Environment",
    "failure": "Failure",
    "feature": "Feature",
    "function": "Function",
    "gateway": "Gateway",
    "infrastructure": "Infrastructure",
    "instance": "Instance",
    "interface": "Interface",
    "interval": "Interval",
    "layer": "Layer",
    "library": "Library",
    "limit": "Limit",
    "middleware": "Middleware",
    "module": "Module",
    "network": "Network",
    "node": "Node",
    "optimization": "Optimization",
    "pipeline": "Pipeline",
    "platform": "Platform",
    "policy": "Policy",
    "process": "Process",
    "protocol": "Protocol",
    "request": "Request",
    "resource": "Resource",
    "role": "Role",
    "route": "Route",
    "schema": "Schema",
    "security": "Security",
    "server": "Server",
    "service": "Service",
    "session": "Session",
    "strategy": "Strategy",
    "system": "System",
    "task": "Task",
    "team": "Team",
    "template": "Template",
    "throttling": "Throttling",
    "token": "Token",
    "user": "User",
    "validator": "Validator",
    "version": "Version",
    "worker": "Worker",
}


def _ner_kind_for_span(
    doc: "spacy.tokens.Doc", chunk_start: int, chunk_abs_start: int, chunk_abs_end: int,
) -> Optional[str]:
    for ent in doc.ents:
        ent_abs_start = chunk_start + ent.start_char
        ent_abs_end = chunk_start + ent.end_char
        if chunk_abs_start < ent_abs_end and ent_abs_start < chunk_abs_end:
            return ent.label_
    return None


def _noun_kind(root: "spacy.tokens.Token") -> str:
    lemma = root.lemma_.lower()
    mapped = _ADDITIONAL_KINDS.get(lemma)
    if mapped is not None:
        return mapped
    if root.pos_ == "PROPN":
        return root.text
    if root.pos_ == "NOUN":
        return root.lemma_.title()
    return "Entity"


def extract_entities(
    graph: Graph,
    docs: List["spacy.tokens.Doc"],
    source_text: str,
    seen_spans: Set[Tuple[int, int]] | None = None,
    **kwargs,
) -> None:
    if seen_spans is None:
        seen_spans = set()

    for doc in docs:
        chunk_start = doc.user_data.get("chunk_start_char", 0)
        for chunk in doc.noun_chunks:
            start = chunk_start + chunk.start_char
            end = chunk_start + chunk.end_char
            key = (start, end)
            if key in seen_spans:
                continue
            seen_spans.add(key)

            text = source_text[start:end]
            if not text.strip():
                continue

            ner_label = _ner_kind_for_span(doc, chunk_start, start, end)
            kind = ner_label if ner_label is not None else _noun_kind(chunk.root)

            entity = Entity(
                id=uuid4(),
                kind=kind,
                name=text.strip(),
                spans=[Span(start=start, end=end, text=text)],
            )
            graph.entities[entity.id] = entity
