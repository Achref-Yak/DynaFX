from __future__ import annotations

import json
import logging
import re
from typing import List, Optional
from uuid import uuid4

from pydantic_ai import Agent

from cognitive_engine.models import (
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeType,
    Opinion,
    ReasoningMode,
    Span,
)

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are an expert cognitive science analyst. Your task is to extract a structured reasoning graph from the given text.

Follow these steps:
1. Identify key claims, evidence, and conditions in the text (L1: text spans)
2. For each span, determine its role: CLAIM (assertion needing support), EVIDENCE (data/observation), CONDITION (qualifier/constraint)
3. Determine the relationships between spans: SUPPORTS, CONTRADICTS, QUALIFIES, INFERS, JUSTIFIES
4. Assign category levels (1=Necessity, 2=Fact, 3=Belief, 4=Concept) — a higher category cannot imply a lower one

Output ONLY raw JSON — no markdown fences, no extra text, no code blocks.
The JSON must have a top-level object with keys "nodes" (array) and "edges" (array).
Do NOT wrap the output in any envelope like `{"ExtractionResult": ...}`.

Constraints:
- Each edge must reference valid source_id and target_id
- Categories must be integers 1-4: 1=Necessity, 2=Fact, 3=Belief, 4=Concept
- Source category cannot be higher than target category
- Each node must have a unique id
- Each edge must have a unique id
"""  # noqa: E501


_EXAMPLE = """{"nodes":[{"id":"a1b2c3d4-...","type":"CLAIM","text":"The system will improve throughput","category":2,"abstraction_level":2,"span":{"start":10,"end":45,"text":"The system will improve throughput"}}],"edges":[{"id":"e5f6g7h8-...","source_id":"a1b2c3d4-...","target_id":"i9j0k1l2-...","type":"SUPPORTS"}]}"""  # noqa: E501

_agent_cache: dict[str, Agent] = {}


def _get_extraction_agent(model_name: str | None = None) -> Agent:
    from pydantic_ai.models.groq import GroqModel

    key = model_name or "default"
    if key not in _agent_cache:
        _agent_cache[key] = Agent(
            GroqModel(model_name or "llama-3.3-70b-versatile"),
            system_prompt=(
                f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                "Example (single line, no markdown):\n"
                f"{_EXAMPLE}"
            ),
        )
    return _agent_cache[key]


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        stripped = m.group(0)
    return json.loads(stripped)


async def extract_graph(
    text: str,
    feedback: str | None = None,
    model_name: str | None = None,
) -> Graph:
    agent = _get_extraction_agent(model_name)

    prompt = f"Extract a reasoning graph from this text:\n\n{text}"
    if feedback:
        prompt += f"\n\nPrevious feedback to address:\n{feedback}"

    result = await agent.run(prompt)
    data = _extract_json(str(result.output))

    return _parse_extraction(data, text)


def _parse_extraction(data: dict, source_text: str) -> Graph:
    id_map: dict[str, UUID] = {}
    nodes: dict[UUID, Node] = {}
    for nd in data.get("nodes", []):
        llm_id = nd.get("id", "") or ""
        node_uuid = uuid4()
        id_map[llm_id] = node_uuid
        span_data = nd.get("span")
        span = Span(
            start=span_data.get("start", 0),
            end=span_data.get("end", 0),
            text=span_data.get("text", ""),
        ) if span_data else None
        nodes[node_uuid] = Node(
            id=node_uuid,
            type=NodeType[nd.get("type", "CLAIM")],
            text=nd.get("text", ""),
            span=span,
            category=nd.get("category", 2),
            abstraction_level=nd.get("abstraction_level", 1),
        )

    edges: list[Edge] = []
    for ed in data.get("edges", []):
        src = ed.get("source_id", "")
        tgt = ed.get("target_id", "")
        warrant = None
        b = ed.get("warrant_belief")
        d = ed.get("warrant_disbelief")
        if b and d and len(b) == 4 and len(d) == 4:
            warrant = (tuple(b), tuple(d))
        edges.append(
            Edge(
                id=uuid4(),
                source_id=id_map.get(src, uuid4()),
                target_id=id_map.get(tgt, uuid4()),
                type=EdgeType[ed.get("type", "SUPPORTS")],
                warrant=warrant,
            )
        )

    return Graph(
        nodes=nodes,
        edges=edges,
        mode=ReasoningMode.ARGUMENT,
        source_text=source_text,
    )
