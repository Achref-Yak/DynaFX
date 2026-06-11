from __future__ import annotations

import json
import logging
import re
from typing import List

from pydantic_ai import Agent

from cognitive_engine.models import Graph, ReviewResult, Violation
from cognitive_engine.validators import format_violations

logger = logging.getLogger(__name__)


REVIEWER_SYSTEM_PROMPT = """You are a reviewer of reasoning graphs extracted from text.

You will receive:
1. The original source text
2. The extracted graph (JSON)
3. Results from deterministic validators (pre-computed)

Your job is to check for MAJOR semantic issues:
- Are the extracted claims faithful to the source text?
- Are the edge types appropriate for the relationship between nodes?
- Are there any missing or spurious nodes/edges?

IGNORE minor issues: exact span offsets, abstraction levels, or edge type
ambiguity when multiple interpretations are reasonable. Only reject if the
graph is clearly wrong or unfaithful to the source.

Do NOT re-check the deterministic validator results — they have already been applied.
Focus only on semantic / textual issues.

Return raw JSON only — no markdown fences, no extra text.
The JSON must be a flat object with exactly these keys:
  "status": "accept" or "reject"
  "feedback": explanation of issues found (if any)
"""  # noqa: E501


_agent_cache: dict[str, Agent] = {}


def _get_reviewer_agent(model_name: str | None = None) -> Agent:
    from pydantic_ai.models.groq import GroqModel

    key = model_name or "default"
    if key not in _agent_cache:
        _agent_cache[key] = Agent(
            GroqModel(model_name or "llama-3.3-70b-versatile"),
            system_prompt=REVIEWER_SYSTEM_PROMPT,
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


async def review_graph(
    graph: Graph,
    source_text: str,
    pre_computed_violations: List[Violation] | None = None,
    model_name: str | None = None,
) -> ReviewResult:
    agent = _get_reviewer_agent(model_name)

    violations_text = format_violations(pre_computed_violations or [])

    prompt = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"EXTRACTED GRAPH:\n{graph.to_json()}\n\n"
        f"VALIDATOR RESULTS (already applied):\n{violations_text}\n\n"
        "Review the graph for semantic issues only."
    )

    result = await agent.run(prompt)
    data = _extract_json(str(result.output))

    status = data.get("status", "reject")
    feedback = data.get("feedback", "")
    return ReviewResult(
        status=status,
        feedback=feedback,
        violations=pre_computed_violations or [],
    )
