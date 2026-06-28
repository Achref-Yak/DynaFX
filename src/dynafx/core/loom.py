from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from dynafx.core.diff import CycleDiff
from dynafx.core.models import Graph
from dynafx.core.state import State
from dynafx.core.workflow import WorkflowDefinition, WorkflowEngine


@dataclass
class Weave:
    state: State
    log: list[CycleDiff] = field(default_factory=list)
    definition: WorkflowDefinition = field(
        default_factory=lambda: WorkflowDefinition(id="", name="")
    )

    @property
    def graph(self) -> Graph:
        return self.state.graph

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively convert UUID keys to hex strings for JSON."""
        if isinstance(obj, dict):
            return {
                k.hex if isinstance(k, UUID) else k: Weave._json_safe(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [Weave._json_safe(v) for v in obj]
        if isinstance(obj, UUID):
            return obj.hex
        return obj

    def to_dict(self) -> dict:
        metadata = self._json_safe(dict(self.state.metadata))
        metadata.pop("source_text", None)
        metadata.pop("text", None)

        # Final graph node states (with post-propagation beliefs)
        final_nodes = {}
        for nid, node in self.state.graph.nodes.items():
            op = node.opinion
            final_nodes[nid.hex] = {
                "id": nid.hex,
                "type": node.type.name,
                "text": node.text[:80],
                "opinion": [round(op.belief, 3), round(op.disbelief, 3),
                            round(op.uncertainty, 3), round(op.prior, 3)],
            }

        return {
            "definition": self.definition.to_dict(),
            "graph_summary": {
                "nodes": len(self.state.graph.nodes),
                "edges": len(self.state.graph.edges),
                "entities": len(self.state.graph.entities),
            },
            "final_nodes": final_nodes,
            "log_count": len(self.log),
            "log": [d.to_compact_dict() for d in self.log],
            "metadata": metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


def weave(
    text_or_state: str | State,
    steps: dict[str, str | dict],
    params: dict[str, Any] | None = None,
    name: str = "weave",
) -> Weave:
    if isinstance(text_or_state, str):
        state = State(graph=Graph(source_text=text_or_state))
    else:
        state = text_or_state

    raw: dict[str, Any] = {
        "name": name,
        "steps": {},
    }
    for sid, spec in steps.items():
        if isinstance(spec, str):
            raw["steps"][sid] = {
                "ref": spec,
                "kind": "operator",
                "params": {},
                "depends_on": [],
            }
        elif isinstance(spec, dict):
            raw["steps"][sid] = dict(spec)

    definition = WorkflowDefinition.from_dict(raw)
    engine = WorkflowEngine()
    log: list[CycleDiff] = []

    def on_diff(diff: CycleDiff) -> None:
        log.append(diff)

    final_state = engine.run_sync(definition, state, on_diff=on_diff)
    return Weave(state=final_state, log=log, definition=definition)
