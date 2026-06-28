"""⏣ (Stock-Flow) operator — Accumulation and rate-of-change modeling.

Identifies stock nodes (accumulators) and flow edges (rates) in a
causal graph. Computes inflow/outflow rates, accumulation levels,
and doubling times for system dynamics analysis.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from dynafx.core.models import Graph
from dynafx.core.state import State


class StockFlowOperator:
    """⏣: Stock-Flow analysis.

    Analyzes the graph as a stock-flow system where:
    - Nodes with high in-degree and accumulated opinion are "stocks"
    - Edges with high weight/confidence are "flows"
    - The ratio of inflow to outflow determines accumulation

    Useful for system dynamics modeling, especially with causal
    feedback loops and temporal reasoning.
    """
    name = "stock_flow"

    def __call__(
        self,
        state: State,
        min_stock_confidence: float = 0.3,
        **kwargs,
    ) -> State:
        if not state.graph.nodes:
            state.metadata["stock_flow"] = {"status": "empty_graph"}
            return state

        stocks = self._identify_stocks(state.graph, min_stock_confidence)
        flows = self._compute_flows(state.graph)
        accumulations = self._compute_accumulations(state.graph, stocks)
        doubling_times = self._compute_doubling_times(accumulations, flows)

        state.metadata["stock_flow"] = {
            "stocks": stocks,
            "flows": flows,
            "accumulations": accumulations,
            "doubling_times": doubling_times,
            "total_stocks": len(stocks),
            "total_flows": len(flows.get("all_flows", [])),
        }

        stock_labels = [s["text"][:50] for s in stocks[:5]]
        accum_summary = [f"{a['node_id'][:8]}: {a['net_accumulation']:+.3f} {'growing' if a['growing'] else 'draining'}" for a in accumulations[:5]]
        dt_summary = [f"{d['node_id'][:8]}: double in {d['doubling_time']:.1f}" for d in doubling_times[:3]]
        state.record(
            self.name,
            f"Identified {len(stocks)} stock variables (accumulators) in the system dynamics. "
            f"Total flows mapped: {len(flows.get('all_flows', []))}. "
            f"Key stocks: {'; '.join(stock_labels)}. "
            f"Net accumulation: {'; '.join(accum_summary)}. "
            f"{'Doubling time estimates: ' + '; '.join(dt_summary) + '. ' if dt_summary else ''}"
            f"The system shows {'net growth' if any(a['growing'] for a in accumulations) else 'net draining'} across identified stocks.",
        )
        return state

    def _identify_stocks(
        self, graph: Graph, min_confidence: float,
    ) -> list[dict]:
        """Score nodes by accumulation potential.

        A stock node has:
        - Multiple incoming edges (accumulation)
        - High belief (opinion[0]) suggesting persisted state
        - High uncertainty (opinion[2]) suggesting variable level
        """
        scores: list[dict] = []
        for nid, node in graph.nodes.items():
            in_degree = sum(
                1 for e in graph.edges.values()
                if e.target_id == nid
            )
            out_degree = sum(
                1 for e in graph.edges.values()
                if e.source_id == nid
            )
            if in_degree == 0 and out_degree == 0:
                continue

            b, d, u, a = node.opinion
            accumulation_score = (
                in_degree * 0.3 +
                b * 0.3 +
                u * 0.2 +
                (in_degree / max(in_degree + out_degree, 1)) * 0.2
            )
            scores.append({
                "node_id": nid.hex,
                "text": node.text[:60],
                "in_degree": in_degree,
                "out_degree": out_degree,
                "belief": b,
                "uncertainty": u,
                "accumulation_score": round(accumulation_score, 4),
            })

        scores.sort(key=lambda x: x["accumulation_score"], reverse=True)
        return scores

    def _compute_flows(self, graph: Graph) -> dict[str, list[dict]]:
        """Categorize edges by flow type.

        Inflow = edges targeting a stock
        Outflow = edges sourced from a stock
        Throughflow = edges between non-stock nodes
        """
        in_flows: list[dict] = []
        out_flows: list[dict] = []
        through_flows: list[dict] = []

        for eid, edge in graph.edges.items():
            flow = {
                "edge_id": eid.hex,
                "source_id": edge.source_id.hex,
                "target_id": edge.target_id.hex,
                "type": edge.type.name,
                "weight": edge.weight,
                "confidence": edge.confidence,
            }
            in_flows.append(flow)

        return {
            "all_flows": in_flows,
            "total": len(in_flows),
        }

    def _compute_accumulations(
        self, graph: Graph, stocks: list[dict],
    ) -> list[dict]:
        """Compute net accumulation for each stock."""
        results = []
        for stock in stocks:
            nid = UUID(hex=stock["node_id"])
            inflow = sum(
                e.weight for e in graph.edges.values()
                if e.target_id == nid
            )
            outflow = sum(
                e.weight for e in graph.edges.values()
                if e.source_id == nid
            )
            net = inflow - outflow
            results.append({
                "node_id": stock["node_id"],
                "inflow": round(inflow, 4),
                "outflow": round(outflow, 4),
                "net_accumulation": round(net, 4),
                "growing": net > 0,
            })
        return results

    def _compute_doubling_times(
        self, accumulations: list[dict], flows: dict,
    ) -> list[dict]:
        """Estimate doubling time for growing stocks."""
        results = []
        for acc in accumulations:
            if acc["growing"] and acc["net_accumulation"] > 0:
                rate = acc["net_accumulation"] / max(acc["inflow"], 1e-6)
                doubling = 0.693 / max(rate, 1e-6)
                results.append({
                    "node_id": acc["node_id"],
                    "doubling_time": round(doubling, 4),
                    "growth_rate": round(rate, 4),
                })
        return results
