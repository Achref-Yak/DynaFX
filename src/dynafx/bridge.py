"""KBSimBridge — formal two-way bridge between TripleStore and SysdModel.

Pre-flight:  KB params → SD simulation params (claim_map)
Mid-flight:  KB_QUERY / KB_ASSERT builtins available via kb= parameter
Post-flight: Simulation results → opinion-bearing triples (evidence)

Usage::

    from dynafx.bridge import KBSimBridge

    bridge = KBSimBridge(store)

    # Pre-flight: extract KB beliefs as simulation params
    params = bridge.params_from_kb(CLAIM_MAP)

    # Mid-flight: run with KB-connected builtins
    result = model.simulate(params=params, kb=store)

    # Post-flight: extract evidence from result
    triples = bridge.evidence_from_result(result, EVIDENCE_MAP)
    for t in triples:
        store.add(t, graph="simulation")
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dynafx.knowledge import parse_sparql, sparql_evaluate
from dynafx.knowledge.model import (
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.production import Action, ActionResult
from dynafx.knowledge.store import TripleStore

logger = logging.getLogger(__name__)

NS_PROV = "http://www.w3.org/ns/prov#"
NS_SIM = "http://dynafx.org/simulation#"


class KBSimBridge:
    """Formal bridge between TripleStore and SysdModel.

    Provides three integration modes:
        1. params_from_kb / params_for_class — KB → simulation params (pre-flight)
        2. run_with_kb     — simulation with KB-connected builtins (mid-flight)
        3. evidence_from_result / evidence_for_stock — simulation → KB triples (post-flight)

    Plus load_queries for external SPARQL management, provenance, and comparison.
    """

    def __init__(self, store: TripleStore, ns_base: str = "http://dynafx.org/"):
        self.store = store
        self.ns_base = ns_base.rstrip("/") + "/"

    def params_from_kb(
        self,
        claim_map: list[tuple[NamedNode, NamedNode, object, str]],
        default: float = 0.5,
        exclude_graphs: set[str] | None = None,
        type_coerce: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Map KB triples to simulation parameters.

        For each (subject, predicate, object, param_name) entry, finds
        matching triples and returns the object value as a typed param.

        Args:
            claim_map: List of (subject, predicate, object_literal, param_name) tuples.
            default: Default value when no matching triple exists.
            exclude_graphs: Graphs to skip.
            type_coerce: Optional dict param_name → type string
                ("float", "int", "str", "bool").

        Returns:
            dict mapping param_name → value (float, int, str, or bool).
        """
        if exclude_graphs is None:
            exclude_graphs = {"schema", "meta", "fused"}
        params: dict[str, Any] = {}
        for subj, pred, obj, param_name in claim_map:
            coerce_type = (type_coerce or {}).get(param_name, "float")
            raw_value: Any = None
            has_any = False
            for g in self.store.graphs():
                if g in exclude_graphs:
                    continue
                for t in self.store.triples(TriplePattern(subj, pred, obj), graph=g):
                    if t.object_ is not None:
                        raw_value = getattr(t.object_, 'value', t.object_)
                        has_any = True
                        break

            if not has_any:
                params[param_name] = default
                continue

            if coerce_type == "float":
                try:
                    params[param_name] = float(raw_value)
                except (ValueError, TypeError):
                    params[param_name] = default
            elif coerce_type == "int":
                try:
                    params[param_name] = int(raw_value)
                except (ValueError, TypeError):
                    params[param_name] = int(default)
            elif coerce_type == "str":
                params[param_name] = str(raw_value)
            elif coerce_type == "bool":
                if isinstance(raw_value, bool):
                    params[param_name] = raw_value
                elif isinstance(raw_value, str):
                    params[param_name] = raw_value.lower() in ('true', '1', 'yes')
                else:
                    params[param_name] = bool(raw_value)
            else:
                params[param_name] = raw_value
        return params

    def evidence_from_result(
        self,
        result: Any,
        evidence_map: list[tuple[str, NamedNode, NamedNode, Callable[[list[float], list[float]], float]]],
        graph: str = "simulation",
    ) -> list[Triple]:
        """Convert simulation results to KB triples.

        Args:
            result: SysdModelResult from model.simulate().
            evidence_map: List of (stock_name, subject, predicate, scoring_fn)
                tuples. The scoring_fn receives (initial_values, final_values)
                and returns a numeric value.
            graph: Named graph to tag triples with (used at add time).

        Returns:
            List of Triple objects with the scored value as literal object.
        """
        triples: list[Triple] = []
        for stock_name, subj, pred, scoring_fn in evidence_map:
            values = result.values.get(stock_name, [])
            if not values:
                continue
            initial_v = values[:max(1, len(values) // 10)]
            final_v = values[-max(1, len(values) // 10):]
            score = scoring_fn(initial_v, final_v)
            obj = Literal(score, datatype="http://www.w3.org/2001/XMLSchema#double")
            triple = Triple(subject=subj, predicate=pred, object_=obj)
            triples.append(triple)
        return triples

    def params_for_class(
        self,
        class_iri: str,
        subject_filter: dict[str, Any] | None = None,
        naming: str = "class_prefix",
        exclude_predicates: set[str] | None = None,
        default: float = 0.5,
    ) -> dict[str, Any]:
        """Introspect KB to extract numeric params for all instances of a class.

        Queries ``?s rdf:type <class_iri>`` to discover subjects, then
        iterates their numeric literal predicates to build a flat param dict.

        Args:
            class_iri: Full IRI string of the RDF class (e.g. ``epc:Portfolio``
                must already be a full IRI, not a prefixed name).
            subject_filter: Optional dict of ``{predicate_iri: value}`` to
                narrow which subjects are included.
            naming: ``"class_prefix"`` (default) prepends local class name;
                ``"subject_prefix"`` prepends the subject local name.
            exclude_predicates: Set of predicate IRIs to skip.
            default: Default value when a predicate has no matching triples.

        Returns:
            dict mapping ``{prefix}_{predicate_localname}`` → float value.
        """
        if exclude_predicates is None:
            exclude_predicates = set()
        ns_rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        rdf_type_node = NamedNode(ns_rdf)
        class_node = NamedNode(class_iri)

        subjects: list[NamedNode] = []
        for t in self.store.triples(TriplePattern(None, rdf_type_node, class_node)):
            subjects.append(t.subject)

        if subject_filter:
            filtered: list[NamedNode] = []
            for subj in subjects:
                match = True
                for pred_str, val_str in subject_filter.items():
                    pred_node = NamedNode(pred_str)
                    matches = list(self.store.triples(TriplePattern(subj, pred_node, None)))
                    if not any(
                        str(getattr(t.object_, "value", t.object_)) == str(val_str)
                        for t in matches
                    ):
                        match = False
                        break
                if match:
                    filtered.append(subj)
            subjects = filtered

        if not subjects:
            return {}

        class_local = class_iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

        params: dict[str, Any] = {}
        for subj in subjects:
            subj_local = str(getattr(subj, "iri", subj)).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            for t in self.store.triples(TriplePattern(subj, None, None)):
                pred_iri = str(getattr(t.predicate, "iri", t.predicate))
                if pred_iri == ns_rdf:
                    continue
                if pred_iri in exclude_predicates:
                    continue
                obj = t.object_
                raw = getattr(obj, "value", obj)
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    continue
                pred_local = pred_iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                if naming == "subject_prefix":
                    key = f"{subj_local}_{pred_local}"
                else:
                    if len(subjects) == 1:
                        key = f"{class_local}_{pred_local}"
                    else:
                        key = f"{class_local}_{subj_local}_{pred_local}"
                key = key.replace("-", "_").replace(".", "_")
                params[key] = val

        return params

    def evidence_for_stock(
        self,
        stock_name: str,
        subject: NamedNode,
        predicate: NamedNode,
        result: Any,
        method: str = "percentile",
        threshold: float = 0.5,
    ) -> Triple:
        """Convert a simulation stock to a KB triple.

        Computes the score using the chosen method:
            ``"percentile"`` — normalises the stock's final values across
                all stocks in the result.
            ``"delta"`` — absolute change relative to initial value.
            ``"threshold"`` — 1.0 if final > threshold else 0.0.

        Args:
            stock_name: Name of the stock in the simulation result.
            subject: Subject IRI for the output triple.
            predicate: Predicate IRI for the output triple.
            result: SysdModelResult from model.simulate().
            method: Scoring method (``"percentile"``, ``"delta"``, or
                ``"threshold"``).
            threshold: Cutoff for ``"threshold"`` method.

        Returns:
            A Triple with the computed score as literal object.
        """
        values = result.values.get(stock_name, [])
        if not values:
            score = 0.0
        else:
            final_val = values[-1]
            if method == "delta":
                initial = values[0]
                denom = max(abs(initial), 1.0)
                score = min(1.0, abs(final_val - initial) / denom)
            elif method == "threshold":
                score = 1.0 if final_val > threshold else 0.0
            else:
                all_finals = []
                for vlist in result.values.values():
                    if vlist:
                        all_finals.append(vlist[-1])
                vmin = min(all_finals) if all_finals else 0.0
                vmax = max(all_finals) if all_finals else 1.0
                score = (final_val - vmin) / (vmax - vmin) if vmax > vmin else 0.5

        obj = Literal(score, datatype="http://www.w3.org/2001/XMLSchema#double")
        return Triple(subject=subject, predicate=predicate, object_=obj)

    @staticmethod
    def load_queries(yaml_path: str) -> dict[str, str]:
        """Load SPARQL queries from a YAML file.

        YAML format::

            query_name:
              sparql: "SELECT ..."
              mode: select     # optional, for documentation
              var: v           # optional, default variable name

        Returns:
            dict mapping query_name → SPARQL query string, ready for use
            as simulation params (e.g. ``params = bridge.load_queries(...)``).
        """
        import yaml as _yaml
        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Query file not found: %s", yaml_path)
            return {}
        with open(path) as f:
            raw = _yaml.safe_load(f)
        if not isinstance(raw, dict):
            return {}
        queries: dict[str, str] = {}
        for name, entry in raw.items():
            if isinstance(entry, dict) and "sparql" in entry:
                queries[name] = entry["sparql"]
            elif isinstance(entry, str):
                queries[name] = entry
        return queries

    def run_with_kb(
        self,
        model: Any,
        params: dict[str, Any] | None = None,
        **sim_kwargs: Any,
    ) -> Any:
        """Run a simulation with KB-connected builtins.

        Convenience wrapper that passes ``kb=self.store`` to
        ``model.simulate()``.

        Args:
            model: A SysdModel instance.
            params: Parameter overrides (name → value).
            **sim_kwargs: Additional kwargs forwarded to simulate().

        Returns:
            SysdModelResult.
        """
        if params is None:
            params = {}
        sim_kwargs.setdefault("kb", self.store)
        return model.simulate(params=params, **sim_kwargs)

    def full_roundtrip(
        self,
        model: Any,
        claim_map: list[tuple[NamedNode, NamedNode, object, str]],
        evidence_map: list[tuple[str, NamedNode, NamedNode, Callable[[list[float], list[float]], float]]],
        params: dict[str, Any] | None = None,
        evidence_graph: str = "simulation",
        param_default: float = 0.5,
        **sim_kwargs: Any,
    ) -> tuple[Any, list[Triple]]:
        """Full KB → Sim → KB round-trip in one call.

        1. Extracts KB beliefs as simulation params via params_from_kb()
        2. Runs simulation with KB-connected builtins via run_with_kb()
        3. Extracts evidence triples from result via evidence_from_result()

        Args:
            model: A SysdModel instance.
            claim_map: For params_from_kb().
            evidence_map: For evidence_from_result().
            params: Additional parameter overrides.
            evidence_graph: Graph name for evidence triples.
            param_default: Default belief for missing triples.
            **sim_kwargs: Additional kwargs for simulate().

        Returns:
            Tuple of (SysdModelResult, list_of_evidence_triples).
        """
        if params is None:
            params = {}
        kb_params = self.params_from_kb(claim_map, default=param_default)
        all_params = {**kb_params, **params}
        result = self.run_with_kb(model, params=all_params, **sim_kwargs)
        triples = self.evidence_from_result(result, evidence_map, graph=evidence_graph)
        for t in triples:
            self.store.add(t, graph=evidence_graph)
        return result, triples

    # ── Provenance ────────────────────────────────────────────────

    def record_provenance(
        self,
        result: Any,
        params: dict[str, Any] | None = None,
        run_id: str | None = None,
        graph: str = "provenance",
        extra_annotations: list[Triple] | None = None,
    ) -> NamedNode:
        """Store a simulation run's provenance as RDF.

        Creates a provenance graph containing:
          - run entity (sim:Run, prov:Activity)
          - start/end times, dt, number of steps
          - parameter snapshot as RDF data
          - stock start/final values
          - link to evidence graph (if provided)

        Args:
            result: SysdModelResult from model.simulate().
            params: The parameter dict used for this run.
            run_id: Optional unique ID; auto-generated if None.
            graph: Named graph to write into.
            extra_annotations: Additional triples to add to the run.

        Returns:
            NamedNode for the run entity.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())
        run_node = NamedNode(f"{self.ns_base}run/{run_id}")

        prov_triples: list[Triple] = []

        # Type + class
        prov_triples.append(Triple(run_node, NamedNode(f"{NS_PROV}type"), NamedNode(f"{NS_SIM}Run")))
        prov_triples.append(Triple(run_node, NamedNode(f"{NS_PROV}type"), NamedNode(f"{NS_PROV}Activity")))

        # Timestamps from result
        if hasattr(result, "times") and result.times:
            t_start = Literal(float(result.times[0]), datatype="http://www.w3.org/2001/XMLSchema#double")
            t_end = Literal(float(result.times[-1]), datatype="http://www.w3.org/2001/XMLSchema#double")
            steps = Literal(len(result.times), datatype="http://www.w3.org/2001/XMLSchema#integer")
            prov_triples.append(Triple(run_node, NamedNode(f"{NS_PROV}startedAtTime"), t_start))
            prov_triples.append(Triple(run_node, NamedNode(f"{NS_PROV}endedAtTime"), t_end))
            prov_triples.append(Triple(run_node, NamedNode(f"{NS_SIM}steps"), steps))

        # Parameter snapshot
        if params:
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    p_obj = Literal(float(v), datatype="http://www.w3.org/2001/XMLSchema#double")
                elif isinstance(v, str):
                    p_obj = Literal(v, datatype="http://www.w3.org/2001/XMLSchema#string")
                else:
                    p_obj = Literal(str(v))
                param_node = NamedNode(f"{self.ns_base}param/{k}")
                prov_triples.append(Triple(run_node, NamedNode(f"{NS_SIM}hasParam"), param_node))
                prov_triples.append(Triple(param_node, NamedNode(f"{NS_SIM}paramName"), Literal(k)))
                prov_triples.append(Triple(param_node, NamedNode(f"{NS_SIM}paramValue"), p_obj))

        # Stock start/final values
        for stock_name in getattr(result, "stocks", []):
            vals = result.values.get(stock_name, [])
            if not vals:
                continue
            stock_node = NamedNode(f"{self.ns_base}stock/{stock_name}")
            prov_triples.append(Triple(run_node, NamedNode(f"{NS_SIM}hasStock"), stock_node))
            prov_triples.append(Triple(stock_node, NamedNode(f"{NS_SIM}stockName"), Literal(stock_name)))
            prov_triples.append(Triple(
                stock_node, NamedNode(f"{NS_SIM}startValue"),
                Literal(float(vals[0]), datatype="http://www.w3.org/2001/XMLSchema#double"),
            ))
            prov_triples.append(Triple(
                stock_node, NamedNode(f"{NS_SIM}finalValue"),
                Literal(float(vals[-1]), datatype="http://www.w3.org/2001/XMLSchema#double"),
            ))

        # Extra annotations
        if extra_annotations:
            prov_triples.extend(extra_annotations)

        for t in prov_triples:
            self.store.add(t, graph=graph)

        return run_node

    @staticmethod
    def _node_val(n: Any) -> Any:
        """Extract raw value from NamedNode (iri) or Literal (value)."""
        if hasattr(n, "iri"):
            return n.iri
        if hasattr(n, "value"):
            return n.value
        return str(n)

    @staticmethod
    def compare_runs(
        store: TripleStore,
        provenance_graph: str = "provenance",
    ) -> dict[str, Any]:
        """Query provenance graph for run comparison.

        Uses SPARQL to retrieve all runs with their parameter values
        and stock start/final values, grouped by run.

        Returns:
            dict with 'runs' (list of run dicts) and 'stock_deltas'
            (dict mapping stock_name → list of deltas across runs).
        """
        SIM_RUN = NamedNode(f"{NS_SIM}Run")
        SIM_HAS_PARAM = NamedNode(f"{NS_SIM}hasParam")
        SIM_PARAM_NAME = NamedNode(f"{NS_SIM}paramName")
        SIM_PARAM_VALUE = NamedNode(f"{NS_SIM}paramValue")
        SIM_HAS_STOCK = NamedNode(f"{NS_SIM}hasStock")
        SIM_STOCK_NAME = NamedNode(f"{NS_SIM}stockName")
        SIM_START_VALUE = NamedNode(f"{NS_SIM}startValue")
        SIM_FINAL_VALUE = NamedNode(f"{NS_SIM}finalValue")
        PROV_TYPE = NamedNode(f"{NS_PROV}type")

        runs_map: dict[str, dict] = {}
        stock_deltas: dict[str, list[float]] = {}

        # Find all sim:Run entities in the provenance graph
        for t in store.triples_in_graph(provenance_graph):
            if t.predicate == PROV_TYPE and t.object_ == SIM_RUN:
                run_iri = KBSimBridge._node_val(t.subject)
                if run_iri not in runs_map:
                    runs_map[run_iri] = {"iri": run_iri, "params": {}, "stocks": {}}

        # Build (subject → param/value) lookup for param nodes
        param_values: dict[str, dict[str, Any]] = {}
        stock_values: dict[str, dict[str, Any]] = {}
        for t in store.triples_in_graph(provenance_graph):
            subj = KBSimBridge._node_val(t.subject)
            pred = KBSimBridge._node_val(t.predicate)
            obj_val = KBSimBridge._node_val(t.object_)
            if pred == KBSimBridge._node_val(SIM_PARAM_NAME):
                param_values.setdefault(subj, {})["name"] = obj_val
            elif pred == KBSimBridge._node_val(SIM_PARAM_VALUE):
                param_values.setdefault(subj, {})["value"] = obj_val
            elif pred == KBSimBridge._node_val(SIM_STOCK_NAME):
                stock_values.setdefault(subj, {})["name"] = obj_val
            elif pred == KBSimBridge._node_val(SIM_START_VALUE):
                stock_values.setdefault(subj, {})["start"] = obj_val
            elif pred == KBSimBridge._node_val(SIM_FINAL_VALUE):
                stock_values.setdefault(subj, {})["final"] = obj_val

        # Map runs to param nodes and stock nodes
        for t in store.triples_in_graph(provenance_graph):
            run_key = KBSimBridge._node_val(t.subject)
            if run_key not in runs_map:
                continue
            pred = KBSimBridge._node_val(t.predicate)
            obj_val = KBSimBridge._node_val(t.object_)
            if pred == KBSimBridge._node_val(SIM_HAS_PARAM):
                pn = param_values.get(obj_val, {})
                if "name" in pn:
                    name = str(pn["name"]).strip('"')
                    try:
                        runs_map[run_key]["params"][name] = float(pn.get("value", 0))
                    except (ValueError, TypeError):
                        runs_map[run_key]["params"][name] = pn.get("value", 0)
            elif pred == KBSimBridge._node_val(SIM_HAS_STOCK):
                sn = stock_values.get(obj_val, {})
                sname = str(sn.get("name", "")).strip('"')
                if sname:
                    try:
                        start = float(sn.get("start", 0))
                        final = float(sn.get("final", 0))
                        runs_map[run_key]["stocks"][sname] = (start, final)
                        stock_deltas.setdefault(sname, []).append(final - start)
                    except (ValueError, TypeError):
                        pass

        return {
            "runs": list(runs_map.values()),
            "stock_deltas": stock_deltas,
        }


# ── Closed-Loop Reasoning ────────────────────────────────────────


@dataclass
class ReasoningPass:
    """A single pass in a closed-loop reasoning pipeline.

    Attributes:
        name: Human-readable label for this pass.
        claim_map: KB triples → simulation params.
        evidence_map: Simulation results → KB triples.
        params_override: Additional param overrides for this pass.
        grade_queries: SPARQL queries that grade this pass's results.
            Each is (query_str, var_name, threshold, penalty_if_fail).
            If the query result falls below threshold, the params
            for the next pass are penalized/nudged.
        param_nudges: Optional adjustments applied to params between
            passes. Dict of param_name → delta (added to previous
            pass's value).
        grade_update: Optional callback invoked after grading. Receives
            (grades dict, TripleStore) and returns a dict of param
            changes to merge into current_params before the next pass.
            Enables conditional param updates: e.g., "if anxiety grade
            > 0.8, set intervention_active = True" by querying the KB
            for the latest evidence and returning updated params.
        max_belief_graph: Which graph to read max-belief from for
            params_from_kb. If None, reads all non-excluded graphs.
    """

    name: str
    claim_map: list[tuple[NamedNode, NamedNode, object, str]]
    evidence_map: list[tuple[str, NamedNode, NamedNode, Callable[[list[float], list[float]], float]]]
    params_override: dict[str, Any] = field(default_factory=dict)
    grade_queries: list[tuple[str, str, float, float]] = field(default_factory=list)
    param_nudges: dict[str, float] = field(default_factory=dict)
    grade_update: Callable[[dict[str, float], TripleStore], dict[str, Any]] | None = None
    max_belief_graph: str | None = None

    def grade(
        self,
        result: Any,
        store: TripleStore,
    ) -> dict[str, float]:
        """Grade this pass by running SPARQL queries on the KB.

        Args:
            result: SysdModelResult from simulation (unused — kept for API compat).
            store: TripleStore with evidence asserted.

        Returns:
            dict mapping grade_name → score in [0, 1].
        """
        prefix = f"grade_{self.name}_"
        return grade_queries(self.grade_queries, store, prefix=prefix)


# ── Standalone grade function ─────────────────────────────────────


def grade_queries(
    grade_specs: list[tuple[str, str, float, float]],
    store: TripleStore,
    prefix: str = "",
) -> dict[str, float]:
    """Run SPARQL queries against a TripleStore and extract float scores.

    Args:
        grade_specs: List of (query_str, var_name, threshold, penalty) tuples.
        store: TripleStore to query.
        prefix: Prepended to result keys (e.g. ``"grade_baseline_"``).

    Returns:
        dict mapping ``f"{prefix}{idx}"`` → float score in [0, 1].
    """
    grades: dict[str, float] = {}
    for q_idx, (query_str, var_name, _, _) in enumerate(grade_specs):
        try:
            ast = parse_sparql(query_str)
            qr = sparql_evaluate(ast, store)
            val = 0.0
            if qr.bindings:
                for row in qr.bindings:
                    if var_name in row:
                        v = row[var_name]
                        raw = v.value if hasattr(v, "value") else v
                        try:
                            val = float(raw)
                        except (ValueError, TypeError):
                            val = 1.0 if raw else 0.0
                        break
            elif qr.cardinality > 0:
                val = 1.0
            grades[f"{prefix}{q_idx}"] = max(0.0, min(1.0, val))
        except Exception:
            grades[f"{prefix}{q_idx}"] = 0.0
    return grades


@dataclass
class ClosedLoopResult:
    """Result of a closed-loop reasoning pipeline."""

    passes: list[ReasoningPass] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    grades: list[dict[str, float]] = field(default_factory=list)
    run_nodes: list[NamedNode] = field(default_factory=list)
    final_params: dict[str, Any] | None = None
    evidence_added: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "passes": len(self.passes),
            "grades": self.grades,
            "final_params": self.final_params,
            "evidence_added": self.evidence_added,
        }


class ClosedLoopReasoner:
    """Orchestrate simulate → grade → update → re-simulate.

    Runs multiple ReasoningPass sequentially. Each pass:
      1. Extracts params from KB via claim_map.
      2. Applies param nudges from previous pass grades.
      3. Simulates.
      4. Asserts evidence back into KB.
      5. Grades results via SPARQL.
      6. Records provenance.
    """

    def __init__(
        self,
        bridge: KBSimBridge,
        model: Any,
        passes: list[ReasoningPass],
        evidence_graph: str = "simulation",
        provenance_graph: str = "provenance",
        param_default: float = 0.5,
    ):
        self.bridge = bridge
        self.model = model
        self.passes = passes
        self.evidence_graph = evidence_graph
        self.provenance_graph = provenance_graph
        self.param_default = param_default

    def run(self) -> ClosedLoopResult:
        """Execute the closed-loop pipeline."""
        cl_result = ClosedLoopResult(passes=self.passes)
        current_params: dict[str, Any] = {}

        for pass_def in self.passes:
            # 1. Extract params from KB
            exclude = None
            if pass_def.max_belief_graph:
                exclude = set()
                for g in self.bridge.store.graphs():
                    if g != pass_def.max_belief_graph:
                        exclude.add(g)
            pass_params = self.bridge.params_from_kb(
                pass_def.claim_map,
                default=self.param_default,
                exclude_graphs=exclude,
            )

            # 2. Merge with accumulated params + overrides + nudges
            # Start with current accumulated params as the base
            final_params = dict(current_params)
            # Update with KB-extracted params (these are the primary source for the current pass)
            final_params.update(pass_params)
            # Apply explicit overrides for this pass
            final_params.update(pass_def.params_override)
            # Apply nudges
            for k, v in pass_def.param_nudges.items():
                final_params[k] = final_params.get(k, 0.0) + v
            current_params = dict(final_params)


            # 3. Simulate with fully merged params
            result = self.bridge.run_with_kb(self.model, params=final_params)

            # 4. Assert evidence into KB
            evidence_triples = self.bridge.evidence_from_result(
                result, pass_def.evidence_map, graph=self.evidence_graph,
            )
            for t in evidence_triples:
                self.bridge.store.add(t, graph=self.evidence_graph)
            cl_result.evidence_added += len(evidence_triples)
            cl_result.results.append(result)

            # 5. Grade
            grades = pass_def.grade(result, self.bridge.store)
            cl_result.grades.append(grades)

            # 5b. Apply grade_update callback to conditionally update params
            if pass_def.grade_update is not None:
                grade_updates = pass_def.grade_update(grades, self.bridge.store)
                if grade_updates:
                    current_params.update(grade_updates)

            # 6. Record provenance
            run_node = self.bridge.record_provenance(
                result, params=pass_params,
                graph=self.provenance_graph,
            )
            cl_result.run_nodes.append(run_node)

        cl_result.final_params = current_params
        return cl_result



# ── Cognitive Orchestrator — event → rule → sim → evidence loop ──


class CognitiveOrchestrator:
    """Wires TransactionStore → ProductionRuleEngine → ExecutionStore
    into a continuous event-driven reasoning loop.

    The loop:
        1. Ingest external event  →  TransactionStore.record()
        2. TripleStore.on_add()   →  ProductionRuleEngine.evaluate()
        3. Rule body matches     →  Head actions execute
        4. Actions recorded      →  ExecutionStore.record()

    Usage::

        from dynafx.bridge import CognitiveOrchestrator

        orb = CognitiveOrchestrator(store)
        orb.add_rule(my_production_rule)
        orb.start()

        # Ingest an event — everything else fires automatically
        orb.ingest_event("ContainerDelayed", {
            "container_id": "C-123",
            "delay_days": 6,
            "port": "Rotterdam",
        })
    """

    def __init__(self, store: Any, bridge: Any | None = None):
        from dynafx.knowledge.execution import ExecutionStore
        from dynafx.knowledge.production import ProductionRuleEngine
        from dynafx.knowledge.transactions import TransactionStore

        self.store = store
        self.bridge = bridge
        self.tx_store = TransactionStore(store)
        self.exec_store = ExecutionStore(store)
        self.rule_engine = ProductionRuleEngine(store)
        self._started: bool = False

    def add_rule(self, rule: Any) -> None:
        """Register a ProductionRule with execution recording."""
        wrapped = self._wrap_rule(rule)
        self.rule_engine.add_rule(wrapped)

    def remove_rule(self, name: str) -> None:
        self.rule_engine.remove_rule(name)

    def start(self) -> None:
        """Start the event-driven loop."""
        self.rule_engine.start()
        self._started = True

    def stop(self) -> None:
        self.rule_engine.stop()
        self._started = False

    def ingest_event(
        self,
        event_type: str,
        payload: dict,
        source: str = "external",
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> Any:
        """Ingest an external event — triggers the full pipeline.

        Args:
            event_type: Event category.
            payload: Key-value data.
            source: Origin identifier.
            confidence: Accuracy confidence [0, 1].
            timestamp: Unix timestamp (default: time.time()).

        Returns:
            Transaction object.
        """
        return self.tx_store.record(
            event_type=event_type,
            payload=payload,
            source=source,
            confidence=confidence,
            timestamp=timestamp,
        )

    def get_causal_chain(self, action_id: str) -> list[dict]:
        """Trace an action back to its rule and recent events.

        Returns:
            List of dicts with keys: type, timestamp, detail.
        """
        chain: list[dict] = []
        exec_rec = self.exec_store.get(action_id)
        if exec_rec is None:
            return chain

        chain.append({
            "type": "action",
            "timestamp": exec_rec.timestamp,
            "detail": f"{exec_rec.action_type} from rule '{exec_rec.rule_name}' — {exec_rec.message}",
        })

        # Find the rule that fired this action
        rule = self.rule_engine.get_rule(exec_rec.rule_name)
        if rule:
            chain.append({
                "type": "rule",
                "timestamp": exec_rec.timestamp,
                "detail": f"Rule '{rule.name}': {rule.description}",
            })

        # Find recent transactions around the execution time
        recent_txs = self.tx_store.recent(n=5)
        for tx in recent_txs:
            if abs(tx.timestamp - exec_rec.timestamp) < 60:
                chain.append({
                    "type": "event",
                    "timestamp": tx.timestamp,
                    "detail": f"Transaction '{tx.event_type}' from {tx.source}: {tx.payload}",
                })

        chain.sort(key=lambda x: x["timestamp"])
        return chain

    def get_rule_status(self) -> list[dict]:
        """Return status of all registered rules."""
        status: list[dict] = []
        for rule in self.rule_engine.rules:
            last_exec = self.exec_store.last_execution(rule.name)
            fired_count = self.rule_engine._fired_count.get(rule.name, 0)
            status.append({
                "name": rule.name,
                "enabled": rule.enabled,
                "fired": fired_count,
                "last_fired": last_exec.timestamp if last_exec else None,
                "last_status": last_exec.status if last_exec else None,
            })
        return status

    # ── Internal: wrap rule actions with execution recording ──

    def _wrap_rule(self, rule: Any) -> Any:
        """Wrap each action in a rule with execution recording."""
        from dataclasses import replace as dataclass_replace

        wrapped_actions: list[Action] = []
        for action in rule.head:
            wrapped = self._wrap_action(action, rule.name)
            wrapped_actions.append(wrapped)

        return dataclass_replace(rule, head=wrapped_actions)

    def _wrap_action(self, action: Action, rule_name: str) -> Action:
        """Wrap a single action to record its execution."""
        original_execute = action.execute

        def recorded_execute(store, bindings):
            try:
                result = original_execute(store, bindings)
                self.exec_store.record(
                    rule_name=rule_name,
                    action_type=getattr(result, "action_type", type(action).__name__),
                    bindings={k: self._node_str(v) for k, v in bindings.items()},
                    output=getattr(result, "output", {}),
                    status="executed" if getattr(result, "success", True) else "failed",
                    message=getattr(result, "message", ""),
                )
                return result
            except Exception as e:
                logger.exception("Action failed for rule '%s'", rule_name)
                self.exec_store.record(
                    rule_name=rule_name,
                    action_type=type(action).__name__,
                    bindings={},
                    output={},
                    status="failed",
                    message=str(e),
                )
                return ActionResult(type(action).__name__, success=False, message=str(e))

        action.execute = recorded_execute
        return action

    @staticmethod
    def _node_str(v: Any) -> str:
        if hasattr(v, "iri"):
            return v.iri
        if hasattr(v, "value"):
            return str(v.value)
        return str(v)
