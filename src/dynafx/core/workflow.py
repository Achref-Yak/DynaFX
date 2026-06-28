from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dynafx.core.diff import CycleDiff, compute_diff
from dynafx.core.math import (
    cumulative_fusion, consensus_compromise, weighted_belief_fusion,
    conditional_deduction, subjective_abduction, conjunction, disjunction,
    trust_transfer, reverse_warrant, propagate_step, master_equation_all,
    dung_semantics, convergence_norm, check_opinion_invariant,
    opinion_conflict, bayes_rule, analogy_warrant_transform,
    tna_propagate, extract_max_dag,
)
from dynafx.core.models import Node, Edge, Opinion, NodeType, EdgeType, Graph
from dynafx.core.operator import Operator
from dynafx.core.state import State


logger = logging.getLogger(__name__)


# ── Primitive Registry ──────────────────────────────────────────────

@dataclass
class PrimitiveInfo:
    id: str
    name: str
    kind: str
    description: str
    params_schema: dict
    callable_ref: Any


class PrimitiveRegistry:
    _instance: Optional[PrimitiveRegistry] = None

    def __init__(self):
        self._primitives: dict[str, PrimitiveInfo] = {}

    @classmethod
    def get_instance(cls) -> PrimitiveRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._discover_all()
        return cls._instance

    def get(self, ref: str) -> Optional[PrimitiveInfo]:
        return self._primitives.get(ref)

    def list_by_kind(self, kind: str) -> list[PrimitiveInfo]:
        return [p for p in self._primitives.values() if p.kind == kind]

    def all(self) -> list[PrimitiveInfo]:
        return list(self._primitives.values())

    def register(self, info: PrimitiveInfo) -> None:
        self._primitives[info.id] = info

    def _discover_all(self) -> None:
        self._discover_operators()
        self._discover_sl_ops()
        self._discover_graph_primitives()
        self._discover_kernel_primitives()
        self._register_gates()

    def _discover_operators(self) -> None:
        op_modules = [
            "dynafx.operators.abduce",
            "dynafx.operators.align",
            "dynafx.operators.analogy",
            "dynafx.operators.attention",
            "dynafx.operators.bottleneck",
            "dynafx.operators.compare",
            "dynafx.operators.compress",
            "dynafx.operators.constraint",
            "dynafx.operators.emergence",
            "dynafx.operators.graph",
            "dynafx.operators.iceberg",
            "dynafx.operators.induce",
            "dynafx.operators.merge",
            "dynafx.operators.plan",
            "dynafx.operators.propagate",
            "dynafx.operators.relate",
            "dynafx.operators.schema",
            "dynafx.operators.simulate",
            "dynafx.operators.stock_flow",
            "dynafx.operators.systems",
            "dynafx.operators.temporal",
            "dynafx.operators.tom",
            "dynafx.operators.update",
        ]
        for mod_path in op_modules:
            try:
                mod = importlib.import_module(mod_path)
                for name in dir(mod):
                    obj = getattr(mod, name)
                    if not inspect.isclass(obj):
                        continue
                    if not hasattr(obj, 'name') or not isinstance(getattr(obj, 'name'), str):
                        continue
                    if not hasattr(obj, '__call__'):
                        continue
                    op_name = obj.name
                    if not op_name:
                        continue
                    self.register(PrimitiveInfo(
                        id=f"op:{op_name}",
                        name=op_name,
                        kind="operator",
                        description=(obj.__doc__ or "").split("\n")[0] if obj.__doc__ else "",
                        params_schema=self._infer_params(obj.__call__),
                        callable_ref=obj,
                    ))
            except Exception as e:
                logger.warning("Failed to discover operators from %s: %s", mod_path, e)

    def _discover_sl_ops(self) -> None:
        sl_ops = [
            ("cumulative_fusion", "Fuse two opinions with independent evidence", cumulative_fusion),
            ("consensus_compromise", "Fuse two opinions with consensus/compromise", consensus_compromise),
            ("weighted_fusion", "Weighted belief fusion", weighted_belief_fusion),
            ("conditional_deduction", "Deduce consequence from premise + warrant", conditional_deduction),
            ("subjective_abduction", "Abduce cause from effect + warrant", subjective_abduction),
            ("conjunction", "Subjective Logic conjunction (AND)", conjunction),
            ("disjunction", "Subjective Logic disjunction (OR)", disjunction),
            ("trust_transfer", "Transfer trust from source via recommender", trust_transfer),
            ("reverse_warrant", "Bayesian inversion of a conditional warrant", reverse_warrant),
            ("propagate_step", "Single-step belief propagation", propagate_step),
            ("master_equation", "Compute unified belief values", master_equation_all),
            ("tna_propagate", "Trust Network Analysis (one-pass topological propagation)", tna_propagate),
            ("extract_max_dag", "Extract max DAG subgraph by dropping back-edges", extract_max_dag),
            ("dung_semantics", "Compute Dung's preferred extension", dung_semantics),
            ("check_opinion_invariant", "Validate opinion b+d+u=1", check_opinion_invariant),
            ("opinion_conflict", "Detect conflict between two opinions", opinion_conflict),
            ("bayes_rule", "Apply Bayes' rule", bayes_rule),
            ("analogy_transform", "Transform opinion for analogy", analogy_warrant_transform),
        ]
        for ref, desc, fn in sl_ops:
            self.register(PrimitiveInfo(
                id=f"sl:{ref}",
                name=ref,
                kind="sl_op",
                description=desc,
                params_schema=self._infer_params(fn),
                callable_ref=fn,
            ))

    def _discover_graph_primitives(self) -> None:
        graph_ops: list[tuple[str, str, Callable]] = [
            ("add_node", "Add a proposition node to the graph", self._primitive_add_node),
            ("add_edge", "Add an edge between two nodes", self._primitive_add_edge),
            ("remove_node", "Remove a node and its edges", self._primitive_remove_node),
            ("remove_edge", "Remove an edge", self._primitive_remove_edge),
            ("set_opinion", "Set opinion on a node", self._primitive_set_opinion),
            ("set_prior", "Set base rate on a node", self._primitive_set_prior),
            ("set_category", "Set category level on a node", self._primitive_set_category),
            ("merge_graphs", "Merge another graph into the current state", self._primitive_merge_graphs),
        ]
        for ref, desc, fn in graph_ops:
            self.register(PrimitiveInfo(
                id=f"graph:{ref}",
                name=ref,
                kind="graph",
                description=desc,
                params_schema=self._infer_params(fn),
                callable_ref=fn,
            ))

    def _discover_kernel_primitives(self) -> None:
        from dynafx.kernel.assertion_gate import AssertionGate
        self.register(PrimitiveInfo(
            id="kernel:assertion_gate",
            name="assertion_gate",
            kind="kernel",
            description="Run assertion gate: type check → opinion assign → invariant check → quarantine",
            params_schema={},
            callable_ref=AssertionGate,
        ))

    def _register_gates(self) -> None:
        self.register(PrimitiveInfo(
            id="gate:if",
            name="if",
            kind="gate",
            description="Conditional branch: evaluate expression on state metadata",
            params_schema={"condition": {"type": "string"}},
            callable_ref=None,
        ))
        self.register(PrimitiveInfo(
            id="gate:parallel",
            name="parallel",
            kind="gate",
            description="Fan-out: run multiple steps in parallel",
            params_schema={"steps": {"type": "array", "items": {"type": "string"}}},
            callable_ref=None,
        ))

    @staticmethod
    def _infer_params(fn: Callable) -> dict:
        try:
            sig = inspect.signature(fn)
            props = {}
            required = []
            for name, param in sig.parameters.items():
                if name in ("self", "state", "args", "kwargs", "opinions", "omega_x", "omega_y",
                            "omega_a", "omega_b", "omega_p", "omega_source", "omega_recommender",
                            "omega_effect", "beliefs", "adjacency", "evidence"):
                    continue
                if param.annotation is inspect.Parameter.empty:
                    ptype = "string"
                else:
                    ann = str(param.annotation)
                    if "float" in ann or "int" in ann:
                        ptype = "number"
                    elif "bool" in ann:
                        ptype = "boolean"
                    elif "dict" in ann or "Any" in ann:
                        ptype = "object"
                    else:
                        ptype = "string"
                props[name] = {"type": ptype, "description": ""}
                if param.default is inspect.Parameter.empty:
                    required.append(name)
            return {"type": "object", "properties": props, "required": required}
        except Exception:
            return {"type": "object", "properties": {}}

    @staticmethod
    def _primitive_add_node(state: State, text: str, node_type: str = "CLAIM", **kwargs) -> State:
        import uuid
        node = Node(
            id=uuid.uuid4(),
            type=NodeType[node_type.upper()] if node_type.upper() in NodeType.__members__ else NodeType.CLAIM,
            text=text,
            opinion=Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, prior=0.5),
        )
        state.graph.nodes[node.id] = node
        state.record("add_node", f"Added {node_type} node: {text[:60]}")
        return state

    @staticmethod
    def _primitive_add_edge(state: State, source_id: str, target_id: str, edge_type: str = "SUPPORTS", **kwargs) -> State:
        import uuid
        src = None
        tgt = None
        for nid, node in state.graph.nodes.items():
            if nid.hex == source_id:
                src = nid
            if nid.hex == target_id:
                tgt = nid
        if src is None or tgt is None:
            logger.warning("add_edge: source or target not found")
            return state
        edge = Edge(
            id=uuid.uuid4(),
            source_id=src,
            target_id=tgt,
            type=EdgeType[edge_type.upper()] if edge_type.upper() in EdgeType.__members__ else EdgeType.SUPPORTS,
        )
        state.graph.edges[edge.id] = edge
        state.record("add_edge", f"Added {edge_type} edge: {source_id[:8]} → {target_id[:8]}")
        return state

    @staticmethod
    def _primitive_remove_node(state: State, node_id: str, **kwargs) -> State:
        to_remove = None
        for nid in state.graph.nodes:
            if nid.hex == node_id:
                to_remove = nid
                break
        if to_remove is None:
            return state
        del state.graph.nodes[to_remove]
        state.graph.edges = {eid: e for eid, e in state.graph.edges.items()
                             if e.source_id != to_remove and e.target_id != to_remove}
        state.record("remove_node", f"Removed node {node_id[:8]}")
        return state

    @staticmethod
    def _primitive_remove_edge(state: State, edge_id: str, **kwargs) -> State:
        to_remove = None
        for eid in state.graph.edges:
            if eid.hex == edge_id:
                to_remove = eid
                break
        if to_remove is not None:
            del state.graph.edges[to_remove]
            state.record("remove_edge", f"Removed edge {edge_id[:8]}")
        return state

    @staticmethod
    def _primitive_set_opinion(state: State, node_id: str, belief: float = 0.0, disbelief: float = 0.0,
                               uncertainty: float = 1.0, prior: float = 0.5, **kwargs) -> State:
        for nid, node in state.graph.nodes.items():
            if nid.hex == node_id:
                node.opinion = Opinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, prior=prior)
                state.record("set_opinion", f"Set opinion on {node_id[:8]}: ({belief},{disbelief},{uncertainty},{prior})")
                break
        return state

    @staticmethod
    def _primitive_set_prior(state: State, node_id: str, prior: float = 0.5, **kwargs) -> State:
        for nid, node in state.graph.nodes.items():
            if nid.hex == node_id:
                op = node.opinion or Opinion()
                node.opinion = Opinion(belief=op.belief, disbelief=op.disbelief, uncertainty=op.uncertainty, prior=prior)
                state.record("set_prior", f"Set prior on {node_id[:8]}: {prior}")
                break
        return state

    @staticmethod
    def _primitive_set_category(state: State, node_id: str, category: int = 2, **kwargs) -> State:
        for nid, node in state.graph.nodes.items():
            if nid.hex == node_id:
                node.category = category
                state.record("set_category", f"Set category on {node_id[:8]}: {category}")
                break
        return state

    @staticmethod
    def _primitive_merge_graphs(state: State, graph_data: dict, **kwargs) -> State:
        g = Graph.from_dict(graph_data)
        for nid, node in g.nodes.items():
            if nid not in state.graph.nodes:
                state.graph.nodes[nid] = node
        for eid, edge in g.edges.items():
            if eid not in state.graph.edges:
                state.graph.edges[eid] = edge
        for eid, entity in g.entities.items():
            if eid not in state.graph.entities:
                state.graph.entities[eid] = entity
        state.record("merge_graphs", f"Merged {len(g.nodes)} nodes, {len(g.edges)} edges")
        return state


# ── Workflow Definition ────────────────────────────────────────────

@dataclass
class WorkflowStep:
    id: str
    kind: str
    ref: str
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    condition: Optional[str] = None
    branches: Optional[dict[str, list[str]]] = None
    output_map: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    id: str
    name: str
    description: str = ""
    steps: dict[str, WorkflowStep] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowDefinition:
        raw_steps = data.get("steps", {})
        steps: dict[str, WorkflowStep] = {}
        if isinstance(raw_steps, list):
            for s in raw_steps:
                if isinstance(s, str):
                    steps[s] = WorkflowStep(id=s, kind="operator", ref=s)
                else:
                    entry = dict(s)
                    entry.setdefault("kind", "operator")
                    entry.setdefault("params", {})
                    entry.setdefault("depends_on", [])
                    steps[entry["id"]] = WorkflowStep(**entry)
        else:
            for sid, spec in raw_steps.items():
                if isinstance(spec, str):
                    steps[sid] = WorkflowStep(id=sid, kind="operator", ref=spec)
                elif isinstance(spec, dict):
                    spec_copy = {k: v for k, v in spec.items() if k != "id"}
                    spec_copy.setdefault("kind", "operator")
                    spec_copy.setdefault("params", {})
                    spec_copy.setdefault("depends_on", [])
                    steps[sid] = WorkflowStep(id=sid, **spec_copy)
        entry_points = data.get("entry_points")
        if not entry_points:
            entry_points = [sid for sid, step in steps.items() if not step.depends_on]
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=steps,
            entry_points=entry_points,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": {sid: {k: v for k, v in vars(s).items() if k != "metadata" or v}
                      for sid, s in self.steps.items()},
            "entry_points": self.entry_points,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False)


# ── Workflow Engine ────────────────────────────────────────────────

class StepResult:
    def __init__(self, step_id: str, state: State, diff: CycleDiff, error: Optional[str] = None):
        self.step_id = step_id
        self.state = state
        self.diff = diff
        self.error = error


class WorkflowEngine:
    def __init__(self, registry: Optional[PrimitiveRegistry] = None):
        self._registry = registry or PrimitiveRegistry.get_instance()

    async def run(
        self,
        definition: WorkflowDefinition,
        initial_state: State,
        on_diff: Optional[Callable[[CycleDiff], None]] = None,
    ) -> State:
        state = initial_state
        order = self._topological_sort(definition)
        completed: dict[str, StepResult] = {}
        current_cycle = 0

        for batch in order:
            tasks = []
            for step_id in batch:
                step = definition.steps[step_id]
                if self._should_skip(step, definition, completed):
                    continue
                tasks.append(self._execute_step(step, state, completed, current_cycle))

            if not tasks:
                continue

            results = await asyncio.gather(*tasks)

            for result in results:
                if result.error:
                    logger.error("Step %s failed: %s", result.step_id, result.error)
                    if on_diff:
                        on_diff(result.diff)
                    completed[result.step_id] = result
                    continue

                state = result.state
                if on_diff:
                    on_diff(result.diff)
                completed[result.step_id] = result
                current_cycle += 1

                if step_id in definition.entry_points:
                    pass

            state.metadata["workflow_step_count"] = len(completed)

        return state

    def run_sync(
        self,
        definition: WorkflowDefinition,
        initial_state: State,
        on_diff: Optional[Callable[[CycleDiff], None]] = None,
    ) -> State:
        state = initial_state
        order = self._topological_sort(definition)
        completed: dict[str, StepResult] = {}
        current_cycle = 0

        for batch in order:
            for step_id in batch:
                step = definition.steps[step_id]
                if self._should_skip(step, definition, completed):
                    continue

                before = state.fork()
                result = self._execute_step_sync(step, state, completed, current_cycle)

                if result.error:
                    logger.error("Step %s failed: %s", result.step_id, result.error)
                    if on_diff:
                        on_diff(result.diff)
                    completed[result.step_id] = result
                    continue

                state = result.state
                if on_diff:
                    on_diff(result.diff)
                completed[result.step_id] = result
                current_cycle += 1

        state.metadata["workflow_step_count"] = len(completed)
        return state

    def _topological_sort(self, definition: WorkflowDefinition) -> list[list[str]]:
        steps = definition.steps
        in_degree: dict[str, int] = {sid: 0 for sid in steps}
        children: dict[str, list[str]] = {sid: [] for sid in steps}

        for sid, step in steps.items():
            for dep in step.depends_on:
                if dep in steps:
                    in_degree[sid] = in_degree.get(sid, 0) + 1
                    children.setdefault(dep, []).append(sid)

        queue = deque()
        entry_set = set(definition.entry_points) if definition.entry_points else set()
        for sid in steps:
            if entry_set:
                if sid in entry_set:
                    queue.append(sid)
            elif in_degree[sid] == 0:
                queue.append(sid)

        levels: list[list[str]] = []
        visited: set[str] = set()

        while queue:
            batch = list(queue)
            queue.clear()
            level: list[str] = []
            for sid in batch:
                if sid in visited:
                    continue
                visited.add(sid)
                level.append(sid)
                for child in children.get(sid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            if level:
                levels.append(level)

        for sid in steps:
            if sid not in visited:
                logger.warning("Step %s not reachable (circular dep or disconnected)", sid)
                levels.append([sid])

        return levels

    def _should_skip(self, step: WorkflowStep, definition: WorkflowDefinition,
                     completed: dict[str, StepResult]) -> bool:
        if step.condition:
            ctx = self._build_condition_context(completed)
            try:
                result = eval(step.condition, {"__builtins__": {}}, ctx)
                if not result:
                    return True
            except Exception:
                logger.warning("Condition '%s' failed to evaluate, skipping step %s", step.condition, step.id)
                return True
        return False

    def _build_condition_context(self, completed: dict[str, StepResult]) -> dict:
        ctx: dict = {}
        for sid, result in completed.items():
            if result.state:
                ctx.update(result.state.metadata)
        return ctx

    async def _execute_step(
        self, step: WorkflowStep, state: State,
        completed: dict[str, StepResult], cycle: int,
    ) -> StepResult:
        return self._execute_step_sync(step, state, completed, cycle)

    def _execute_step_sync(
        self, step: WorkflowStep, state: State,
        completed: dict[str, StepResult], cycle: int,
    ) -> StepResult:
        before = state.fork()
        info = self._registry.get(f"{step.kind}:{step.ref}")
        if info is None:
            info = self._registry.get(f"op:{step.ref}")
        if info is None:
            err = f"Primitive not found: {step.kind}:{step.ref}"
            logger.error(err)
            return StepResult(step.id, state, compute_diff(before, state, step.id, cycle), error=err)

        try:
            if step.kind in ("operator",):
                op_cls = info.callable_ref
                if inspect.isclass(op_cls):
                    op_instance = op_cls()
                else:
                    op_instance = op_cls
                state = op_instance(state, **step.params)
            elif step.kind == "sl_op":
                state = self._apply_sl_op(info.callable_ref, state, step.params)
            elif step.kind == "graph":
                state = info.callable_ref(state, **step.params)
            elif step.kind == "kernel":
                gate = info.callable_ref()
                result = gate.evaluate(state, **step.params)
                state = result.state if hasattr(result, 'state') else state
                state.metadata["gate_result"] = str(result)
            elif step.kind == "gate":
                pass
            else:
                err = f"Unknown step kind: {step.kind}"
                return StepResult(step.id, state, compute_diff(before, state, step.id, cycle), error=err)
        except Exception as e:
            logger.exception("Step %s failed", step.id)
            return StepResult(step.id, state, compute_diff(before, state, step.id, cycle), error=str(e))

        diff = compute_diff(before, state, step.id, cycle)
        return StepResult(step.id, state, diff)

    @staticmethod
    def _apply_sl_op(fn: Callable, state: State, params: dict) -> State:
        fn_name = fn.__name__

        if fn_name in ("cumulative_fusion", "consensus_compromise", "weighted_belief_fusion"):
            source = params.get("source_nodes", [])
            opinions = []
            for nid_str in source:
                for nid, node in state.graph.nodes.items():
                    if nid.hex == nid_str:
                        op = node.opinion or Opinion()
                        opinions.append((op.belief, op.disbelief, op.uncertainty, op.prior))
                        break
            if len(opinions) >= 2:
                if fn_name == "weighted_belief_fusion":
                    wa = params.get("weight_a", 0.5)
                    wb = params.get("weight_b", 0.5)
                    result = fn(opinions[0], opinions[1], wa, wb)
                else:
                    result = fn(opinions[0], opinions[1])
                target = params.get("target_node")
                if target:
                    for nid, node in state.graph.nodes.items():
                        if nid.hex == target:
                            node.opinion = Opinion.from_tuple(result)
                            break
                state.metadata[f"sl:{fn_name}"] = result
                state.record(f"sl:{fn_name}", f"Applied {fn_name} on {len(source)} nodes")

        elif fn_name == "conditional_deduction":
            premise = params.get("premise_id")
            warrant_source = params.get("warrant_source_id")
            if premise and warrant_source:
                premise_op = None
                warrant_op = None
                for nid, node in state.graph.nodes.items():
                    if nid.hex == premise:
                        premise_op = node.opinion or Opinion()
                    if nid.hex == warrant_source:
                        warrant_op = node.opinion or Opinion()
                if premise_op and warrant_op:
                    wp = (premise_op.belief, premise_op.disbelief, premise_op.uncertainty, premise_op.prior)
                    wr = ((warrant_op.belief, warrant_op.disbelief, warrant_op.uncertainty, warrant_op.prior),
                          (0.0, 0.0, 1.0, 0.5))
                    result = fn(wp, wr)
                    target = params.get("target_node")
                    if target:
                        for nid, node in state.graph.nodes.items():
                            if nid.hex == target:
                                node.opinion = Opinion.from_tuple(result)
                                break
                    state.metadata["sl:conditional_deduction"] = result

        elif fn_name == "subjective_abduction":
            effect = params.get("effect_id")
            warrant_source = params.get("warrant_source_id")
            if effect and warrant_source:
                effect_op = None
                for nid, node in state.graph.nodes.items():
                    if nid.hex == effect:
                        effect_op = node.opinion or Opinion()
                if effect_op:
                    wp = (effect_op.belief, effect_op.disbelief, effect_op.uncertainty, effect_op.prior)
                    wr = ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5))
                    result = fn(wp, wr)
                    state.metadata["sl:subjective_abduction"] = result

        elif fn_name in ("conjunction", "disjunction"):
            a_id = params.get("source_a")
            b_id = params.get("source_b")
            if a_id and b_id:
                a_op = b_op = None
                for nid, node in state.graph.nodes.items():
                    if nid.hex == a_id:
                        a_op = node.opinion or Opinion()
                    if nid.hex == b_id:
                        b_op = node.opinion or Opinion()
                if a_op and b_op:
                    result = fn((a_op.belief, a_op.disbelief, a_op.uncertainty, a_op.prior),
                                (b_op.belief, b_op.disbelief, b_op.uncertainty, b_op.prior))
                    state.metadata[f"sl:{fn_name}"] = result

        return state
