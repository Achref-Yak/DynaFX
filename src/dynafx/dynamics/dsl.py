"""System dynamics DSL — parse .sysd files into simulation-ready models.

Syntax:
    model "Name"
      dt 0.5
      from 0 to 100

      stock "Stock Name": initial_value
        + "Inflow Name": rate_expression
        - "Outflow Name": rate_expression

      table "Table Name"
        x: [0, 10, 20]
        y: [5, 15, 5]

Expressions support: +, -, *, /, parentheses, comparison (<, >, <=, >=, =),
MIN(a,b), MAX(a,b), IF(cond,a,b), ABS(x), EXP(x), LN(x), SQRT(x), SIN(x),
COS(x), PI, SMOOTH(x,delay), SMOOTHI(x,delay,init), DELAY3(input,delay),
DELAYN(input,delay,n), DELAY_FIXED(input,delay), CONVEY(input,delay),
CONVEY_BATCH(input,delay,batch), PULSE(vol,start,width), STEP(height,start),
RAMP(slope,start,end), NOISE(amplitude), UNIFORM(a,b), LOGNORMAL(mu,sigma),
ALLOCATE_FRACTION(available,demand,total), KB_QUERY(sparql,var),
KB_QUERY_TEMPLATE(template,subject_iri,var), KB_ASSERT(s,p,o,belief,graph),
and user-defined func() macros.
"""

from __future__ import annotations

import logging
import math
import random
import re

logger = logging.getLogger(__name__)
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, Optional

from dynafx.core.decomposer import SystemDecomposer
from dynafx.dynamics._parser import (
    ExprBinOp,
    ExprFuncCall,
    ExprLiteral,
    ExprNode,
    ExprParser,
    ExprRef,
    _compile_expr,
    _expand_func_calls,
    _replace_smooths,
    _serialize_expr,
    _topo_sort,
)
from dynafx.dynamics.equations import euler_step, rk4_step

# ── AST Nodes ───────────────────────────────────────────────────

@dataclass
class FlowDef:
    """A single flow attached to a stock."""
    name: str
    direction: str
    expr: str
    units: str = ""


@dataclass
class StockDef:
    """A stock (level/state variable) with its flows."""
    name: str
    initial: float
    flows: list[FlowDef] = field(default_factory=list)
    units: str = ""


@dataclass
class TableDef:
    name: str
    x: list[float]
    y: list[float]


@dataclass
class AuxDef:
    """An auxiliary variable computed from an expression each step."""
    name: str
    expr: str
    units: str = ""


@dataclass
class AgentPropDef:
    """Agent property definition."""
    name: str
    initial: float = 0.0
    min: float = 0.0
    max: float = 1e18


@dataclass
class AgentRuleDef:
    """Agent behavioral rule: when condition → effects."""
    name: str
    condition: str
    effects: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class AgentStrategy:
    """Named rule set for strategy switching."""
    name: str
    rules: list[AgentRuleDef] = field(default_factory=list)


@dataclass
class AgentDef:
    """Agent type definition with properties and behavioral rules.

    If ``strategies`` is non-empty, agent uses strategy-scoped rules
    instead of the flat ``rules`` list. ``meta_rules`` are always
    evaluated regardless of active strategy.
    """
    name: str
    count: int = 1
    properties: list[AgentPropDef] = field(default_factory=list)
    rules: list[AgentRuleDef] = field(default_factory=list)
    strategies: list[AgentStrategy] = field(default_factory=list)
    meta_rules: list[AgentRuleDef] = field(default_factory=list)
    network_type: str = "none"  # none, complete, random, small-world, scale-free


@dataclass
class QueueDef:
    """DES queue definition."""
    name: str
    capacity: int = -1       # -1 = unlimited
    initial: int = 0
    service_time: str = ""   # expression or distribution name
    arrival_rate: str = ""   # optional arrival rate expression
    servers: int = 1         # number of parallel servers
    event_driven: bool = False  # use event-driven (vs time-sliced) service


@dataclass
class ResourceDef:
    """DES resource definition (e.g., server pool)."""
    name: str
    capacity: int = 1
    cost_per_unit: float = 0.0


@dataclass
class EventDef:
    """DES event definition."""
    name: str
    rate: str = ""           # rate expression or distribution
    target_queue: str = ""
    effects: list[str] = field(default_factory=list)
    payload: Any = None
    enqueue_to: list[str] = field(default_factory=list)


@dataclass
class SubmodelDef:
    """Reusable submodel template."""
    name: str
    stocks: list[StockDef] = field(default_factory=list)
    aux_vars: list[AuxDef] = field(default_factory=list)
    tables: list[TableDef] = field(default_factory=list)


@dataclass
class IncludeDef:
    """Submodel instantiation directive."""
    submodel_name: str    # template name to include
    instance_name: str    # prefix for namespaced variables
    params: dict[str, float] = field(default_factory=dict)  # parameter overrides


@dataclass
class FuncDef:
    """A user-defined function (macro) for use in expressions."""
    name: str
    params: list[str]
    body: str


# ── Python-native DSL context managers ──────────────────────────

class _StockCtx:
    """Context manager returned by ``SysdModel.stock()``.

    Usage::

        with model.stock("x", 0.0) as s:
            s.inflow("dx", "v")
            s.outflow("leak", "k * x")
    """
    def __init__(self, model: SysdModel, name: str, initial: float = 0.0, unit: str = "") -> None:
        self._model = model
        self._stock = StockDef(name, initial, [], unit)

    def __enter__(self) -> _StockCtx:
        self._model.stocks.append(self._stock)
        self._model._bump_revision()
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def inflow(self, name: str, expr: str = "", unit: str = "") -> _StockCtx:
        self._stock.flows.append(FlowDef(name, "+", expr or name, unit))
        return self

    def outflow(self, name: str, expr: str = "", unit: str = "") -> _StockCtx:
        self._stock.flows.append(FlowDef(name, "-", expr or name, unit))
        return self


class _AgentCtx:
    """Context manager returned by ``SysdModel.agent()``.

    Usage::

        with model.agent("customer", 100) as a:
            a.prop("satisfaction", 1.0, min_val=0, max_val=1)
            a.rule("churn", "satisfaction < 0.3", ["churn_risk += 0.1"])
            a.strategy("crisis").rule("ration", "inventory < 10", ["safety_stock += 50"])
            a.meta_rule("detect", "KB_QUERY(q) > 0.5", ["SWITCH_STRATEGY('crisis')"])
    """
    def __init__(self, model: SysdModel, name: str, count: int = 1) -> None:
        self._model = model
        self._agent = AgentDef(name, count, [], [])

    def __enter__(self) -> _AgentCtx:
        self._model.agents.append(self._agent)
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def prop(self, name: str, initial: float = 0.0, min_val: float = 0.0, max_val: float = 1e18) -> _AgentCtx:
        self._agent.properties.append(AgentPropDef(name, initial, min_val, max_val))
        return self

    def rule(self, name: str, condition: str, effects: Optional[list[str]] = None, priority: int = 0) -> _AgentCtx:
        self._agent.rules.append(AgentRuleDef(name, condition, effects or [], priority))
        return self

    def strategy(self, name: str) -> _StrategyCtx:
        return _StrategyCtx(self._agent, name)

    def meta_rule(self, name: str, condition: str, effects: Optional[list[str]] = None, priority: int = 0) -> _AgentCtx:
        self._agent.meta_rules.append(AgentRuleDef(name, condition, effects or [], priority))
        return self

    def network(self, network_type: str = "none") -> _AgentCtx:
        self._agent.network_type = network_type
        return self


class _StrategyCtx:
    """Context manager returned by ``_AgentCtx.strategy()``."""
    def __init__(self, agent: AgentDef, name: str) -> None:
        self._strategy = AgentStrategy(name)
        agent.strategies.append(self._strategy)

    def __enter__(self) -> _StrategyCtx:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def rule(self, name: str, condition: str, effects: Optional[list[str]] = None, priority: int = 0) -> _StrategyCtx:
        self._strategy.rules.append(AgentRuleDef(name, condition, effects or [], priority))
        return self


class _SubmodelStockCtx:
    """Context manager for a stock inside a submodel definition."""
    def __init__(self, stock: StockDef) -> None:
        self._stock = stock

    def __enter__(self) -> _SubmodelStockCtx:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def inflow(self, name: str, expr: str = "", unit: str = "") -> _SubmodelStockCtx:
        self._stock.flows.append(FlowDef(name, "+", expr or name, unit))
        return self

    def outflow(self, name: str, expr: str = "", unit: str = "") -> _SubmodelStockCtx:
        self._stock.flows.append(FlowDef(name, "-", expr or name, unit))
        return self


class _SubmodelCtx:
    """Context manager returned by ``SysdModel.submodel()``.

    Usage::

        with model.submodel("sector") as sm:
            with sm.stock("population", 1000) as s:
                s.inflow("births", "population * birth_rate")
            sm.aux("birth_rate", "0.02")
    """
    def __init__(self, model: SysdModel, name: str) -> None:
        self._model = model
        self._submodel = SubmodelDef(name, [], [])

    def __enter__(self) -> _SubmodelCtx:
        self._model.submodels.append(self._submodel)
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def stock(self, name: str, initial: float = 0.0, unit: str = "") -> _SubmodelStockCtx:
        s = StockDef(name, initial, [], unit)
        self._submodel.stocks.append(s)
        return _SubmodelStockCtx(s)

    def aux(self, name: str, expr: str, unit: str = "") -> _SubmodelCtx:
        self._submodel.aux_vars.append(AuxDef(name, expr, unit))
        return self


@dataclass
class SysdModel:
    """A system dynamics model composed of stocks, flows, auxiliaries, and optionally ABM/DES components.

    Parse from a ``.sysd`` file with ``parse_sysd_file()`` or construct
    programmatically via the Python-native DSL::

        model = SysdModel("vibration")
        with model.stock("x", 0.0) as s:
            s.inflow("dx", "v")
        model.aux("v", "dx/dt")
        model.dt = 0.01
        result = model.simulate()
    """

    name: str = ""
    dt: float = 1.0
    t_span: tuple[float, float] = (0.0, 100.0)
    stocks: list[StockDef] = field(default_factory=list)
    tables: list[TableDef] = field(default_factory=list)
    aux_vars: list[AuxDef] = field(default_factory=list)
    emergent_props: list = field(default_factory=list)
    agents: list[AgentDef] = field(default_factory=list)
    queues: list[QueueDef] = field(default_factory=list)
    resources: list[ResourceDef] = field(default_factory=list)
    events: list[EventDef] = field(default_factory=list)
    submodels: list[SubmodelDef] = field(default_factory=list)
    includes: list[IncludeDef] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    func_defs: list[FuncDef] = field(default_factory=list)
    _compiled_cache: Any = field(default=None, repr=False)
    _model_revision: int = 0

    def _bump_revision(self) -> None:
        self._model_revision += 1

    def to_decomposer(self) -> SystemDecomposer:
        d = SystemDecomposer(name=self.name)
        for s in self.stocks:
            nid = d.add_node(s.name, type="STOCK")
            if nid and s.initial != 0.0:
                node = d.graph.nodes.get(nid)
                if node:
                    node.metadata["parameter"] = s.initial
        for s in self.stocks:
            for f in s.flows:
                pol = 1 if f.direction == "+" else -1
                d.add_node(f.name, type="FLOW")
                d.add_edge(f.name, s.name, "CAUSES", polarity=pol)
        for t in self.tables:
            d.graph.metadata.setdefault("sysd_tables", {})[t.name] = {
                "x": t.x, "y": t.y,
            }
        d.graph.metadata["sysd_model"] = {
            "name": self.name,
            "dt": self.dt,
            "t_span": list(self.t_span),
        }
        return d

    # ── Python-native DSL methods ────────────────────────────────

    def stock(self, name: str, initial: float = 0.0, unit: str = "") -> _StockCtx:
        """Define a stock with optional flows. Use as context manager::

            with model.stock("x", 0.0) as s:
                s.inflow("dx", "v")
                s.outflow("leak", "k * x")
        """
        return _StockCtx(self, name, initial, unit)

    def aux(self, name: str, expr: str, unit: str = "") -> SysdModel:
        """Add an auxiliary variable computed each step."""
        self.aux_vars.append(AuxDef(name, expr, unit))
        self._bump_revision()
        return self

    def table(self, name: str, x: list[float], y: list[float]) -> SysdModel:
        """Add a lookup table."""
        self.tables.append(TableDef(name, x, y))
        self._bump_revision()
        return self

    def param(self, name: str, value: float) -> SysdModel:
        """Add a default parameter value. Override at simulate time::

            model.param("k", 2.0)
            result = model.simulate(params={"k": 3.0})  # overrides default

        """
        self.params[name] = value
        return self

    def func(self, name: str, params: list[str], body: str) -> SysdModel:
        """Define a reusable function for use in expressions.

        Usage::

            model.func("square", ["x"], "x * x")
            model.aux("KE", "0.5 * m * square(v)")

        Functions are expanded at compile time (macros).
        """
        self.func_defs.append(FuncDef(name, params, body))
        self._bump_revision()
        return self

    def agent(self, name: str, count: int = 1) -> _AgentCtx:
        """Define an agent type with properties and rules. Use as context manager::

            with model.agent("customer", 100) as a:
                a.prop("satisfaction", 1.0, min_val=0, max_val=1)
                a.rule("churn", "satisfaction < 0.3", ["churn_risk += 0.1"])
        """
        return _AgentCtx(self, name, count)

    def queue(self, name: str, capacity: int = -1, service_time: str = "",
              arrival_rate: str = "", initial: int = 0, servers: int = 1,
              event_driven: bool = False) -> SysdModel:
        """Add a DES queue."""
        self.queues.append(QueueDef(name, capacity, initial, service_time, arrival_rate, servers, event_driven))
        return self

    def resource(self, name: str, capacity: int = 1, cost_per_unit: float = 0.0) -> SysdModel:
        """Add a DES resource."""
        self.resources.append(ResourceDef(name, capacity, cost_per_unit))
        return self

    def event(self, name: str, rate: str = "",
              target_queue: str = "", effects: Optional[list[str]] = None) -> SysdModel:
        """Add a DES event."""
        self.events.append(EventDef(name, rate, target_queue, effects or []))
        return self

    def submodel(self, name: str) -> _SubmodelCtx:
        """Define a submodel template. Use as context manager::

            with model.submodel("sector") as sm:
                with sm.stock("population", 1000) as s:
                    s.inflow("births", "population * birth_rate")
                sm.aux("birth_rate", "0.02")
        """
        return _SubmodelCtx(self, name)

    def include(self, name: str, alias: str = "",
                params: Optional[dict[str, float]] = None) -> SysdModel:
        """Instantiate a submodel with optional parameter overrides."""
        self.includes.append(IncludeDef(name, alias or f"{name}_inst", params or {}))
        return self

    def import_data(
        self,
        path: str,
        fill: str = "forward",
        time_unit: str = "auto",
    ) -> dict[str, Any]:
        """Import time series data from a CSV file.

        CSV format: first column is time, subsequent columns are variable names.
        Returns a dict of {variable_name: [(time, value), ...]} for use as
        forcing functions or calibration data.

        The data is also stored in self._imported_data for use during simulation.

        Args:
            path: Path to CSV file.
            fill: Missing data strategy: "forward" (forward-fill),
                  "interpolate" (linear), "zero" (fill with 0.0).
            time_unit: "auto" (auto-detect datetime or float),
                       "hours", "days", "seconds" (convert datetime to this unit).
                       For float columns, parsed directly regardless of time_unit.
        """

        raw = self._read_csv_auto(path, time_unit)
        data = self._fill_missing(raw, fill)
        self._imported_data = data
        return data

    def _read_csv_auto(
        self, path: str, time_unit: str = "auto",
    ) -> dict[str, list[tuple[float, float]]]:
        """Read CSV, auto-detecting datetime vs float time column."""
        import csv
        from datetime import datetime

        data: dict[str, list[tuple[float, float]]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for col_name in header[1:]:
                data[col_name.strip()] = []

            rows = list(reader)
            if not rows:
                return data

            # Detect time column type from first non-empty row
            time_is_datetime = False
            time_ref: Optional[datetime] = None
            for row in rows:
                if row and row[0].strip():
                    first_val = row[0].strip()
                    try:
                        float(first_val)
                    except ValueError:
                        try:
                            time_ref = datetime.fromisoformat(first_val)
                            time_is_datetime = True
                        except (ValueError, TypeError):
                            pass
                    break

            for row in rows:
                if not row or not row[0].strip():
                    continue
                raw_t = row[0].strip()
                try:
                    if time_is_datetime:
                        dt = datetime.fromisoformat(raw_t)
                        if time_ref is None:
                            time_ref = dt
                        t = (dt - time_ref).total_seconds() / 3600.0  # hours
                        if time_unit == "days":
                            t /= 24.0
                        elif time_unit == "seconds":
                            t *= 3600.0
                    else:
                        t = float(raw_t)
                except (ValueError, TypeError):
                    continue

                for i, col_name in enumerate(header[1:], 1):
                    if i < len(row):
                        raw_val = row[i].strip()
                        if raw_val:
                            try:
                                val = float(raw_val)
                                data[col_name.strip()].append((t, val))
                            except ValueError:
                                data[col_name.strip()].append((t, None))
                        else:
                            data[col_name.strip()].append((t, None))

        return data

    @staticmethod
    def _fill_missing(
        data: dict[str, list[tuple[float, float | None]]],
        fill: str = "forward",
    ) -> dict[str, list[tuple[float, float]]]:
        """Fill missing (None) values in imported time series."""
        result: dict[str, list[tuple[float, float]]] = {}
        for name, series in data.items():
            filled: list[tuple[float, float]] = []
            # First pass: collect valid indices
            valid_indices = [i for i, (_, v) in enumerate(series) if v is not None]
            if not valid_indices:
                result[name] = [(t, 0.0) for t, _ in series]
                continue

            if fill == "zero":
                result[name] = [(t, v if v is not None else 0.0) for t, v in series]
                continue

            last_val: float = series[valid_indices[0]][1]  # type: ignore[assignment]
            for i, (t, v) in enumerate(series):
                if v is not None:
                    filled.append((t, v))
                    last_val = v
                elif fill == "forward":
                    filled.append((t, last_val))
                elif fill == "interpolate":
                    # Find next valid index after i
                    next_valid = next((j for j in valid_indices if j > i), None)
                    if next_valid is not None:
                        prev_valid = valid_indices[max(k for k, idx in enumerate(valid_indices) if idx < i)] if any(idx < i for idx in valid_indices) else i
                        if prev_valid != i and next_valid > prev_valid:
                            prev_t, prev_v = series[prev_valid]
                            next_t, next_v = series[next_valid]
                            frac = ((t - prev_t) / (next_t - prev_t)) if next_t > prev_t else 1.0
                            interp_val = prev_v + frac * (next_v - prev_v)
                            filled.append((t, interp_val))
                        else:
                            filled.append((t, last_val))
                    else:
                        filled.append((t, last_val))
            result[name] = filled
        return result

    def merge_data(self, paths: list[str], fill: str = "forward", time_unit: str = "auto") -> dict[str, Any]:
        """Import and merge data from multiple CSV files.

        All files are read with the same time reference. Returns a single
        dict combining all variable columns. Conflicts raise ValueError.
        """
        merged: dict[str, list[tuple[float, float]]] = {}
        for path in paths:
            raw = self._read_csv_auto(path, time_unit)
            for name, series in raw.items():
                if name in merged:
                    raise ValueError(f"Duplicated variable '{name}' across CSV files")
                merged[name] = series
        merged = self._fill_missing(merged, fill)
        self._imported_data = merged
        return merged

    def get_imported_interpolator(self, name: str):
        """Get a linear interpolation function for imported data.

        Returns a callable f(t) that interpolates the imported data.
        """
        if not hasattr(self, "_imported_data") or name not in self._imported_data:
            return lambda t: 0.0
        series = self._imported_data[name]
        if not series:
            return lambda t: 0.0
        times = [p[0] for p in series]
        values = [p[1] for p in series]
        def interpolator(t: float) -> float:
            if t <= times[0]:
                return values[0]
            if t >= times[-1]:
                return values[-1]
            # Linear interpolation
            for i in range(len(times) - 1):
                if times[i] <= t <= times[i + 1]:
                    denom = times[i + 1] - times[i]
                    frac = (t - times[i]) / denom if denom > 0 else 0.0
                    return values[i] + frac * (values[i + 1] - values[i])
            return values[-1]
        return interpolator

    def simulate(
        self,
        method: str = "rk4",
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        params: Optional[dict[str, Any]] = None,
        kb: Any = None,
    ) -> SysdModelResult:
        """Run a simulation and return the trajectory.

        Args:
            method: Integration method (``"rk4"`` or ``"euler"``).
            t_span: Override time range (start, end).
            dt: Override time step.
            params: Parameter overrides (name → value).
            kb: Optional TripleStore — enables KB_QUERY/KB_ASSERT builtins
                for expressions, ABM rules, and DES rates.

        Returns:
            SysdModelResult with stocks, values, times, and optional
            ABM/DES engine references.
        """
        t_span = t_span or self.t_span
        step = dt or self.dt
        step_fn = rk4_step if method == "rk4" else euler_step
        if params is None:
            params = {}

        # Merge model-level default params (call-site wins)
        params = {**self.params, **params}

        for t in self.tables:
            params[t.name] = LookupTable(t.x, t.y)
        params["dt"] = step

        # Merge aux defaults into params (so calibration/optimization can override)
        for a in self.aux_vars:
            if a.name not in params:
                try:
                    params[a.name] = float(a.expr)
                except ValueError:
                    pass  # Expression auxes stay as expressions

        # Cache compiled artifacts for subsequent simulate() calls
        if (self._compiled_cache is None
                or self._compiled_cache.revision != self._model_revision
                or len(self._compiled_cache.stock_names) != len(self.stocks)
                or len(self._compiled_cache.aux_names_ordered) != len(self.aux_vars)):
            self._compiled_cache = _compile_system(self)
        build_result = _build_system(self, params, self.emergent_props, seed=42, cache=self._compiled_cache, kb=kb)
        f = build_result[0]
        stock_names = build_result[1]
        y0 = build_result[2]
        aux_count = build_result[3]
        pipeline_info = build_result[4]

        # Extract KB builtins from f() closure for injection into eval namespaces
        _kb_builtins_sim: dict[str, Any] = getattr(f, '_kb_builtins', {})

        # Initialize pipeline delays at t0
        if pipeline_info is not None:
            y0 = list(y0)
            pipeline_info["process"](t_span[0], y0, params)

        # Initialize ABM engine if agents are defined
        abm_engine = None
        abm_initial_metrics: dict[str, float] = {}
        if self.agents:
            from dynafx.dynamics.agent import ABMEngine
            abm_engine = ABMEngine(self.agents, seed=42, kb_builtins=_kb_builtins_sim)
            abm_engine.initialize()
            # Compute initial ABM metrics and merge into params for t=0 aux eval
            abm_initial_metrics = abm_engine.get_metrics()
            params.update(abm_initial_metrics)

        # Initialize DES engine if queues/resources/events are defined
        des_engine = None
        if self.queues or self.resources or self.events:
            from dynafx.dynamics.des import (
                DESEngine,
                Queue,
                Resource,
            )
            des_engine = DESEngine()
            for q in self.queues:
                q_obj = Queue(q.name, q.capacity, q.service_time, servers=q.servers, event_driven=q.event_driven)
                # Compile service_time expression if provided
                if q.service_time:
                    try:
                        from dynafx.dynamics._parser import ExprParser, _compile_expr
                        st_node = ExprParser(q.service_time).parse()
                        st_compiled = _compile_expr(st_node, set(), set())
                        q_obj._compiled_service_time = lambda _c=st_compiled: eval(
                            _c, {"__builtins__": {}}, {**params, **dict(zip(stock_names, y0, strict=False))}
                        )
                    except Exception as _e:
                        logger.warning("Failed to compile service_time '%s' — %s", q.service_time, _e)
                des_engine.add_queue(q_obj)
            for r in self.resources:
                r_obj = Resource(r.name, r.capacity, r.cost_per_unit)
                des_engine.add_resource(r_obj)
            for ev in self.events:
                if ev.rate:
                    des_engine.schedule_event(0.0, ev.name, ev.payload)
                for enq in ev.enqueue_to:
                    if enq in des_engine.queues:
                        des_engine.queues[enq].enqueue(
                            {"event": ev.name, "time": 0.0}, 0.0,
                            event_queue=des_engine.event_queue,
                        )

        # Compile DES arrival rate expressions for queue injection
        des_arrival_injectors: list[tuple[str, Any]] = []
        des_arrival_accum: dict[str, float] = {}
        if des_engine:
            for q in self.queues:
                if q.arrival_rate:
                    try:
                        from dynafx.dynamics._parser import ExprParser, _compile_expr
                        ar_node = ExprParser(q.arrival_rate).parse()
                        ar_compiled = _compile_expr(ar_node, set(), set())
                        ar_code = compile(ar_compiled, "<arrival_rate>", "eval")
                        des_arrival_injectors.append((q.name, ar_code))
                        des_arrival_accum[q.name] = 0.0
                    except Exception as _e:
                        logger.warning("Failed to compile arrival_rate '%s' — %s", q.arrival_rate, _e)

        t0, t_end = t_span
        direction = 1 if t_end >= t0 else -1
        y = list(y0)
        times = [t0]
        y_hist = [list(y)]
        params_history = [dict(params)]
        abm_metrics_history: list[dict[str, float]] = [dict(abm_initial_metrics)] if abm_engine else []
        des_metrics_history: list[dict[str, float]] = [{}] if des_engine else []

        _rng = random.Random(42)  # seeded for reproducibility (ABM + aux-replay)

        while abs(t0 - t_end) > 1e-12:
            remaining = abs(t_end - t0)
            actual_step = direction * (remaining if remaining < abs(step) else abs(step))

            # Compute aux values from current state for ABM/DES visibility
            _aux_info = getattr(f, '_aux_info', None)
            _a_abm: dict[str, float] = {}
            if _aux_info is not None:
                _np_abm_build, _cp_abm, _b_abm, _nb_abm, _anames_abm, _acode_abm, _sc_abm = _aux_info
                _np_abm = {k: v for k, v in params.items() if isinstance(v, (int, float))}
                _sp_abm = {k: v for k, v in params.items() if isinstance(v, str)}
                _s_abm = dict(zip(stock_names, y, strict=False))
                _s_abm.update(_np_abm)
                if _sp_abm:
                    _s_abm.update(_sp_abm)
                _ns_abm = {
                    **_b_abm, **_kb_builtins_sim, **_np_abm, **_sp_abm, "_s": _s_abm, "_a": _a_abm, "_p": params, "t": t0,
                    "PULSE": lambda volume, start, width, _t=t0: volume if start <= _t < start + width else 0.0,
                    "STEP": lambda height, start, _t=t0: height if _t >= start else 0.0,
                    "RAMP": lambda slope, start, end, _t=t0: 0.0 if _t < start else slope * (_t - start) if _t <= end else slope * (end - start),
                    "NOISE": lambda amplitude, _rng=_rng: _rng.uniform(-amplitude, amplitude),
                    "UNIFORM": lambda a, b, _rng=_rng: _rng.uniform(a, b),
                    "LOGNORMAL": lambda mu, sigma, _rng=_rng: _rng.lognormvariate(mu, sigma) if sigma > 0 else mu,
                }
                for _k, _v in _cp_abm:
                    _ns_abm[_k] = _v
                for _i, _aname in enumerate(_anames_abm):
                    if _aname in _np_abm:
                        _a_abm[_aname] = _np_abm[_aname]
                    else:
                        _a_abm[_aname] = eval(_acode_abm[_i], _nb_abm, _ns_abm)
                _s_abm.update(_a_abm)

            # Run ABM step BEFORE stock advancement so metrics feed current step
            if abm_engine:
                shared_state = dict(zip(stock_names, y, strict=False))
                shared_state["t"] = t0
                shared_state.update(params)
                if _aux_info is not None:
                    shared_state.update(_a_abm)
                if _kb_builtins_sim:
                    shared_state.update(_kb_builtins_sim)
                abm_metrics = abm_engine.step(t0, actual_step, shared_state)
                params.update(abm_metrics)
                abm_metrics_history.append(dict(abm_metrics))

            # Inject DES arrivals from arrival_rate expressions
            if des_arrival_injectors:
                import math
                _builtins_ar = {
                    "__builtins__": {},
                    "MIN": min, "MAX": max,
                    "IF": lambda c, a, b: a if c else b,
                    "ABS": abs, "EXP": math.exp, "LN": math.log,
                    "SQRT": math.sqrt, "SIN": math.sin, "COS": math.cos,
                    "PI": math.pi,
                    **_kb_builtins_sim,
                }
                _s_ar = dict(zip(stock_names, y, strict=False))
                _s_ar.update({k: v for k, v in params.items() if isinstance(v, (int, float))})
                _sp_ar = {k: v for k, v in params.items() if isinstance(v, str)}
                if _sp_ar:
                    _s_ar.update(_sp_ar)
                _p_ar = dict(params)
                _all_vars_ar = dict(_p_ar)
                _ns_arrival = {"_s": _s_ar, "_p": _p_ar, "t": t0, **_all_vars_ar}
                if _a_abm:
                    _s_ar.update(_a_abm)
                for _qname, _ar_code in des_arrival_injectors:
                    _rate = eval(_ar_code, _builtins_ar, _ns_arrival)
                    if _rate > 0 and _qname in des_engine.queues:
                        des_arrival_accum[_qname] += _rate * actual_step
                        while des_arrival_accum[_qname] >= 1.0:
                            des_engine.queues[_qname].enqueue(
                                {"source": _qname, "time": t0}, t0,
                                event_queue=des_engine.event_queue,
                            )
                            des_arrival_accum[_qname] -= 1.0

            # Run DES step BEFORE stock advancement so metrics feed current step
            if des_engine:
                shared_state = dict(zip(stock_names, y, strict=False))
                for q in self.queues:
                    if q.service_time and q.name in des_engine.queues:
                        q_obj = des_engine.queues[q.name]
                        try:
                            from dynafx.dynamics._parser import ExprParser, _compile_expr
                            st_node = ExprParser(q.service_time).parse()
                            st_compiled = _compile_expr(st_node, set(), set())
                            _state_snapshot = dict(shared_state)
                            _state_snapshot.update(_kb_builtins_sim)
                            q_obj._compiled_service_time = lambda _c=st_compiled, _s=_state_snapshot: eval(
                                _c, {"__builtins__": {}}, {**params, **_s}
                            )
                        except Exception as _e:
                            logger.warning("Failed to recompile service_time '%s' — %s", q.service_time, _e)
                des_metrics = des_engine.step(max(t0, 0.0), actual_step)
                des_metrics_history.append(dict(des_metrics))
                step_params = {**params, **des_metrics}
            else:
                step_params = params

            # Stock advancement using current params (includes ABM/DES metrics)
            y = step_fn(f, t0, y, actual_step, step_params)
            if remaining < abs(step):
                t0 = t_end
            else:
                t0 += actual_step

            # Process pipeline delays (DELAY_FIXED / CONVEY) — ONCE per step
            if pipeline_info is not None:
                pipeline_info["process"](t0, y, params)

            times.append(t0)
            y_hist.append(list(y))
            params_history.append(dict(params))

        # Record aux variable values at each timestep (post-hoc)
        # Uses per-step params_history so ABM/DES metrics are visible.
        aux_values: dict[str, list[float]] = {}
        if self._compiled_cache and self._compiled_cache.aux_names_ordered:
            _aux_anames = self._compiled_cache.aux_names_ordered
            _aux_code = self._compiled_cache.aux_code
            _aux_builtins = self._compiled_cache.builtins
            _full_names = self._compiled_cache.all_names
            _no_builtins = {"__builtins__": {}}
            for _si, (_t, _y_full) in enumerate(zip(times, y_hist, strict=False)):
                _step_p = dict(params_history[_si]) if _si < len(params_history) else dict(params)
                for _tbl in self.tables:
                    _step_p[_tbl.name] = LookupTable(_tbl.x, _tbl.y)
                _step_p["dt"] = step
                _num_p = {k: v for k, v in _step_p.items() if isinstance(v, (int, float))}
                _str_p = {k: v for k, v in _step_p.items() if isinstance(v, str)}
                _call_p = [(k, v) for k, v in _step_p.items() if callable(v)]
                _s = dict(zip(_full_names, _y_full, strict=False))
                _s.update(_num_p)
                if _str_p:
                    _s.update(_str_p)
                _a: dict[str, float] = {}
                _ns = {
                    **_aux_builtins, **_kb_builtins_sim, **_num_p, **_str_p, "_s": _s, "_p": _step_p, "_a": _a, "t": _t,
                    "PULSE": lambda volume, start, width, _t=_t: volume if start <= _t < start + width else 0.0,
                    "STEP": lambda height, start, _t=_t: height if _t >= start else 0.0,
                    "RAMP": lambda slope, start, end, _t=_t: (
                        0.0 if _t < start else slope * (_t - start) if _t <= end else slope * (end - start)
                    ),
                    "NOISE": lambda amplitude, _rng=_rng: _rng.uniform(-amplitude, amplitude),
                    "UNIFORM": lambda a, b, _rng=_rng: _rng.uniform(a, b),
                    "LOGNORMAL": lambda mu, sigma, _rng=_rng: _rng.lognormvariate(mu, sigma) if sigma > 0 else mu,
                }
                for _k, _v in _call_p:
                    _ns[_k] = _v
                for _i, _aname in enumerate(_aux_anames):
                    if _aname in _num_p:
                        _a[_aname] = _num_p[_aname]
                    else:
                        _a[_aname] = eval(_aux_code[_i], _no_builtins, _ns)
                    _ns["_a"] = _a
                for _aname in _aux_anames:
                    aux_values.setdefault(_aname, []).append(_a[_aname])

        if aux_count:
            pure_stocks = len(stock_names) - aux_count
            y_hist = [row[:pure_stocks] for row in y_hist]
            stock_names = stock_names[:pure_stocks]

        return SysdModelResult(
            times=times,
            stocks=stock_names,
            values={
                name: [row[i] for row in y_hist]
                for i, name in enumerate(stock_names)
            },
            final_state=y_hist[-1],
            method=method,
            steps=len(times) - 1,
            model_name=self.name,
            abm_engine=abm_engine,
            des_engine=des_engine,
            aux_values=aux_values,
            abm_metrics_history=abm_metrics_history,
            des_metrics_history=des_metrics_history,
        )

    def validate(self, params: Optional[set[str]] = None) -> ValidationResult:
        result = ValidationResult()
        all_names: set[str] = set()
        all_names.update(_get_builtin_names())
        for s in self.stocks:
            all_names.add(s.name)
        for a in self.aux_vars:
            all_names.add(a.name)
        for t in self.tables:
            all_names.add(t.name)
        if params:
            all_names.update(params)

        flow_name_counts: dict[str, list[tuple[str, str]]] = {}

        def _check_expr(expr_str: str, location: str) -> None:
            try:
                node = ExprParser(expr_str).parse()
            except SyntaxError:
                result.errors.append(ValidationIssue("error", f"Syntax error in expression: {expr_str}", location))
                return
            _walk_refs(node, location)

        def _walk_refs(node: ExprNode, location: str) -> None:
            if isinstance(node, ExprRef):
                if node.name not in all_names:
                    result.errors.append(ValidationIssue(
                        "error", f"Unknown identifier '{node.name}'", location
                    ))
            elif isinstance(node, ExprBinOp):
                _walk_refs(node.left, location)
                _walk_refs(node.right, location)
            elif isinstance(node, ExprFuncCall):
                for a in node.args:
                    _walk_refs(a, location)

        def _has_stock_protection(expr_str: str, stock_name: str) -> bool:
            """Check if expression uses MAX(0, stock_name) or MIN(stock_name/dt, ...)."""
            pattern = re.compile(
                rf"\bMAX\s*\(\s*0\s*,\s*{re.escape(stock_name)}\b",
                re.IGNORECASE
            )
            if pattern.search(expr_str):
                return True
            pattern2 = re.compile(
                rf"\bMIN\s*\(\s*{re.escape(stock_name)}\s*/\s*dt",
                re.IGNORECASE
            )
            return bool(pattern2.search(expr_str))

        for s in self.stocks:
            for f in s.flows:
                loc = f"stock '{s.name}': flow '{f.name}'"
                _check_expr(f.expr, loc)
                key = f.name
                flow_name_counts.setdefault(key, [])
                flow_name_counts[key].append((f.direction, s.name))

                if f.direction == "-" and s.initial > 0:
                    if not _has_stock_protection(f.expr, s.name):
                        result.warnings.append(ValidationIssue(
                            "warning",
                            f"Outflow '{f.name}' may drive stock '{s.name}' negative — "
                            f"consider using MAX(0, {s.name}) or MIN({s.name} / dt, ...)",
                            loc,
                        ))

        for a in self.aux_vars:
            _check_expr(a.expr, f"aux '{a.name}'")

        # Check for multi-outflow stocks (potential double-drain)
        for s in self.stocks:
            outflows = [f for f in s.flows if f.direction == "-"]
            if len(outflows) >= 2:
                names = [f.name for f in outflows]
                all_min_dt = all(
                    f.expr.strip().startswith("MIN(")
                    and re.search(r'/\s*dt', f.expr)
                    and "," in f.expr
                    for f in outflows
                )
                if all_min_dt:
                    result.infos.append(ValidationIssue(
                        "info",
                        f"Stock '{s.name}' has {len(outflows)} outflows ({', '.join(names)}) "
                        f"using MIN(…/dt, demand) pattern — compiler will auto-apply "
                        f"proportional allocation via ALLOCATE_FRACTION",
                        f"stock '{s.name}'",
                    ))
                else:
                    has_allocate = any("ALLOCATE_FRACTION" in f.expr for f in outflows)
                    if not has_allocate:
                        result.warnings.append(ValidationIssue(
                            "warning",
                            f"Stock '{s.name}' has {len(outflows)} outflows ({', '.join(names)}) "
                            f"not following the MIN(…/dt, demand) pattern — compiler cannot "
                            f"auto-allocate. Use ALLOCATE_FRACTION or explicit cascading.",
                            f"stock '{s.name}'",
                        ))

        for name, sides in flow_name_counts.items():
            if len(sides) == 1:
                result.warnings.append(ValidationIssue(
                    "warning",
                    f"Flow '{name}' has only one side ({sides[0][0]} in stock '{sides[0][1]}') "
                    f"— check conservation",
                ))
            elif len(sides) > 2:
                result.errors.append(ValidationIssue(
                    "error",
                    f"Flow '{name}' appears {len(sides)} times, expected exactly 2 (one +, one -)",
                ))

        # Check for stocks with initial=0 used as divisor
        zero_stocks = {s.name for s in self.stocks if s.initial == 0}
        for s in self.stocks:
            for f in s.flows:
                if "/" in f.expr:
                    for zs in zero_stocks:
                        if re.search(rf"\b{re.escape(zs)}\b", f.expr.split("/", 1)[1]):
                            result.warnings.append(ValidationIssue(
                                "warning",
                                f"Stock '{zs}' has initial value 0 and appears as a divisor in flow '{f.name}'",
                                f"stock '{s.name}': flow '{f.name}'",
                            ))

        return result

    def check_consistency(self):
        """Run stock-flow structural consistency checks.

        Returns ConsistencyResult with violations for partition sums,
        one-sided flows, zero divisors, and cross-type flows.
        """
        from dynafx.dynamics.emergent import run_consistency_checks
        return run_consistency_checks(self)

    def simulate_ensemble(
        self,
        params: dict[str, tuple[float, float, str]],
        n: int = 100,
        method: str = "rk4",
        seed: int = 42,
        fixed_params: Optional[dict[str, Any]] = None,
        **sim_kwargs: Any,
    ) -> dict[str, Any]:
        """Run ensemble simulation with parameter uncertainty.

        Args:
            params: dict of param_name -> (low, high, [dist]) where dist is
                    "uniform" (default), "normal", or "lognormal"
            n: number of ensemble members
            method: solver method
            seed: random seed for reproducibility
            fixed_params: non-varying parameters to include in every sample

        Returns dict with times, stocks, mean, std, p5, p95, trajectories.
        """
        rng = random.Random(seed)
        trajectories: list[SysdModelResult] = []
        base_params: dict[str, Any] = dict(fixed_params) if fixed_params else {}

        for t in self.tables:
            base_params[t.name] = LookupTable(t.x, t.y)
        base_params["dt"] = self.dt

        samples: list[dict[str, Any]] = []
        for _ in range(n):
            sample = dict(base_params)
            for pname, spec in params.items():
                low, high = spec[0], spec[1]
                dist = spec[2] if len(spec) > 2 else "uniform"
                if dist == "uniform":
                    val = low + rng.random() * (high - low)
                elif dist == "normal":
                    mean_val = (low + high) / 2
                    std_val = (high - low) / 4
                    val = rng.gauss(mean_val, std_val)
                elif dist == "lognormal":
                    mu = math.log(max(low, 1e-10))
                    sigma = (math.log(high) - mu) / 3
                    val = rng.lognormvariate(mu, sigma)
                else:
                    val = low + rng.random() * (high - low)
                sample[pname] = val
            samples.append(sample)

        if not samples:
            return {"times": [], "stocks": {}, "mean": {}, "std": {}, "p5": {}, "p95": {}, "trajectories": []}

        for sample in samples:
            result = self.simulate(method=method, params=sample, **sim_kwargs)
            trajectories.append(result)

        n_stocks = len(trajectories[0].stocks)
        stock_names = list(trajectories[0].stocks)
        times = list(trajectories[0].times)
        n_t = len(times)

        mean: dict[str, list[float]] = {s: [0.0] * n_t for s in stock_names}
        std: dict[str, list[float]] = {s: [0.0] * n_t for s in stock_names}
        p5: dict[str, list[float]] = {s: [0.0] * n_t for s in stock_names}
        p95: dict[str, list[float]] = {s: [0.0] * n_t for s in stock_names}

        for ti in range(n_t):
            for si, s in enumerate(stock_names):
                vals = sorted(traj.values[s][ti] for traj in trajectories)
                mean[s][ti] = sum(vals) / len(vals)
                if len(vals) >= 2:
                    var = sum((v - mean[s][ti]) ** 2 for v in vals) / (len(vals) - 1)
                    std[s][ti] = math.sqrt(var)
                idx5 = max(0, int(0.05 * len(vals)))
                idx95 = min(len(vals) - 1, int(0.95 * len(vals)))
                p5[s][ti] = vals[idx5]
                p95[s][ti] = vals[idx95]

        return {
            "times": times,
            "stocks": stock_names,
            "mean": mean,
            "std": std,
            "p5": p5,
            "p95": p95,
            "trajectories": trajectories,
        }


# ── Simulation Result ────────────────────────────────────────────

@dataclass
class SysdModelResult:
    """Returned by ``SysdModel.simulate()`` — holds the full trajectory.

    Access per-stock values via ``result.values[stock_name]``.
    Supports dict-style access (``result["times"]``) for backward compat.
    """

    times: list[float]
    stocks: list[str]
    values: dict[str, list[float]]
    final_state: list[float]
    method: str
    steps: int
    model_name: str = ""
    abm_engine: Any = None
    des_engine: Any = None
    aux_values: dict[str, list[float]] = field(default_factory=dict)
    abm_metrics_history: list[dict[str, float]] = field(default_factory=list)
    des_metrics_history: list[dict[str, float]] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def plot(
        self,
        path: str,
        stocks: Optional[list[str]] = None,
        subplots: bool = False,
        title: Optional[str] = None,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")
            return
        names = stocks or self.stocks
        if subplots:
            fig, axes = plt.subplots(len(names), 1, figsize=(8, 2 * len(names)), sharex=True)
            if len(names) == 1:
                axes = [axes]
            for ax, name in zip(axes, names, strict=False):
                ax.plot(self.times, self.values[name], label=name)
                ax.set_ylabel(name)
                ax.legend()
                ax.grid(True)
            axes[-1].set_xlabel("Time")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            for name in names:
                ax.plot(self.times, self.values[name], label=name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(title or self.model_name)
            ax.legend()
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_with_bands(
        self,
        path: str,
        mean: dict[str, list[float]],
        std: dict[str, list[float]],
        p5: dict[str, list[float]],
        p95: dict[str, list[float]],
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        t = self.times
        for stock in self.stocks:
            ax.plot(t, mean[stock], label=stock)
            ax.fill_between(t, p5[stock], p95[stock], alpha=0.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"{self.model_name} — Sensitivity (5th–95th percentile)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def export_results(self, path: str) -> None:
        """Export simulation results to a CSV file.

        First column is time, subsequent columns are variable values.
        """
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["time"] + self.stocks)
            # Data rows
            for i, t in enumerate(self.times):
                row = [t] + [self.values[name][i] for name in self.stocks]
                writer.writerow(row)


# ── Validation Result ────────────────────────────────────────────

@dataclass
class ValidationIssue:
    level: str
    message: str
    location: str = ""

@dataclass
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    infos: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: ValidationResult) -> ValidationResult:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.infos.extend(other.infos)
        return self

    def print_report(self) -> None:
        for issue in self.errors:
            print(f"  ERROR: {issue.message}")
        for issue in self.warnings:
            print(f"  WARNING: {issue.message}")
        for issue in self.infos:
            print(f"  INFO: {issue.message}")
        if self.is_valid:
            print("  ✅ Model is valid")
        else:
            print(f"  ❌ {len(self.errors)} error(s) found")


# ── Builtin names for validation ─────────────────────────────────

_BASE_BUILTIN_NAMES: set[str] = {
    "t", "dt",
    # Math
    "MIN", "MAX", "IF", "ABS", "EXP", "LN", "SQRT", "SIN", "COS", "PI",
    # Smoothing / delays
    "SMOOTH", "SMOOTHI", "DELAY3", "DELAYN", "DELAY_FIXED", "CONVEY", "CONVEY_BATCH",
    # Time functions
    "PULSE", "STEP", "RAMP", "NOISE",
    # Stochastic distributions
    "UNIFORM", "LOGNORMAL",
}

def _get_builtin_names() -> set[str]:
    """Return base builtin names merged with any registered plugin builtins."""
    from dynafx.registry import get_registered_builtins
    return _BASE_BUILTIN_NAMES | set(get_registered_builtins().keys())


# ── Lookup table ────────────────────────────────────────────────

class LookupTable:
    """Linear-interpolated lookup table for time-varying parameters."""
    def __init__(self, x: list[float], y: list[float]):
        self.x = x
        self.y = y

    def __call__(self, t: float) -> float:
        if not self.x:
            return 0.0
        if t <= self.x[0]:
            return self.y[0]
        if t >= self.x[-1]:
            return self.y[-1]
        for i in range(len(self.x) - 1):
            if self.x[i] <= t < self.x[i + 1]:
                denom = self.x[i + 1] - self.x[i]
                frac = (t - self.x[i]) / denom if denom > 0 else 0.0
                return self.y[i] + frac * (self.y[i + 1] - self.y[i])
        return self.y[-1]





# ── System builder ──────────────────────────────────────────────


@dataclass
class CompiledSystem:
    """Cached compilation artifacts for _build_system.

    Generated once per model, reused across all simulate() calls.
    """
    stock_names: list[str]
    name_set: set[str]
    aux_names: list[str]
    aux_set: set[str]
    all_names: list[str]
    base_y0: list[float]
    inflow_strs: list[str]
    outflow_strs: list[str]
    aux_names_ordered: list[str]
    aux_compile_ordered: list[str]
    smooth_names: list[str]
    smooth_ode_strs: list[str]
    smooth_delay_exprs: list[str]
    smooth_init_exprs: list[str]
    delay_fixed_compiled: list[tuple[str, str, str]]
    cbatch_compiled: list[tuple[str, str, str, str, str]]
    builtins: dict[str, Any]
    # Pre-compiled code objects (avoids string re-parsing on every eval)
    aux_code: list[CodeType]
    inflow_code: list[CodeType]
    outflow_code: list[CodeType]
    smooth_ode_code: list[CodeType]
    df_input_code: list[CodeType]
    df_delay_code: list[CodeType]
    revision: int = 0


def _compile_system(model: SysdModel) -> CompiledSystem:
    """Compile model once — parse expressions, topo-sort, build strings.

    All string/AST work is done here and cached. _build_system only
    needs to build the closure f(t, y, p) from these pre-computed artifacts.
    """
    stock_names = [s.name for s in model.stocks]
    name_set = set(stock_names)
    aux_names = [a.name for a in model.aux_vars]
    aux_set = set(aux_names)
    all_names = list(stock_names)
    base_y0 = [s.initial for s in model.stocks]

    smooth_params: list[tuple[str, str, str, float, float]] = []
    stock_inflow: list[list[str]] = [[] for _ in model.stocks]
    stock_outflow: list[list[str]] = [[] for _ in model.stocks]

    # Build function expansion map
    func_map: dict[str, tuple[list[str], ExprNode]] = {}
    for fd in model.func_defs:
        body_node = ExprParser(fd.body).parse()
        func_map[fd.name] = (fd.params, body_node)

    # Phase 1: Parse all stock flows into AST nodes (after smooth expansion)
    stock_inflow_nodes: list[list[ExprNode]] = [[] for _ in model.stocks]
    stock_outflow_nodes: list[list[ExprNode]] = [[] for _ in model.stocks]
    for si, s in enumerate(model.stocks):
        for fl in s.flows:
            if not fl.expr.strip():
                continue
            node = ExprParser(fl.expr).parse()
            if func_map:
                node = _expand_func_calls(node, func_map)
            modified = _replace_smooths(node, smooth_params)
            if fl.direction == "+":
                stock_inflow_nodes[si].append(modified)
            else:
                stock_outflow_nodes[si].append(modified)

    # Phase 2: Auto-apply proportional allocation for multi-outflow stocks.
    # When a stock has >=2 outflows each following the MIN(<ref>/dt, demand) pattern,
    # the outflows independently gate against the full stock, causing over-draft.
    # Transform to proportional allocation via ALLOCATE_FRACTION.
    for si, s in enumerate(model.stocks):
        nodes = stock_outflow_nodes[si]
        if len(nodes) >= 2:
            demands: list[ExprNode] = []
            for node in nodes:
                if (isinstance(node, ExprFuncCall) and node.name == "MIN"
                        and len(node.args) == 2
                        and isinstance(node.args[0], ExprBinOp)
                        and node.args[0].op == "/"
                        and isinstance(node.args[0].right, ExprRef)
                        and node.args[0].right.name == "dt"):
                    demands.append(node.args[1])
                else:
                    demands = []
                    break
            if len(demands) == len(nodes):
                available = ExprBinOp(
                    "/",
                    ExprFuncCall("MAX", [ExprLiteral(0.0), ExprRef(s.name)]),
                    ExprRef("dt"),
                )
                total = demands[0]
                for d in demands[1:]:
                    total = ExprBinOp("+", total, d)
                transformed: list[ExprNode] = []
                for d in demands:
                    alloc = ExprFuncCall("ALLOCATE_FRACTION", [available, d, total])
                    transformed.append(ExprFuncCall("MIN", [alloc, d]))
                stock_outflow_nodes[si] = transformed

    # Phase 3: Compile all nodes to strings
    for i in range(len(model.stocks)):
        for node in stock_inflow_nodes[i]:
            stock_inflow[i].append(_compile_expr(node, name_set, aux_set))
        for node in stock_outflow_nodes[i]:
            stock_outflow[i].append(_compile_expr(node, name_set, aux_set))

    aux_expr_nodes: list[ExprNode] = []
    for a in model.aux_vars:
        node = ExprParser(a.expr).parse()
        if func_map:
            node = _expand_func_calls(node, func_map)
        modified = _replace_smooths(node, smooth_params)
        aux_expr_nodes.append(modified)

    smooth_names: list[str] = []
    delay_fixed_entries: list[tuple[str, str, str]] = []
    cbatch_entries: list[tuple[str, str, str, float, str]] = []  # (acc_name, input_expr, delay_str, batch_size, out_name)
    smooth_delay_exprs: list[str] = []
    smooth_init_exprs: list[str] = []
    _last_cbatch_acc: str | None = None
    for entry in smooth_params:
        entry_type, aux_name, input_expr_str, delay_time, fifth = entry
        smooth_names.append(aux_name)
        all_names.append(aux_name)
        delay_str = _serialize_expr(delay_time) if isinstance(delay_time, ExprNode) else str(delay_time)
        if entry_type == "convey_batch":
            _last_cbatch_acc = aux_name
            batch_size_expr = fifth
            batch_str = _serialize_expr(batch_size_expr) if isinstance(batch_size_expr, ExprNode) else str(batch_size_expr)
            try:
                _cb_delay_compiled = _compile_expr(ExprParser(delay_str).parse(), name_set, aux_set)
                cb_delay_str = _cb_delay_compiled
            except Exception:
                cb_delay_str = delay_str
            try:
                _cb_batch_compiled = _compile_expr(ExprParser(batch_str).parse(), name_set, aux_set)
                cb_batch_str = _cb_batch_compiled
            except Exception:
                cb_batch_str = batch_str
            cbatch_entries.append((aux_name, input_expr_str, cb_delay_str, cb_batch_str, ""))
            smooth_delay_exprs.append(delay_str)
            smooth_init_exprs.append("0.0")
            base_y0.append(0.0)
        elif entry_type == "convey_batch_out":
            # Output slot: no ODE, no delay, set by pipeline processing
            if _last_cbatch_acc is not None:
                # Find and update the matching cbatch entry
                for _cbe_i in range(len(cbatch_entries)):
                    if cbatch_entries[_cbe_i][0] == _last_cbatch_acc:
                        cbatch_entries[_cbe_i] = (
                            cbatch_entries[_cbe_i][0], cbatch_entries[_cbe_i][1],
                            cbatch_entries[_cbe_i][2], cbatch_entries[_cbe_i][3], aux_name,
                        )
                        break
            smooth_delay_exprs.append("0.0")
            smooth_init_exprs.append("0.0")
            base_y0.append(0.0)
        else:
            init_str = _serialize_expr(fifth) if isinstance(fifth, ExprNode) else str(fifth)
            smooth_delay_exprs.append(delay_str)
            smooth_init_exprs.append(init_str)
            try:
                base_y0.append(float(eval(init_str, {"__builtins__": {}})))
            except Exception:
                base_y0.append(0.0)
            if entry_type in ("delay_fixed", "convey"):
                delay_fixed_entries.append((aux_name, input_expr_str, delay_str))

    inflow_strs: list[str] = []
    outflow_strs: list[str] = []
    for i in range(len(model.stocks)):
        inf = " + ".join(stock_inflow[i]) if stock_inflow[i] else "0.0"
        outf = " + ".join(stock_outflow[i]) if stock_outflow[i] else "0.0"
        inflow_strs.append(inf)
        outflow_strs.append(outf)

    smooth_ode_strs: list[str] = []
    for i, entry in enumerate(smooth_params):
        entry_type, aux_name, input_expr_str, delay_time, init_val = entry
        input_node = ExprParser(input_expr_str).parse()
        input_compiled = _compile_expr(input_node, name_set, aux_set)
        delay_str = smooth_delay_exprs[i]
        if entry_type in ("delay_fixed", "convey", "convey_batch_out"):
            smooth_ode_strs.append("0.0")
        elif entry_type == "convey_batch":
            smooth_ode_strs.append(f"{input_compiled}")
        else:
            try:
                _delay_compiled = _compile_expr(ExprParser(delay_str).parse(), name_set, aux_set)
            except Exception:
                _delay_compiled = delay_str
            smooth_ode_strs.append(
                f"({input_compiled} - _s.get('{aux_name}', 0.0)) / MAX({_delay_compiled}, 1e-10)"
            )

    aux_compile_strs: list[str] = []
    for node in aux_expr_nodes:
        aux_compile_strs.append(_compile_expr(node, name_set, aux_set))

    aux_order = _topo_sort(aux_names, aux_expr_nodes, name_set | aux_set)
    aux_names_ordered = [aux_names[i] for i in aux_order]
    aux_compile_ordered = [aux_compile_strs[i] for i in aux_order]

    import math

    from dynafx.registry import get_registered_builtins
    def _allocate_frac(available: float, demand: float, total_demand: float) -> float:
        """Proportional split: out = demand * min(1, available / max(1, total_demand))."""
        frac = min(1.0, available / max(1.0, total_demand))
        return demand * frac

    builtins = {
        "MIN": min, "MAX": max,
        "IF": lambda c, a, b: a if c else b,
        "ABS": abs,
        "EXP": math.exp,
        "LN": math.log,
        "SQRT": math.sqrt,
        "SIN": math.sin,
        "COS": math.cos,
        "PI": math.pi,
        "ALLOCATE_FRACTION": _allocate_frac,
    }
    builtins.update(get_registered_builtins())

    delay_fixed_compiled: list[tuple[str, str, str]] = []
    for df_name, df_input_str, df_delay in delay_fixed_entries:
        df_node = ExprParser(df_input_str).parse()
        df_compiled = _compile_expr(df_node, name_set, aux_set)
        try:
            df_delay_compiled = _compile_expr(ExprParser(df_delay).parse(), name_set, aux_set)
        except Exception:
            df_delay_compiled = df_delay
        delay_fixed_compiled.append((df_name, df_compiled, df_delay_compiled))

    # Pre-compile all expression strings to code objects (avoids re-parsing on every eval)
    _co = "<compiled>"
    aux_code = [compile(s, _co, "eval") for s in aux_compile_ordered]
    inflow_code = [compile(s, _co, "eval") for s in inflow_strs]
    outflow_code = [compile(s, _co, "eval") for s in outflow_strs]
    smooth_ode_code = [compile(s, _co, "eval") for s in smooth_ode_strs]
    df_input_code = [compile(entry[1], _co, "eval") for entry in delay_fixed_compiled]
    df_delay_code = [compile(entry[2], _co, "eval") for entry in delay_fixed_compiled]

    return CompiledSystem(
        stock_names=stock_names,
        name_set=name_set,
        aux_names=aux_names,
        aux_set=aux_set,
        all_names=all_names,
        base_y0=base_y0,
        inflow_strs=inflow_strs,
        outflow_strs=outflow_strs,
        aux_names_ordered=aux_names_ordered,
        aux_compile_ordered=aux_compile_ordered,
        smooth_names=smooth_names,
        smooth_ode_strs=smooth_ode_strs,
        smooth_delay_exprs=smooth_delay_exprs,
        smooth_init_exprs=smooth_init_exprs,
        delay_fixed_compiled=delay_fixed_compiled,
        cbatch_compiled=cbatch_entries,
        builtins=builtins,
        revision=model._model_revision,
        aux_code=aux_code,
        inflow_code=inflow_code,
        outflow_code=outflow_code,
        smooth_ode_code=smooth_ode_code,
        df_input_code=df_input_code,
        df_delay_code=df_delay_code,
    )


def _make_kb_builtins(kb_store: Any = None) -> dict[str, Any]:
    """Create KB_QUERY and KB_ASSERT builtins that close over a TripleStore.

    When kb_store is None, returns no-op stubs that return 0.0 so expressions
    using KB functions don't crash when no KB is provided.
    """
    if kb_store is None:
        return {
            "KB_QUERY": lambda sparql_str="", var="v": 0.0,
            "KB_QUERY_TEMPLATE": lambda template="", subject_iri="", var="v": 0.0,
            "KB_ASSERT": lambda s="", p="", o="", belief=1.0, graph="simulation": 0.0,
        }

    from dynafx.core.models import Opinion as _Opinion
    from dynafx.knowledge.model import (
        BlankNode as _BlankNode,
    )
    from dynafx.knowledge.model import (
        Literal as _Literal,
    )
    from dynafx.knowledge.model import (
        NamedNode as _NamedNode,
    )
    from dynafx.knowledge.model import (
        Triple as _Triple,
    )
    from dynafx.knowledge.sparql import Ask as _SparqlAsk
    from dynafx.knowledge.sparql import evaluate as _sparql_evaluate
    from dynafx.knowledge.sparql import parse_sparql

    _sparql_cache: dict[str, Any] = {}
    _SPARQL_CACHE_MAX = 256

    def _kb_query(sparql_str: str, var: str = "v") -> float:
        if len(_sparql_cache) > _SPARQL_CACHE_MAX:
            _sparql_cache.clear()
        if sparql_str not in _sparql_cache:
            _sparql_cache[sparql_str] = parse_sparql(sparql_str)
        algebra = _sparql_cache[sparql_str]

        if isinstance(algebra, _SparqlAsk):
            result = _sparql_evaluate(algebra, kb_store)
            return 1.0 if result.cardinality > 0 else 0.0

        result = _sparql_evaluate(algebra, kb_store)
        if result.bindings:
            val = result.bindings[0].get(var)
            if val is not None:
                return float(getattr(val, "value", val))
        return 0.0

    def _kb_query_template(sparql_template: str, subject_iri: str = "", var: str = "v") -> float:
        resolved = sparql_template.replace("$subject", str(subject_iri))
        return _kb_query(resolved, var)

    def _resolve_kb_node(x: Any, force_literal: bool = False) -> Any:
        if isinstance(x, (_NamedNode, _BlankNode, _Literal)):
            return x
        if isinstance(x, str):
            if x.startswith("_:"):
                return _BlankNode(x[2:])
            if force_literal or ("://" not in x):
                return _Literal(x)
            return _NamedNode(x)
        if isinstance(x, (int, float, bool)):
            return _Literal(x)
        return _NamedNode(str(x))

    def _kb_assert(
        s: Any, p: Any, o: Any, belief: float = 1.0, graph: str = "simulation"
    ) -> float:
        s_node = _resolve_kb_node(s)
        p_node = _resolve_kb_node(p)
        o_node = _resolve_kb_node(o, force_literal=True)
        triple = _Triple(s_node, p_node, o_node, opinion=_Opinion(belief, 1.0 - belief, 0.0))
        kb_store.add(triple, graph=graph)
        return 1.0

    return {"KB_QUERY": _kb_query, "KB_QUERY_TEMPLATE": _kb_query_template, "KB_ASSERT": _kb_assert}


def _build_system(
    model: SysdModel,
    params: dict[str, Any],
    emergent_props: Optional[list] = None,
    seed: int = 42,
    cache: Optional[CompiledSystem] = None,
    kb: Any = None,
) -> tuple[Callable, list[str], list[float], int, dict[str, Any] | None]:
    """Build ODE system from SysdModel.

    Returns: (f(t, y, params), all_names, y0, aux_state_count, pipeline_delay_infos)
    where pipeline_delay_infos is a dict containing shared buffer and processing
    callable for DELAY_FIXED/CONVEY, or None if no pipeline delays exist.
    """
    if cache is None:
        cache = _compile_system(model)

    all_names = list(cache.all_names)
    y0 = list(cache.base_y0)
    # Overwrite stock portion with current model initial values
    for i, s in enumerate(model.stocks):
        y0[i] = s.initial
    stock_names = cache.stock_names
    inflow_strs = cache.inflow_strs
    outflow_strs = cache.outflow_strs
    aux_names_ordered = cache.aux_names_ordered
    smooth_names = cache.smooth_names
    smooth_ode_strs = cache.smooth_ode_strs
    delay_fixed_compiled = cache.delay_fixed_compiled
    cbatch_compiled = cache.cbatch_compiled
    _builtins = cache.builtins
    # Pre-compiled code objects
    _aux_code = cache.aux_code
    _inflow_code = cache.inflow_code
    _outflow_code = cache.outflow_code
    _smooth_ode_code = cache.smooth_ode_code
    _df_input_code = cache.df_input_code
    _df_delay_code = cache.df_delay_code

    import random as _random
    _rng = _random.Random(seed)  # seeded for reproducibility

    # ── Pipeline delay buffers (DELAY_FIXED / CONVEY) ─────────────
    # State is managed OUTSIDE f() to avoid RK4 intermediate corruption.
    # Each buffer is a deque of (exit_time, value) pairs.
    # Buffers are seeded with initial value so delays work from t=0.
    _pipeline_buffers: dict[str, deque[tuple[float, float]]] = {}
    for df_name, _, df_delay in delay_fixed_compiled:
        _pipeline_buffers[df_name] = deque()
    # CONVEY_BATCH pipeline buffers (separate from delay_fixed)
    _cbatch_pipe_buffers: dict[str, deque[tuple[float, float]]] = {}
    _cbatch_count = len(cbatch_compiled)

    # Pre-compute things that don't change between f() calls
    _no_builtins = {"__builtins__": {}}
    _stock_count = len(model.stocks)
    _df_count = len(delay_fixed_compiled)

    # Pre-filter params to only numeric values (avoids per-call isinstance check)
    _numeric_params = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    # Non-numeric params (strings, etc.) — needed for KB_QUERY SPARQL strings
    _string_params = {k: v for k, v in params.items() if isinstance(v, str)}

    # Evaluate smooth delay/init expressions at runtime with params
    _eval_ns = {"__builtins__": {}, **_builtins, **_numeric_params}
    _smooth_delays = []
    for expr_str in cache.smooth_delay_exprs:
        try:
            _smooth_delays.append(float(eval(expr_str, _eval_ns)))
        except Exception:
            _smooth_delays.append(1.0)
    _smooth_inits = []
    for expr_str in cache.smooth_init_exprs:
        try:
            _smooth_inits.append(float(eval(expr_str, _eval_ns)))
        except Exception:
            _smooth_inits.append(0.0)
    # Update y0 with runtime-resolved init values
    _stock_count_rt = len(model.stocks)
    for i, init_val in enumerate(_smooth_inits):
        y0[_stock_count_rt + i] = init_val

    # Pre-compute callable params (lookup tables etc.)
    _callable_params = [(k, v) for k, v in params.items() if callable(v)]

    _kb_builtins = _make_kb_builtins(kb)

    _cbatch_remainder: dict[str, float] = {}

    def f(t: float, y: list[float], p: dict) -> list[float]:
        _s = dict(zip(all_names, y, strict=False))
        # Merge pre-computed params + runtime params (includes string SPARQL queries for KB_QUERY)
        _s.update(_numeric_params)
        if _string_params:
            _s.update(_string_params)
        # Restore CONVEY_BATCH accumulator state (pipeline overwrites y with output)
        for _cba_name, _cba_val in _cbatch_remainder.items():
            _s[_cba_name] = _cba_val
        if p:
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    _s[k] = v
        _a: dict[str, float] = {}
        # Build eval namespace — minimal dict, reuse _no_builtins
        _ns: dict = {
            **_builtins, **_kb_builtins, **_numeric_params, **_string_params, "_s": _s, "_p": params, "_a": _a, "t": t,
            "PULSE": lambda volume, start, width: volume if start <= t < start + width else 0.0,
            "STEP": lambda height, start: height if t >= start else 0.0,
            "RAMP": lambda slope, start, end: (
                0.0 if t < start else
                slope * (t - start) if t <= end else
                slope * (end - start)
            ),
            "NOISE": lambda amplitude: _rng.uniform(-amplitude, amplitude),
            "UNIFORM": lambda a, b: _rng.uniform(a, b),
            "LOGNORMAL": lambda mu, sigma: (
                _rng.lognormvariate(mu, sigma) if sigma > 0 else mu
            ),
        }
        # Inject lookup tables and callables into eval namespace only (not _s)
        for _k, _v in _callable_params:
            _ns[_k] = _v
        if p:
            for _k, _v in p.items():
                if callable(_v):
                    _ns[_k] = _v
        # Evaluate auxes in dependency order (using pre-compiled code objects)
        # Numeric params override aux expressions when the name matches
        for _i in range(len(aux_names_ordered)):
            _aname = aux_names_ordered[_i]
            if _aname in _numeric_params:
                _a[_aname] = _numeric_params[_aname]
            else:
                _a[_aname] = eval(_aux_code[_i], _no_builtins, _ns)
        _ns["_a"] = _a
        # Check emergent properties — threshold crossings modify state
        if emergent_props:
            for _ep in emergent_props:
                _check_state = {**_s, **_a}
                _ep.check(_check_state, t)
                if _ep.active:
                    _modified = _ep.apply_effects(_check_state)
                    _s.update(_modified)
                    _a.update(_modified)
                    _ns["_s"] = _s
                    _ns["_a"] = _a
        # NO pipeline delay processing here — that's done in simulate() loop
        # to avoid RK4 buffer corruption. Delay values are pre-set in y.
        # Stock equations (using pre-compiled code objects)
        dydt: list[float] = []
        for i in range(_stock_count):
            inflow_val = eval(_inflow_code[i], _no_builtins, _ns)
            outflow_val = eval(_outflow_code[i], _no_builtins, _ns)
            dydt.append(inflow_val - outflow_val)
        # SMOOTH ODEs (using pre-compiled code objects)
        for _si in range(len(smooth_ode_strs)):
            dydt.append(eval(_smooth_ode_code[_si], _no_builtins, _ns))
        return dydt

    if _df_count == 0 and _cbatch_count == 0:
        f._aux_info = (_numeric_params, _callable_params, _builtins, _no_builtins, aux_names_ordered, cache.aux_code, _stock_count)
        f._kb_builtins = _kb_builtins
        return f, all_names, y0, len(smooth_names), None

    # ── Pipeline delay processing (called ONCE per step from simulate) ──
    _pipeline_delay_values: dict[str, float] = {}

    def _process_pipeline_delays(t: float, y: list[float], p: dict) -> None:
        """Update pipeline delay buffers and inject values into y.
        Must be called exactly ONCE per step, after f() completes.
        """
        _s = dict(zip(all_names, y, strict=False))
        _s.update(_numeric_params)
        if _string_params:
            _s.update(_string_params)
        if p:
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    _s[k] = v
        _a: dict[str, float] = {}
        _ns: dict = {
            **_builtins, **_kb_builtins, **_numeric_params, **_string_params, "_s": _s, "_p": params, "_a": _a, "t": t,
            "PULSE": lambda volume, start, width: volume if start <= t < start + width else 0.0,
            "STEP": lambda height, start: height if t >= start else 0.0,
            "RAMP": lambda slope, start, end: (
                0.0 if t < start else
                slope * (t - start) if t <= end else
                slope * (end - start)
            ),
            "NOISE": lambda amplitude: _rng.uniform(-amplitude, amplitude),
            "UNIFORM": lambda a, b: _rng.uniform(a, b),
            "LOGNORMAL": lambda mu, sigma: (
                _rng.lognormvariate(mu, sigma) if sigma > 0 else mu
            ),
        }
        for _k, _v in _callable_params:
            _ns[_k] = _v
        if p:
            for _k, _v in p.items():
                if callable(_v):
                    _ns[_k] = _v
        for _i in range(len(aux_names_ordered)):
            _aname = aux_names_ordered[_i]
            if _aname in _numeric_params:
                _a[_aname] = _numeric_params[_aname]
            else:
                _a[_aname] = eval(_aux_code[_i], _no_builtins, _ns)
        _ns["_a"] = _a
        _ns.update(_a)

        for _dfi in range(_df_count):
            df_name, df_compiled, df_delay_str = delay_fixed_compiled[_dfi]
            try:
                input_val = eval(_df_input_code[_dfi], _no_builtins, _ns)
            except Exception:
                input_val = 0.0
            # Evaluate delay at runtime (supports stochastic: UNIFORM, LOGNORMAL)
            try:
                df_delay = float(eval(_df_delay_code[_dfi], _no_builtins, _ns))
            except Exception:
                df_delay = 1.0
            df_delay = max(0.0, df_delay)
            # Append the input with its exit time to the FIFO buffer
            exit_t = t + df_delay
            _pipeline_buffers[df_name].append((exit_t, input_val))
            # Emit all entries whose exit time has passed
            emitted = 0.0
            while _pipeline_buffers[df_name]:
                e_time, e_val = _pipeline_buffers[df_name][0]
                if e_time <= t:
                    emitted = e_val
                    _pipeline_buffers[df_name].popleft()
                else:
                    break
            # If nothing emitted yet, use the oldest entry's value
            if not _pipeline_buffers[df_name]:
                _pipeline_delay_values[df_name] = emitted
            else:
                _pipeline_delay_values[df_name] = emitted
            # Inject into state vector y so next f() sees it
            idx = all_names.index(df_name)
            y[idx] = _pipeline_delay_values[df_name]

        # ── CONVEY_BATCH processing ─────────────────────────────────
        if _cbatch_count > 0:
            for _cbi in range(_cbatch_count):
                acc_name, cb_input_str, cb_delay_str, cb_batch, out_name = cbatch_compiled[_cbi]
                acc_idx = all_names.index(acc_name)
                acc_val = y[acc_idx]
                pipe_name = f"{acc_name}_pipe"
                if pipe_name not in _cbatch_pipe_buffers:
                    _cbatch_pipe_buffers[pipe_name] = deque()
                # Evaluate batch size at runtime
                try:
                    batch_val = float(eval(cb_batch, {"__builtins__": {}}, _ns))
                except Exception:
                    batch_val = 1.0
                batch_val = max(0.1, batch_val)
                # Emit batches if accumulator >= batch_size
                if acc_val >= batch_val:
                    batches = int(acc_val / batch_val)
                    emit_amount = batch_val * batches
                    try:
                        batch_delay = float(eval(cb_delay_str, {"__builtins__": {}}, _ns))
                    except Exception:
                        batch_delay = 1.0
                    batch_delay = max(0.0, batch_delay)
                    _cbatch_pipe_buffers[pipe_name].append((t + batch_delay, emit_amount))
                    acc_val -= emit_amount
                # Emit matured batches from pipeline
                emitted_batch = 0.0
                while _cbatch_pipe_buffers[pipe_name]:
                    e_time, e_val = _cbatch_pipe_buffers[pipe_name][0]
                    if e_time <= t:
                        emitted_batch += e_val
                        _cbatch_pipe_buffers[pipe_name].popleft()
                    else:
                        break
                # Save remainder and keep y[acc] in sync
                _cbatch_remainder[acc_name] = acc_val
                y[acc_idx] = acc_val
                # Write output to the out_name slot (not accumulator)
                if out_name:
                    out_idx = all_names.index(out_name)
                    y[out_idx] = emitted_batch

    pipeline_info = {
        "buffers": _pipeline_buffers,
        "process": _process_pipeline_delays,
        "names": [entry[0] for entry in delay_fixed_compiled],
        "compiled": delay_fixed_compiled,
    }
    f._aux_info = (_numeric_params, _callable_params, _builtins, _no_builtins, aux_names_ordered, cache.aux_code, _stock_count)
    f._kb_builtins = _kb_builtins
    return f, all_names, y0, len(smooth_names), pipeline_info





# ── Sysd Lexer / Structure Parser ───────────────────────────────

_COMMENT_RE = re.compile(r"(?:^|\s)//.*$")
_STRIP_RE = re.compile(r'^["\']|["\']$')

_TokenLine = tuple[int, str, str]  # (indent, keyword, args)

_UNIT_RE = re.compile(r"~([^~]+)~")


def _extract_units(s: str) -> tuple[str, str]:
    """Extract ~Unit~ annotation from a string.

    Returns (cleaned_string, unit_string).
    Example: '"Population": 1000 ~people~' → ('"Population": 1000', 'people')
    """
    m = _UNIT_RE.search(s)
    if m:
        unit = m.group(1).strip()
        cleaned = s[:m.start()] + s[m.end():]
        return cleaned.strip(), unit
    return s, ""


def _lex_sysd(source: str) -> list[_TokenLine]:
    lines: list[_TokenLine] = []
    for line in source.split("\n"):
        raw = _COMMENT_RE.sub("", line).rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        content = raw.strip()
        m = re.match(r"(\w[\w.]*|[+\-])\s*(.*)", content)
        if not m:
            continue
        keyword = m.group(1)
        args = m.group(2).strip()
        lines.append((indent, keyword, args))
    return lines


def _build_tree(lines: list[_TokenLine]) -> SysdModel:
    """Convert indent-aware token lines into a SysdModel AST."""
    model = SysdModel()
    stack: list[tuple[int, StockDef | TableDef | AgentDef | None]] = [(-1, None)]

    for line_idx, (indent, keyword, args) in enumerate(lines):
        if keyword == "model":
            model.name = _STRIP_RE.sub("", args)
            # Pop any remaining submodel from stack
            while stack and stack[-1][0] >= 0:
                stack.pop()
            continue
        if keyword == "dt":
            model.dt = float(args)
            continue
        if keyword == "from":
            parts = args.split()
            if parts:
                t0 = float(parts[0])
                if len(parts) >= 3 and parts[1] == "to":
                    t1 = float(parts[2])
                    model.t_span = (t0, t1)
                else:
                    model.t_span = (t0, model.t_span[1])
            continue
        if keyword == "stock":
            name, initial = _parse_name_value(args)
            name = name.replace(" ", "_")
            # Extract ~Unit~ annotation
            _, units_str = _extract_units(args)
            sd = StockDef(name=name, initial=initial, units=units_str)
            # Pop stack first to find the correct parent
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            # Check if parent is a SubmodelDef — append to its stocks
            if isinstance(parent, SubmodelDef):
                parent.stocks.append(sd)
            else:
                model.stocks.append(sd)
            stack.append((indent, sd))
            continue

        if keyword == "table":
            name = _STRIP_RE.sub("", args)
            td = TableDef(name=name, x=[], y=[])
            model.tables.append(td)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, td))
            continue

        if keyword in ("x", "y"):
            parent = stack[-1][1] if stack else None
            if isinstance(parent, TableDef):
                vals = _parse_list(args.lstrip(":"))
                if keyword == "x":
                    parent.x = vals
                else:
                    parent.y = vals
                continue
            # Fall through: x/y can also be agent rule effect lines

        if keyword in ("+", "-"):
            parent = stack[-1][1] if stack else None
            if isinstance(parent, StockDef):
                name, _ = _parse_name_value(args)
                expr = _split_expr(args)
                # Extract ~Unit~ annotation from the raw args
                _, units_str = _extract_units(args)
                parent.flows.append(FlowDef(name=name, direction=keyword, expr=expr, units=units_str))
            continue

        if keyword == "aux":
            name, expr = _parse_name_expr(args)
            # Extract ~Unit~ annotation
            _, units_str = _extract_units(args)
            # Check if we're inside a SubmodelDef (check all ancestors)
            in_submodel = any(isinstance(entry[1], SubmodelDef) for entry in stack if entry[1] is not None)
            if in_submodel:
                # Find the SubmodelDef ancestor
                for entry in reversed(stack):
                    if isinstance(entry[1], SubmodelDef):
                        entry[1].aux_vars.append(AuxDef(name=name, expr=expr, units=units_str))
                        break
            else:
                model.aux_vars.append(AuxDef(name=name, expr=expr, units=units_str))
            continue

        # ── ABM keywords ────────────────────────────────────────
        if keyword == "agent":
            name, count = _parse_name_value(args)
            if count < 1:
                count = 1
            ad = AgentDef(name=name, count=int(count))
            model.agents.append(ad)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, ad))
            continue

        if keyword == "property":
            parent = stack[-1][1] if stack else None
            if isinstance(parent, AgentDef):
                prop = _parse_agent_property(args)
                parent.properties.append(prop)
            continue

        if keyword == "rule":
            # Pop stack to find the owning AgentDef or AgentStrategy
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            rule = _parse_agent_rule(args)
            if isinstance(parent, AgentDef):
                parent.rules.append(rule)
            elif isinstance(parent, AgentStrategy):
                parent.rules.append(rule)
            else:
                continue  # orphan rule, skip
            # Push rule onto stack so indented effect lines attach to it
            stack.append((indent, rule))
            continue

        if keyword == "network":
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if isinstance(parent, AgentDef):
                parent.network_type = _STRIP_RE.sub("", args.strip()).lower() or "none"
            continue

        if keyword == "strategy":
            # strategy "name" — creates scoped rule block
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if isinstance(parent, AgentDef):
                strategy_name = _STRIP_RE.sub("", args.strip()).strip('"\' ')
                strategy = AgentStrategy(strategy_name)
                parent.strategies.append(strategy)
                stack.append((indent, strategy))
            continue

        if keyword == "meta_rule":
            # meta_rule "name": when condition — always-evaluated rules
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if isinstance(parent, AgentDef):
                rule = _parse_agent_rule(args)
                parent.meta_rules.append(rule)
                stack.append((indent, rule))
            continue

        # Rule effect lines (indented under a rule)
        parent = stack[-1][1] if stack else None
        if isinstance(parent, AgentRuleDef):
            # Lines like "budget -= Price" or "satisfaction += 1"
            # or "x += 1" lexed as keyword="x", args="+= 1"
            if args and args.lstrip().startswith(("+=", "-=", "*=", "/=", "=")):
                parent.effects.append(f"{keyword} {args}")
                continue
            # Also handle full-line effects like "budget -= Price"
            full_line = f"{keyword} {args}".strip()
            if any(op in full_line for op in ("+=", "-=", "*=", "/=")):
                parent.effects.append(full_line)
                continue
            # Handle KB_ASSERT side-effect
            if keyword == "KB_ASSERT":
                parent.effects.append(full_line)
                continue
            # Handle SEND and SWITCH_STRATEGY side-effects
            if keyword in ("SEND", "SWITCH_STRATEGY"):
                parent.effects.append(full_line)
                continue

        # ── Function / macro keyword ─────────────────────────────
        if keyword == "func":
            # func "name(p1, p2)" = expression
            name_and_params = args
            body = ""
            if "=" in args:
                name_and_params, body = args.split("=", 1)
                body = body.strip()
            name_and_params = _STRIP_RE.sub("", name_and_params.strip())
            # Parse name and param list from e.g. "square(x)" or "clamp(v, lo, hi)"
            func_name = name_and_params
            func_params: list[str] = []
            if "(" in name_and_params:
                pstart = name_and_params.index("(")
                func_name = _STRIP_RE.sub("", name_and_params[:pstart].strip())
                pinside = name_and_params[pstart+1:]
                if ")" in pinside:
                    pinside = pinside[:pinside.index(")")]
                func_params = [p.strip() for p in pinside.split(",") if p.strip()]
            model.func_defs.append(FuncDef(name=func_name, params=func_params, body=body))
            continue

        # ── DES keywords ────────────────────────────────────────
        if keyword == "queue":
            name, _ = _parse_name_value(args)
            # Parse capacity/service_time from args like "Q": capacity 3, service_time 2
            capacity = -1
            service_time = ""
            servers = 1
            event_driven = False
            if ":" in args:
                after_colon = args.split(":", 1)[1]
                for part in after_colon.split(","):
                    part = part.strip()
                    pl = part.lower()
                    if pl.startswith("capacity"):
                        val = ""
                        if "=" in part:
                            val = part.split("=", 1)[1]
                        elif " " in part:
                            val = part.split(None, 1)[1]
                        if val:
                            try:
                                capacity = int(float(val.strip()))
                            except ValueError:
                                pass
                    elif pl.startswith("service_time"):
                        val = ""
                        if "=" in part:
                            val = part.split("=", 1)[1]
                        elif " " in part:
                            val = part.split(None, 1)[1]
                        if val:
                            service_time = val.strip()
                    elif pl.startswith("server"):
                        val = ""
                        if "=" in part:
                            val = part.split("=", 1)[1]
                        elif " " in part:
                            val = part.split(None, 1)[1]
                        if val:
                            try:
                                servers = max(1, int(float(val.strip())))
                            except ValueError:
                                pass
                    elif pl.startswith("event_driven") or pl == "event_driven":
                        event_driven = True
            qd = QueueDef(name=name, capacity=capacity, service_time=service_time, servers=servers, event_driven=event_driven)
            model.queues.append(qd)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, qd))
            continue

        if keyword == "service_time":
            parent = stack[-1][1] if stack else None
            if isinstance(parent, QueueDef):
                parent.service_time = _split_expr(args)
            continue

        if keyword == "arrival_rate":
            parent = stack[-1][1] if stack else None
            if isinstance(parent, QueueDef):
                parent.arrival_rate = _split_expr(args)
            continue

        if keyword == "resource":
            name, _ = _parse_name_value(args)
            # Parse capacity from args like "Server": capacity 5
            capacity = 1
            if ":" in args:
                after_colon = args.split(":", 1)[1]
                for part in after_colon.split(","):
                    part = part.strip().lower()
                    if part.startswith("capacity"):
                        if "=" in part:
                            val = part.split("=", 1)[1]
                        elif " " in part:
                            val = part.split(None, 1)[1]
                        else:
                            continue
                        try:
                            capacity = int(float(val.strip()))
                        except ValueError:
                            pass
            rd = ResourceDef(name=name, capacity=capacity)
            model.resources.append(rd)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, rd))
            continue

        if keyword == "cost_per_unit":
            parent = stack[-1][1] if stack else None
            if isinstance(parent, ResourceDef):
                parent.cost_per_unit = float(_split_expr(args))
            continue

        if keyword == "event":
            name = _STRIP_RE.sub("", args)
            ed = EventDef(name=name)
            model.events.append(ed)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, ed))
            continue

        if keyword == "rate":
            parent = stack[-1][1] if stack else None
            if isinstance(parent, EventDef):
                parent.rate = _split_expr(args)
            continue

        if keyword == "enqueue":
            parent = stack[-1][1] if stack else None
            if isinstance(parent, EventDef):
                parent.target_queue = _STRIP_RE.sub("", args)
            continue

        if keyword in ("resource_request", "resource_release", "drop"):
            parent = stack[-1][1] if stack else None
            if isinstance(parent, EventDef):
                parent.effects.append(f"{keyword} {_STRIP_RE.sub('', args)}")
            continue

        # ── Submodel keywords ───────────────────────────────────
        if keyword == "submodel":
            name = _STRIP_RE.sub("", args)
            smd = SubmodelDef(name=name)
            model.submodels.append(smd)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, smd))
            continue

        if keyword == "include":
            # include SubModelName as instance_name
            # params: key=value, key=value
            parts = args.split(" as ", 1)
            submodel_name = _STRIP_RE.sub("", parts[0].strip())
            instance_name = ""
            params = {}
            if len(parts) > 1:
                rest = parts[1]
                if " params:" in rest:
                    instance_name, params_str = rest.split(" params:", 1)
                    instance_name = _STRIP_RE.sub("", instance_name.strip())
                    # Parse params from same line
                    for p in params_str.split(","):
                        if "=" in p:
                            k, v = p.split("=", 1)
                            try:
                                params[k.strip()] = float(v.strip())
                            except ValueError:
                                pass
                else:
                    instance_name = _STRIP_RE.sub("", rest.strip())

            # If params: was found but no params on same line, check continuation lines
            if " params:" in args and not params:
                # Look ahead for indented continuation lines
                for j in range(line_idx + 1, len(lines)):
                    next_indent, next_kw, next_args = lines[j]
                    if next_indent <= indent:
                        break  # Back to same level, stop
                    # Parse continuation line for key=value pairs
                    if next_kw and next_kw.startswith("//"):
                        continue
                    # Reconstruct line text from keyword and args
                    line_text = f"{next_kw} {next_args}".strip() if next_args else next_kw
                    for p in line_text.split(","):
                        p = p.strip()
                        if "=" in p:
                            k, v = p.split("=", 1)
                            try:
                                params[k.strip()] = float(v.strip())
                            except ValueError:
                                pass

            inc = IncludeDef(submodel_name=submodel_name, instance_name=instance_name, params=params)
            model.includes.append(inc)
            continue

    return model


def _parse_name_value(args: str) -> tuple[str, float]:
    """Parse 'Warehouse Stock: 100' or 'items = 5' → name, value."""
    sep = ":" if ":" in args else "=" if "=" in args else None
    if sep:
        name, val = args.split(sep, 1)
        name = _STRIP_RE.sub("", name.strip())
        # Strip ~Unit~ annotation from both name and value
        name = re.sub(r'~[^~]*~', '', name).strip()
        val = re.sub(r'~[^~]*~', '', val).strip()
        try:
            return name, float(val)
        except ValueError:
            return name, 0.0
    name = _STRIP_RE.sub("", args.strip())
    name = re.sub(r'~[^~]*~', '', name).strip()
    return name, 0.0


def _parse_name_expr(args: str) -> tuple[str, str]:
    """Parse 'rate: S * 0.1' → ('rate', 'S * 0.1')."""
    if ":" in args:
        name, expr = args.split(":", 1)
        name = _STRIP_RE.sub("", name.strip())
        return name, expr.strip()
    return _STRIP_RE.sub("", args.strip()), "0"


def _split_expr(args: str) -> str:
    """Extract the expression part after 'name: expr'."""
    if ":" in args:
        _, expr = args.split(":", 1)
        return expr.strip()
    return args


def _parse_list(args: str) -> list[float]:
    """Parse '[1, 2, 3]' or '1, 2, 3' into [1.0, 2.0, 3.0]."""
    args = args.strip().strip("[]")
    parts = [p.strip() for p in args.split(",") if p.strip()]
    result: list[float] = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            logger.error("Cannot parse table value '%s' as float", p)
            raise
    return result


def _parse_agent_property(args: str) -> AgentPropDef:
    """Parse '"budget": 1000, min=0, max=5000' into AgentPropDef."""
    # Split on comma, first part is name:initial
    parts = [p.strip() for p in args.split(",")]
    name = _STRIP_RE.sub("", parts[0].split(":")[0].strip()) if ":" in parts[0] else _STRIP_RE.sub("", parts[0].strip())
    initial = 0.0
    if ":" in parts[0]:
        try:
            initial = float(parts[0].split(":", 1)[1].strip())
        except ValueError:
            pass

    prop = AgentPropDef(name=name, initial=initial)
    for p in parts[1:]:
        p = p.strip()
        if "=" in p:
            k, v = p.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "min":
                prop.min = float(v)
            elif k == "max":
                prop.max = float(v)
    return prop


def _parse_agent_rule(args: str) -> AgentRuleDef:
    """Parse 'buy: when satisfaction < 0.5 and budget > Price' into AgentRuleDef."""
    if ":" in args:
        name, rest = args.split(":", 1)
        name = _STRIP_RE.sub("", name.strip())
        rest = rest.strip()
    else:
        name = _STRIP_RE.sub("", args.strip())
        rest = ""

    condition = ""
    if rest.lower().startswith("when "):
        condition = rest[5:].strip()
    elif rest:
        condition = rest

    return AgentRuleDef(name=name, condition=condition)


# ── Public API ──────────────────────────────────────────────────

def _expand_includes(model: SysdModel) -> SysdModel:
    """Expand submodel includes by copying components with namespaced names."""
    if not model.includes:
        return model

    # Build submodel registry
    submodel_registry: dict[str, SubmodelDef] = {
        sm.name: sm for sm in model.submodels
    }

    for inc in model.includes:
        template = submodel_registry.get(inc.submodel_name)
        if template is None:
            logger.warning("Unknown submodel '%s' in include, skipping", inc.submodel_name)
            continue

        prefix = inc.instance_name
        sep = "_" if prefix else ""

        # Build replacement map from original names to prefixed names
        # Only replace names that are actual variables (not function names)
        from dynafx.registry import get_registered_builtins
        _FUNC_NAMES = {"SMOOTH", "SMOOTHI", "DELAY3", "DELAYN", "DELAY_FIXED",
                       "CONVEY", "CONVEY_BATCH",
                       "MIN", "MAX", "IF", "ABS", "EXP", "LN", "SQRT",
                        "SIN", "COS", "PI", "PULSE", "STEP", "RAMP", "NOISE",
                        "UNIFORM", "LOGNORMAL"} | set(get_registered_builtins().keys())
        replacements: dict[str, str] = {}

        # Collect all known names from the template
        known_names: set[str] = set()
        for var_stock in template.stocks:
            known_names.add(var_stock.name)
            for var_flow in var_stock.flows:
                known_names.add(var_flow.name)
        for var_aux in template.aux_vars:
            known_names.add(var_aux.name)
        for var_table in template.tables:
            known_names.add(var_table.name)

        # Add known names to replacements
        for name in known_names:
            if name not in _FUNC_NAMES:
                replacements[name] = f"{prefix}{sep}{name}" if prefix else name

        # Scan expressions for parameter references (names not in known_names)
        # These are runtime parameters that also need prefixing
        import re as _re
        all_exprs = []
        for stock in template.stocks:
            for flow in stock.flows:
                all_exprs.append(flow.expr)
        for aux in template.aux_vars:
            all_exprs.append(aux.expr)
        for expr in all_exprs:
            for ref in _re.findall(r'\b([A-Za-z_]\w*)\b', expr):
                if ref not in known_names and ref not in _FUNC_NAMES and ref not in replacements:
                    replacements[ref] = f"{prefix}{sep}{ref}" if prefix else ref

        # Sort by length (longest first) to avoid partial matches
        sorted_names = sorted(replacements.keys(), key=len, reverse=True)

        def _replace_refs(expr: str) -> str:
            """Replace variable references using word-boundary matching."""
            import re as _re
            result = expr
            for old_name in sorted_names:
                # Use word boundary to avoid replacing parts of function names
                pattern = r'\b' + _re.escape(old_name) + r'\b'
                result = _re.sub(pattern, replacements[old_name], result)
            return result

        # Copy stocks with prefixed names
        existing_stock_names = {s.name for s in model.stocks}
        for stock in template.stocks:
            new_name = f"{prefix}{sep}{stock.name}" if prefix else stock.name
            if new_name in existing_stock_names:
                logger.warning("Duplicate stock '%s' from include '%s' — may cause collisions", new_name, inc.submodel_name)
            new_flows = []
            for flow in stock.flows:
                # Update flow names and expressions
                new_flow_name = f"{prefix}{sep}{flow.name}" if prefix else flow.name
                new_expr = _replace_refs(flow.expr)
                new_flows.append(FlowDef(
                    name=new_flow_name,
                    direction=flow.direction,
                    expr=new_expr,
                    units=flow.units,
                ))
            new_stock = StockDef(
                name=new_name,
                initial=stock.initial,
                flows=new_flows,
                units=stock.units,
            )
            # Apply parameter overrides to initial value
            if stock.name in inc.params:
                new_stock.initial = inc.params[stock.name]
            model.stocks.append(new_stock)

        # Create aux entries for include parameters that aren't in the template
        # These are runtime parameters that need to exist as auxes
        # Insert BEFORE template auxes so they're evaluated first (topo-sort order)
        param_auxes: list[AuxDef] = []
        for param_name, param_val in inc.params.items():
            new_name = f"{prefix}{sep}{param_name}" if prefix else param_name
            # Check if this param already exists as an aux (from template expansion)
            if not any(a.name == new_name for a in model.aux_vars) and \
               not any(a.name == new_name for a in param_auxes):
                param_auxes.append(AuxDef(
                    name=new_name,
                    expr=str(param_val),
                    units="",
                ))

        # Copy auxes with prefixed names and parameter overrides
        for aux in template.aux_vars:
            new_name = f"{prefix}{sep}{aux.name}" if prefix else aux.name
            new_expr = _replace_refs(aux.expr)
            # Apply parameter overrides to expression
            if aux.name in inc.params:
                new_expr = str(inc.params[aux.name])
            param_auxes.append(AuxDef(
                name=new_name,
                expr=new_expr,
                units=aux.units,
            ))

        # Insert all auxes (param + template) before any existing auxes
        # Find insertion point (before any existing auxes from this include)
        insert_idx = 0
        for k, a in enumerate(model.aux_vars):
            if not a.name.startswith(prefix):
                break
            insert_idx = k + 1
        model.aux_vars[insert_idx:insert_idx] = param_auxes

        # Copy tables with prefixed names
        existing_table_names = {t.name for t in model.tables}
        for table in template.tables:
            new_name = f"{prefix}{sep}{table.name}" if prefix else table.name
            if new_name in existing_table_names:
                logger.warning("Duplicate table '%s' from include '%s' — may cause collisions", new_name, inc.submodel_name)
            model.tables.append(TableDef(
                name=new_name,
                x=list(table.x),
                y=list(table.y),
            ))

    return model


def parse_sysd(source: str) -> SysdModel:
    """Parse a .sysd source string into a SysdModel."""
    lines = _lex_sysd(source)
    model = _build_tree(lines)
    return _expand_includes(model)


def parse_sysd_file(path: str) -> SysdModel:
    """Load and parse a .sysd file."""
    with open(path, encoding="utf-8") as f:
        return parse_sysd(f.read())
