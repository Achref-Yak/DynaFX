"""System dynamics DSL — parse .sysd files into simulation-ready models.

Syntax (Vensim-inspired):
    model "Name"
      dt 0.5
      from 0 to 100

      stock "Stock Name": initial_value
        + "Inflow Name": rate_expression
        - "Outflow Name": rate_expression

      table "Table Name"
        x: [0, 10, 20]
        y: [5, 15, 5]

Expressions support: +, -, *, /, parentheses, MIN(a,b), MAX(a,b),
IF(cond,a,b), SMOOTH(x,delay), and references to other stocks/flows.
"""

from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from types import CodeType
from uuid import uuid4

from cognitive_engine.core.decomposer import SystemDecomposer
from cognitive_engine.system.equations import rk4_step, euler_step


# ── AST Nodes ───────────────────────────────────────────────────

@dataclass
class FlowDef:
    name: str
    direction: str     # "+" inflow, "-" outflow
    expr: str          # raw source expression
    units: str = ""    # optional unit annotation ~Unit~


@dataclass
class StockDef:
    name: str
    initial: float
    flows: list[FlowDef] = field(default_factory=list)
    units: str = ""    # optional unit annotation ~Unit~


@dataclass
class TableDef:
    name: str
    x: list[float]
    y: list[float]


@dataclass
class AuxDef:
    name: str
    expr: str
    units: str = ""    # optional unit annotation ~Unit~


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
class AgentDef:
    """Agent type definition with properties and behavioral rules."""
    name: str
    count: int = 1
    properties: list[AgentPropDef] = field(default_factory=list)
    rules: list[AgentRuleDef] = field(default_factory=list)


@dataclass
class QueueDef:
    """DES queue definition."""
    name: str
    capacity: int = -1       # -1 = unlimited
    initial: int = 0
    service_time: str = ""   # expression or distribution name


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
class SysdModel:
    name: str = ""
    dt: float = 1.0
    t_span: tuple[float, float] = (0.0, 100.0)
    stocks: list[StockDef] = field(default_factory=list)
    tables: list[TableDef] = field(default_factory=list)
    aux_vars: list[AuxDef] = field(default_factory=list)
    emergent_props: list = field(default_factory=list)  # list[EmergentProperty]
    agents: list[AgentDef] = field(default_factory=list)
    queues: list[QueueDef] = field(default_factory=list)
    resources: list[ResourceDef] = field(default_factory=list)
    events: list[EventDef] = field(default_factory=list)
    submodels: list[SubmodelDef] = field(default_factory=list)  # templates
    includes: list[IncludeDef] = field(default_factory=list)    # instantiations
    _compiled_cache: Any = field(default=None, repr=False)  # CompiledSystem, set on first simulate()

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

    def import_data(self, path: str) -> dict[str, Any]:
        """Import time series data from a CSV file.

        CSV format: first column is time, subsequent columns are variable names.
        Returns a dict of {variable_name: [(time, value), ...]} for use as
        forcing functions or calibration data.

        The data is also stored in self._imported_data for use during simulation.
        """
        import csv
        data: dict[str, list[tuple[float, float]]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for col_name in header[1:]:
                data[col_name.strip()] = []
            for row in reader:
                if not row:
                    continue
                try:
                    t = float(row[0])
                except (ValueError, IndexError):
                    continue
                for i, col_name in enumerate(header[1:], 1):
                    if i < len(row):
                        try:
                            val = float(row[i])
                            data[col_name.strip()].append((t, val))
                        except ValueError:
                            pass
        self._imported_data = data
        return data

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
                    frac = (t - times[i]) / (times[i + 1] - times[i])
                    return values[i] + frac * (values[i + 1] - values[i])
            return values[-1]
        return interpolator

    def simulate(
        self,
        method: str = "rk4",
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> SysdModelResult:
        t_span = t_span or self.t_span
        step = dt or self.dt
        step_fn = rk4_step if method == "rk4" else euler_step
        if params is None:
            params = {}

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
        if self._compiled_cache is None:
            self._compiled_cache = _compile_system(self)
        f, stock_names, y0, aux_count = _build_system(self, params, self.emergent_props, seed=42, cache=self._compiled_cache)

        # Initialize ABM engine if agents are defined
        abm_engine = None
        if self.agents:
            from cognitive_engine.system.agent import ABMEngine
            abm_engine = ABMEngine(self.agents)
            abm_engine.initialize()

        # Initialize DES engine if queues/resources/events are defined
        des_engine = None
        if self.queues or self.resources or self.events:
            from cognitive_engine.system.des import (
                DESEngine, Queue, Resource, Event, DESClock,
            )
            des_engine = DESEngine()
            for q in self.queues:
                q_obj = Queue(q.name, q.capacity, q.service_time)
                # Compile service_time expression if provided
                if q.service_time:
                    try:
                        from cognitive_engine.system.dsl import ExprParser, _compile_expr
                        st_node = ExprParser(q.service_time).parse()
                        st_compiled = _compile_expr(st_node, set(), set())
                        q_obj._compiled_service_time = lambda _c=st_compiled: eval(
                            _c, {"__builtins__": {}}, {**params, **dict(zip(stock_names, y))}
                        )
                    except Exception:
                        pass
                des_engine.add_queue(q_obj)
            for r in self.resources:
                r_obj = Resource(r.name, r.capacity, r.cost_per_unit)
                des_engine.add_resource(r_obj)
            for ev in self.events:
                # Schedule initial events
                if ev.rate > 0:
                    des_engine.schedule_event(0.0, ev.name, ev.payload)
                for enq in ev.enqueue_to:
                    if enq in des_engine.queues:
                        des_engine.queues[enq].enqueue(
                            {"event": ev.name, "time": 0.0}, 0.0,
                            event_queue=des_engine.event_queue,
                        )

        t0, t_end = t_span
        direction = 1 if t_end >= t0 else -1
        y = list(y0)
        times = [t0]
        y_hist = [list(y)]

        while abs(t0 - t_end) > 1e-12:
            remaining = abs(t_end - t0)
            if remaining < abs(step):
                y = step_fn(f, t0, y, direction * remaining, params)
                t0 = t_end
            else:
                y = step_fn(f, t0, y, direction * step, params)
                t0 += direction * step

            # Run ABM step if agents exist
            if abm_engine:
                shared_state = dict(zip(stock_names, y))
                shared_state.update(params)  # include parameters for agent conditions
                abm_metrics = abm_engine.step(t0, step, shared_state)
                params.update(abm_metrics)

            # Run DES step if queues/resources/events exist
            if des_engine:
                shared_state = dict(zip(stock_names, y))
                params.update(shared_state)
                # Recompile service_time lambdas with current state
                for q in self.queues:
                    if q.service_time and q.name in des_engine.queues:
                        q_obj = des_engine.queues[q.name]
                        try:
                            from cognitive_engine.system.dsl import ExprParser, _compile_expr
                            st_node = ExprParser(q.service_time).parse()
                            st_compiled = _compile_expr(st_node, set(), set())
                            _state_snapshot = dict(shared_state)
                            q_obj._compiled_service_time = lambda _c=st_compiled, _s=_state_snapshot: eval(
                                _c, {"__builtins__": {}}, {**params, **_s}
                            )
                        except Exception:
                            pass
                des_metrics = des_engine.step(t0 - direction * step, step)
                params.update(des_metrics)

            times.append(t0)
            y_hist.append(list(y))

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
        )

    def validate(self, params: Optional[set[str]] = None) -> ValidationResult:
        result = ValidationResult()
        all_names: set[str] = set()
        all_names.update(_BUILTIN_NAMES)
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
        from cognitive_engine.system.emergent import run_consistency_checks
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
                    mu = math.log(low)
                    sigma = (math.log(high) - mu) / 3
                    val = rng.lognormvariate(mu, sigma)
                else:
                    val = low + rng.random() * (high - low)
                sample[pname] = val
            samples.append(sample)

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
    times: list[float]
    stocks: list[str]
    values: dict[str, list[float]]
    final_state: list[float]
    method: str
    steps: int
    model_name: str = ""
    abm_engine: Any = None  # ABMEngine if agents were simulated
    des_engine: Any = None  # DESEngine if queues/resources/events were simulated

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
            for ax, name in zip(axes, names):
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

_BUILTIN_NAMES: set[str] = {
    "t", "dt",
    # Math
    "MIN", "MAX", "IF", "ABS", "EXP", "LN", "SQRT", "SIN", "COS", "PI",
    # Smoothing / delays
    "SMOOTH", "SMOOTHI", "DELAY3", "DELAYN", "DELAY_FIXED",
    # Time functions
    "PULSE", "STEP", "RAMP", "NOISE",
}


# ── Lookup table ────────────────────────────────────────────────

class LookupTable:
    """Linear-interpolated lookup table for time-varying parameters."""
    def __init__(self, x: list[float], y: list[float]):
        self.x = x
        self.y = y

    def __call__(self, t: float) -> float:
        if t <= self.x[0]:
            return self.y[0]
        if t >= self.x[-1]:
            return self.y[-1]
        for i in range(len(self.x) - 1):
            if self.x[i] <= t < self.x[i + 1]:
                frac = (t - self.x[i]) / (self.x[i + 1] - self.x[i])
                return self.y[i] + frac * (self.y[i + 1] - self.y[i])
        return self.y[-1]


# ── Expression AST and Parser ───────────────────────────────────

_TOKEN_RE = re.compile(r"""
    \s*(?:((?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)  # number (incl. sci notation)
         |([a-zA-Z_]\w*)           # identifier
         |(>=|<=|!=|==|[><+\-*/(),\[\]])  # operators
         |(\S)                     # unexpected char
    )""", re.VERBOSE)

_OP_CHARS = frozenset(["+", "-", "*", "/", "(", ")", ",", "[", "]", ">", "<", "=", ">=", "<=", "==", "!="])

Token = tuple[str, str]  # (type, value)


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            pos += 1
            continue
        if m.group(1):
            tokens.append(("num", m.group(1)))
        elif m.group(2):
            tokens.append(("id", m.group(2)))
        elif m.group(3):
            ch = m.group(3)
            if ch in _OP_CHARS:
                tokens.append(("op", ch))
        pos = m.end()
    return tokens


class ExprNode: pass

@dataclass
class ExprLiteral(ExprNode):
    value: float

@dataclass
class ExprRef(ExprNode):
    name: str

@dataclass
class ExprBinOp(ExprNode):
    op: str
    left: ExprNode
    right: ExprNode

@dataclass
class ExprFuncCall(ExprNode):
    name: str
    args: list[ExprNode]


class ExprParser:
    def __init__(self, source: str):
        self.tokens = _tokenize(source)
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> Token:
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected end of expression")
        if expected and t[1] != expected:
            raise SyntaxError(f"Expected '{expected}', got '{t[1]}'")
        self.pos += 1
        return t

    _COMP_OPS = frozenset([">", "<", ">=", "<=", "==", "!="])

    def parse(self) -> ExprNode:
        return self._comparison()

    def _comparison(self) -> ExprNode:
        left = self._expression()
        while self.peek() and self.peek()[1] in self._COMP_OPS:
            op = self.consume()[1]
            right = self._expression()
            left = ExprBinOp(op, left, right)
        return left

    def _expression(self) -> ExprNode:
        left = self._term()
        while self.peek() and self.peek()[1] in ("+", "-"):
            op = self.consume()[1]
            right = self._term()
            left = ExprBinOp(op, left, right)
        return left

    def _term(self) -> ExprNode:
        left = self._unary()
        while self.peek() and self.peek()[1] in ("*", "/"):
            op = self.consume()[1]
            right = self._unary()
            left = ExprBinOp(op, left, right)
        return left

    def _unary(self) -> ExprNode:
        if self.peek() and self.peek()[1] == "-":
            self.consume()
            return ExprBinOp("*", ExprLiteral(-1.0), self._unary())
        return self._primary()

    def _primary(self) -> ExprNode:
        t = self.peek()
        if t is None:
            raise SyntaxError("Expected expression, got end")
        if t[0] == "num":
            self.consume()
            return ExprLiteral(float(t[1]))
        if t[0] == "id":
            name = self.consume()[1]
            if self.peek() and self.peek()[1] == "(":
                self.consume()  # (
                args = []
                if self.peek() and self.peek()[1] != ")":
                    args.append(self._comparison())
                    while self.peek() and self.peek()[1] == ",":
                        self.consume()
                        args.append(self._comparison())
                self.consume(")")
                return ExprFuncCall(name, args)
            return ExprRef(name)
        if t[1] == "(":
            self.consume()
            node = self._comparison()
            self.consume(")")
            return node
        raise SyntaxError(f"Unexpected token '{t[1]}'")


# ── Expression compiler ─────────────────────────────────────────

_COMPILED_CONSTANTS: dict[str, str] = {
    "PI": repr(math.pi),
}

def _compile_expr(node: ExprNode, stock_names: set[str], aux_names: set[str] = frozenset()) -> str:
    if isinstance(node, ExprLiteral):
        return repr(node.value)
    if isinstance(node, ExprRef):
        if node.name == "dt":
            return "_p['dt']"
        if node.name == "t":
            return "t"
        if node.name in _COMPILED_CONSTANTS:
            return _COMPILED_CONSTANTS[node.name]
        if node.name in aux_names:
            return f"_a['{node.name}']"
        return f"_s.get('{node.name}', 0.0)"
    if isinstance(node, ExprBinOp):
        left = _compile_expr(node.left, stock_names, aux_names)
        right = _compile_expr(node.right, stock_names, aux_names)
        return f"({left} {node.op} {right})"
    if isinstance(node, ExprFuncCall):
        args = ", ".join(
            _compile_expr(a, stock_names, aux_names) for a in node.args
        )
        return f"{node.name}({args})"
    raise TypeError(f"Unknown node: {node}")


# ── System builder ──────────────────────────────────────────────

def _find_refs(node: ExprNode, names: set[str]) -> set[str]:
    """Collect all ExprRef names from an AST that are in `names`."""
    if isinstance(node, ExprRef):
        return {node.name} if node.name in names else set()
    if isinstance(node, ExprBinOp):
        return _find_refs(node.left, names) | _find_refs(node.right, names)
    if isinstance(node, ExprFuncCall):
        result: set[str] = set()
        for a in node.args:
            result |= _find_refs(a, names)
        return result
    return set()


def _topo_sort(names: list[str], expr_nodes: list[ExprNode], all_names: set[str]) -> list[int]:
    """Topological sort of indices based on cross-references in expression trees.
    
    Returns a list of indices in evaluation order. Falls back to definition
    order if a cycle is detected.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    # adj[i] = set of indices that expression i depends on
    adj: list[set[int]] = []
    for node in expr_nodes:
        refs = _find_refs(node, set(names))
        adj.append({name_to_idx[r] for r in refs if r in name_to_idx})

    in_degree = [len(deps) for deps in adj]
    queue = [i for i, d in enumerate(in_degree) if d == 0]
    # Build reverse adjacency for Kahn's algorithm
    rev_adj: list[set[int]] = [set() for _ in names]
    for j, deps in enumerate(adj):
        for i in deps:
            rev_adj[i].add(j)

    order: list[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j in rev_adj[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                queue.append(j)

    if len(order) == len(names):
        return order
    # Cycle — fall back to definition order
    return list(range(len(names)))


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
    delay_fixed_compiled: list[tuple[str, str, float]]
    builtins: dict[str, Any]
    # Pre-compiled code objects (avoids string re-parsing on every eval)
    aux_code: list[CodeType]
    inflow_code: list[CodeType]
    outflow_code: list[CodeType]
    smooth_ode_code: list[CodeType]
    df_input_code: list[CodeType]


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

    for si, s in enumerate(model.stocks):
        for fl in s.flows:
            node = ExprParser(fl.expr).parse()
            modified = _replace_smooths(node, smooth_params)
            compiled = _compile_expr(modified, name_set, aux_set)
            if fl.direction == "+":
                stock_inflow[si].append(compiled)
            else:
                stock_outflow[si].append(compiled)

    aux_expr_nodes: list[ExprNode] = []
    for a in model.aux_vars:
        node = ExprParser(a.expr).parse()
        modified = _replace_smooths(node, smooth_params)
        aux_expr_nodes.append(modified)

    smooth_names: list[str] = []
    delay_fixed_entries: list[tuple[str, str, float]] = []
    smooth_delay_exprs: list[str] = []
    smooth_init_exprs: list[str] = []
    for entry in smooth_params:
        entry_type, aux_name, input_expr_str, delay_time, init_val = entry
        smooth_names.append(aux_name)
        all_names.append(aux_name)
        # Store delay/init as expression strings for runtime evaluation
        delay_str = _serialize_expr(delay_time) if isinstance(delay_time, ExprNode) else str(delay_time)
        init_str = _serialize_expr(init_val) if isinstance(init_val, ExprNode) else str(init_val)
        smooth_delay_exprs.append(delay_str)
        smooth_init_exprs.append(init_str)
        # Evaluate init at parse time for base_y0 (fallback 0.0)
        try:
            base_y0.append(float(eval(init_str, {"__builtins__": {}})))
        except Exception:
            base_y0.append(0.0)
        if entry_type == "delay_fixed":
            try:
                delay_val = float(eval(delay_str, {"__builtins__": {}}))
            except Exception:
                delay_val = 1.0
            delay_fixed_entries.append((aux_name, input_expr_str, delay_val))

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
        if entry_type == "delay_fixed":
            smooth_ode_strs.append("0.0")
        else:
            smooth_ode_strs.append(
                f"({input_compiled} - _s.get('{aux_name}', 0.0)) / ({delay_str})"
            )

    aux_compile_strs: list[str] = []
    for node in aux_expr_nodes:
        aux_compile_strs.append(_compile_expr(node, name_set, aux_set))

    aux_order = _topo_sort(aux_names, aux_expr_nodes, name_set | aux_set)
    aux_names_ordered = [aux_names[i] for i in aux_order]
    aux_compile_ordered = [aux_compile_strs[i] for i in aux_order]

    import math
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
    }

    delay_fixed_compiled: list[tuple[str, str, float]] = []
    for df_name, df_input_str, df_delay in delay_fixed_entries:
        df_node = ExprParser(df_input_str).parse()
        df_compiled = _compile_expr(df_node, name_set, aux_set)
        delay_fixed_compiled.append((df_name, df_compiled, df_delay))

    # Pre-compile all expression strings to code objects (avoids re-parsing on every eval)
    _co = "<compiled>"
    aux_code = [compile(s, _co, "eval") for s in aux_compile_ordered]
    inflow_code = [compile(s, _co, "eval") for s in inflow_strs]
    outflow_code = [compile(s, _co, "eval") for s in outflow_strs]
    smooth_ode_code = [compile(s, _co, "eval") for s in smooth_ode_strs]
    df_input_code = [compile(entry[1], _co, "eval") for entry in delay_fixed_compiled]

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
        builtins=builtins,
        aux_code=aux_code,
        inflow_code=inflow_code,
        outflow_code=outflow_code,
        smooth_ode_code=smooth_ode_code,
        df_input_code=df_input_code,
    )


def _build_system(
    model: SysdModel,
    params: dict[str, Any],
    emergent_props: Optional[list] = None,
    seed: int = 42,
    cache: Optional[CompiledSystem] = None,
) -> tuple[Callable, list[str], list[float], int]:
    """Build ODE system from SysdModel.

    Returns: (f(t, y, params), all_names, y0, aux_state_count)
    """
    if cache is None:
        cache = _compile_system(model)

    all_names = list(cache.all_names)
    y0 = list(cache.base_y0)
    stock_names = cache.stock_names
    inflow_strs = cache.inflow_strs
    outflow_strs = cache.outflow_strs
    aux_names_ordered = cache.aux_names_ordered
    smooth_names = cache.smooth_names
    smooth_ode_strs = cache.smooth_ode_strs
    delay_fixed_compiled = cache.delay_fixed_compiled
    _builtins = cache.builtins
    # Pre-compiled code objects
    _aux_code = cache.aux_code
    _inflow_code = cache.inflow_code
    _outflow_code = cache.outflow_code
    _smooth_ode_code = cache.smooth_ode_code
    _df_input_code = cache.df_input_code

    import random as _random
    _rng = _random.Random(seed)  # seeded for reproducibility

    # Module-level buffer for DELAY_FIXED transport delays: {name: [(t, val), ...]}
    _delay_fixed_buffers: dict[str, list[tuple[float, float]]] = {
        name: [] for name, _, _ in delay_fixed_compiled
    }

    # Pre-compute things that don't change between f() calls
    _no_builtins = {"__builtins__": {}}
    _stock_count = len(model.stocks)
    _df_count = len(delay_fixed_compiled)

    # Pre-filter params to only numeric values (avoids per-call isinstance check)
    _numeric_params = {k: v for k, v in params.items() if isinstance(v, (int, float))}

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
    _callable_params = [(k, v) for k, v in params.items() if hasattr(v, "__call__")]

    def f(t: float, y: list[float], p: dict) -> list[float]:
        _s = dict(zip(all_names, y))
        # Merge pre-computed numeric params + runtime numeric params
        _s.update(_numeric_params)
        if p:
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    _s[k] = v
        _a: dict[str, float] = {}
        # Build eval namespace — minimal dict, reuse _no_builtins
        _ns: dict = {
            **_builtins, **_numeric_params, "_s": _s, "_p": params, "_a": _a, "t": t,
            "PULSE": lambda volume, start, width: volume if start <= t < start + width else 0.0,
            "STEP": lambda height, start: height if t >= start else 0.0,
            "RAMP": lambda slope, start, end: (
                0.0 if t < start else
                slope * (t - start) if t <= end else
                slope * (end - start)
            ),
            "NOISE": lambda amplitude: _rng.uniform(-amplitude, amplitude),
        }
        # Inject lookup tables and callables into eval namespace only (not _s)
        for _k, _v in _callable_params:
            _ns[_k] = _v
        if p:
            for _k, _v in p.items():
                if hasattr(_v, "__call__"):
                    _ns[_k] = _v
        # Inject resolved smooth delay/init values into namespace
        # The delay expression strings reference parameter names (e.g., "sentiment_delay")
        # which are already in _numeric_params. But we also need to handle the case
        # where the delay is a literal number embedded in the expression.
        # The smooth_ode_strs already contain the delay expression inline,
        # so we just need to make sure any variable references resolve.
        # Evaluate auxes in dependency order (using pre-compiled code objects)
        for _i in range(len(aux_names_ordered)):
            _a[aux_names_ordered[_i]] = eval(_aux_code[_i], _no_builtins, _ns)
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
        # Process DELAY_FIXED transport delays
        for _dfi in range(_df_count):
            df_name, df_compiled, df_delay = delay_fixed_compiled[_dfi]
            try:
                input_val = eval(_df_input_code[_dfi], _no_builtins, _ns)
            except Exception:
                input_val = 0.0
            _delay_fixed_buffers[df_name].append((t, input_val))
            target_t = t - df_delay
            delayed_val = 0.0
            buf = _delay_fixed_buffers[df_name]
            if buf:
                if target_t <= buf[0][0]:
                    delayed_val = buf[0][1]
                else:
                    for i in range(len(buf) - 1, -1, -1):
                        if buf[i][0] <= target_t:
                            delayed_val = buf[i][1]
                            break
            _s[df_name] = delayed_val
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

    return f, all_names, y0, len(smooth_names)


def _replace_smooths(
    node: ExprNode,
    smooth_params: list[tuple[str, str, str, float, float]],
) -> ExprNode:
    """Walk expression tree, replace SMOOTH/SMOOTHI/DELAY3/DELAYN/DELAY_FIXED calls with aux variable references."""
    if isinstance(node, ExprBinOp):
        return ExprBinOp(
            node.op,
            _replace_smooths(node.left, smooth_params),
            _replace_smooths(node.right, smooth_params),
        )
    if isinstance(node, ExprFuncCall):
        args = [_replace_smooths(a, smooth_params) for a in node.args]
        if node.name == "SMOOTH" and len(args) >= 2:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                delay = delay_node.value
            else:
                delay = delay_node  # store ExprNode for runtime eval
            aux_name = f"_smooth_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("smooth", aux_name, input_expr, delay, 0.0))
            return ExprRef(aux_name)
        if node.name == "SMOOTHI" and len(args) >= 3:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                delay = delay_node.value
            else:
                delay = delay_node  # store ExprNode for runtime eval
            init_node = args[2]
            if isinstance(init_node, ExprLiteral):
                init_val = init_node.value
            else:
                init_val = init_node  # store ExprNode for runtime eval
            aux_name = f"_smooth_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("smooth", aux_name, input_expr, delay, init_val))
            return ExprRef(aux_name)
        if node.name == "DELAY3" and len(args) >= 2:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                total_delay = delay_node.value
                stage_delay = total_delay / 3.0
            else:
                # Non-literal: store expression, divide at runtime
                stage_delay = ExprBinOp("/", delay_node, ExprLiteral(3.0))
            current_input = _serialize_expr(args[0])
            for i in range(3):
                aux_name = f"_delay3_{len(smooth_params)}"
                smooth_params.append(("smooth", aux_name, current_input, stage_delay, 0.0))
                current_input = aux_name
            return ExprRef(current_input)
        if node.name == "DELAYN" and len(args) >= 3:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                total_delay = delay_node.value
            else:
                total_delay = delay_node  # store ExprNode for runtime eval
            n_node = args[2]
            if isinstance(n_node, ExprLiteral):
                n_stages = max(1, int(n_node.value))
            else:
                n_stages = 3
            if isinstance(total_delay, (int, float)):
                stage_delay = total_delay / n_stages
            else:
                stage_delay = ExprBinOp("/", total_delay, ExprLiteral(n_stages))
            current_input = _serialize_expr(args[0])
            for i in range(n_stages):
                aux_name = f"_delayn_{len(smooth_params)}"
                smooth_params.append(("smooth", aux_name, current_input, stage_delay, 0.0))
                current_input = aux_name
            return ExprRef(current_input)
        if node.name == "DELAY_FIXED" and len(args) >= 2:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                delay = delay_node.value
            else:
                delay = delay_node  # store ExprNode for runtime eval
            aux_name = f"_delay_fixed_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("delay_fixed", aux_name, input_expr, delay, 0.0))
            return ExprRef(aux_name)
        return ExprFuncCall(node.name, args)
    return node


def _serialize_expr(node: ExprNode) -> str:
    if isinstance(node, ExprLiteral):
        return repr(node.value)
    if isinstance(node, ExprRef):
        return node.name
    if isinstance(node, ExprBinOp):
        return f"({_serialize_expr(node.left)} {node.op} {_serialize_expr(node.right)})"
    if isinstance(node, ExprFuncCall):
        args = ", ".join(_serialize_expr(a) for a in node.args)
        return f"{node.name}({args})"
    return "0"


# ── Sysd Lexer / Structure Parser ───────────────────────────────

_COMMENT_RE = re.compile(r"//.*$")
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
            # Fall through if parent is not TableDef (e.g., agent rule effect)

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
            # Pop stack to find the owning AgentDef (may need to pop prior rule)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if isinstance(parent, AgentDef):
                rule = _parse_agent_rule(args)
                parent.rules.append(rule)
                # Push rule onto stack so indented effect lines attach to it
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

        # ── DES keywords ────────────────────────────────────────
        if keyword == "queue":
            name, _ = _parse_name_value(args)
            # Parse capacity from args like "Line": capacity 3, service_time 2
            capacity = -1
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
            qd = QueueDef(name=name, capacity=capacity)
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
        # Strip ~Unit~ annotation from value before parsing
        val = re.sub(r'~[^~]*~', '', val).strip()
        try:
            return name, float(val)
        except ValueError:
            return name, 0.0
    return _STRIP_RE.sub("", args.strip()), 0.0


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
    return [float(p) for p in parts]


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
            continue  # unknown submodel, skip

        prefix = inc.instance_name
        sep = "_" if prefix else ""

        # Build replacement map from original names to prefixed names
        # Only replace names that are actual variables (not function names)
        _FUNC_NAMES = {"SMOOTH", "SMOOTHI", "DELAY3", "DELAYN", "DELAY_FIXED",
                       "MIN", "MAX", "IF", "ABS", "EXP", "LN", "SQRT",
                       "SIN", "COS", "PI", "PULSE", "STEP", "RAMP", "NOISE"}
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
        for stock in template.stocks:
            new_name = f"{prefix}{sep}{stock.name}" if prefix else stock.name
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
        for table in template.tables:
            new_name = f"{prefix}{sep}{table.name}" if prefix else table.name
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
