"""SimulationController — step-by-step simulation control.

Provides single-step advance, state injection, param hot-swap,
and event callbacks. Wraps SysdModel for interactive/gaming use.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dynafx.system.dsl import (
    SysdModel,
    SysdModelResult,
    _build_system,
    _compile_system,
)
from dynafx.system.equations import rk4_step, euler_step


@dataclass
class SimulationSnapshot:
    """Point-in-time simulation state."""
    t: float
    y: list[float]
    params: dict[str, Any]
    step_count: int


class SimulationController:
    """Step-by-step simulation controller with callbacks, state injection,
    and param hot-swap support.

    Usage:
        ctrl = SimulationController(model)
        ctrl.initialize(params={"beta": 0.3})
        while ctrl.t < 100:
            ctrl.step()
            if ctrl.t == 50:
                ctrl.set_param("beta", 0.1)
        result = ctrl.get_result()
    """

    def __init__(
        self,
        model: SysdModel,
        on_step: Optional[Callable[[SimulationController], None]] = None,
        on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
        on_state_change: Optional[Callable[[str, float, float], None]] = None,
    ):
        self.model = model
        self.on_step = on_step
        self.on_event = on_event
        self.on_state_change = on_state_change

        # Internal state
        self._f: Optional[Callable] = None
        self._all_names: list[str] = []
        self._y: list[float] = []
        self._y0: list[float] = []
        self._t: float = 0.0
        self._t_span: tuple[float, float] = model.t_span
        self._dt: float = model.dt
        self._step_fn: Optional[Callable] = None
        self._params: dict[str, Any] = {}
        self._step_count: int = 0
        self._aux_count: int = 0
        self._stock_names: list[str] = []
        self._pipeline_info: Optional[dict[str, Any]] = None
        self._is_initialized: bool = False
        self._is_done: bool = False

        # ABM engine
        self._abm_engine: Any = None

        # DES engine
        self._des_engine: Any = None

        # History trackers
        self._times: list[float] = []
        self._y_hist: list[list[float]] = []

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self._t

    @property
    def dt(self) -> float:
        """Current time step."""
        return self._dt

    @property
    def step_count(self) -> int:
        """Number of steps taken."""
        return self._step_count

    @property
    def is_running(self) -> bool:
        """True if initialized and not yet finished."""
        return self._is_initialized and not self._is_done

    @property
    def params(self) -> dict[str, Any]:
        """Current parameters (read-write via set_param)."""
        return self._params

    def initialize(
        self,
        params: Optional[dict[str, Any]] = None,
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        method: str = "rk4",
    ) -> None:
        """Set up the simulation for step-by-step control.

        Args:
            params: Parameter overrides.
            t_span: (t_start, t_end). Defaults to model's t_span.
            dt: Time step. Defaults to model's dt.
            method: "rk4" or "euler".
        """
        t_span = t_span or self.model.t_span
        dt = dt or self.model.dt
        self._dt = dt
        self._t_span = t_span
        self._step_fn = rk4_step if method == "rk4" else euler_step

        self._params = dict(params or {})
        for t in self.model.tables:
            from dynafx.system.dsl import LookupTable
            self._params[t.name] = LookupTable(t.x, t.y)
        self._params["dt"] = dt

        for a in self.model.aux_vars:
            if a.name not in self._params:
                try:
                    self._params[a.name] = float(a.expr)
                except ValueError:
                    pass

        if self.model._compiled_cache is None:
            self.model._compiled_cache = _compile_system(self.model)

        build_result = _build_system(
            self.model, self._params, self.model.emergent_props,
            seed=42, cache=self.model._compiled_cache,
        )
        self._f = build_result[0]
        self._all_names = build_result[1]
        self._y0 = build_result[2]
        self._aux_count = build_result[3]
        self._pipeline_info = build_result[4]

        self._stock_names = [s.name for s in self.model.stocks]
        self._y = list(self._y0)
        self._t = t_span[0]

        # Initialize pipeline delays at t0
        if self._pipeline_info is not None:
            self._pipeline_info["process"](self._t, self._y, self._params)

        # ABM engine
        self._abm_engine = None
        if self.model.agents:
            from dynafx.system.agent import ABMEngine
            self._abm_engine = ABMEngine(self.model.agents)
            self._abm_engine.initialize()

        # DES engine
        self._des_engine = None
        if self.model.queues or self.model.resources or self.model.events:
            from dynafx.system.des import (
                DESEngine, Queue, Resource, Event, DESClock,
            )
            self._des_engine = DESEngine()
            for q in self.model.queues:
                q_obj = Queue(q.name, q.capacity, q.service_time)
                if q.service_time:
                    try:
                        from dynafx.system.dsl import ExprParser, _compile_expr
                        st_node = ExprParser(q.service_time).parse()
                        st_compiled = _compile_expr(st_node, set(), set())
                        q_obj._compiled_service_time = lambda _c=st_compiled: eval(
                            _c, {"__builtins__": {}},
                            {**self._params, **dict(zip(self._stock_names, self._y))},
                        )
                    except Exception:
                        pass
                self._des_engine.add_queue(q_obj)
            for r in self.model.resources:
                r_obj = Resource(r.name, r.capacity, r.cost_per_unit)
                self._des_engine.add_resource(r_obj)
            for ev in self.model.events:
                if ev.rate > 0:
                    self._des_engine.schedule_event(0.0, ev.name, ev.payload)
                for enq in ev.enqueue_to:
                    if enq in self._des_engine.queues:
                        self._des_engine.queues[enq].enqueue(
                            {"event": ev.name, "time": 0.0}, 0.0,
                            event_queue=self._des_engine.event_queue,
                        )

        self._times = [self._t]
        self._y_hist = [list(self._y)]
        self._step_count = 0
        self._is_initialized = True
        self._is_done = False

    def step(self) -> bool:
        """Advance one timestep.

        Returns:
            True if the simulation advanced, False if finished.
        """
        if not self._is_initialized:
            raise RuntimeError("SimulationController not initialized. Call initialize() first.")
        if self._is_done:
            return False

        direction = 1 if self._t_span[1] >= self._t_span[0] else -1
        remaining = abs(self._t_span[1] - self._t)
        if remaining < 1e-12:
            self._is_done = True
            return False

        step_size = min(abs(self._dt), remaining) * direction
        old_y = list(self._y)
        old_t = self._t

        self._y = self._step_fn(self._f, self._t, self._y, step_size, self._params)
        self._t += step_size

        # Process pipeline delays (once per step)
        if self._pipeline_info is not None:
            self._pipeline_info["process"](self._t, self._y, self._params)

        # ABM step
        if self._abm_engine:
            shared_state = dict(zip(self._stock_names, self._y))
            shared_state.update(self._params)
            abm_metrics = self._abm_engine.step(self._t, abs(step_size), shared_state)
            self._params.update(abm_metrics)

        # DES step
        if self._des_engine:
            shared_state = dict(zip(self._stock_names, self._y))
            self._params.update(shared_state)
            for q in self.model.queues:
                if q.service_time and q.name in self._des_engine.queues:
                    q_obj = self._des_engine.queues[q.name]
                    try:
                        from dynafx.system.dsl import ExprParser, _compile_expr
                        st_node = ExprParser(q.service_time).parse()
                        st_compiled = _compile_expr(st_node, set(), set())
                        _state_snapshot = dict(shared_state)
                        q_obj._compiled_service_time = lambda _c=st_compiled, _s=_state_snapshot: eval(
                            _c, {"__builtins__": {}}, {**self._params, **_s}
                        )
                    except Exception:
                        pass
            des_metrics = self._des_engine.step(self._t - abs(step_size), abs(step_size))
            self._params.update(des_metrics)

        self._step_count += 1
        self._times.append(self._t)
        self._y_hist.append(list(self._y))

        # Callbacks
        if self.on_step is not None:
            self.on_step(self)
        if self.on_state_change is not None and old_y != self._y:
            for i, name in enumerate(self._all_names):
                if i < len(old_y) and i < len(self._y):
                    if old_y[i] != self._y[i]:
                        self.on_state_change(name, old_y[i], self._y[i])

        if abs(self._t - self._t_span[1]) < 1e-12:
            self._is_done = True

        return True

    def get_state(self) -> dict[str, float]:
        """Return current state as {name: value} for stocks and auxes."""
        return dict(zip(self._all_names, self._y))

    def get_stock_state(self) -> dict[str, float]:
        """Return current stock values only (excludes smooth/delay auxes)."""
        pure_count = len(self._stock_names)
        return dict(zip(self._stock_names, self._y[:pure_count]))

    def set_param(self, name: str, value: Any) -> None:
        """Change a parameter mid-simulation (hot-swap).
        Takes effect on the next step() call.
        """
        old = self._params.get(name)
        self._params[name] = value
        if self.on_state_change is not None and old != value:
            self.on_state_change(name, old, value)

    def inject_state(self, name: str, value: float) -> None:
        """Directly set a state variable (stock, aux, delay, smooth).
        Takes effect immediately on the next f() call.
        """
        if name in self._all_names:
            idx = self._all_names.index(name)
            old = self._y[idx]
            self._y[idx] = value
            if self.on_state_change is not None and old != value:
                self.on_state_change(name, old, value)
            # Update y_hist so get_result is consistent
            if self._y_hist:
                self._y_hist[-1] = list(self._y)
        else:
            # Try as a parameter
            self.set_param(name, value)

    def get_result(self) -> SysdModelResult:
        """Build a SysdModelResult from accumulated history.
        Call after simulation completes.
        """
        pure_stocks = len(self._stock_names)
        out_stock_names = self._stock_names
        out_y_hist = self._y_hist
        if self._aux_count:
            out_y_hist = [row[:pure_stocks] for row in self._y_hist]

        return SysdModelResult(
            times=list(self._times),
            stocks=out_stock_names,
            values={
                name: [row[i] for row in out_y_hist]
                for i, name in enumerate(out_stock_names)
            },
            final_state=out_y_hist[-1] if out_y_hist else [],
            method="rk4" if self._step_fn == rk4_step else "euler",
            steps=self._step_count,
            model_name=self.model.name,
            abm_engine=self._abm_engine,
            des_engine=self._des_engine,
        )

    def snapshot(self) -> SimulationSnapshot:
        """Capture current simulation state for later restore."""
        return SimulationSnapshot(
            t=self._t,
            y=list(self._y),
            params=dict(self._params),
            step_count=self._step_count,
        )

    def restore(self, snap: SimulationSnapshot) -> None:
        """Restore simulation state from a snapshot."""
        self._t = snap.t
        self._y = list(snap.y)
        self._params = dict(snap.params)
        self._step_count = snap.step_count
        self._is_done = False
        if self._times and self._y_hist:
            self._times.append(self._t)
            self._y_hist.append(list(self._y))

    def run_until(self, t_end: float) -> None:
        """Advance simulation until time reaches t_end."""
        max_steps = 100000
        steps = 0
        while self._t < t_end - 1e-12 and steps < max_steps:
            if not self.step():
                break
            steps += 1


def run_scenario(args: tuple) -> SysdModelResult:
    """Execute a single scenario. Used by BatchRunner for multiprocessing.

    Args:
        (model, params, method, t_span, dt, seed)

    Returns: SysdModelResult
    """
    model, params, method, t_span, dt, seed = args
    return model.simulate(
        params=params, method=method, t_span=t_span, dt=dt,
    )


def batch_run(
    model: SysdModel,
    scenario_params: list[dict[str, Any]],
    method: str = "rk4",
    t_span: Optional[tuple[float, float]] = None,
    dt: Optional[float] = None,
    n_jobs: int = 1,
) -> list[SysdModelResult]:
    """Run multiple scenarios, optionally in parallel.

    Args:
        model: The SysdModel to simulate.
        scenario_params: List of param dicts, one per scenario.
        method: "rk4" or "euler".
        t_span: Override time span (default model.t_span).
        dt: Override time step (default model.dt).
        n_jobs: Number of parallel workers. 1 = sequential.

    Returns:
        List of SysdModelResult in the same order as scenario_params.
    """
    args_list = [
        (model, params, method, t_span, dt, 42)
        for params in scenario_params
    ]

    if n_jobs <= 1:
        return [run_scenario(args) for args in args_list]

    with mp.Pool(min(n_jobs, len(args_list))) as pool:
        results = pool.map(run_scenario, args_list)
    return results
