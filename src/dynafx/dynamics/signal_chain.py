"""SignalChain — reusable leading-indicator → outcome model builder.

Builds a SysdModel for the pattern:

    trace signal → DELAY3 (detection) → SMOOTH (interpretation)
    → IF (threshold gate) → impact → modifies core stock outflow

The trace is a **leading indicator** that modulates an existing process.
It is NOT the same thing that feeds the core stock — that's what makes
it a useful signal rather than a direct measurement.

Usage (Python API):

    model = SignalChain.build(
        name="food_delivery",
        trace_expr="pizza_orders * seasonality",
        detection_delay=14,
        decision_lag=7,
        outcome_threshold=0.3,
    )
    result = model.simulate(params={"pizza_orders": 10, "seasonality": 1.2})
"""

from __future__ import annotations

import warnings as _warnings

from dynafx.dynamics.dsl import AuxDef, FlowDef, StockDef, SysdModel


def _clip_delay(v):
    if isinstance(v, (int, float)):
        return max(0.1, float(v))
    return [max(0.1, float(d)) for d in v]


class SignalChain:
    """Factory that builds a SysdModel for a leading-indicator → outcome pattern.

    Returns a fully constructed SysdModel ready to simulate. No .sysd file needed.

    Usage::

        model = SignalChain.build(
            name="food_delivery",
            trace_expr="pizza_orders",
            detection_delay=14,
            decision_lag=7,
        )
        result = model.simulate()
    """

    @classmethod
    def build(
        cls,
        name: str,
        trace_expr: str,
        detection_delay: float | list[float] = 14.0,
        decision_lag: float = 7.0,
        outcome_threshold: float = 0.3,
        outcome_sensitivity: float = 1.0,
        threshold_direction: str = "above",
        core_stock_name: str = "Active_Users",
        core_stock_initial: float = 10000.0,
        base_inflow_rate: float = 15.0,
        base_outflow_rate: float = 0.003,
        has_feedback: bool = True,
        has_tracking: bool = True,
        aux_prefix: str = "sc",
    ) -> SysdModel:
        """Build a signal chain SysdModel.

        Args:
            name: Model name (e.g. 'food_delivery').
            trace_expr: Expression for the raw leading indicator signal.
            detection_delay: Time to detect the signal (DELAY3 parameter).
                Pass a list for multi-hop detection: [14, 7] chains two delays.
            decision_lag: Time to interpret the signal (SMOOTH parameter).
            outcome_threshold: IF threshold — impact activates when signal
                crosses this value.
            outcome_sensitivity: How strongly the impact affects the outflow.
            threshold_direction: "above" (signal > threshold triggers impact)
                or "below" (signal < threshold triggers impact).
            core_stock_name: Name of the main accumulating stock.
            core_stock_initial: Initial value of the main stock.
            base_inflow_rate: Constant rate that feeds the core stock.
            base_outflow_rate: Base rate at which the core stock drains.
            has_feedback: Whether to include a balancing feedback loop.
            has_tracking: Whether to include tracking stocks for observability.
            aux_prefix: Prefix for generated aux variable names.
        """
        model = SysdModel(name=name)

        # Normalize detection_delay to a list
        delays = [detection_delay] if isinstance(detection_delay, (int, float)) else list(detection_delay)
        delays = _clip_delay(delays)

        # ── 1. Trace signal ────────────────────────────────────
        # The raw observable that leaks early from the decision subsystem.
        # Includes a crisis pulse + seasonal + noise by default.
        model.aux_vars.append(AuxDef(name=f"{aux_prefix}_raw", expr=trace_expr))

        # ── 2. Detection pipeline ──────────────────────────────
        # Multi-hop DELAY3 chain: each hop is a detection stage.
        prev = f"{aux_prefix}_raw"
        for i, d in enumerate(delays):
            hop = f"{aux_prefix}_detected_{i}"
            model.aux_vars.append(AuxDef(name=hop, expr=f"DELAY3({prev}, {d})"))
            prev = hop
        detected_name = prev

        # ── 3. Interpretation smoothing ────────────────────────
        model.aux_vars.append(AuxDef(
            name=f"{aux_prefix}_interpreted",
            expr=f"SMOOTH({detected_name}, {decision_lag})",
        ))

        # ── 4. Impact (threshold-gated outcome) ────────────────
        # The impact modulates the outflow of the core stock.
        # When signal crosses threshold, the outflow changes proportionally.
        if threshold_direction == "below":
            cond = f"{aux_prefix}_interpreted < {outcome_threshold}"
            delta = f"({outcome_threshold} - {aux_prefix}_interpreted)"
        else:
            cond = f"{aux_prefix}_interpreted > {outcome_threshold}"
            delta = f"({aux_prefix}_interpreted - {outcome_threshold})"
        model.aux_vars.append(AuxDef(
            name="signal_impact",
            expr=(
                f"IF({cond}, "
                f"{delta} * {outcome_sensitivity}, 0)"
            ),
        ))

        # ── 5. Adjusted outflow ────────────────────────────────
        # Base outflow + signal impact. Higher impact = more outflow.
        model.aux_vars.append(AuxDef(
            name="adjusted_outflow",
            expr=f"base_outflow * (1 + signal_impact)",
        ))

        # ── 6. Core stock ──────────────────────────────────────
        # The stock accumulates from base inflow and drains via the
        # adjusted outflow.
        model.stocks.append(StockDef(
            name=core_stock_name,
            initial=core_stock_initial,
            flows=[
                FlowDef(name=f"+ {core_stock_name}_in", direction="+", expr="base_inflow"),
                FlowDef(name=f"- {core_stock_name}_out", direction="-",
                        expr=f"{core_stock_name} * adjusted_outflow"),
            ],
        ))

        # Add base_inflow and base_outflow as params (aux defaults)
        model.aux_vars.append(AuxDef(name="base_inflow", expr=str(base_inflow_rate)))
        model.aux_vars.append(AuxDef(name="base_outflow", expr=str(base_outflow_rate)))

        # ── 7. Tracking stocks ─────────────────────────────────
        if has_tracking:
            signal_suffixes = ["_raw", "_detected_0", "_interpreted"]
            display_names = ["Trace", "Detected", "Interpreted"]
            for suffix, display in zip(signal_suffixes, display_names):
                src = f"{aux_prefix}{suffix}"
                stock_name = f"signal_{display}"
                model.stocks.append(StockDef(
                    name=stock_name,
                    initial=0.0,
                    flows=[
                        FlowDef(name=f"+ {stock_name}_in", direction="+", expr=src),
                        FlowDef(name=f"- {stock_name}_out", direction="-", expr=stock_name),
                    ],
                ))

        # ── 8. Feedback loop (balancing) ───────────────────────
        if has_feedback:
            # When impact is high, feedback reduces the raw signal
            # (representing organizational response to the detected threat)
            model.aux_vars.append(AuxDef(
                name=f"{aux_prefix}_feedback",
                expr="IF(signal_impact > 0, -signal_impact * 0.3, 0)",
            ))
            # Modulate the raw signal with feedback
            model.aux_vars.append(AuxDef(
                name=f"{aux_prefix}_modulated",
                expr=f"MAX(0, {aux_prefix}_raw + {aux_prefix}_feedback)",
            ))
            # Re-point first detection stage to use modulated value
            first_hop = f"{aux_prefix}_detected_0"
            for i, a in enumerate(model.aux_vars):
                if a.name == first_hop:
                    model.aux_vars[i] = AuxDef(
                        name=first_hop,
                        expr=f"DELAY3({aux_prefix}_modulated, {delays[0]})",
                    )
                    break

        return model

    def __new__(cls, *args, **kwargs):
        _warnings.warn(
            "SignalChain(...) is deprecated, use SignalChain.build(...)",
            DeprecationWarning, stacklevel=2,
        )
        return cls.build(*args, **kwargs)
