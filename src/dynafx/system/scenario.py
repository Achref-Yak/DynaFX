"""Scenario comparison: compare runs, deviation charts, tornado diagrams."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from dynafx.system.dsl import SysdModel, SysdModelResult


@dataclass
class ScenarioDef:
    """Definition of a scenario to run."""
    name: str
    params: dict[str, Any]


@dataclass
class ScenarioResult:
    """A single scenario run result."""
    name: str
    result: SysdModelResult
    params: dict[str, Any]

    def __getitem__(self, stock: str) -> list[float]:
        return self.result.values[stock]


class ScenarioComparison:
    """Compare multiple simulation scenarios.

    Usage:
        model = SysdModel(...)
        comp = ScenarioComparison(model, [
            ScenarioDef("Baseline", {}),
            ScenarioDef("High demand", {"demand": 150}),
            ScenarioDef("Low capacity", {"capacity": 50}),
        ])
        comp.plot_comparison("comparison.png", ["Stock1", "Stock2"])
        comp.plot_deviation("deviation.png", ["Stock1"])
        comp.tornado("tornado.png", {"demand": (50, 200), "capacity": (30, 100)},
                     output_stock="Stock1")
        print(comp.summary())
    """

    def __init__(
        self,
        model: SysdModel,
        scenarios: list[ScenarioDef],
        method: str = "rk4",
    ):
        self.model = model
        self.method = method
        self.scenarios: list[ScenarioResult] = []
        for sd in scenarios:
            result = model.simulate(params=sd.params, method=method)
            self.scenarios.append(ScenarioResult(sd.name, result, sd.params))

    def get(self, name: str) -> Optional[ScenarioResult]:
        for s in self.scenarios:
            if s.name == name:
                return s
        return None

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.scenarios]

    @property
    def times(self) -> list[float]:
        return self.scenarios[0].result.times if self.scenarios else []

    def _get_mpl(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            return None

    # ── Comparison plot ───────────────────────────────────────────

    def plot_comparison(
        self,
        path: str,
        stocks: Optional[list[str]] = None,
        title: Optional[str] = None,
        return_fig: bool = False,
    ) -> None:
        """Overlay all scenarios for each specified stock.

        Args:
            path: Output path. Ignored when return_fig=True.
            stocks: Stock names to include (default: all).
            title: Optional plot title.
            return_fig: If True, return the Figure instead of saving.
        """
        plt = self._get_mpl()
        if plt is None:
            return
        if not self.scenarios:
            return
        stock_names = stocks or self.scenarios[0].result.stocks
        t = self.times
        n = len(stock_names)

        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(title or f"Scenario Comparison — {self.model.name}")

        for ax, stock in zip(axes, stock_names):
            for sc in self.scenarios:
                ax.plot(t, sc.result.values[stock], label=sc.name)
            ax.set_ylabel(stock)
            ax.legend()
            ax.grid(True)
        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        if return_fig:
            return fig
        fig.savefig(path)
        plt.close(fig)

    # ── Deviation plot ────────────────────────────────────────────

    def plot_deviation(
        self,
        path: str,
        stocks: Optional[list[str]] = None,
        baseline: int = 0,
        mode: str = "absolute",
        title: Optional[str] = None,
        return_fig: bool = False,
    ) -> None:
        """Plot deviation of each scenario from a baseline.

        Args:
            path: Output path. Ignored when return_fig=True.
            stocks: Stock names to include (default: all).
            baseline: Index of the baseline scenario (default: 0).
            mode: "absolute" or "relative" (fractional deviation).
            return_fig: If True, return the Figure instead of saving.
        """
        plt = self._get_mpl()
        if plt is None:
            return
        if not self.scenarios:
            return
        baseline_sc = self.scenarios[baseline]
        stock_names = stocks or self.scenarios[0].result.stocks
        t = self.times
        n = len(stock_names)

        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        title_text = title or f"Deviation from {baseline_sc.name} ({mode})"
        fig.suptitle(title_text)

        for ax, stock in zip(axes, stock_names):
            base_vals = baseline_sc.result.values[stock]
            for sc in self.scenarios:
                vals = sc.result.values[stock]
                if mode == "relative":
                    dev = [
                        (v - b) / b if abs(b) > 1e-12 else (v - b)
                        for v, b in zip(vals, base_vals)
                    ]
                else:
                    dev = [v - b for v, b in zip(vals, base_vals)]
                ax.plot(t, dev, label=sc.name)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax.set_ylabel(stock)
            ax.legend()
            ax.grid(True)
        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        if return_fig:
            return fig
        fig.savefig(path)
        plt.close(fig)

    # ── Tornado diagram ───────────────────────────────────────────

    def tornado(
        self,
        path: str,
        param_ranges: dict[str, tuple[float, float]],
        output_stock: str,
        t: Optional[float] = None,
        n_steps: int = 20,
        title: Optional[str] = None,
        return_fig: bool = False,
    ) -> None:
        """Generate a tornado diagram for parameter sensitivity.

        Each parameter is varied between its low and high bound
        while holding all other parameters at their baseline (midpoint).
        The output_stock value at time t is measured for each extreme.

        Args:
            path: Output path. Ignored when return_fig=True.
            param_ranges: param_name -> (low, high).
            output_stock: The stock whose value to measure.
            t: Time point to measure (default: final time).
            n_steps: Number of steps between low and high for the sweep.
            return_fig: If True, return the Figure instead of saving.
        """
        plt = self._get_mpl()
        if plt is None:
            return
        if not self.scenarios:
            return

        baseline_sc = self.scenarios[0]
        base_params = dict(baseline_sc.params)
        t_measure = t if t is not None else self.times[-1]

        impacts: list[tuple[str, float, float, float]] = []
        for pname, (low, high) in param_ranges.items():
            mid = (low + high) / 2.0
            params_low = dict(base_params)
            params_low[pname] = low
            params_high = dict(base_params)
            params_high[pname] = high

            result_low = self.model.simulate(params=params_low,
                                             method=self.method)
            result_high = self.model.simulate(params=params_high,
                                              method=self.method)

            val_low = self._interp_at(result_low, output_stock, t_measure)
            val_high = self._interp_at(result_high, output_stock, t_measure)
            spread = abs(val_high - val_low)
            impacts.append((pname, val_low, val_high, spread))

        impacts.sort(key=lambda x: x[3])

        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(impacts))))
        y_pos = list(range(len(impacts)))
        labels = [i[0] for i in impacts]
        low_vals = [i[1] for i in impacts]
        high_vals = [i[2] for i in impacts]
        mids = [(lv + hv) / 2.0 for lv, hv in zip(low_vals, high_vals)]

        bar_width = 0.4
        for i, (pn, lv, hv, sp) in enumerate(impacts):
            left = min(lv, hv)
            right = max(lv, hv)
            color = "steelblue" if (hv - lv) > 0 else "coral"
            ax.barh(i, right - left, left=left, height=bar_width,
                    color=color, edgecolor="black")
            ax.text(lv, i, f"{lv:.2f}", va="center", ha="right",
                    fontsize=8)
            ax.text(hv, i, f"{hv:.2f}", va="center", ha="left",
                    fontsize=8)

        ax.set_yticks(list(range(len(impacts))))
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"{output_stock} at t={t_measure}")
        ax.set_title(
            title or f"Tornado Diagram — {self.model.name}"
        )
        ax.axvline(mids[len(mids) // 2] if mids else 0,
                   color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, axis="x")
        fig.tight_layout()
        if return_fig:
            return fig, impacts
        fig.savefig(path)
        plt.close(fig)

    @staticmethod
    def _interp_at(result: SysdModelResult, stock: str,
                   t: float) -> float:
        vals = result.values[stock]
        times = result.times
        if t <= times[0]:
            return vals[0]
        if t >= times[-1]:
            return vals[-1]
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                frac = (t - times[i]) / (times[i + 1] - times[i])
                return vals[i] + frac * (vals[i + 1] - vals[i])
        return vals[-1]

    # ── Summary ────────────────────────────────────────────────

    def summary(self) -> dict[str, dict[str, float]]:
        """Return dict of scenario_name -> {stock_name: final_value}."""
        result: dict[str, dict[str, float]] = {}
        for sc in self.scenarios:
            final: dict[str, float] = {}
            for stock in sc.result.stocks:
                final[stock] = sc.result.values[stock][-1]
            result[sc.name] = final
        return result

    def summary_dataframe(self) -> Any:
        """Return a pandas DataFrame summary."""
        import pandas as pd
        rows: list[dict[str, Any]] = []
        for sc in self.scenarios:
            row: dict[str, Any] = {"scenario": sc.name}
            row.update(sc.params)
            for stock in sc.result.stocks:
                row[f"{stock} (final)"] = sc.result.values[stock][-1]
            rows.append(row)
        return pd.DataFrame(rows)

    def deviation_table(self, baseline: int = 0,
                        mode: str = "relative") -> dict[str, dict[str, float]]:
        """Return dict of scenario_name -> {stock_name: deviation}."""
        base = self.scenarios[baseline]
        result: dict[str, dict[str, float]] = {}
        base_final = {
            stock: base.result.values[stock][-1]
            for stock in base.result.stocks
        }
        for sc in self.scenarios:
            deviations: dict[str, float] = {}
            for stock in sc.result.stocks:
                val = sc.result.values[stock][-1]
                bval = base_final[stock]
                if mode == "relative" and abs(bval) > 1e-12:
                    deviations[stock] = (val - bval) / bval
                else:
                    deviations[stock] = val - bval
            result[sc.name] = deviations
        return result
