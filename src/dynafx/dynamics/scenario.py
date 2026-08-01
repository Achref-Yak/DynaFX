"""Scenario comparison: compare runs, deviation charts, tornado diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynafx.dynamics.dsl import SysdModel, SysdModelResult


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
        **sim_kwargs: Any,
    ):
        self.model = model
        self.method = method
        self.scenarios: list[ScenarioResult] = []
        for sd in scenarios:
            result = model.simulate(params=sd.params, method=method, **sim_kwargs)
            self.scenarios.append(ScenarioResult(sd.name, result, sd.params))

    def get(self, name: str) -> ScenarioResult | None:
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
        stocks: list[str] | None = None,
        title: str | None = None,
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

        for ax, stock in zip(axes, stock_names, strict=False):
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
        stocks: list[str] | None = None,
        baseline: int = 0,
        mode: str = "absolute",
        title: str | None = None,
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

        for ax, stock in zip(axes, stock_names, strict=False):
            base_vals = baseline_sc.result.values[stock]
            for sc in self.scenarios:
                vals = sc.result.values[stock]
                if mode == "relative":
                    dev = [
                        (v - b) / b if abs(b) > 1e-12 else (v - b)
                        for v, b in zip(vals, base_vals, strict=False)
                    ]
                else:
                    dev = [v - b for v, b in zip(vals, base_vals, strict=False)]
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
        t: float | None = None,
        n_steps: int = 20,
        title: str | None = None,
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
        labels = [i[0] for i in impacts]
        low_vals = [i[1] for i in impacts]
        high_vals = [i[2] for i in impacts]
        mids = [(lv + hv) / 2.0 for lv, hv in zip(low_vals, high_vals, strict=False)]

        bar_width = 0.4
        for i, (lv, hv) in enumerate(zip(low_vals, high_vals, strict=False)):
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

    # ── Grading per scenario ───────────────────────────────────────

    def grade_scenarios(
        self,
        grade_specs: list[tuple[str, str, float, float]],
        store: Any,
        evidence_map: list | None = None,
        bridge: Any | None = None,
        temp_graph: str = "_ranking",
    ) -> dict[str, dict[str, float]]:
        """Grade each scenario result and return per-goal breakdown.

        Asserts each scenario's evidence into a temp graph, runs
        ``grade_queries()``, then cleans up.

        Args:
            grade_specs: List of (query_str, var_name, threshold, penalty)
                tuples — same format as :meth:`ReasoningPass.grade`.
            store: TripleStore to query and temporarily write into.
            evidence_map: Optional (stock, subj, pred, fn) list for
                :meth:`KBSimBridge.evidence_from_result`. If provided,
                ``bridge`` must also be provided.
            bridge: A :class:`KBSimBridge` instance (required if
                ``evidence_map`` is given).
            temp_graph: Named graph for temporary evidence (removed after
                grading).

        Returns:
            ``{scenario_name: {grade_name: score}}`` — per-query scores
            for every scenario.
        """
        from dynafx.bridge import grade_queries

        result: dict[str, dict[str, float]] = {}
        for sr in self.scenarios:
            if evidence_map is not None and bridge is not None:
                triples = bridge.evidence_from_result(
                    sr.result, evidence_map, graph=temp_graph,
                )
                for t in triples:
                    store.add(t, graph=temp_graph)
            else:
                triples = []

            grades = grade_queries(grade_specs, store, prefix=f"{sr.name}_")

            if evidence_map is not None and bridge is not None:
                from dynafx.knowledge.model import TriplePattern
                for t in triples:
                    store.remove(
                        TriplePattern(t.subject, t.predicate, t.object_),
                        graph=temp_graph,
                    )

            result[sr.name] = grades

        return result

    # ── Ranking ─────────────────────────────────────────────────

    def rank(
        self,
        grade_specs: list[tuple[str, str, float, float]],
        store: Any,
        evidence_map: list | None = None,
        bridge: Any | None = None,
        agg: str = "mean",
        temp_graph: str = "_ranking",
    ) -> list[tuple[str, float]]:
        """Grade each scenario result and return ranked by aggregate score.

        Delegates to :meth:`grade_scenarios` for per-scenario grading,
        then aggregates and sorts.

        Args:
            grade_specs: List of (query_str, var_name, threshold, penalty)
                tuples — same format as :meth:`ReasoningPass.grade`.
            store: TripleStore to query and temporarily write into.
            evidence_map: Optional (stock, subj, pred, fn) list for
                :meth:`KBSimBridge.evidence_from_result`. If provided,
                ``bridge`` must also be provided.
            bridge: A :class:`KBSimBridge` instance (required if
                ``evidence_map`` is given).
            agg: Aggregation across multiple grade values per scenario.
                One of ``"mean"``, ``"min"``, ``"max"``, ``"sum"``.
            temp_graph: Named graph for temporary evidence (removed after
                grading).

        Returns:
            List of ``(scenario_name, score)`` sorted descending by score.
        """
        all_grades = self.grade_scenarios(
            grade_specs, store, evidence_map=evidence_map,
            bridge=bridge, temp_graph=temp_graph,
        )
        scored: list[tuple[str, float]] = []
        for name, grades in all_grades.items():
            vals = list(grades.values())
            if agg == "min":
                score = min(vals) if vals else 0.0
            elif agg == "max":
                score = max(vals) if vals else 0.0
            elif agg == "sum":
                score = sum(vals)
            else:
                score = sum(vals) / len(vals) if vals else 0.0
            scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Constraint filter ─────────────────────────────────────────

    def filter(
        self,
        store: Any,
        constraint_queries: list[str],
        evidence_map: list | None = None,
        bridge: Any | None = None,
        temp_graph: str = "_filter_temp",
    ) -> ScenarioComparison:
        """Remove scenarios that violate SPARQL ASK constraints.

        For each scenario, asserts evidence into a temp graph, runs
        each ASK query. If any returns ``cardinality == 0`` the scenario
        is removed from ``self.scenarios``.

        Args:
            store: TripleStore to query and temporarily write into.
            constraint_queries: SPARQL ``ASK`` query strings. A scenario
                passes only when every ASK returns ``True``.
            evidence_map: Optional (stock, subj, pred, fn) list for
                :meth:`KBSimBridge.evidence_from_result`. If provided,
                ``bridge`` must also be provided.
            bridge: A :class:`KBSimBridge` instance (required if
                ``evidence_map`` is given).
            temp_graph: Named graph for temporary evidence (removed after
                filtering).

        Returns:
            ``self`` for chaining.
        """
        from dynafx.knowledge import parse_sparql, sparql_evaluate

        surviving: list[Any] = []
        for sr in self.scenarios:
            if evidence_map is not None and bridge is not None:
                triples = bridge.evidence_from_result(
                    sr.result, evidence_map, graph=temp_graph,
                )
                for t in triples:
                    store.add(t, graph=temp_graph)
            else:
                triples = []

            passed = True
            for q_str in constraint_queries:
                try:
                    ast = parse_sparql(q_str)
                    qr = sparql_evaluate(ast, store)
                    if qr.cardinality == 0:
                        passed = False
                        break
                except Exception:
                    passed = False
                    break

            if evidence_map is not None and bridge is not None:
                from dynafx.knowledge.model import TriplePattern
                for t in triples:
                    store.remove(
                        TriplePattern(t.subject, t.predicate, t.object_),
                        graph=temp_graph,
                    )

            if passed:
                surviving.append(sr)

        self.scenarios = surviving
        return self

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

    # ── Explanation ───────────────────────────────────────────────

    def explain_scenario(
        self,
        name: str,
        store: Any,
        evidence_map: list[tuple[str, Any, Any, Any]],
        bridge: Any,
        grade_specs: list[tuple[str, str, float, float]] | None = None,
        grades: dict[str, dict[str, float]] | None = None,
        ranked: list[tuple[str, float]] | None = None,
        agg: str = "mean",
    ) -> dict[str, Any]:
        """Return structured explanation for one scenario.

        The output dict has keys:

        - ``name`` — scenario name
        - ``rank`` — rank position (1-indexed, 0 = no rank)
        - ``total_score`` — aggregate score
        - ``num_scenarios`` — how many survived filtering
        - ``params`` — the scenario's param dict
        - ``goals`` — list of per-goal explanations, each with ``label``,
          ``score``, ``causal_chain`` (variable list), ``kb_facts``
        - ``kb_facts`` — global KB facts sharing the evidence prefix

        Args:
            name: Scenario name to explain.
            store: TripleStore with the knowledge base.
            evidence_map: List of ``(stock_name, subject, predicate, scoring_fn)``
                tuples — same as :meth:`grade_scenarios`. The predicate IRI
                namespace is used to scope KB facts.
            bridge: A :class:`KBSimBridge` instance.
            grade_specs: Optional — if omitted, grades / ranked must be
                provided.
            grades: Pre-computed grades dict (from :meth:`grade_scenarios`).
            ranked: Pre-computed ranked list (from :meth:`rank`).
            agg: Aggregation for the total score (default ``"mean"``).

        Returns:
            Explanation dict (see above).
        """
        sr = self.get(name)
        if sr is None:
            raise ValueError(f"Scenario {name!r} not found in comparison")

        # Resolve rank / score
        rank = 0
        total_score = 0.0
        if ranked is not None:
            for i, (rn, rs) in enumerate(ranked, 1):
                if rn == name:
                    rank, total_score = i, rs
                    break
        if rank == 0 and grade_specs is not None:
            if grades is None:
                grades = self.grade_scenarios(
                    grade_specs, store, evidence_map=evidence_map,
                    bridge=bridge,
                )
            g = grades.get(name, {})
            vals = list(g.values())
            total_score = sum(vals) / len(vals) if vals else 0.0

        # Infer the base namespace from the first evidence predicate
        kb_ns = ""
        if evidence_map:
            first_pred = evidence_map[0][2]
            if hasattr(first_pred, "iri"):
                uri = first_pred.iri
                if "#" in uri:
                    kb_ns = uri[:uri.rindex("#") + 1]
                elif "/" in uri:
                    kb_ns = uri[:uri.rindex("/") + 1]

        # Collect KB facts (triples sharing the evidence namespace)
        kb_facts_list: list[str] = []
        if store and kb_ns:
            all_store_triples = list(store.all_triples())
            for t in all_store_triples[:50]:
                subj_iri = getattr(t.subject, "iri", "")
                pred_iri = getattr(t.predicate, "iri", "")
                obj_iri = getattr(t.object_, "iri", "")
                if any(kb_ns in x for x in [subj_iri, pred_iri, obj_iri]):
                    local_subj = subj_iri.replace(kb_ns, "")
                    local_pred = pred_iri.replace(kb_ns, "")
                    if hasattr(t.object_, "value"):
                        obj_display = str(t.object_.value)
                    else:
                        obj_display = obj_iri.replace(kb_ns, "")
                    kb_facts_list.append(f"{local_subj} → {local_pred}: {obj_display}")

        # Build dependency graph once (used by all goals)
        from dynafx.dynamics.causal import _get_dependencies
        deps = _get_dependencies(self.model)

        # grade_specs are positionally aligned with evidence_map:
        #   grade_specs[i] → evidence_map[i] → grade key f"{name}_{i}"
        goals_list: list[dict[str, Any]] = []
        for idx, (stock_name, _subj, pred, _scoring_fn) in enumerate(evidence_map):
            label = getattr(pred, "iri", str(pred))
            if kb_ns:
                label = label.replace(kb_ns, "")

            # Get score from positional grade key
            score = 0.0
            if grades and name in grades:
                grade_key = f"{name}_{idx}"
                score = grades[name].get(grade_key, 0.0)

            # Causal chain — walk the dependency graph directly.
            # Intermediate variables (those referenced by something else) first,
            # then leaf params.
            chain: list[str] = [stock_name]
            try:
                seen: set[str] = {stock_name}

                def _walk_refs(n: str, _seen=seen, depth: int = 0, max_depth: int = 8) -> list[str]:
                    if depth >= max_depth:
                        return []
                    _, refs = deps.get(n, ("", set()))
                    result: list[str] = []
                    for ref in sorted(refs):
                        if ref not in _seen:
                            _seen.add(ref)
                            result.append(ref)
                            result.extend(_walk_refs(ref, _seen, depth + 1, max_depth))
                    return result

                raw = _walk_refs(stock_name)
                nonleaves: list[str] = []
                leaf_names: list[str] = []
                for dep_name in raw:
                    _, refs = deps.get(dep_name, ("", set()))
                    if refs:
                        nonleaves.append(dep_name)
                    else:
                        leaf_names.append(dep_name)
                chain.extend(nonleaves)
                chain.extend(leaf_names)
            except Exception:
                chain = [stock_name]

            goals_list.append({
                "label": label,
                "score": round(score, 4),
                "causal_chain": chain,
            })

        return {
            "name": name,
            "rank": rank,
            "total_score": round(total_score, 4),
            "num_scenarios": len(self.scenarios),
            "params": dict(sr.params),
            "goals": goals_list,
            "kb_facts": kb_facts_list,
        }

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
