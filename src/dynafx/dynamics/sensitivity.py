"""Variance-based sensitivity analysis for SysdModel parameters.

Provides:
  - SensitivityResult: structured output dataclass
  - SensitivityAnalyzer: class with Sobol, Morris, PRCC, OAT, SRC methods

Usage::

    from dynafx.dynamics import SensitivityAnalyzer, SensitivityResult

    model = SysdModel("my_model")
    model.aux("y", "A * x + B")
    with model.stock("x", 0): ...

    sa = SensitivityAnalyzer(model)
    result = sa.sobol(
        param_spec={"A": (0, 2), "B": (0, 5)},
        output="y", n_base=512,
    )
    print(result.ranking("total_order"))
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dynafx.dynamics.dsl import SysdModel, SysdModelResult

# ═══════════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensitivityResult:
    """Structured result of a sensitivity analysis run.

    Exactly one of the index dicts is populated depending on the method.
    """

    method: str
    param_names: list[str]
    output: str
    n_samples: int

    # Sobol indices
    first_order: dict[str, float] | None = None
    total_order: dict[str, float] | None = None
    first_order_ci: dict[str, tuple[float, float]] | None = None
    total_order_ci: dict[str, tuple[float, float]] | None = None

    # Morris screening
    mu: dict[str, float] | None = None
    mu_star: dict[str, float] | None = None
    sigma: dict[str, float] | None = None

    # PRCC
    prcc: dict[str, float] | None = None
    prcc_pvalue: dict[str, float] | None = None

    # OAT
    oat_low: dict[str, float] | None = None
    oat_high: dict[str, float] | None = None

    # SRC
    src: dict[str, float] | None = None

    # Metadata
    execution_time: float = 0.0
    converged: bool = True
    _raw_outputs: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.first_order is not None and not self.param_names:
            self.param_names = list(self.first_order.keys())

    def ranking(self, by: str = "first_order") -> list[tuple[str, float]]:
        """Return params sorted by sensitivity index, descending.

        Args:
            by: Which index to sort by — ``"first_order"``, ``"total_order"``,
                ``"mu_star"``, ``"prcc"``, or ``"src"``.

        Returns:
            List of ``(param_name, value)`` from most to least influential.
        """
        index = getattr(self, by, None)
        if index is None:
            msg = f"No index '{by}' available in this {self.method} result"
            raise ValueError(msg)
        return sorted(index.items(), key=lambda kv: -abs(kv[1]))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (JSON-friendly)."""
        d: dict[str, Any] = {
            "method": self.method,
            "param_names": list(self.param_names),
            "output": self.output,
            "n_samples": self.n_samples,
            "execution_time": self.execution_time,
            "converged": self.converged,
        }
        for attr in ("first_order", "total_order",
                      "mu", "mu_star", "sigma",
                      "prcc", "prcc_pvalue",
                      "oat_low", "oat_high",
                      "src"):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = dict(val)
        for attr in ("first_order_ci", "total_order_ci"):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = {k: list(v) for k, v in val.items()}
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class SensitivityAnalyzer:
    """Variance-based sensitivity analysis for ``SysdModel`` parameters.

    Example::

        model = SysdModel("example")
        model.aux("y", "A * t + B")
        ...
        sa = SensitivityAnalyzer(model, method="rk4")

        # Sobol first-order + total-order indices
        result = sa.sobol({"A": (0, 2), "B": (0, 5)}, output="y")
        print(result.ranking("total_order"))

        # Morris screening (cheaper, good for 10+ params)
        result = sa.morris({"A": (0, 2), "B": (0, 5), "C": (1, 3)}, output="y")

        # PRCC (partial rank correlation)
        result = sa.prcc({"A": (0, 2), "B": (0, 5)}, output="y")

        # OAT tornado-style
        result = sa.oat({"A": (0, 2), "B": (0, 5)}, output="y")
    """

    def __init__(self, model: SysdModel, method: str = "rk4") -> None:
        self._model = model
        self._method = method

    # ── Public methods ──────────────────────────────────────────────

    def sobol(
        self,
        param_spec: dict[str, tuple[float, float]],
        output: str,
        n_base: int = 512,
        seed: int = 42,
        t: float | None = None,
        n_bootstrap: int = 100,
        **sim_kwargs: Any,
    ) -> SensitivityResult:
        """Sobol first-order and total-order sensitivity indices.

        Uses Saltelli's sampling scheme (A, B, A_B matrices) via
        ``scipy.stats.qmc.Sobol`` low-discrepancy sequences.

        Args:
            param_spec: ``{name: (low, high)}`` for each parameter.
            output: Variable name (stock or aux) to measure.
            n_base: Base sample size (must be power of 2 for Sobol).
            seed: RNG seed.
            t: Time point to evaluate output (None = final value).
            n_bootstrap: Number of bootstrap resamples for CIs (0 = skip).

        Returns:
            SensitivityResult with ``first_order``, ``total_order``,
            and optional ``first_order_ci`` / ``total_order_ci``.
        """
        t0 = time.time()
        from scipy.stats.qmc import Sobol as SobolSeq

        param_names = list(param_spec.keys())
        k = len(param_names)
        n_base = _round_pow2(n_base)

        # Single Sobol sequence of dimension 2k, split into A and B
        sampler = SobolSeq(2 * k, seed=seed)
        samples = sampler.random_base2(m=int(math.log2(n_base)))
        A_raw = samples[:, :k]
        B_raw = samples[:, k:]
        A = _scale_samples(A_raw, param_names, param_spec)
        B = _scale_samples(B_raw, param_names, param_spec)

        # Build AB_i: A with column i replaced by B's column i
        AB: list[np.ndarray] = []
        for i in range(k):
            ab = A.copy()
            ab[:, i] = B[:, i]
            AB.append(ab)

        # Evaluate: A, B, all AB_i
        all_matrices = [A, B, *AB]
        y_blocks: list[np.ndarray] = []
        for mat in all_matrices:
            y_blocks.append(self._evaluate_batch(mat, param_names, output, t, **sim_kwargs))

        yA = y_blocks[0]
        yB = y_blocks[1]
        yAB = y_blocks[2:]            # f(AB_i) for each i

        # Saltelli estimator (Saltelli et al. 2010):
        #   S_i  = (E[yB * yAB_i]  - f0^2) / Var(Y)    — shares column i
        #   S_Ti = 1 - (E[yA * yAB_i] - f0^2) / Var(Y)  — shares columns -i
        f0 = float(np.mean(np.concatenate([yA, yB])))
        varY = float(np.var(np.concatenate([yA, yB]), ddof=0))

        first_order: dict[str, float] = {}
        total_order: dict[str, float] = {}

        for i, pname in enumerate(param_names):
            Si = (np.mean(yB * yAB[i]) - f0 ** 2) / varY
            STi = 1.0 - (np.mean(yA * yAB[i]) - f0 ** 2) / varY
            first_order[pname] = _clamp01(Si)
            total_order[pname] = _clamp01(STi)

        # Bootstrap confidence intervals
        first_order_ci: dict[str, tuple[float, float]] | None = None
        total_order_ci: dict[str, tuple[float, float]] | None = None
        if n_bootstrap > 0 and n_base >= 8:
            boot_Si: dict[str, list[float]] = {p: [] for p in param_names}
            boot_STi: dict[str, list[float]] = {p: [] for p in param_names}
            rng = np.random.default_rng(seed + 2)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n_base, size=n_base)
                yA_b = yA[idx]
                yB_b = yB[idx]
                yAB_b = [yAB[i][idx] for i in range(k)]
                f0_b = float(np.mean(np.concatenate([yA_b, yB_b])))
                varY_b = float(np.var(np.concatenate([yA_b, yB_b]), ddof=0))
                if varY_b < 1e-15:
                    continue
                for i, pname in enumerate(param_names):
                    Si_b = (np.mean(yB_b * yAB_b[i]) - f0_b ** 2) / varY_b
                    STi_b = 1.0 - (np.mean(yA_b * yAB_b[i]) - f0_b ** 2) / varY_b
                    boot_Si[pname].append(_clamp01(Si_b))
                    boot_STi[pname].append(_clamp01(STi_b))

            first_order_ci = {}
            total_order_ci = {}
            for pname in param_names:
                if len(boot_Si[pname]) >= 20:
                    s_arr = np.sort(boot_Si[pname])
                    first_order_ci[pname] = (float(s_arr[2]), float(s_arr[-3]))
                if len(boot_STi[pname]) >= 20:
                    s_arr = np.sort(boot_STi[pname])
                    total_order_ci[pname] = (float(s_arr[2]), float(s_arr[-3]))

        elapsed = time.time() - t0
        return SensitivityResult(
            method="sobol",
            param_names=param_names,
            output=output,
            n_samples=n_base * (2 * k + 2),
            first_order=first_order,
            total_order=total_order,
            first_order_ci=first_order_ci,
            total_order_ci=total_order_ci,
            execution_time=elapsed,
            _raw_outputs={"yA": yA, "yB": yB},
        )

    def morris(
        self,
        param_spec: dict[str, tuple[float, float]],
        output: str,
        n_trajectories: int = 20,
        n_levels: int = 4,
        seed: int = 42,
        t: float | None = None,
        **sim_kwargs: Any,
    ) -> SensitivityResult:
        """Morris screening (elementary effects).

        Each trajectory varies one parameter at a time across
        ``n_levels`` discretized levels. Returns ``mu_star``
        (mean absolute elementary effect) and ``sigma`` (std of
        effects — detects interactions / nonlinearity).

        Args:
            param_spec: ``{name: (low, high)}``.
            output: Variable name to measure.
            n_trajectories: Number of Morris trajectories.
            n_levels: Discretization levels per parameter.
            seed: RNG seed.

        Returns:
            SensitivityResult with ``mu``, ``mu_star``, ``sigma``.
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)

        param_names = list(param_spec.keys())
        k = len(param_names)
        delta = n_levels / (2.0 * (n_levels - 1))  # step size in [0,1] space

        ee: dict[str, list[float]] = {p: [] for p in param_names}

        for _ in range(n_trajectories):
            # Generate trajectory: random base point in [0,1]^k
            x_star = rng.uniform(0, 1 - delta, size=k)
            # Random permutation of which order to vary
            perm = rng.permutation(k)

            # Build trajectory points in [0,1] space
            traj = np.zeros((k + 1, k))
            traj[0] = x_star
            for j in range(k):
                traj[j + 1] = traj[j].copy()
                pi = perm[j]
                # Random direction: +delta or -delta (clamped to [0,1])
                if traj[j, pi] + delta <= 1.0:
                    traj[j + 1, pi] = traj[j, pi] + delta
                else:
                    traj[j + 1, pi] = traj[j, pi] - delta

            # Scale to parameter ranges and evaluate
            traj_scaled = _scale_samples(traj, param_names, param_spec)
            y_vals = self._evaluate_batch(traj_scaled, param_names, output, t, **sim_kwargs)

            # Elementary effects
            for j in range(k):
                pi = perm[j]
                dy = y_vals[j + 1] - y_vals[j]
                dx = (traj[j + 1, pi] - traj[j, pi]) * (param_spec[param_names[pi]][1]
                                                         - param_spec[param_names[pi]][0])
                if abs(dx) > 1e-15:
                    ee[param_names[pi]].append(dy / dx)

        mu: dict[str, float] = {}
        mu_star: dict[str, float] = {}
        sigma: dict[str, float] = {}
        for p in param_names:
            arr = np.array(ee[p])
            mu[p] = float(np.mean(arr)) if len(arr) > 0 else 0.0
            mu_star[p] = float(np.mean(np.abs(arr))) if len(arr) > 0 else 0.0
            sigma[p] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        elapsed = time.time() - t0
        return SensitivityResult(
            method="morris",
            param_names=param_names,
            output=output,
            n_samples=n_trajectories * (k + 1),
            mu=mu,
            mu_star=mu_star,
            sigma=sigma,
            execution_time=elapsed,
        )

    def prcc(
        self,
        param_spec: dict[str, tuple[float, float]],
        output: str,
        n_samples: int = 500,
        seed: int = 42,
        t: float | None = None,
        **sim_kwargs: Any,
    ) -> SensitivityResult:
        """Partial Rank Correlation Coefficient.

        Uses Latin Hypercube sampling (``scipy.stats.qmc.LatinHypercube``)
        for efficient space-filling. PRCC measures the monotonic
        relationship between each parameter and the output after
        removing the linear effects of all other parameters.

        Args:
            param_spec: ``{name: (low, high)}``.
            output: Variable name to measure.
            n_samples: Number of LHS samples.
            seed: RNG seed.

        Returns:
            SensitivityResult with ``prcc`` and ``prcc_pvalue``.
        """
        t0 = time.time()
        from scipy import stats as sp_stats

        param_names = list(param_spec.keys())

        X = _latin_hypercube(param_names, param_spec, n_samples, seed)
        y = self._evaluate_batch(X, param_names, output, t, **sim_kwargs)

        # Rank transform
        X_rank = _rank_transform(X)
        y_rank = _rank_transform(y.reshape(-1, 1))

        prcc_vals: dict[str, float] = {}
        pvals: dict[str, float] = {}
        n = n_samples

        for i, pname in enumerate(param_names):
            # Build regression model: X_i ~ X_{-i}
            Xi = X_rank[:, i]
            X_other = np.delete(X_rank, i, axis=1)
            # Add constant term for intercept
            X_design = np.column_stack([np.ones(n), X_other])
            try:
                beta = np.linalg.lstsq(X_design, Xi, rcond=None)[0]
                Xi_resid = Xi - X_design @ beta

                # Y ~ X_{-i}
                beta_y = np.linalg.lstsq(X_design, y_rank.flatten(), rcond=None)[0]
                y_resid = y_rank.flatten() - X_design @ beta_y

                # PRCC = correlation of residuals
                r_val, p_val = sp_stats.pearsonr(Xi_resid, y_resid)
                prcc_vals[pname] = float(r_val)
                pvals[pname] = float(p_val)
            except np.linalg.LinAlgError:
                prcc_vals[pname] = 0.0
                pvals[pname] = 1.0

        elapsed = time.time() - t0
        return SensitivityResult(
            method="prcc",
            param_names=param_names,
            output=output,
            n_samples=n_samples,
            prcc=prcc_vals,
            prcc_pvalue=pvals,
            execution_time=elapsed,
        )

    def oat(
        self,
        param_spec: dict[str, tuple[float, float]],
        output: str,
        t: float | None = None,
        **sim_kwargs: Any,
    ) -> SensitivityResult:
        """One-at-a-time sensitivity (tornado-style).

        Each parameter is varied from its low to high bound while
        all others are held at the midpoint.

        Args:
            param_spec: ``{name: (low, high)}``.
            output: Variable name to measure.
            t: Time point (None = final).

        Returns:
            SensitivityResult with ``oat_low`` and ``oat_high``.
        """
        t0 = time.time()
        param_names = list(param_spec.keys())

        # Midpoint params
        mid_params: dict[str, float] = {}
        for pname, (lo, hi) in param_spec.items():
            mid_params[pname] = (lo + hi) / 2.0

        oat_low: dict[str, float] = {}
        oat_high: dict[str, float] = {}

        for pname, (lo, hi) in param_spec.items():
            # Low
            plo = dict(mid_params)
            plo[pname] = lo
            ylo = self._evaluate_output(plo, output, t, **sim_kwargs)

            # High
            phi = dict(mid_params)
            phi[pname] = hi
            yhi = self._evaluate_output(phi, output, t, **sim_kwargs)

            oat_low[pname] = float(ylo)
            oat_high[pname] = float(yhi)

        elapsed = time.time() - t0
        return SensitivityResult(
            method="oat",
            param_names=param_names,
            output=output,
            n_samples=2 * len(param_names) + 1,
            oat_low=oat_low,
            oat_high=oat_high,
            execution_time=elapsed,
        )

    def src(
        self,
        param_spec: dict[str, tuple[float, float]],
        output: str,
        n_samples: int = 500,
        seed: int = 42,
        t: float | None = None,
        **sim_kwargs: Any,
    ) -> SensitivityResult:
        """Standardized Regression Coefficients.

        Fits a linear model ``Y = sum(beta_i * X_i)`` on normalized
        (zero-mean, unit-variance) data. ``beta_i`` is the SRC
        for parameter ``i``.

        Args:
            param_spec: ``{name: (low, high)}``.
            output: Variable name to measure.
            n_samples: Number of LHS samples.
            seed: RNG seed.

        Returns:
            SensitivityResult with ``src``.
        """
        t0 = time.time()
        param_names = list(param_spec.keys())

        X = _latin_hypercube(param_names, param_spec, n_samples, seed)
        y = self._evaluate_batch(X, param_names, output, t, **sim_kwargs)

        # Standardize
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        y_std = (y - y.mean()) / y.std(ddof=1)

        # OLS
        beta, *_ = np.linalg.lstsq(X_std, y_std, rcond=None)
        src_vals: dict[str, float] = {}
        for i, pname in enumerate(param_names):
            src_vals[pname] = float(beta[i])

        elapsed = time.time() - t0
        return SensitivityResult(
            method="src",
            param_names=param_names,
            output=output,
            n_samples=n_samples,
            src=src_vals,
            execution_time=elapsed,
        )

    # ── Plotting ───────────────────────────────────────────────────

    def plot_sobol(
        self,
        result: SensitivityResult,
        path: str = "",
        figsize: tuple[float, float] = (9, 5),
    ) -> Any | None:
        """Grouped bar chart: first-order vs total-order Sobol indices.

        Args:
            result: SensitivityResult from ``sobol()``.
            path: Save path (empty = return figure).
            figsize: Figure dimensions.

        Returns:
            Matplotlib figure if ``path`` is empty, else ``None``.
        """
        import matplotlib.pyplot as plt

        if result.method != "sobol":
            msg = f"Expected sobol result, got {result.method}"
            raise ValueError(msg)
        if result.first_order is None:
            msg = "No first-order indices in result"
            raise ValueError(msg)

        names = result.param_names
        x = np.arange(len(names))
        w = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        Si_vals = [result.first_order[n] for n in names]
        STi_vals = [result.total_order[n] if result.total_order else 0 for n in names]

        ax.bar(x - w / 2, Si_vals, w, label="First-order S\u1d62",
                        color="steelblue", alpha=0.85)

        # Add CI error bars
        if result.first_order_ci:
            err_lo = [Si_vals[i] - result.first_order_ci[n][0] for i, n in enumerate(names)]
            err_hi = [result.first_order_ci[n][1] - Si_vals[i] for i, n in enumerate(names)]
            ax.errorbar(x - w / 2, Si_vals,
                         yerr=[err_lo, err_hi],
                         fmt="none", capsize=3, color="navy")

        ax.bar(x + w / 2, STi_vals, w, label="Total-order S\u1d54\u1d62",
                        color="coral", alpha=0.85)

        if result.total_order_ci:
            err_lo = [STi_vals[i] - result.total_order_ci[n][0] for i, n in enumerate(names)]
            err_hi = [result.total_order_ci[n][1] - STi_vals[i] for i, n in enumerate(names)]
            ax.errorbar(x + w / 2, STi_vals,
                         yerr=[err_lo, err_hi],
                         fmt="none", capsize=3, color="darkred")

        ax.axhline(y=0, color="gray", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Sobol Index")
        ax.set_title(f"Sobol Sensitivity  —  {result.output}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_morris(
        self,
        result: SensitivityResult,
        path: str = "",
        figsize: tuple[float, float] = (7, 7),
    ) -> Any | None:
        r"""mu\ :sup:`*` -sigma scatter plot.

        Parameters with high ``mu_star`` are important.
        Those with high ``sigma`` relative to ``mu_star`` have
        interactions or nonlinear effects.

        Args:
            result: SensitivityResult from ``morris()``.
            path: Save path (empty = return figure).
            figsize: Figure dimensions.

        Returns:
            Matplotlib figure if ``path`` is empty, else ``None``.
        """
        import matplotlib.pyplot as plt

        if result.method != "morris":
            msg = f"Expected morris result, got {result.method}"
            raise ValueError(msg)
        if result.mu_star is None or result.sigma is None:
            msg = "No mu_star/sigma in result"
            raise ValueError(msg)

        names = result.param_names
        mu_vals = [result.mu_star[n] for n in names]
        sg_vals = [result.sigma[n] for n in names]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(mu_vals, sg_vals, s=80, c="steelblue", edgecolors="navy")
        for i, name in enumerate(names):
            ax.annotate(name, (mu_vals[i], sg_vals[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)
        diag_max = max(max(mu_vals), max(sg_vals)) * 1.1
        ax.plot([0, diag_max], [0, diag_max], "k--", lw=0.8, alpha=0.4,
                 label="\u03c3 = \u03bc* (linear)")

        ax.set_xlabel("\u03bc* (mean |EE|)")
        ax.set_ylabel("\u03c3 (std EE)")
        ax.set_title(f"Morris Screening  —  {result.output}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_tornado(
        self,
        result: SensitivityResult,
        path: str = "",
        figsize: tuple[float, float] = (8, 5),
    ) -> Any | None:
        """Horizontal tornado plot for OAT sensitivity.

        Args:
            result: SensitivityResult from ``oat()``.
            path: Save path (empty = return figure).
            figsize: Figure dimensions.

        Returns:
            Matplotlib figure if ``path`` is empty, else ``None``.
        """
        import matplotlib.pyplot as plt

        if result.method != "oat":
            msg = f"Expected oat result, got {result.method}"
            raise ValueError(msg)
        if result.oat_low is None or result.oat_high is None:
            msg = "No oat_low/oat_high in result"
            raise ValueError(msg)

        names = list(reversed(result.param_names))
        lo_vals = [result.oat_low[n] for n in names]
        hi_vals = [result.oat_high[n] for n in names]
        mid_vals = [(l + h) / 2.0 for l, h in zip(lo_vals, hi_vals, strict=False)]
        spread = [abs(h - l) for h, l in zip(hi_vals, lo_vals, strict=False)]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))
        colors = plt.cm.RdYlGn(np.array(spread) / max(spread) if max(spread) > 0 else 0.5)

        for i in range(len(names)):
            ax.barh(y_pos[i], hi_vals[i] - mid_vals[i], left=mid_vals[i],
                     height=0.6, color=colors[i], edgecolor="gray", alpha=0.85)
            ax.barh(y_pos[i], lo_vals[i] - mid_vals[i], left=mid_vals[i],
                     height=0.6, color=colors[i], edgecolor="gray", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel(result.output)
        ax.set_title(f"One-at-a-Time Sensitivity  —  {result.output}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return None
        return fig

    def plot_prcc_heatmap(
        self,
        results: SensitivityResult | list[SensitivityResult],
        path: str = "",
        figsize: tuple[float, float] = (7, 5),
    ) -> Any | None:
        """Heatmap for PRCC values across outputs.

        Args:
            results: Single SensitivityResult or list of results
                     for multiple outputs.
            path: Save path (empty = return figure).
            figsize: Figure dimensions.

        Returns:
            Matplotlib figure if ``path`` is empty, else ``None``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        results_list = [results] if isinstance(results, SensitivityResult) else results

        param_names = results_list[0].param_names
        output_names: list[str] = []
        data: list[list[float]] = []

        for res in results_list:
            if res.prcc is None:
                continue
            output_names.append(res.output)
            data.append([res.prcc[p] for p in param_names])

        if not data:
            msg = "No PRCC data to plot"
            raise ValueError(msg)

        fig, ax = plt.subplots(figsize=figsize)
        arr = np.array(data)
        vmax = max(abs(arr.min()), abs(arr.max()), 0.01)
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_yticks(range(len(output_names)))
        ax.set_yticklabels(output_names)
        ax.set_title("PRCC Sensitivity")

        # Annotate cells
        for i in range(len(output_names)):
            for j in range(len(param_names)):
                val = data[i][j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=8, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="PRCC")
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return None
        return fig

    # ── Internal helpers ───────────────────────────────────────────

    def _evaluate_output(
        self,
        params: dict[str, float],
        output: str,
        t: float | None = None,
        **sim_kwargs: Any,
    ) -> float:
        """Run a single simulation and return the output value."""
        result = self._model.simulate(method=self._method,
                                       params=params, **sim_kwargs)
        return _extract_value(result, output, t)

    def _evaluate_batch(
        self,
        X: np.ndarray,
        param_names: list[str],
        output: str,
        t: float | None = None,
        **sim_kwargs: Any,
    ) -> np.ndarray:
        """Evaluate the model for every row of X.

        X has shape (n_rows, len(param_names)).  Returns array of shape
        (n_rows,) with the output value for each row.
        """
        n = X.shape[0]
        y = np.empty(n)
        for i in range(n):
            params = dict(zip(param_names, X[i], strict=False))
            y[i] = self._evaluate_output(params, output, t, **sim_kwargs)
        return y


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_value(
    result: SysdModelResult,
    output: str,
    t: float | None = None,
) -> float:
    """Extract the value of ``output`` at time ``t`` (or final).

    Checks stocks (``result.values``) first, then auxes
    (``result.aux_values``).
    """
    if output in result.values:
        series = result.values[output]
    elif output in result.aux_values:
        series = result.aux_values[output]
    else:
        msg = f"Variable '{output}' not found in stocks or auxes"
        raise ValueError(msg)

    if t is None:
        return float(series[-1])

    # Interpolate to the requested time
    times = result.times
    if t <= times[0]:
        return float(series[0])
    if t >= times[-1]:
        return float(series[-1])
    idx = np.searchsorted(times, t, side="right") - 1
    if idx < 0:
        return float(series[0])
    if idx >= len(times) - 1:
        return float(series[-1])
    # Linear interpolation
    t0, t1 = times[idx], times[idx + 1]
    y0, y1 = series[idx], series[idx + 1]
    frac = (t - t0) / (t1 - t0)
    return float(y0 + frac * (y1 - y0))


def _scale_samples(
    X: np.ndarray,
    param_names: list[str],
    param_spec: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Scale a [0,1]^k matrix to parameter ranges.

    ``X`` has shape ``(n, k)`` with values in ``[0, 1]``.
    Returns array scaled to each parameter's ``(low, high)``.
    """
    X_scaled = np.empty_like(X)
    for i, pname in enumerate(param_names):
        lo, hi = param_spec[pname]
        X_scaled[:, i] = lo + X[:, i] * (hi - lo)
    return X_scaled


def _latin_hypercube(
    param_names: list[str],
    param_spec: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Generate LHS samples and scale to parameter ranges."""
    from scipy.stats.qmc import LatinHypercube

    k = len(param_names)
    sampler = LatinHypercube(k, seed=seed)
    samples = sampler.random(n=n_samples)
    return _scale_samples(samples, param_names, param_spec)


def _rank_transform(X: np.ndarray) -> np.ndarray:
    """Replace each column with its rank (mean for ties)."""
    from scipy.stats import rankdata

    ranks = np.empty_like(X)
    for i in range(X.shape[1]):
        ranks[:, i] = rankdata(X[:, i])
    return ranks


def _round_pow2(n: int) -> int:
    """Round up to the nearest power of 2."""
    return 1 << (n - 1).bit_length()


def _clamp01(x: float) -> float:
    """Clamp to ``[0, 1]``."""
    return max(0.0, min(1.0, x))
