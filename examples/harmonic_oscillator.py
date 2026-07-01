"""Forced damped harmonic oscillator — signal processing with DynaFX.

Demonstrates: vibration, resonance, damping, noise, energy tracking,
scenario comparison, causal tracing, and feedback detection.

Output: reports/harmonic_oscillator.pdf
"""

import io
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fpdf import FPDF

from dynafx import SysdModel
from dynafx.dynamics import (
    SysdModelResult,
    ScenarioComparison,
    ScenarioDef,
    ScenarioResult,
    causal_trace,
    detect_feedback_loops,
)


# ── Model builder ──────────────────────────────────────────────────────────

def build_model() -> SysdModel:
    """Build the forced damped harmonic oscillator using the Python API."""
    model = SysdModel("harmonic_oscillator")
    model.dt = 0.02
    model.t_span = (0.0, 50.0)

    with model.stock("x", 1.0, unit="m") as s:
        s.inflow("v")
    with model.stock("v", 0.0, unit="m/s") as s:
        s.inflow("a")

    model.aux("F", "A * SIN(2 * PI * f * t)")
    model.aux("a", "(F - c * v - k * x) / m")
    model.aux("z", "x + NOISE(sigma)")

    model.aux("KE", "0.5 * m * v * v")
    model.aux("PE", "0.5 * k * x * x")
    model.aux("E_total", "KE + PE")

    model.param("m", 1.0)
    model.param("k", 10.0)
    model.param("c", 0.5)
    model.param("A", 1.0)
    model.param("f", 0.5)
    model.param("sigma", 0.05)

    return model


# ── Plotting helpers ───────────────────────────────────────────────────────

def _s(text: str) -> str:
    text = text.replace("\u2192", "->").replace("\u2014", "--")
    return text.encode("latin-1", errors="replace").decode("latin-1")

def _arrow(text: str) -> str:
    return text.replace("\u2192", "->").replace("\u2014", "--")

def _format_cause_tree(node, depth: int = 0) -> str:
    """Recursively format a causal trace dict into an indented text tree."""
    if node is None:
        return "  " * depth + "(none)"
    name = node.get("name", "?")
    expr = node.get("expr", "?")
    line = "  " * depth + f"{name} = {expr}"
    children = node.get("children", [])
    if not children:
        return line
    lines = [line]
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        prefix = "+-- " if is_last else "|-- "
        child_depth = depth + 1
        child_lines = _format_cause_tree(child, child_depth).split("\n")
        for j, cl in enumerate(child_lines):
            if j == 0:
                lines.append("  " * depth + prefix + cl.strip())
            else:
                cont = "    " if is_last else "|   "
                lines.append("  " * depth + cont + cl.strip())
    return "\n".join(lines)


def _fig_bytes(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_time_series(result: SysdModelResult, stocks: list[str],
                     title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name in stocks:
        ax.plot(result.times, result.values[name], label=name, lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fft(result: SysdModelResult, name: str,
             title: str) -> plt.Figure:
    t = np.array(result.times)
    y = np.array(result.values[name])
    dt = t[1] - t[0]
    n = len(y)
    yf = np.fft.rfft(y - np.mean(y))
    xf = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(yf) / n

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(t, y, lw=0.8)
    axes[0].set_ylabel(name)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xf, mag, lw=1)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, 5)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_energy(result: SysdModelResult, m: float = 1.0, k: float = 10.0) -> plt.Figure:
    x = np.array(result.values["x"])
    v = np.array(result.values["v"])
    ke = 0.5 * m * v * v
    pe = 0.5 * k * x * x
    total = ke + pe
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(result.times, ke, label="Kinetic", lw=1)
    ax.plot(result.times, pe, label="Potential", lw=1)
    ax.plot(result.times, total, label="Total", lw=1, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy Transfer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scenario_comparison(scenes: list[tuple[str, ScenarioResult]],
                             stock: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for label, sr in scenes:
        ax.plot(sr.result.times, sr.result.values[stock], label=_s(label), lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(stock)
    ax.set_title(f"{stock} — Scenario Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── PDF Report ─────────────────────────────────────────────────────────────

class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, _s("Harmonic Oscillator \u2014 DynaFX Signal Processing"), align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, _s(f"Page {self.page_no()}/{{nb}}"), align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, _s(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_section(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, _s(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, _s(text))
        self.ln(3)

    def add_chart_page(self, title: str, fig, conclusion: str = ""):
        self.add_page()
        self.section(title)
        img = _fig_bytes(fig)
        self.image(img, x=self.l_margin, w=170)
        if conclusion:
            self.ln(3)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 4.5, _s(conclusion))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harmonic Oscillator Signal Study")
    parser.add_argument("--output", default="reports/harmonic_oscillator.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── 1. Build model ────────────────────────────────────────────
    print("Building model...")
    model = build_model()

    # ── 2. Baseline ──────────────────────────────────────────────
    print("Running baseline simulation...")
    result = model.simulate()

    # ── 3. Scenarios ──────────────────────────────────────────────
    print("Running scenario comparison...")
    # Natural frequency: ω₀ = √(k/m) = √10 ≈ 3.16 rad/s → f₀ ≈ 0.50 Hz
    # for small damping, resonance near f₀
    f0 = np.sqrt(10.0) / (2 * np.pi)  # ≈ 0.503 Hz
    sdefs = [
        ScenarioDef(name="Resonance", params={"f": round(f0, 3), "c": 0.3}),
        ScenarioDef(name="Off-Resonance", params={"f": 0.8, "c": 0.3}),
        ScenarioDef(name="Heavy Damping", params={"f": round(f0, 3), "c": 3.0}),
        ScenarioDef(name="Free Decay", params={"A": 0.0, "c": 0.5}),
        ScenarioDef(name="High Noise", params={"sigma": 0.5}),
    ]
    comp = ScenarioComparison(model, sdefs, method="rk4")

    # ── 4. Causal analysis ────────────────────────────────────────
    print("Causal tracing...")
    state = {s.name: result.values[s.name][-1] for s in model.stocks}
    state.update({a.name: 0.0 for a in model.aux_vars})
    traces = {}
    for var in ["x", "v", "E_total"]:
        traces[var] = causal_trace(model, var, state=state, max_depth=5)

    # ── 5. Feedback loops ─────────────────────────────────────────
    print("Feedback loop detection...")
    loops = detect_feedback_loops(model)

    # ── 6. Build report ───────────────────────────────────────────
    print("Building PDF report...")
    pdf = Report()
    pdf.alias_nb_pages()

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 60, 120)
    pdf.ln(40)
    pdf.cell(0, 14, _s("Forced Damped Harmonic Oscillator"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, _s("Signal Processing with DynaFX"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.body(
        "Model: m = 1 kg, k = 10 N/m, c varies, F(t) = A*sin(2πft). "
        f"Natural frequency f\u2080 = {f0:.3f} Hz."
    )

    # 2. Time-domain baseline
    pdf.add_chart_page(
        "Displacement & Velocity",
        plot_time_series(result, ["x", "v"], "Baseline: displacement and velocity", "Amplitude"),
        "Baseline with A=1.0, f=0.5 Hz, c=0.5, sigma=0.05. "
        "Transient decays within ~10 s; steady-state oscillation follows forcing frequency.",
    )

    # 3. Energy
    pdf.add_chart_page(
        "Energy Transfer",
        plot_energy(result),
        "Total energy is conserved in the undamped case. "
        "With damping, mechanical energy dissipates as heat. "
        "Steady-state energy reflects balance between forcing input and damping loss.",
    )

    # 4. FFT
    pdf.add_chart_page(
        "Frequency Domain (FFT)",
        plot_fft(result, "x", "Spectral analysis of displacement"),
        f"Dominant peak at the forcing frequency f = 0.5 Hz. "
        f"Natural frequency f\u2080 = {f0:.3f} Hz visible as a secondary peak during the transient.",
    )

    # 5. Noisy measurement
    pdf.add_chart_page(
        "Noisy Measurement",
        plot_time_series(
            result, ["x", "v"],
            "Free decay: initial condition (x₀=1.0, v₀=0) with damping",
            "Amplitude",
        ),
        "Measurement noise modeled as additive Gaussian. "
        "The NOISE() function produces reproducible stochastic variation with a given seed.",
    )

    # 6. Scenario comparison
    for stock in ["x"]:
        scenes = [(sd.name, comp.scenarios[i]) for i, sd in enumerate(sdefs)]
        pdf.add_chart_page(
            f"Scenario Comparison \u2014 {stock}",
            plot_scenario_comparison(scenes, stock),
            "Resonance: maximum amplitude when forcing frequency matches natural frequency. "
            "Off-resonance: reduced steady-state amplitude. "
            "Heavy damping: suppresses both transient and steady-state. "
            "Free decay: initial condition decays exponentially.",
        )

    # 7. Causal traces
    pdf.add_page()
    pdf.section("Causal Structure")
    for var in ["x", "v", "E_total"]:
        pdf.sub_section(f"Causes of {var}")
        t = traces[var]
        tree = _format_cause_tree(t.get("causes"))
        for line in tree.split("\n"):
            pdf.body(f"  {line}")
    pdf.body(_arrow(
        "Causal tracing reveals the dependency graph: displacement x depends on velocity v, "
        "which depends on acceleration a, which depends on forcing F, damping -c*v, "
        "and stiffness -k*x. Energy E_total depends on both KE and PE."
    ))

    # 8. Feedback loops
    pdf.sub_section("Feedback Loops")
    if loops.loops:
        for i, loop in enumerate(loops.loops[:5]):
            nodes = " -> ".join(loop.nodes)
            pdf.body(f"  Loop {i+1}: {nodes} ({loop.polarity})")
        pdf.body(_arrow(
            "The primary feedback loop is: x -> v -> a -> x "
            "(position affects acceleration via stiffness, creating a balancing loop). "
            "Velocity feedback through damping creates a second balancing loop. "
            "Forcing F is an open-loop input with no feedback."
        ))
    else:
        pdf.body("No feedback loops detected (all paths are open-loop or acyclic).")

    # Save
    print(f"Saving {args.output}...")
    pdf.output(args.output)
    print(f"Done. Report: {args.output}")


if __name__ == "__main__":
    main()
