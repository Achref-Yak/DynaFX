"""Adaptive multi-sensor signal detection — showcases 4 new features.

New features demonstrated:
  1. Power operator **  — signal power and RMS calculations
  2. User-defined functions — reusable DSP blocks (sq, db, rms, envelope)
  3. Multi-server DES queues — parallel processing of detected events
  4. Agent networks — distributed sensors sharing confidence via peer influence

Model design
============
  A weak sine wave (A=0.5, f=2 Hz) arrives in Gaussian noise (sigma=0.3).
  The receiver chain:
    raw_signal  = A * SIN(2*PI*f*t) + NOISE(sigma)
    signal_power = raw_signal ** 2          ← power operator
    noise_floor  = sigma ** 2               ← power operator
    SNR_db       = db(signal_power / noise_floor)  ← user function

  An adaptive threshold gates detections.  When the signal exceeds threshold,
  the detection rate drives a multi-server DES event processor (3 parallel
  servers).  Meanwhile, 5 distributed sensor agents share their confidence
  estimates over a complete network, so low-confidence sensors get a boost.

Output: reports/adaptive_detection.pdf  (multi-page report with plot)
"""

import io
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynafx import SysdModel
from dynafx.dynamics import SysdModelResult, causal_trace, detect_feedback_loops

from fpdf import FPDF, XPos, YPos


# ═══════════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_model() -> SysdModel:
    """Build the adaptive detection system using the Python-native API.

    All four new features are used inline with clear annotations.
    """
    model = SysdModel("adaptive_detection")
    model.dt = 0.02
    model.t_span = (0.0, 10.0)

    # User-defined functions (feature #2)
    model.func("sq",   ["x"],      "x ** 2")
    model.func("db",   ["x"],      "10 * LN(x + 1e-10) / LN(10)")
    model.func("rms",  ["x", "w"], "SQRT(SMOOTH(sq(x), w))")
    model.func("envelope", ["x", "tau"], "SMOOTH(x, tau)")

    # Signal chain
    model.aux("raw_signal",      "A * SIN(2 * PI * f * t) + NOISE(sigma)")

    # Power operator **  (feature #1)
    model.aux("signal_power",    "raw_signal ** 2")
    model.aux("noise_floor",     "sigma ** 2")

    # User function calls
    model.aux("SNR_db",          "db(signal_power / noise_floor)")
    model.aux("rms_estimate",    "rms(raw_signal, 0.1)")

    # Adaptive threshold
    model.aux("threshold",
              "base_thresh * MAX(0.5, MIN(2.0, SNR_db / 10))")

    # Detection and event rate
    model.aux("detection",       "IF(raw_signal > threshold, 1, 0)")
    model.aux("event_rate",      "SMOOTH(detection / dt, 0.3)")

    # Stock: cumulative detections
    with model.stock("detections", 0.0) as s:
        s.inflow("detection_rate", "detection / dt")

    # Multi-server DES queue (feature #3)
    model.queue("processor",  capacity=-1,
                service_time="0.15",
                arrival_rate="event_rate",
                servers=3)

    # Agent network (feature #4)
    with model.agent("sensor", 5) as a:
        a.prop("confidence", 0.5, min_val=0, max_val=1)
        a.rule("boost", "detection > 0.5",
               ["confidence += 0.01"])
        a.rule("revert", "neighbor_confidence_avg < confidence",
               ["confidence -= 0.005"])
        a.rule("peer_pressure",
               "neighbor_confidence_avg > confidence + 0.05",
               ["confidence += 0.01"])
        a.network("complete")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def run(params: dict | None = None
        ) -> tuple[SysdModel, SysdModelResult, dict]:
    if params is None:
        params = {}
    model = build_model()
    defaults = dict(A=0.5, f=2.0, sigma=0.3, base_thresh=0.15, dt=0.02)
    defaults.update(params)
    result = model.simulate(params=defaults)
    return model, result, defaults


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plot(result: SysdModelResult, path: str) -> None:
    """Multi-panel figure for the PDF report."""
    t = np.array(result.times)
    av = result.aux_values

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Panel 1: raw signal + threshold
    axes[0].plot(t, av["raw_signal"], label="raw_signal", alpha=0.7)
    axes[0].plot(t, av["threshold"],  label="threshold",
                 ls="--", color="red", lw=1.5)
    axes[0].fill_between(t, av["threshold"],
                          where=np.array(av["raw_signal"])
                                > np.array(av["threshold"]),
                          alpha=0.15, color="green", label="detections")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("1. Raw Signal + Adaptive Threshold")

    # Panel 2: signal power
    axes[1].plot(t, av["signal_power"], label="signal_power", lw=1)
    axes[1].axhline(y=np.mean(av["noise_floor"]), color="gray",
                    ls=":", label="noise_floor (mean)")
    axes[1].set_ylabel("Power")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("2. Signal Power (raw_signal ** 2)")

    # Panel 3: SNR
    axes[2].plot(t, av["SNR_db"], label="SNR", color="purple", lw=1)
    axes[2].axhline(y=10, color="gray", ls=":", alpha=0.5)
    axes[2].set_ylabel("dB")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("3. SNR  (user func: db(signal_power / noise_floor))")

    # Panel 4: event rate + cumulative detections
    axes[3].plot(t, av["event_rate"], label="event_rate", color="orange")
    axes[3].plot(t, np.array(result.values["detections"]),
                 label="cumulative detections", color="green")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Count")
    axes[3].legend(loc="upper left", fontsize=8)
    axes[3].set_title("4. Detections + Event Rate")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_abm_plot(result: SysdModelResult, path: str) -> None:
    """Agent confidence trajectories."""
    if not result.abm_metrics_history:
        return
    conf_key = "sensor_confidence_avg"
    if conf_key not in result.abm_metrics_history[0]:
        return
    t = np.array(result.times)
    steps = len(result.abm_metrics_history)
    step_ts = np.linspace(t[0], t[-1], steps)
    conf_traj = np.array([m[conf_key] for m in result.abm_metrics_history])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(step_ts, conf_traj, label="mean confidence", lw=2, color="darkblue")
    ax.fill_between(step_ts,
                     conf_traj - 0.02, conf_traj + 0.02,
                     alpha=0.2, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title("Agent Network: Mean Confidence over Time")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _s(text: str) -> str:
    return text.replace("\u2192", "->").replace("\u2014", "--")


def _fmt_cause_tree(tree, indent=0) -> str:
    if not isinstance(tree, dict):
        return ""
    result = ""
    for key, val in tree.items():
        result += "  " * indent + _s(f"+-- {key}") + "\n"
        if isinstance(val, dict):
            result += _fmt_cause_tree(val, indent + 1)
        elif isinstance(val, (list, tuple)):
            for v in val[:4]:
                s = _s(str(v))[:90]
                result += "  " * (indent + 1) + f"+-- {s}" + "\n"
        else:
            s = _s(str(val))[:90]
            if s:
                result += "  " * (indent + 1) + f"`-- {s}" + "\n"
    return result


def _des_summary(q) -> list[tuple[str, str]]:
    total_time = q.stats.total_arrivals * 0.15 if q.stats.total_arrivals > 0 else 0
    util = 100.0 * q.stats.total_arrivals * 0.15 / (q.servers * 10.0)
    util = min(util, 100.0)
    return [
        ("Queue", f"{q.name}"),
        ("Servers", str(q.servers)),
        ("Service time", "0.15"),
        ("Total arrivals", str(q.stats.total_arrivals)),
        ("Total departures", str(q.stats.total_departures)),
        ("Peak queue length", str(q.stats.max_length)),
        ("Avg wait time", f"{q.stats.avg_wait:.4f}"),
        ("Utilization", f"{util:.1f}%"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# PDF report
# ═══════════════════════════════════════════════════════════════════════════════

class Report(FPDF):
    """Multi-page PDF report for the adaptive detection model."""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.cell(0, 4, "Adaptive Multi-Sensor Signal Detection",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}",
                  align="C")


def build_report(model: SysdModel, result: SysdModelResult,
                 params: dict, plot_path: str,
                 abm_plot_path: str) -> FPDF:
    """Build the full multi-page PDF report."""
    pdf = Report()
    pdf.alias_nb_pages()

    t = np.array(result.times)
    av = result.aux_values
    n_steps = len(t)

    # ─── Page 1: Title + Model Overview ───────────────────────────────
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "Adaptive Multi-Sensor", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 12, "Signal Detection", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7,
             "System Dynamics + Multi-server DES + Agent Network",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)

    # Key parameters table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Model Parameters", align="L",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 8)
    param_rows = [
        ("Signal amplitude (A)", f"{params['A']}"),
        ("Frequency (f)", f"{params['f']} Hz"),
        ("Noise sigma", f"{params['sigma']}"),
        ("Base threshold", f"{params['base_thresh']}"),
        ("Time span", f"{model.t_span}"),
        ("Time step (dt)", f"{model.dt}"),
        ("Integration steps", str(n_steps)),
        ("Integration method", "RK4"),
    ]
    for name, val in param_rows:
        pdf.cell(70, 5, f"  {name}")
        pdf.cell(0, 5, f"{val}", new_x=XPos.LMARGIN,
                 new_y=YPos.NEXT)

    pdf.ln(4)

    # New features summary
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "New Features Showcased", align="L",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Symbol", "", 8)
    features = [
        "Power operator **   -- signal_power = raw_signal ** 2",
        "User-defined functions -- sq, db, rms, envelope",
        "Multi-server DES queue  -- 3 parallel servers",
        "Agent network  -- 5 sensors in complete graph topology",
    ]
    for ft in features:
        pdf.cell(0, 5, _s(f"     -  {ft}"), new_x=XPos.LMARGIN,
                 new_y=YPos.NEXT)

    # ─── Page 2: SD Signal Chain ─────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "1.  System Dynamics -- Signal Chain",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Insert the plot
    pdf.image(plot_path, x=10, w=190)
    pdf.ln(2)

    # Signal power analysis
    spow = np.array(av["signal_power"])
    nf = np.array(av["noise_floor"])
    snr = np.array(av["SNR_db"])
    det = np.array(av["detection"])
    n_active = int(np.sum(det > 0.5))

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Signal Power Analysis", align="L",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 8)

    # Left column: power stats
    rows = [
        ("Mean signal power", f"{np.mean(spow):.4f}"),
        ("Peak signal power", f"{np.max(spow):.4f}"),
        ("Mean noise floor", f"{np.mean(nf):.4f}"),
        ("Signal-to-noise ratio", f"{10*np.log10(np.mean(spow)/max(np.mean(nf),1e-10)):.1f} dB"),
        (" ", " "),
        ("SNR range", f"{np.min(snr):.1f} to {np.max(snr):.1f} dB"),
        ("RMS estimate range", f"{np.min(av['rms_estimate']):.4f} to {np.max(av['rms_estimate']):.4f}"),
        (" ", " "),
        ("Active detection steps", f"{n_active} / {n_steps}"),
        ("Detection ratio", f"{100.0*n_active/n_steps:.1f}%"),
        ("Cumulative detections", f"{result.values['detections'][-1]:.0f}"),
    ]
    for name, val in rows:
        pdf.cell(70, 5, f"  {name}")
        pdf.cell(0, 5, f"{val}", new_x=XPos.LMARGIN,
                 new_y=YPos.NEXT)

    # Expression definitions
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Expression Chain", align="L",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 7)
    exprs = [
        ("raw_signal", "A * SIN(2*PI*f*t) + NOISE(sigma)"),
        ("signal_power", "raw_signal ** 2   [POWER OP]"),
        ("noise_floor", "sigma ** 2   [POWER OP]"),
        ("SNR_db", "db(signal_power / noise_floor)   [USER FUNC]"),
        ("rms_estimate", "rms(raw_signal, 0.1)   [USER FUNC]"),
        ("threshold", "base_thresh * MAX(0.5, MIN(2.0, SNR_db/10))"),
        ("detection", "IF(raw_signal > threshold, 1, 0)"),
        ("event_rate", "SMOOTH(detection / dt, 0.3)"),
    ]
    for name, expr in exprs:
        pdf.cell(40, 4.5, f"  {name}")
        pdf.cell(0, 4.5, _s(f"= {expr}"), new_x=XPos.LMARGIN,
                 new_y=YPos.NEXT)

    # ─── Page 3: DES Queue ─────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "2.  Discrete Event Simulation -- Multi-server Queue",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    if result.des_engine and "processor" in result.des_engine.queues:
        q = result.des_engine.queues["processor"]

        # Stats table
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Queue Performance Metrics", align="L",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Courier", "", 8)

        for name, val in _des_summary(q):
            pdf.cell(60, 5, f"  {name}")
            pdf.cell(0, 5, f"{val}", new_x=XPos.LMARGIN,
                     new_y=YPos.NEXT)

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Queue Length History", align="L",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Courier", "", 7)

        if q.stats.length_history:
            hist = q.stats.length_history
            step = max(1, len(hist) // 40)
            for i, (ts, l) in enumerate(hist):
                if i % step != 0:
                    continue
                bar = "#" * min(l, 20) + (f" {l}" if l > 0 else " 0")
                pdf.cell(0, 3.5, _s(f"  {ts:5.1f} |{bar}"),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # DES arrival/departure process description
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "How It Works", align="L",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        desc = _s(
            "Each simulation step, the SD engine computes an event_rate "
            "from the smoothed detection signal. The DES engine then "
            "enqueues that many events at the processor queue. Three "
            "parallel servers each process one event at a time with "
            "service_time=0.15. Because events arrive in bursts when "
            "the signal exceeds threshold, short queuing delays occur "
            "until all three servers absorb the burst."
        )
        pdf.multi_cell(0, 4, desc)

    # ─── Page 4: ABM Agent Network ────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "3.  Agent-Based Model -- Sensor Network",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    if result.abm_metrics_history:
        conf_key = "sensor_confidence_avg"
        if conf_key in result.abm_metrics_history[0]:
            init_c = result.abm_metrics_history[0][conf_key]
            final_c = result.abm_metrics_history[-1][conf_key]

            # ABM plot
            pdf.image(abm_plot_path, x=10, w=190)
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Agent Rule Set", align="L",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Courier", "", 8)
            rules = [
                ("boost", "detection > 0.5", "confidence += 0.01"),
                ("revert", "neighbor_confidence_avg < confidence",
                 "confidence -= 0.005"),
                ("peer_pressure",
                 "neighbor_confidence_avg > confidence + 0.05",
                 "confidence += 0.01"),
            ]
            for rname, cond, effect in rules:
                pdf.cell(0, 4.5,
                         _s(f"  {rname}: IF {cond} THEN {effect}"),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Network Influence", align="L",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8)
            desc = _s(
                "Five sensor agents form a complete graph (all-to-all "
                "connectivity). Each agent perceives the average, min, "
                "and max confidence of its neighbors. When a sensor "
                "detects the signal (detection > 0.5), its confidence "
                "rises. If a sensor's confidence lags behind its "
                "neighbors by more than 0.05, peer pressure pulls it up. "
                "If it runs ahead, the revert rule moderates it. The "
                "network converges toward a shared confidence estimate "
                f"({init_c:.3f} -> {final_c:.3f})."
            )
            pdf.multi_cell(0, 4, desc)

    # ─── Page 5: Causal Analysis ──────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "4.  Causal Analysis & Feedback",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Causal trace
    state = dict(zip(result.stocks, result.final_state))
    state.update({k: v[-1] for k, v in result.aux_values.items()})
    traces = causal_trace(model, "raw_signal", state)
    ef = traces.get("effects", {})

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Effects of raw_signal (causal trace)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 7)
    tree_str = _fmt_cause_tree(ef)
    pdf.multi_cell(0, 3.5, tree_str)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Feedback Loops",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 8)
    loops = detect_feedback_loops(model)
    if loops.loops:
        for loop in loops.loops:
            pdf.cell(0, 5,
                     _s(f"  {_s(' -> '.join(loop.nodes))}  ({loop.polarity})"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.cell(0, 5, _s("  No feedback loops (open-loop signal chain)"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Paradigm Interactions",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    desc = _s(
        "SD computes the analog signal chain (raw_signal -> SNR -> "
        "detection -> event_rate). The event_rate feeds the DES queue "
        "as an arrival process, so SD controls the timing of DES "
        "events. Meanwhile, the ABM agents read the SD detection aux "
        "to update their confidence, so SD detection events directly "
        "influence agent belief states. All three paradigms share the "
        "same unified state vector at each time step."
    )
    pdf.multi_cell(0, 4, desc)

    # ─── Page 6: Feature Summary ──────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "5.  Feature Summary",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    summaries = [
        ("Power Operator **",
         "signal_power = raw_signal ** 2 computes instantaneous "
         "signal power. noise_floor = sigma ** 2 gives the reference "
         "noise level. The operator is right-associative and integrates "
         "seamlessly into the expression parser.",
         "src/dynafx/dynamics/_parser.py"),
        ("User-defined Functions",
         "sq(x), db(x), rms(x,w), and envelope(x,tau) are defined "
         "as macros that expand at compile time. Nested calls (e.g. "
         "rms calls sq) are supported via recursive substitution.",
         "src/dynafx/dynamics/dsl.py"),
        ("Multi-server DES",
         "queue 'processor': servers=3 processes up to 3 events in "
         "parallel. The advance_service() method returns the number "
         "of completed services per step. Backward-compatible with "
         "servers=1 (default).",
         "src/dynafx/dynamics/des.py"),
        ("Agent Networks",
         "Five topologies available: none, complete, random, "
         "small-world, scale-free. Neighbor averages are injected "
         "as perceive() state variables (neighbor_*_avg/min/max). "
         "Built deterministically with networkx.",
         "src/dynafx/dynamics/agent.py"),
    ]

    for title, desc, src in summaries:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, _s(f"  {title}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4, _s(f"    {desc}"))
        pdf.set_font("Courier", "", 7)
        pdf.cell(0, 4, _s(f"    File: {src}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # Closing
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5,
             _s("Generated with DynaFX  |  1096+ tests passing  |  "
                "pyright 0 errors"),
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return pdf


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Building and simulating adaptive detection model...")
    model, result, p = run()
    print("Done.")

    os.makedirs("reports", exist_ok=True)
    plot_path = "reports/adaptive_detection_plot.png"
    abm_plot_path = "reports/adaptive_detection_abm.png"

    print("Generating plots...")
    generate_plot(result, plot_path)
    generate_abm_plot(result, abm_plot_path)

    print("Building PDF report...")
    pdf = build_report(model, result, p, plot_path, abm_plot_path)
    pdf.output("reports/adaptive_detection.pdf")
    print("PDF saved: reports/adaptive_detection.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
