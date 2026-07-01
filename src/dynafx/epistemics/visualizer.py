"""Line-of-reasoning visualizer — exposes the inference black box.

Renders SL reasoning artifacts as matplotlib figures:
- Claim chain cards (one per query result binding)
- Evidence matrix heatmap (sources x claims)
- Argumentation attack graph (DAG with typed edges)
- Trust evolution (KBT scores over EM iterations)
- Consensus landscape (opinion bars per claim)

Each function returns a ``plt.Figure``. The ``recommend_pages()``
function selects which figures are worth including in a report.
"""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dynafx.core.models import Opinion
from dynafx.knowledge.model import NamedNode, Literal, Triple
from dynafx.knowledge.sparql import QueryResult
from dynafx.knowledge.store import TripleStore
from dynafx.epistemics.argumentation import (
    Attack,
    AttackType,
    ArgumentationFramework,
)
from dynafx.epistemics.evidence import (
    ConsensusLevel,
    EvidenceMatrixResult,
)
from dynafx.epistemics.kbt import KBTResult


# ── Helpers ────────────────────────────────────────────────────────

_COLORS = plt.cm.tab10.colors


def _shorten(iri: str, max_len: int = 28) -> str:
    if "#" in iri:
        name = iri.split("#")[-1]
    elif "/" in iri:
        name = iri.rstrip("/").split("/")[-1]
    else:
        name = iri
    if len(name) > max_len:
        name = name[: max_len - 3] + "..."
    return name


def _node_label(n: NamedNode | Literal) -> str:
    if hasattr(n, "iri"):
        return _shorten(str(n.iri))
    if hasattr(n, "value"):
        s = str(n.value)
        return s if len(s) < 24 else s[:21] + "..."
    return str(n)


def _fig_to_bytes(fig: plt.Figure) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _opinion_badge(op: Opinion) -> tuple[str, tuple]:
    """Return (label, color) for a fused opinion."""
    if op.belief >= 0.7 and op.uncertainty < 0.3:
        return "HIGH", (0.15, 0.55, 0.15)
    if op.belief >= 0.4 or op.uncertainty < 0.5:
        return "MEDIUM", (0.7, 0.55, 0.1)
    return "LOW", (0.7, 0.2, 0.15)


def _trust_color(t: float) -> tuple:
    if t >= 0.7:
        return (0.15, 0.55, 0.15)
    if t >= 0.4:
        return (0.7, 0.55, 0.1)
    return (0.7, 0.2, 0.15)


# ── 1. Claim Chain (single card) ──────────────────────────────────


def plot_claim_chain(
    claim_label: str,
    source_opinions: dict[str, Opinion],
    trust_scores: dict[str, float],
    attacks: list[tuple[str, str, str, float]],
    fused_opinion: Opinion,
    consensus: str = "medium",
) -> plt.Figure:
    """One compact card showing the evidence path for a single claim.

    Args:
        claim_label: Human-readable claim (e.g. ``revenue=5000000``).
        source_opinions: ``{source_name: Opinion}`` per source graph.
        trust_scores: ``{source_name: float}`` KBT trust scores.
        attacks: ``[(src, tgt, type_str, strength)]`` attack edges.
        fused_opinion: Fused Opinion across all sources.
        consensus: Consensus level string (``"high"``/``"medium"``/``"low"``).

    Returns:
        A ``plt.Figure`` (6×3.5 inches) ready for PDF embedding.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")
    y0 = 0.95
    row_h = 0.10
    col1_x = 0.05
    col2_x = 0.32
    col3_x = 0.55
    bar_w = 0.35

    grade, grade_color = _opinion_badge(fused_opinion)

    ax.text(col1_x, y0, claim_label, fontsize=10, fontweight="bold",
            va="top", transform=ax.transAxes)
    ax.text(0.95, y0, f"{grade} (b={fused_opinion.belief:.2f})",
            fontsize=8, fontweight="bold", color=grade_color,
            va="top", ha="right", transform=ax.transAxes)

    sources = sorted(source_opinions.keys())
    y = y0 - 0.08
    ax.text(col1_x, y, "Sources", fontsize=7, fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(col2_x, y, "Trust", fontsize=7, fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(col3_x, y, "Opinion", fontsize=7, fontweight="bold", va="top",
            transform=ax.transAxes)
    y -= row_h * 0.7

    for src in sources:
        op = source_opinions.get(src, Opinion())
        trust = trust_scores.get(src, 0.5)
        tc = _trust_color(trust)

        ax.text(col1_x, y, src, fontsize=7, va="center", transform=ax.transAxes)

        trust_w = trust * bar_w
        ax.barh(y, trust_w, height=0.03, left=col2_x, color=tc,
                edgecolor="none", transform=ax.transAxes)
        ax.text(col2_x + bar_w + 0.01, y, f"{trust:.2f}", fontsize=6,
                va="center", transform=ax.transAxes)

        parts = [("b", op.belief, (0.15, 0.55, 0.15)),
                 ("d", op.disbelief, (0.7, 0.2, 0.15)),
                 ("u", op.uncertainty, (0.6, 0.6, 0.6))]
        x = col3_x
        for label, val, color in parts:
            w = val * bar_w * 0.7
            if w > 0.005:
                ax.barh(y, w, height=0.03, left=x, color=color,
                        edgecolor="none", transform=ax.transAxes)
                x += w

        y -= row_h

    # Attacks section
    if attacks:
        y -= row_h * 0.5
        ax.text(col1_x, y, "Attacks:", fontsize=7, fontweight="bold",
                va="top", transform=ax.transAxes)
        y -= row_h * 0.55
        for a_src, a_tgt, a_typ, a_str in attacks:
            ae_color = {"rebut": (0.7, 0.2, 0.15),
                        "undermine": (0.8, 0.5, 0.1),
                        "undercut": (0.5, 0.3, 0.7)}.get(a_typ, (0.5, 0.5, 0.5))
            ax.text(col1_x + 0.02, y, f"{a_src}  {a_typ}  {a_tgt}",
                    fontsize=6, color=ae_color, va="top",
                    transform=ax.transAxes)
            y -= row_h * 0.55

    # Fused opinion bar at bottom
    y_fuse = 0.04
    ax.text(col1_x, y_fuse, "Fused:", fontsize=7, fontweight="bold",
            va="bottom", transform=ax.transAxes)
    for i, (label, val, color) in enumerate([
        ("b", fused_opinion.belief, (0.15, 0.55, 0.15)),
        ("d", fused_opinion.disbelief, (0.7, 0.2, 0.15)),
        ("u", fused_opinion.uncertainty, (0.6, 0.6, 0.6)),
    ]):
        w = val * 0.2
        if w > 0.005:
            x_pos = col2_x + sum(
                p[1] * 0.2 for p in [
                    ("b", fused_opinion.belief, ...),
                    ("d", fused_opinion.disbelief, ...),
                    ("u", fused_opinion.uncertainty, ...),
                ][:i]
            )
            # Recompute x_pos properly
            prev = 0.0
            for j in range(i):
                _, v_j, _ = [("b", fused_opinion.belief, (0.15, 0.55, 0.15)),
                             ("d", fused_opinion.disbelief, (0.7, 0.2, 0.15)),
                             ("u", fused_opinion.uncertainty, (0.6, 0.6, 0.6))][j]
                prev += v_j * 0.2
            x_pos = col2_x + prev
            ax.barh(y_fuse, w, height=0.035, left=x_pos, color=color,
                    edgecolor="none", transform=ax.transAxes)
    ax.text(col2_x + 0.22, y_fuse,
            f"b={fused_opinion.belief:.2f}  d={fused_opinion.disbelief:.2f}  u={fused_opinion.uncertainty:.2f}",
            fontsize=6, va="center", transform=ax.transAxes)

    fig.tight_layout(pad=0.3)
    return fig


# ── 2. Line of Reasoning (grid of cards) ──────────────────────────


def plot_line_of_reasoning(
    claims_data: list[dict[str, Any]],
) -> plt.Figure:
    """Grid of claim chain cards, one per query result binding.

    Args:
        claims_data: Each dict has keys ``claim_label``, ``source_opinions``,
            ``trust_scores``, ``attacks``, ``fused_opinion``, ``consensus``.

    Returns:
        A ``plt.Figure`` with one subplot per claim.
    """
    n = len(claims_data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No claims to display", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return fig

    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, cd in enumerate(claims_data):
        ax = axes[idx]
        ax.axis("off")
        _render_chain_on_ax(ax, cd)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Line-of-Reasoning Analysis", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=1.5)
    return fig


def _render_chain_on_ax(ax, cd: dict[str, Any]):
    """Render a single claim chain on a given Axes."""
    y0 = 0.92
    row_h = 0.10
    c1 = 0.02
    c2 = 0.30
    c3 = 0.52
    bw = 0.35
    clabel = cd.get("claim_label", "?")
    src_ops = cd.get("source_opinions", {})
    trust = cd.get("trust_scores", {})
    attacks_list = cd.get("attacks", [])
    fused = cd.get("fused_opinion", Opinion())

    grade, gc = _opinion_badge(fused)
    ax.text(c1, y0, clabel, fontsize=9, fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(0.97, y0, f"{grade} (b={fused.belief:.2f})", fontsize=7,
            fontweight="bold", color=gc, va="top", ha="right",
            transform=ax.transAxes)

    sources = sorted(src_ops.keys())
    y = y0 - 0.07
    ax.text(c1, y, "Source", fontsize=6, fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(c2, y, "Trust", fontsize=6, fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(c3, y, "b d u", fontsize=6, fontweight="bold", va="top",
            transform=ax.transAxes)
    y -= row_h * 0.6

    for src in sources:
        op = src_ops.get(src, Opinion())
        t = trust.get(src, 0.5)
        tc = _trust_color(t)
        ax.text(c1, y, src, fontsize=6, va="center", transform=ax.transAxes)
        ax.barh(y, t * bw, 0.025, left=c2, color=tc, edgecolor="none",
                transform=ax.transAxes)
        ax.text(c2 + bw + 0.01, y, f"{t:.2f}", fontsize=5, va="center",
                transform=ax.transAxes)

        x = c3
        for val, clr in [(op.belief, (0.15, 0.55, 0.15)),
                         (op.disbelief, (0.7, 0.2, 0.15)),
                         (op.uncertainty, (0.6, 0.6, 0.6))]:
            if val * bw * 0.7 > 0.005:
                ax.barh(y, val * bw * 0.7, 0.025, left=x, color=clr,
                        edgecolor="none", transform=ax.transAxes)
                x += val * bw * 0.7
        y -= row_h * 0.55

    if attacks_list:
        y -= row_h * 0.3
        ax.text(c1, y, "Attacks:", fontsize=6, fontweight="bold", va="top",
                transform=ax.transAxes)
        y -= row_h * 0.4
        for a_src, a_tgt, a_typ, _ in attacks_list[:3]:
            ec = {"rebut": (0.7, 0.2, 0.15), "undermine": (0.8, 0.5, 0.1),
                  "undercut": (0.5, 0.3, 0.7)}.get(a_typ, (0.5, 0.5, 0.5))
            ax.text(c1 + 0.02, y, f"{a_src} {a_typ} {a_tgt}",
                    fontsize=5, color=ec, va="top", transform=ax.transAxes)
            y -= row_h * 0.4

    yf = 0.03
    ax.text(c1, yf, "Fused:", fontsize=6, fontweight="bold", va="bottom",
            transform=ax.transAxes)
    x = c2
    for val, clr in [(fused.belief, (0.15, 0.55, 0.15)),
                     (fused.disbelief, (0.7, 0.2, 0.15)),
                     (fused.uncertainty, (0.6, 0.6, 0.6))]:
        w = val * 0.18
        if w > 0.005:
            ax.barh(yf, w, 0.03, left=x, color=clr, edgecolor="none",
                    transform=ax.transAxes)
            x += w
    ax.text(c2 + 0.2, yf,
            f"b={fused.belief:.2f} d={fused.disbelief:.2f} u={fused.uncertainty:.2f}",
            fontsize=5, va="center", transform=ax.transAxes)


# ── 3. Evidence Matrix Heatmap ────────────────────────────────────


def plot_evidence_matrix(result: EvidenceMatrixResult) -> plt.Figure:
    """Sources x claims heatmap colored by belief + pairwise agreement.

    Left panel: sources x claims (belief as fill; cross-hatch if contested).
    Right panel: pairwise agreement matrix (upper triangle).

    Args:
        result: An ``EvidenceMatrixResult`` from ``EvidenceMatrix.compute()``.

    Returns:
        A ``plt.Figure`` (10×5 inches).
    """
    sources = result.source_names
    claims = sorted(result.claim_names)
    ns, nc = len(sources), len(claims)
    if ns == 0 or nc == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(4, ns * 0.5 + 2)))

    # Matrix: sources x claims, value = belief
    matrix = np.zeros((ns, nc))
    masked = np.zeros((ns, nc), dtype=bool)
    for i, s in enumerate(sources):
        for j, c in enumerate(claims):
            ca = result.claims.get(c)
            if ca and s in ca.opinions:
                matrix[i, j] = ca.opinions[s].belief
            else:
                masked[i, j] = True

    im = ax1.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                    interpolation="nearest")
    # Cross-hatch missing
    if np.any(masked):
        for i in range(ns):
            for j in range(nc):
                if masked[i, j]:
                    ax1.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                        color="gray", alpha=0.3))

    ax1.set_xticks(range(nc))
    ax1.set_xticklabels(claims, fontsize=7, rotation=45, ha="right")
    ax1.set_yticks(range(ns))
    ax1.set_yticklabels(sources, fontsize=7)
    ax1.set_title("Belief per Source × Claim", fontsize=10)

    # Annotate contested claims
    contested = set(result.contested_claims())
    for j, c in enumerate(claims):
        if c in contested:
            ax1.text(j, ns + 0.3, "⚡", fontsize=8, ha="center",
                     va="bottom", transform=ax1.get_xaxis_transform())

    # Pairwise agreement matrix (upper triangle)
    n_pairs = ns * (ns - 1) // 2
    if n_pairs > 0:
        agree = np.ones((ns, ns))
        agree_mask = np.ones((ns, ns), dtype=bool)
        for i, s1 in enumerate(sources):
            for j, s2 in enumerate(sources):
                if i >= j:
                    agree_mask[i, j] = True
                    continue
                # Average agreement across all shared claims
                vals = []
                for c in claims:
                    ca = result.claims.get(c)
                    if ca:
                        for pa in ca.pairwise:
                            if ((pa.source_a == s1 and pa.source_b == s2) or
                                (pa.source_a == s2 and pa.source_b == s1)):
                                vals.append(pa.agreement)
                agree[i, j] = np.mean(vals) if vals else 0.5
                agree_mask[i, j] = False

        im2 = ax2.imshow(agree, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                         interpolation="nearest")
        # Mask lower triangle + diagonal
        for i in range(ns):
            for j in range(ns):
                if i >= j:
                    ax2.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                        color="gray", alpha=0.3))

        ax2.set_xticks(range(ns))
        ax2.set_xticklabels(sources, fontsize=7, rotation=45, ha="right")
        ax2.set_yticks(range(ns))
        ax2.set_yticklabels(sources, fontsize=7)
        ax2.set_title("Pairwise Agreement", fontsize=10)

        # Annotate agreement values
        for i in range(ns):
            for j in range(i + 1, ns):
                if not agree_mask[i, j]:
                    ax2.text(j, i, f"{agree[i, j]:.2f}", ha="center",
                             va="center", fontsize=6, color="black",
                             fontweight="bold")

    fig.colorbar(im, ax=ax1, shrink=0.6)
    if n_pairs > 0:
        fig.colorbar(im2, ax=ax2, shrink=0.6)
    fig.tight_layout()
    return fig


# ── 4. Attack Graph ──────────────────────────────────────────────


def plot_attack_graph(
    framework: ArgumentationFramework,
    extension: set[str],
    max_nodes: int = 40,
) -> plt.Figure:
    """Attack DAG with colored edges by type, grounded extension highlighted.

    Args:
        framework: An ``ArgumentationFramework`` with arguments and attacks.
        extension: Set of accepted argument IDs (from ``compute_grounded()``).
        max_nodes: Max arguments to show (to keep layout readable).

    Returns:
        A ``plt.Figure`` (10×7 inches).
    """
    args = framework.arguments
    attacks = framework.attacks

    if not args:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No arguments to display", ha="center",
                va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Subsample if too many nodes
    arg_items = list(args.items())
    if len(arg_items) > max_nodes:
        accepted = {k for k, v in arg_items if k in extension}
        rejected = [k for k, v in arg_items if k not in extension]
        # Keep all accepted, trim rejected
        keep_n = max_nodes - len(accepted)
        rejected = rejected[:keep_n]
        keep_ids = accepted | set(rejected)
        arg_items = [(k, v) for k, v in arg_items if k in keep_ids]
        attacks = [a for a in attacks if a.source_id in keep_ids and a.target_id in keep_ids]

    G = nx.DiGraph()
    for aid, arg in arg_items:
        label_parts = [_shorten(str(arg.triple.predicate.iri))]
        if hasattr(arg.triple.object_, "value"):
            label_parts.append(str(arg.triple.object_.value)[:12])
        else:
            label_parts.append(_shorten(str(arg.triple.object_.iri)))
        label = "=".join(label_parts)
        in_ext = aid in extension
        G.add_node(aid, label=label, accepted=in_ext,
                   source=str(arg.source_graph or "?")[:8])

    edge_color_map = {AttackType.REBUT: "#c0392b",
                      AttackType.UNDERMINE: "#e67e22",
                      AttackType.UNDERCUT: "#8e44ad"}
    edge_colors = []
    for a in attacks:
        G.add_edge(a.source_id, a.target_id, atype=a.attack_type.value)
        edge_colors.append(edge_color_map.get(a.attack_type, "#7f8c8d"))

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    accepted_nodes = [n for n, d in G.nodes(data=True) if d.get("accepted")]
    rejected_nodes = [n for n, d in G.nodes(data=True) if not d.get("accepted")]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           arrows=True, arrowsize=12, width=1.5,
                           alpha=0.7, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=accepted_nodes,
                           node_color="#2ecc71", node_size=400,
                           edgecolors="#27ae60", linewidths=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=rejected_nodes,
                           node_color="#e74c3c", node_size=300,
                           edgecolors="#c0392b", linewidths=1, alpha=0.5)

    labels = {n: d["label"][:16] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)

    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Accepted (grounded)"),
        mpatches.Patch(color="#e74c3c", label="Defeated", alpha=0.5),
        mpatches.Patch(color="#c0392b", label="Rebut"),
        mpatches.Patch(color="#e67e22", label="Undermine"),
        mpatches.Patch(color="#8e44ad", label="Undercut"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax.set_title(
        f"Argumentation Framework — {len(extension)}/{len(args)} accepted",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── 5. Trust Evolution ───────────────────────────────────────────


def plot_trust_evolution(kbt: KBTResult, max_sources: int = 10) -> plt.Figure:
    """Per-source trust scores over EM iterations + final bar chart.

    Args:
        kbt: A ``KBTResult`` from ``compute_kbt()``.
        max_sources: Max sources to include.

    Returns:
        A ``plt.Figure`` (8×4 inches).
    """
    if not kbt.source_trust:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No trust data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    sources = sorted(kbt.source_trust.keys())[:max_sources]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Left: trust evolution
    for i, src in enumerate(sources):
        hist = kbt.trust_history.get(src, [0.5])
        ax1.plot(range(len(hist)), hist, color=_COLORS[i % len(_COLORS)],
                 label=src, marker=".", markersize=4, linewidth=1.2)
    ax1.set_xlabel("EM Iteration")
    ax1.set_ylabel("Trust Score")
    ax1.set_title("Trust Evolution")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    if kbt.converged:
        ax1.axvline(x=kbt.iterations - 1, color="green", linestyle="--",
                    alpha=0.5, label=f"converged ({kbt.iterations} iter)")
    else:
        ax1.text(0.95, 0.05, f"max iter ({kbt.iterations})", fontsize=8,
                 transform=ax1.transAxes, ha="right", va="bottom",
                 color="orange")

    # Right: final trust bar chart
    final_trusts = [kbt.source_trust.get(src, 0.5) for src in sources]
    bars = ax2.barh(range(len(sources)), final_trusts, color=[
        _trust_color(t) for t in final_trusts
    ])
    ax2.set_yticks(range(len(sources)))
    ax2.set_yticklabels(sources, fontsize=7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Final Trust")
    ax2.set_title("Source Reliability")
    for i, (bar, t) in enumerate(zip(bars, final_trusts)):
        ax2.text(t + 0.02, i, f"{t:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    return fig


# ── 6. Consensus Landscape ────────────────────────────────────────


def plot_consensus_landscape(result: EvidenceMatrixResult) -> plt.Figure:
    """Per-claim belief/disbelief/uncertainty bars across sources.

    Each claim gets a group of bars (one per source) with the
    fused opinion overlaid as a stacked bar.

    Args:
        result: An ``EvidenceMatrixResult`` from ``EvidenceMatrix.compute()``.

        Returns:
        A ``plt.Figure`` (10×4 inches).
    """
    claims = sorted(result.claim_names)
    sources = result.source_names
    if not claims or not sources:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(10, max(3, len(claims) * 0.6 + 2)))

    for i, c in enumerate(claims):
        ca = result.claims.get(c)
        if not ca:
            continue
        y_base = i * (len(sources) + 2)
        for j, src in enumerate(sources):
            op = ca.opinions.get(src)
            if op is None:
                continue
            y = y_base + j
            ax.barh(y, op.belief, 0.7, left=0, color=(0.15, 0.55, 0.15),
                    edgecolor="none")
            ax.barh(y, op.disbelief, 0.7, left=op.belief,
                    color=(0.7, 0.2, 0.15), edgecolor="none")
            ax.barh(y, op.uncertainty, 0.7, left=op.belief + op.disbelief,
                    color=(0.6, 0.6, 0.6), edgecolor="none")

        # Fused opinion
        y_fuse = y_base + len(sources)
        fused = ca.fused
        ax.barh(y_fuse, fused.belief, 0.7, left=0, color=(0.0, 0.4, 0.0),
                edgecolor="black", linewidth=1.5)
        ax.barh(y_fuse, fused.disbelief, 0.7, left=fused.belief,
                color=(0.5, 0.0, 0.0), edgecolor="black", linewidth=1.5)
        ax.barh(y_fuse, fused.uncertainty, 0.7,
                left=fused.belief + fused.disbelief,
                color=(0.4, 0.4, 0.4), edgecolor="black", linewidth=1.5)

        ax.text(-0.02, y_base + len(sources) / 2, c, fontsize=7,
                va="center", ha="right", transform=ax.get_yaxis_transform())

    # Y-axis labels
    y_ticks = []
    y_labels = []
    for i, c in enumerate(claims):
        y_base = i * (len(sources) + 2)
        for j, src in enumerate(sources):
            y_ticks.append(y_base + j)
            y_labels.append(src)
        y_ticks.append(y_base + len(sources))
        y_labels.append("FUSED")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Opinion Mass")
    ax.set_title("Consensus Landscape — per Claim per Source", fontsize=10)

    legend_elements = [
        mpatches.Patch(color=(0.15, 0.55, 0.15), label="Belief"),
        mpatches.Patch(color=(0.7, 0.2, 0.15), label="Disbelief"),
        mpatches.Patch(color=(0.6, 0.6, 0.6), label="Uncertainty"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ── 7. Adaptive Page Recommender ──────────────────────────────────


def recommend_pages(
    query_result: QueryResult,
    kbt: KBTResult,
    framework: ArgumentationFramework,
    evidence_result: EvidenceMatrixResult,
    claims_data: list[dict[str, Any]],
) -> list[tuple[str, Callable[[], plt.Figure]]]:
    """Decide which visualizations are worth including in a report.

    Returns a list of ``(title, render_fn)`` tuples. The first entry
    is always the line-of-reasoning grid. Subsequent entries are
    included only when there's something interesting to show.

    Args:
        query_result: SPARQL ``QueryResult``.
        kbt: ``KBTResult`` from ``compute_kbt()``.
        framework: ``ArgumentationFramework``.
        evidence_result: ``EvidenceMatrixResult``.
        claims_data: Data for ``plot_line_of_reasoning()``.

    Returns:
        List of ``(title, zero-arg-callable->Figure)``.
    """
    pages: list[tuple[str, Callable[[], plt.Figure]]] = []

    # Always show the line-of-reasoning grid
    pages.append((
        "Line of Reasoning",
        lambda cd=claims_data: plot_line_of_reasoning(cd),
    ))

    # Trust evolution — only if trust < 0.8 for any source
    if kbt.source_trust and any(
        t < 0.8 for t in kbt.source_trust.values()
    ):
        pages.append((
            "Source Trust Evolution",
            lambda k=kbt: plot_trust_evolution(k),
        ))

    # Attack graph — only if there are actual attacks
    if framework.attacks:
        ext = framework.compute_grounded()
        pages.append((
            "Argumentation Conflicts",
            lambda fw=framework, ex=ext: plot_attack_graph(fw, ex),
        ))

    # Evidence matrix — only if there are contested claims
    contested = evidence_result.contested_claims()
    if contested:
        pages.append((
            "Evidence Matrix",
            lambda er=evidence_result: plot_evidence_matrix(er),
        ))

    # Consensus landscape — only if 2+ sources and 2+ claims
    if (evidence_result.source_count >= 2 and
            evidence_result.claim_count >= 2):
        pages.append((
            "Consensus Landscape",
            lambda er=evidence_result: plot_consensus_landscape(er),
        ))

    return pages
