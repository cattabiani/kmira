"""Figure: the total arm-to-arm gap, and its split into "freedom" and "reach".

freedom = learn7 - control      (weights allowed to move, over mira's own 7 blocks)
reach   = learned_mix - learn7  (the other 17, mostly shallower, blocks made available)
total   = learned_mix - control (Experiment 1's headline)

Left panel runs to wherever control and learned_mix are both scored; right panel only over steps
where ALL THREE arms are scored, which is what a decomposition requires.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from lib import (
    COLOR,
    INK_MUTED,
    INK_SOFT,
    SURFACE,
    apply_style,
    load_rows,
    matched_steps,
    save,
    series,
    spines,
    thousands,
)


def build():
    rows = load_rows()
    apply_style()
    fig, (ax_gap, ax_dec) = plt.subplots(1, 2, figsize=(10.6, 4.2))

    # ---- left: the total gap, over its full measured range ------------------------------------
    pair_steps = matched_steps(rows, ("control", "learned_mix"))
    ctrl, lm = dict(series(rows, "control")), dict(series(rows, "learned_mix"))
    gaps = [lm[s] - ctrl[s] for s in pair_steps]

    ax_gap.axhline(0, color=INK_MUTED, linewidth=0.8, alpha=0.55)
    ax_gap.plot(pair_steps, gaps, color=COLOR["learned_mix"], marker="o", markersize=3.2)
    ax_gap.annotate(
        f"  +{gaps[-1]:.3f} dB",
        xy=(pair_steps[-1], gaps[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=COLOR["learned_mix"],
        fontweight="bold",
        va="center",
    )
    widened = sum(1 for i in range(1, len(gaps)) if gaps[i] > gaps[i - 1])
    ax_gap.set_title("learned_mix − control, at matched steps", loc="left")
    ax_gap.set_xlabel("training step")
    ax_gap.set_ylabel("PSNR gap (dB)  ↑ bigger effect")
    ax_gap.set_xlim(0, pair_steps[-1] * 1.16)
    ax_gap.annotate(
        f"widens at {widened} of {len(gaps) - 1} step-to-step transitions",
        xy=(0.03, 0.955),
        xycoords="axes fraction",
        fontsize=8,
        color=INK_SOFT,
        va="top",
    )
    thousands(ax_gap)
    spines(ax_gap)

    # ---- right: freedom / reach, only where all three arms exist -------------------------------
    tri = matched_steps(rows, ("control", "learn7", "learned_mix"))
    l7 = dict(series(rows, "learn7"))
    freedom = [l7[s] - ctrl[s] for s in tri]
    reach = [lm[s] - l7[s] for s in tri]

    # Stacked, because freedom + reach IS the total by construction -- a part-to-whole over time.
    # Drawn as two fills with a surface-coloured line between them rather than a border, so the
    # boundary reads as a gap instead of an outline.
    ax_dec.fill_between(tri, 0, freedom, color=COLOR["learn7"], alpha=0.85, linewidth=0)
    ax_dec.fill_between(
        tri,
        freedom,
        [f + r for f, r in zip(freedom, reach, strict=True)],
        color=COLOR["learned_mix"],
        alpha=0.85,
        linewidth=0,
    )
    ax_dec.plot(tri, freedom, color=SURFACE, linewidth=2)

    mid_f = freedom[len(tri) // 2] / 2
    ax_dec.annotate(
        "freedom\n(learn7 − control)",
        xy=(tri[len(tri) // 2], mid_f),
        fontsize=8.5,
        color="#7a3413",
        ha="center",
        va="center",
        fontweight="bold",
    )
    mid_r = freedom[-2] + reach[-2] / 2
    ax_dec.annotate(
        "reach\n(learned_mix − learn7)",
        xy=(tri[-2], mid_r),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color="#0c5e42",
        ha="right",
        va="center",
        fontweight="bold",
    )

    share = 100 * freedom[-1] / (freedom[-1] + reach[-1])
    ax_dec.set_title("...split into freedom and reach", loc="left")
    ax_dec.set_xlabel("training step")
    ax_dec.set_ylabel("contribution to the gap (dB)")
    ax_dec.set_xlim(tri[0], tri[-1])
    ax_dec.annotate(
        f"freedom is {share:.0f}% of the gap at {tri[-1] // 1000}k,\ndown from "
        f"{100 * freedom[0] / (freedom[0] + reach[0]):.0f}% at {tri[0] // 1000}k",
        xy=(0.03, 0.955),
        xycoords="axes fraction",
        fontsize=8,
        color=INK_SOFT,
        va="top",
    )
    thousands(ax_dec)
    spines(ax_dec)

    fig.text(
        0.5,
        -0.03,
        f"learn7 is scored to {tri[-1]:,} steps against the other arms' {pair_steps[-1]:,}; "
        "the split is provisional until it is matched.",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
    )

    stats = [
        "### Freedom / reach decomposition (steps where all three arms are scored)\n",
        "| step | freedom (dB) | reach (dB) | total (dB) | freedom share |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(tri):
        tot = freedom[i] + reach[i]
        stats.append(
            f"| {s} | {freedom[i]:+.3f} | {reach[i]:+.3f} | {tot:+.3f} | {100 * freedom[i] / tot:.0f}% |"
        )
    stats.append("\n### Total gap, learned_mix − control, at every matched step\n")
    stats.append("| step | gap (dB) |")
    stats.append("|---|---|")
    for s, g in zip(pair_steps, gaps, strict=True):
        stats.append(f"| {s} | {g:+.3f} |")
    stats.append(f"\nGap widens at {widened} of {len(gaps) - 1} step-to-step transitions.\n")
    return fig, "\n".join(stats)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "gap_decomposition"))
