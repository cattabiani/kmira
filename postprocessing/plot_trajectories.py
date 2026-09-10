"""Figure: PSNR against training step for all three arms, with the two baselines marked.

The headline figure. Reads codec/results/benchmark.jsonl only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from lib import (
    ARM_LABEL,
    ARMS,
    COLOR,
    INK_MUTED,
    INK_SOFT,
    SURFACE,
    apply_style,
    baselines,
    load_rows,
    matched_steps,
    save,
    series,
    spines,
    thousands,
)


def build():
    rows = load_rows()
    base = baselines(rows)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for y in (base["anneal"], base["plateau"]):
        ax.axhline(y, color=INK_MUTED, linewidth=0.8, linestyle="-", alpha=0.55, zorder=1)

    present = []
    for arm in ARMS:
        pts = series(rows, arm)
        if not pts:
            continue
        present.append(arm)
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, color=COLOR[arm], marker="o", markersize=3.2, zorder=3, label=ARM_LABEL[arm])
        # Direct-label the endpoint of every series: identity is never colour-alone, and aqua
        # carries a contrast WARN that obligates a visible label.
        ax.annotate(
            f"  {ARM_LABEL[arm]}  {ys[-1]:.3f}",
            xy=(xs[-1], ys[-1]),
            xytext=(4, -1),
            textcoords="offset points",
            fontsize=8.5,
            color=COLOR[arm],
            fontweight="bold",
            va="center",
        )

    ax.set_xlabel("training step (warm start from checkpoint-304000)")
    ax.set_ylabel("PSNR (dB)  ↑ better")
    ax.set_title("Learned DINO layer aggregation: three paired arms", loc="left")
    ax.set_xlim(0, max(s for arm in present for s, _ in series(rows, arm)) * 1.19)
    lo = min(v for arm in present for _, v in series(rows, arm))
    hi = max(v for arm in present for _, v in series(rows, arm))
    ax.set_ylim(lo - 0.42, hi + 0.22)  # bottom headroom for the legend

    # Reference-line labels, placed once the x-range is known. They are separated HORIZONTALLY
    # rather than vertically: the two lines are 0.138 dB apart, which is less than one line of
    # 7.5pt text at this scale, so stacking them makes one rule strike through the other's label.
    # Each carries a surface-coloured box so the hairline it crosses is masked locally.
    xmax = ax.get_xlim()[1]
    for name, y, x in (
        ("plateau", base["plateau"], xmax * 0.005),
        ("annealed", base["anneal"], xmax * 0.62),
    ):
        ax.annotate(
            f"{name} {y:.3f}",
            xy=(x, y),
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_MUTED,
            va="bottom",
            ha="left",
            zorder=4,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2},
        )

    thousands(ax)
    spines(ax)
    ax.grid(axis="x", alpha=0.5)
    ax.legend(loc="lower right", ncols=3, fontsize=8.5)

    last = matched_steps(rows, ("control", "learned_mix"))[-1]
    c = dict(series(rows, "control"))[last]
    m = dict(series(rows, "learned_mix"))[last]
    ax.annotate(
        f"matched-step gap at {last // 1000}k: +{m - c:.3f} dB",
        xy=(0.015, 0.955),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_SOFT,
        va="top",
    )

    stats = [
        "### PSNR by arm and step\n",
        "| step | " + " | ".join(ARM_LABEL[a] for a in present) + " |",
        "|---" * (len(present) + 1) + "|",
    ]
    all_steps = sorted({s for a in present for s, _ in series(rows, a)})
    for step in all_steps:
        cells = []
        for arm in present:
            v = dict(series(rows, arm)).get(step)
            cells.append("—" if v is None else f"{v:.3f}")
        stats.append(f"| {step} | " + " | ".join(cells) + " |")
    stats.append(
        f"\nBaselines: constant-LR plateau {base['plateau']:.3f} dB "
        f"(`{'plateau-272000'}`), annealed {base['anneal']:.3f} dB (`anneal-304000`).\n"
    )
    return fig, "\n".join(stats)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "arm_trajectories"))
    print(stats)
