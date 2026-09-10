"""Figure: the baseline's constant-LR plateau search, its elbow, and the anneal that followed.

The single most load-bearing figure in this folder: everything downstream compares against the
checkpoint this curve ends at. It is also the evidence for the project's most expensive lesson --
this run hit TWO stretches that looked like convergence and were not.

Top panel is scored PSNR. Bottom panel is the per-reading increment, which is what makes "flat"
falsifiable: a stretch near zero followed by a jump is an apparent plateau, and reading only the
top panel is how you stop in one.

Both stretches were later traced to the per-chunk seed schedule -- the same seeds stall and
unstall the warm-start arms 100k+ steps away (see plot_seed_effects.py). This figure is still the
right one for "do not call an elbow on one flat stretch"; it just is not evidence about the
optimiser. The annotations say "apparent", not "false", for that reason.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from lib import (
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    SURFACE,
    apply_style,
    load_rows,
    save,
    series,
    spines,
    thousands,
)

PLATEAU_C = "#2a78d6"  # categorical slot 1: the constant-LR search
ANNEAL_C = "#eb6834"  # slot 2: the cosine anneal, a different phase of the same run


def build():
    rows = load_rows()
    plateau = series(rows, "plateau")
    anneal = series(rows, "anneal")
    apply_style()
    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(8.6, 5.9), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    px = [s for s, _ in plateau]
    py = [v for _, v in plateau]
    ax.plot(px, py, color=PLATEAU_C, marker="o", markersize=3.2, label="constant LR 1e-4", zorder=3)
    ax_ = [s for s, _ in anneal]
    ay = [v for _, v in anneal]
    ax.plot(ax_, ay, color=ANNEAL_C, marker="o", markersize=3.2, label="cosine anneal 1e-4 → 1e-6", zorder=4)

    # The two stretches that looked converged and were not. Both are real readings, so the step
    # numbers and deltas here are computed from the record rather than named by hand.
    by_step = dict(plateau)
    jump1 = max(range(1, len(px)), key=lambda i: py[i] - py[i - 1])
    ax.annotate(
        f"apparent plateau: flat to {px[jump1 - 1] // 1000}k,\nthen +{py[jump1] - py[jump1 - 1]:.2f} dB\n"
        f"(chunk seed 40 — see 0c)",
        xy=(px[jump1], py[jump1]),
        xytext=(px[jump1] + 12000, py[jump1] - 2.05),
        fontsize=8,
        color=INK_SOFT,
        arrowprops={"arrowstyle": "->", "color": INK_MUTED, "linewidth": 0.9, "shrinkB": 4},
    )
    # The second one: the flattest three-reading window that is followed by a >0.1 dB step.
    cands = [
        i
        for i in range(3, len(px) - 1)
        if max(abs(py[j] - py[j - 1]) for j in range(i - 2, i + 1)) < 0.05 and py[i + 1] - py[i] > 0.1
    ]
    if cands:
        i = cands[-1]
        ax.annotate(
            f"flat {px[i - 2] // 1000}k–{px[i] // 1000}k (<0.05 dB/8k),\nthen +{py[i + 1] - py[i]:.2f} dB",
            xy=(px[i + 1], py[i + 1]),
            xytext=(px[i + 1] - 42000, py[i + 1] - 1.45),
            fontsize=8,
            color=INK_SOFT,
            arrowprops={"arrowstyle": "->", "color": INK_MUTED, "linewidth": 0.9, "shrinkB": 4},
        )

    ax.axvline(px[-1], color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)
    ax.annotate(
        f"elbow, step {px[-1] // 1000}k\n{py[-1]:.3f} dB",
        xy=(px[-1], py[0] + 2.4),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    ax.annotate(
        f"  anneal buys +{ay[-1] - ay[0]:.3f} dB\n  → locked baseline {ay[-1]:.3f}",
        xy=(ax_[-1], ay[-1]),
        xytext=(5, -2),
        textcoords="offset points",
        fontsize=8.5,
        color=ANNEAL_C,
        fontweight="bold",
        va="center",
    )

    ax.set_ylabel("PSNR (dB)  ↑ better")
    ax.set_title("Baseline codec: constant-LR plateau search, elbow, then anneal", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    spines(ax)

    # ---- increments -----------------------------------------------------------------------------
    dx = px[1:]
    dy = [py[i] - py[i - 1] for i in range(1, len(px))]
    ax_d.axhline(0, color=INK_MUTED, linewidth=0.9, alpha=0.7, zorder=2)
    ax_d.bar(dx, dy, width=5200, color=PLATEAU_C, linewidth=0, zorder=3)
    adx = ax_[1:]
    ady = [ay[i] - ay[i - 1] for i in range(1, len(ax_))]
    ax_d.bar(adx, ady, width=5200, color=ANNEAL_C, linewidth=0, zorder=3)
    ax_d.axvline(px[-1], color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)
    ax_d.set_ylabel("Δ PSNR per 8k\nsteps (dB)", fontsize=8)
    ax_d.set_xlabel("training step")
    ax_d.annotate(
        "a near-zero stretch is not convergence — twice here it was followed by a jump",
        xy=(0.34, 0.95),
        xycoords="axes fraction",
        fontsize=7.5,
        color=INK_SOFT,
        va="top",
    )
    thousands(ax_d)
    spines(ax_d)
    ax_d.grid(axis="x", color=GRID, alpha=0.5)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()

    # The readings just before the big jump, so "it looked converged" is a number not an adjective.
    pre_window = [py[i] - py[i - 1] for i in range(max(1, jump1 - 6), jump1)]
    stats = [
        "### Baseline plateau search and anneal\n",
        (
            f"- Constant LR 1e-4 from step {px[0]:,} to the elbow at **{px[-1]:,}** steps, "
            f"PSNR {py[0]:.4f} → **{py[-1]:.4f}** dB."
        ),
        (
            f"- Cosine anneal 1e-4 → 1e-6 over {ax_[-1] - ax_[0]:,} further steps: "
            f"{ay[0]:.4f} → **{ay[-1]:.4f}** dB, a gain of **+{ay[-1] - ay[0]:.4f}** dB."
        ),
        (
            f"- Largest single jump: **+{py[jump1] - py[jump1 - 1]:.3f} dB** at step {px[jump1]:,} "
            f"(from {py[jump1 - 1]:.4f} to {py[jump1]:.4f})."
        ),
        (
            f"  - The {len(pre_window)} readings immediately before it ({px[jump1 - len(pre_window)]:,}"
            f"–{px[jump1 - 1]:,}) averaged **{sum(pre_window) / len(pre_window):+.3f}** dB per 8k and "
            f"never exceeded **{max(pre_window):+.3f}**."
        ),
    ]
    if cands:
        i = cands[-1]
        stats.append(
            f"- Second apparent plateau: steps {px[i - 2]:,}–{px[i]:,} all moved <0.05 dB per 8k "
            f"({py[i - 2]:.4f} → {py[i]:.4f}), then +{py[i + 1] - py[i]:.3f} dB at {px[i + 1]:,}."
        )
    stats.append(
        "- Final 3 readings before the elbow: "
        + ", ".join(f"{s // 1000}k {by_step[s]:.4f}" for s in px[-3:])
        + " — the flatness the elbow was called on.\n"
    )
    stats.append("| step | PSNR (dB) | Δ per 8k | phase |")
    stats.append("|---|---|---|---|")
    for i, s in enumerate(px):
        d = "" if i == 0 else f"{py[i] - py[i - 1]:+.3f}"
        stats.append(f"| {s} | {py[i]:.4f} | {d} | constant LR |")
    for i, s in enumerate(ax_):
        if i == 0:
            continue
        stats.append(f"| {s} | {ay[i]:.4f} | {ay[i] - ay[i - 1]:+.3f} | cosine anneal |")
    return fig, "\n".join(stats) + "\n"


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "baseline_elbow"))
