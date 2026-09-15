"""Figure: the baseline's full trajectory, and the first paired comparison against it.

THREE CURVES. The constant-LR stretch and the cosine anneal are one run -- `baseline_v2`, continued
into its own output directory -- and were first drawn in one colour separated by line style, on the
argument that they are one entity and only two things were compared. That was correct and useless:
against 264,000 steps of solid line, a 32,000-step dotted tail is invisible. The anneal now carries
its own colour. It is still a phase of the baseline rather than a third arm, which the label and the
marked transition say instead.

WHY TWO PANELS. The ablation has run 32,000 steps against the baseline's 296,000. On one axis it is
a stub in the corner and the comparison is invisible, so the right panel re-plots the matched range
alone. Both panels are PSNR on a single linear axis -- no second scale anywhere -- and the right
panel is a zoom of the left, not a different measurement.

READ THE RIGHT PANEL AT MATCHED STEPS ONLY. The two arms share a seed base, so at any given step
they have trained on the same data; the gap at a step is therefore attributable to the frozen
bottleneck. Nothing here compares the ablation to the baseline's ENDPOINT, which would confound the
intervention with 264,000 extra steps.
"""

from __future__ import annotations

import re

import lib
import matplotlib.pyplot as plt
from lib import GRID, INK, INK_SOFT, SURFACE

# Colour follows the entity: the baseline keeps slot 1 wherever it appears, the ablation slot 2.
C_BASE = "#2a78d6"
C_ANNEAL = "#1baf7a"
C_FROZEN = "#eb6834"

RUNS = {"baseline": "ablation_baseline", "frozen": "ablation_frozen_bneck"}


def scored(run_dir: str) -> list[tuple[int, float]]:
    """(step, psnr) for one run, keyed off the checkpoint path rather than the tag.

    The tag is per-arm and one of them deliberately no longer matches its own directory, so a tag
    prefix can pair one run's curve with another's numbers. The path cannot drift from the run that
    wrote it.
    """
    out = []
    for row in lib.load_rows():
        m = re.search(rf"/{re.escape(run_dir)}/checkpoint-(\d+)/", row.get("checkpoint", ""))
        if m:
            out.append((int(m.group(1)), row["psnr"]))
    return sorted(out)


def anneal_base() -> int | None:
    """The step the anneal began at, read from the window the launcher wrote."""
    path = lib.REPO / "checkpoints" / "calibration" / RUNS["baseline"] / "anneal_window.env"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("BASE_STEP="):
            return int(line.split("=", 1)[1])
    return None


def build():
    base = scored(RUNS["baseline"])
    frozen = scored(RUNS["frozen"])
    cut = anneal_base()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.6), facecolor=SURFACE)

    # ---- left: the whole baseline, constant then annealed -------------------------------------
    const = [(s, p) for s, p in base if cut is None or s <= cut]
    ann = [(s, p) for s, p in base if cut is not None and s >= cut]  # shares the joining point
    ax.plot([s for s, _ in const], [p for _, p in const], color=C_BASE, linewidth=2, zorder=3)
    if ann:
        ax.plot(
            [s for s, _ in ann],
            [p for _, p in ann],
            color=C_ANNEAL,
            linewidth=2.6,
            marker="o",
            markersize=4,
            zorder=4,
        )
        ax.axvline(cut, color=INK_SOFT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            f"anneal starts\n{cut:,}",
            xy=(cut, base[0][1] + 0.35),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=INK_SOFT,
        )
        ax.annotate(
            f"cosine anneal\n{ann[-1][1]:.3f} dB",
            xy=(ann[-1][0], ann[-1][1]),
            xytext=(-6, -34),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=C_ANNEAL,
        )
    if frozen:
        ax.plot(
            [s for s, _ in frozen],
            [p for _, p in frozen],
            color=C_FROZEN,
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=5,
        )
        ax.annotate(
            "frozen bottleneck",
            xy=(frozen[-1][0], frozen[-1][1]),
            xytext=(10, -2),
            textcoords="offset points",
            fontsize=9,
            color=C_FROZEN,
            fontweight="bold",
        )
    ax.annotate(
        "baseline, constant LR",
        xy=(base[len(base) // 3][0], base[len(base) // 3][1]),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=C_BASE,
        fontweight="bold",
    )
    ax.set_title("The baseline, trained then annealed", loc="left", fontweight="bold")
    ax.set_ylabel("PSNR (dB), 2048 held-out frames")
    ax.set_xlabel("step")

    # ---- right: the matched range, where the comparison actually lives -------------------------
    if frozen:
        hi = frozen[-1][0]
        bsub = [(s, p) for s, p in base if s <= hi]
        bx.plot(
            [s for s, _ in bsub],
            [p for _, p in bsub],
            color=C_BASE,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        bx.plot(
            [s for s, _ in frozen],
            [p for _, p in frozen],
            color=C_FROZEN,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        bd = dict(base)
        for s, p in frozen:
            bx.plot([s, s], [p, bd[s]], color=INK_SOFT, linewidth=0.8, zorder=1)
        gaps = [bd[s] - p for s, p in frozen]
        mid = frozen[len(frozen) // 2]
        bx.annotate(
            f"gap {min(gaps):.2f}–{max(gaps):.2f} dB",
            xy=(mid[0], (mid[1] + bd[mid[0]]) / 2),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            fontweight="bold",
            color=INK,
            va="center",
        )
        bx.annotate(
            "baseline",
            xy=bsub[-1],
            xytext=(-4, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=C_BASE,
            fontweight="bold",
        )
        bx.annotate(
            "frozen bottleneck",
            xy=frozen[-1],
            xytext=(-4, -16),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=C_FROZEN,
            fontweight="bold",
        )
        bx.set_title(
            f"Matched steps, to {hi:,} — the paired comparison", loc="left", fontweight="bold"
        )
        bx.set_xlabel("step")
        bx.set_ylabel("PSNR (dB)")

    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.7)
        a.set_axisbelow(True)
        lib.spines(a)
        lib.thousands(a)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(
        0.008,
        0.012,
        "Both arms run the locked recipe (constant LR 1e-4, 8,000-step chunks) from the same seed "
        "base, so at any step they have trained on the same data.\nThe ablation is still running; "
        "it is read only against the baseline at matched steps, never against the baseline's "
        "endpoint.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    return fig, stats_for(base, frozen, cut)


def stats_for(base, frozen, cut) -> str:
    lines = ["### The baseline, and the first arm read against it\n"]
    if cut is not None:
        pre = dict(base).get(cut)
        post = base[-1]
        lines.append(
            f"**The anneal.** Constant LR to {cut:,} reached **{pre:.4f} dB**; a "
            f"{post[0] - cut:,}-step cosine decay to min_lr took it to **{post[1]:.4f} dB** at step "
            f"**{post[0]:,}**, a gain of **{post[1] - pre:+.4f} dB**. That is the fixed reference "
            f"every arm is read against.\n"
        )
    if frozen:
        bd = dict(base)
        rows = "\n".join(
            f"| {s:,} | {p:.4f} | {bd[s]:.4f} | **{bd[s] - p:+.4f}** |" for s, p in frozen
        )
        gaps = [bd[s] - p for s, p in frozen]
        lines.append(
            "**Frozen bottleneck, at matched steps.** A random frozen bottleneck against the "
            "trained one, same seed base so the same data chunk for chunk:\n\n"
            "| step | frozen | baseline | gap |\n|---|---|---|---|\n"
            f"{rows}\n\n"
            f"The gap runs **{min(gaps):.4f}** to **{max(gaps):.4f} dB** over "
            f"{len(frozen)} matched readings. Still running.\n"
        )
    return "\n".join(lines)


def main() -> None:
    fig, _ = build()
    lib.save(fig, "arms")


if __name__ == "__main__":
    main()
