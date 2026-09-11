"""Figure: where the clean baseline is heading, and how far it still has to go.

`baseline_v2` is the first run on this rig with a per-chunk seed from step 0, so it is the first
one whose trajectory is a property of the rig rather than of the data-repetition bug. It is still
training, so this figure is a WIP readout, not a result.

Six panels, two questions.

WHERE IS THE ELBOW? The scored PSNR curve and, beside it, the per-chunk increment. The increment is
the panel to read: this project has twice called an elbow off a flat-looking curve and been wrong,
and both times the increment panel would have shown the flat stretch was still buying 0.1 dB a
chunk. An elbow needs several trailing increments near zero, not one.

WHERE ARE THE LOSSES CEILING OUT? mira's four validation terms, each in its own panel because their
scales differ by three orders of magnitude (dino latent consistency runs ~1e-4 where lpips runs
~0.5) and a shared axis would flatten three of them into a line. Each panel carries its own
trailing rate, so "has this one stopped moving" is answerable per term rather than by eye on
loss_total, which is dominated by whichever term is largest.

ONLY `baseline_v2` APPEARS HERE. The legacy baseline saw 53.4% of the training data and every
number derived from it is retired; overlaying it would invite exactly the comparison that is not
valid. See postprocessing/archive/RESULTS-legacy-baseline.md.
"""

from __future__ import annotations

import itertools
import json

import matplotlib.pyplot as plt
from lib import (
    DATA,
    INK,
    INK_SOFT,
    apply_style,
    load_rows,
    save,
    series,
    spines,
    thousands,
)

RUN = "ablation_baseline"  # the arm RESULTS.md calls baseline_v2; its tags are abl_baseline-*
TAG = "abl_baseline"
C = "#2a78d6"

# (key, panel title, how to say "better")
TERMS = [
    ("loss_total", "loss_total", "the sum, dominated by lpips"),
    ("loss_mae", "loss_mae", "pixel L1"),
    ("loss_lpips_perceptual", "loss_lpips_perceptual", "perceptual"),
    ("loss_dino_latent_consistency", "loss_dino_latent_consistency", "latent consistency"),
]


def trailing_rate(xs, ys, n=8):
    """Least-squares slope over the last n readings, per 10,000 steps.

    Reported rather than eyeballed because the whole point of the panel is whether the curve has
    stopped, and a curve that looks flat at this zoom can still be moving.
    """
    xs, ys = xs[-n:], ys[-n:]
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return 10_000 * sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den


def build():
    readings = json.loads((DATA / "run_metadata.json").read_text())["runs"][RUN]["readings"]
    psnr = series(load_rows(), TAG)

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6))
    steps = [r["step"] for r in readings]
    last_step = max(steps)

    # --- PSNR, and the increment that decides the elbow -------------------------------------
    ax = axes[0][0]
    px = [s for s, _ in psnr]
    py = [v for _, v in psnr]
    ax.plot(px, py, color=C, marker="o", markersize=4)
    ax.set_title("PSNR, scored every 8k  ↑ better", loc="left")
    ax.set_ylabel("PSNR (dB), 2048 held-out frames")
    ax.annotate(
        f"{py[-1]:.3f} dB at {px[-1]:,}",
        xy=(0.97, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        fontweight="bold",
        color=C,
    )
    thousands(ax)
    spines(ax)

    ax = axes[0][1]
    dx = [b for (b, _), (_, _) in zip(psnr[1:], psnr[:-1], strict=True)]
    dy = [b - a for (_, a), (_, b) in itertools.pairwise([(s, v) for s, v in psnr])]
    ax.bar(dx, dy, width=6200, color=C)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9)
    ax.set_title("Δ PSNR per 8k chunk — read THIS for the elbow", loc="left")
    ax.set_ylabel("Δ dB per chunk")
    ax.annotate(
        f"last 3: {', '.join(f'{d:+.3f}' for d in dy[-3:])}\n"
        "an elbow needs several near zero,\nnot one",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK_SOFT,
    )
    thousands(ax)
    spines(ax)

    # --- the four loss terms, each on its own scale ------------------------------------------
    for ax, (key, title, sub) in zip([axes[0][2], *axes[1]], TERMS, strict=True):
        ys = [r[key] for r in readings]
        ax.plot(steps, ys, color=C, linewidth=1.6)
        rate = trailing_rate(steps, ys)
        ax.set_title(f"{title}  ↓ better", loc="left")
        ax.set_xlabel(sub)
        # mira logs these to 4 decimal places. A term small enough to sit on that grid stops being
        # a curve and becomes a staircase, and its "trailing slope" is then rounding, not training.
        # Say so on the panel rather than letting a flat line be read as convergence.
        note = f"{ys[-1]:.4g}\ntrailing {rate:+.2g}/10k"
        if len(set(ys[-8:])) == 1:
            note = f"{ys[-1]:.4g}\nAT THE LOG'S 4-dp FLOOR —\nslope here is rounding, not training"
        ax.annotate(
            note,
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=INK_SOFT,
        )
        thousands(ax)
        spines(ax)

    fig.suptitle(
        f"baseline_v2 — WIP at step {last_step:,}, still training",
        x=0.006,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    fig.text(
        0.008,
        0.012,
        "The first run on this rig with a per-chunk seed from step 0, so the first whose trajectory "
        "is a property of the rig and not of the data-repetition bug.\nNothing from the retired "
        "baseline is plotted here. Validation terms are mira's own val loop (512 samples, every "
        "1,000 steps); PSNR is eval_codec on 2,048 held-out frames.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    return fig, stats_for(psnr, readings, steps)


def stats_for(psnr, readings, steps) -> str:
    lines = [
        f"### baseline_v2 — WIP readout at step {max(steps):,}\n",
        (
            "The clean baseline: per-chunk seeds from step 0, 100% training-data coverage, constant "
            "LR 1e-4, no anneal yet. Still training, so every number here moves.\n"
        ),
        "| step | PSNR (dB) | Δ per 8k |",
        "|---|---|---|",
    ]
    prev = None
    for s, v in psnr:
        d = "" if prev is None else f"{v - prev:+.3f}"
        lines.append(f"| {s:,} | {v:.4f} | {d} |")
        prev = v
    lines += ["", "Validation loss terms, trailing slope over the last 8 readings:\n"]
    lines += ["| term | latest | trailing slope per 10k steps |", "|---|---|---|"]
    for key, title, _ in TERMS:
        ys = [r[key] for r in readings]
        floored = len(set(ys[-8:])) == 1
        rate = "— at the log's 4-dp floor" if floored else f"{trailing_rate(steps, ys):+.3g}"
        lines.append(f"| `{title}` | {ys[-1]:.6g} | {rate} |")
    lines.append("")
    lines.append(
        "A term whose trailing slope is small relative to its own value has stopped moving; read "
        "each separately, because `loss_total` is dominated by `loss_lpips_perceptual` and hides "
        "the other two.\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "baseline_v2_progress"))
