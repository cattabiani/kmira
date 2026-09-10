"""Figure: the three-arm calibration that decided how this benchmark had to be run.

Three runs of 15,299 steps each, before any experiment:

    A  baseline
    B  identical but with the bottleneck frozen at a random projection -- a published ~1.4 dB effect
    C  baseline again, different seed -- the noise floor

A−B says whether the benchmark can SEE a bottleneck-sized change. |A−C| says how big a gap has to
be before it means anything. The script that ran these demanded A−B > 3·|A−C| to call the setup
usable, and it did not clear that. The figure shows why, because that failure is what set the
protocol for everything after it.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from lib import (
    DATA,
    GRID,
    INK,
    INK_SOFT,
    RECESSIVE,
    SURFACE,
    apply_style,
    load_rows,
    save,
    spines,
)

BASELINE_C = "#2a78d6"  # slot 1: both baseline arms, since they are the same configuration
FROZEN_C = "#eb6834"  # slot 2: the intervention
ARMS = [
    ("A_baseline", "A\nbaseline", BASELINE_C),
    ("B_frozen_bneck", "B\nfrozen bottleneck", FROZEN_C),
    ("C_baseline_seed2", "C\nbaseline, seed 2", BASELINE_C),
]


def build():
    rows = {r["tag"]: r for r in load_rows()}
    target = json.loads((DATA / "mira_bottleneck_ablation.json").read_text())
    expected = target["calibration_target"]["expected_drop_db"]

    a, b, c = (rows[t] for t, _, _ in ARMS)
    effect = a["psnr"] - b["psnr"]
    noise = abs(a["psnr"] - c["psnr"])
    step = int(a["checkpoint"].split("checkpoint-")[1].split("/")[0])

    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.9))

    # A dot plot, not bars. The three values span 18.7-21.2 and the whole question is how they sit
    # RELATIVE to each other, which bars anchored at zero compress into near-identical heights. Dots
    # carry no area, so a restricted axis is legitimate for them where it would mislead on bars.
    lo, hi = min(a["psnr"], c["psnr"]), max(a["psnr"], c["psnr"])
    ax.axvspan(lo, hi, color=RECESSIVE, alpha=0.6, zorder=1)
    ax.annotate(
        f"seed spread of the IDENTICAL configuration: {noise:.2f} dB",
        xy=((lo + hi) / 2, 2.62),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK_SOFT,
    )

    for i, (tag, label, color) in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        r = rows[tag]
        ax.plot(
            [r["psnr"]],
            [y],
            marker="o",
            markersize=10,
            color=color,
            zorder=4,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            f"  {r['psnr']:.3f} dB",
            xy=(r["psnr"], y),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([label.replace("\n", " — ") for _, label, _ in reversed(ARMS)], fontsize=9)
    ax.set_ylim(-0.75, 2.95)
    ax.set_xlim(min(r["psnr"] for r in (a, b, c)) - 0.45, hi + 0.95)
    ax.set_xlabel("PSNR (dB)  ↑ better", labelpad=4)
    ax.set_title(f"Three calibration runs, {step:,} steps each", loc="left")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, alpha=0.7)
    spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)

    fig.suptitle(
        "Why every experiment here is long and paired",
        x=0.008,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    # Caption in FIGURE coordinates: tight_layout does not account for annotations placed outside
    # the axes, so anchoring this to the axes collapsed the plot area instead of making room.
    fig.text(
        0.012,
        0.03,
        f"effect A−B = {effect:+.2f} dB, reproducing the published ≈{expected:.1f} dB — "
        f"but only {effect / noise:.2f}× the seed spread.\n"
        f"The launcher required {3 * noise:.2f} dB (3× noise) to call the setup usable. "
        f"VERDICT at this length: TOO NOISY.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )

    stats = [
        f"### Three-arm calibration, {step:,} steps per arm\n",
        "| arm | config | PSNR (dB) | SSIM | LPIPS | rFDD |",
        "|---|---|---|---|---|---|",
    ]
    for tag, label, _ in ARMS:
        r = rows[tag]
        stats.append(
            f"| {label.replace(chr(10), ' ')} | `{tag}` | {r['psnr']:.4f} | {r['ssim']:.4f} | "
            f"{r['lpips']:.4f} | {r['r_fdd']:.4f} |"
        )
    stats += [
        "",
        (
            f"- **effect** A − B = **{effect:+.4f} dB**, against a published target of ~{expected} dB "
            f"(`data/mira_bottleneck_ablation.json`, mira table `tab:exp-bottleneck`: "
            f"{target['rows']['learned_convolution']['psnr']} vs "
            f"{target['rows']['random_frozen_projection']['psnr']}). The effect size reproduces."
        ),
        (
            f"- **noise** |A − C| = **{noise:.4f} dB** between two runs of the identical configuration "
            "differing only in seed."
        ),
        (
            f"- effect / noise = **{effect / noise:.2f}×**. The launcher's own criterion for calling the "
            f"setup usable was effect > 3 × noise = {3 * noise:.4f} dB, which this does not meet: "
            "**TOO NOISY** at this run length.\n"
        ),
    ]
    return fig, "\n".join(stats)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "calibration_three_arm"))
