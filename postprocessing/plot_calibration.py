"""Figure: the three-arm calibration that decided how this benchmark had to be run.

Three runs of 15,299 steps each, before any experiment:

    A  baseline
    B  identical but with the bottleneck frozen at a random projection -- a published ~1.4 dB effect
    C  baseline again, different seed -- the noise floor

A−B says whether the benchmark can SEE a bottleneck-sized change. |A−C| says how big a gap has to
be before it means anything. The launcher demanded A−B > 3·|A−C| to call the setup usable, and it
did not clear that. This figure is why the protocol for everything after looks the way it does.

TWO PANELS, TWO DIFFERENT QUANTITIES, deliberately not on one axis:

* left  -- mira's validation loss over training (512 samples, from the run logs). The only per-step
           record these arms have, and what shows whether they had settled. They had not.
* right -- PSNR of held-out reconstructions at the final step (2048 frames, eval_codec). The dB
           numbers everything else on the page is quoted in, and the only step where they exist.

COLOUR FOLLOWS THE CONFIGURATION, not the run: A and C are the *same* configuration differing only
in seed, so they share a hue and are separated by marker and label. Two blue curves landing far
apart is the finding, and making them one colour is what says so.
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
    thousands,
)

BASELINE_C = "#2a78d6"  # slot 1: the baseline configuration, both of its runs
FROZEN_C = "#eb6834"  # slot 2: the frozen-bottleneck configuration
# (tag, short label, colour, marker)
ARMS = [
    ("A_baseline", "A  baseline", BASELINE_C, "o"),
    ("B_frozen_bneck", "B  frozen bottleneck", FROZEN_C, "o"),
    ("C_baseline_seed2", "C  baseline, seed 2", BASELINE_C, "s"),
]
METRIC = "loss_total"


def build():
    scored = {r["tag"]: r for r in load_rows()}
    curves = json.loads((DATA / "calibration_curves.json").read_text())["arms"]
    target = json.loads((DATA / "mira_bottleneck_ablation.json").read_text())
    expected = target["calibration_target"]["expected_drop_db"]

    a, b, c = (scored[t] for t, _, _, _ in ARMS)
    effect = a["psnr"] - b["psnr"]
    noise = abs(a["psnr"] - c["psnr"])
    step = int(a["checkpoint"].split("checkpoint-")[1].split("/")[0])
    n_frames = a["n_frames"]

    apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.65, 1]})

    # ---- left: validation curves ----------------------------------------------------------------
    for tag, label, color, marker in ARMS:
        rec = curves.get(tag)
        if not rec:
            continue
        xs = [r["step"] for r in rec["readings"]]
        ys = [r[METRIC] for r in rec["readings"]]
        ax.plot(xs, ys, color=color, marker=marker, markersize=3.6, label=label, zorder=3)
        ax.annotate(
            f"  {label.split('  ')[0]}",
            xy=(xs[-1], ys[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel(f"training step  (each arm ran to {step:,})")
    ax.set_ylabel("validation loss, mira's own val loop\n(512 samples)  ↓ better")
    ax.set_title("None of the three had settled", loc="left")
    ax.set_xlim(-400, max(r["step"] for r in curves["A_baseline"]["readings"]) * 1.10)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.annotate(
        "A and C are the SAME configuration — only the seed differs.\n"
        "In this metric they separate by MORE than the frozen\nbottleneck costs.",
        xy=(0.30, 0.62),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_SOFT,
        va="top",
    )
    thousands(ax)
    spines(ax)

    # ---- right: final scored PSNR, dB on the y axis ---------------------------------------------
    lo, hi = min(a["psnr"], c["psnr"]), max(a["psnr"], c["psnr"])
    ax2.axhspan(lo, hi, color=RECESSIVE, alpha=0.6, zorder=1)
    ax2.annotate(
        f"seed spread\n{noise:.2f} dB",
        xy=(2.44, (lo + hi) / 2),
        ha="right",
        va="center",
        fontsize=8,
        color=INK_SOFT,
    )
    for i, (tag, label, color, marker) in enumerate(ARMS):
        r = scored[tag]
        ax2.plot(
            [i],
            [r["psnr"]],
            marker=marker,
            markersize=10,
            color=color,
            zorder=4,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax2.annotate(
            f"{r['psnr']:.3f}",
            xy=(i, r["psnr"]),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
    ax2.set_xticks(range(len(ARMS)))
    ax2.set_xticklabels([label.split("  ")[0] for _, label, _, _ in ARMS], fontsize=9.5)
    ax2.set_xlim(-0.55, 2.5)
    ax2.set_ylim(min(r["psnr"] for r in (a, b, c)) - 0.45, hi + 0.60)
    ax2.set_ylabel(f"PSNR of held-out reconstructions\nat step {step:,} ({n_frames} frames), dB  ↑ better")
    ax2.set_title("...and the dB they were scored at", loc="left")
    ax2.grid(axis="x", visible=False)
    ax2.grid(axis="y", color=GRID, alpha=0.7)
    spines(ax2)

    fig.suptitle(
        "Why every experiment here is long and paired",
        x=0.008,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.text(
        0.012,
        0.025,
        f"effect A−B = {effect:+.2f} dB, reproducing the published ≈{expected:.1f} dB — but only "
        f"{effect / noise:.2f}× the seed spread. The launcher required {3 * noise:.2f} dB "
        f"(3× noise) to call the setup usable.\nVERDICT at this length: TOO NOISY. Hence long runs, "
        "and arms paired on one warm start and one seed schedule so the spread cancels rather than "
        "being averaged over.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )

    stats = [
        f"### Three-arm calibration, {step:,} steps per arm\n",
        f"Final scored PSNR, `eval_codec` on {n_frames} held-out frames:\n",
        "| arm | config | PSNR (dB) | SSIM | LPIPS | rFDD |",
        "|---|---|---|---|---|---|",
    ]
    for tag, label, _, _ in ARMS:
        r = scored[tag]
        stats.append(
            f"| {label} | `{tag}` | {r['psnr']:.4f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['r_fdd']:.4f} |"
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
            f"- **noise** |A − C| = **{noise:.4f} dB** between two runs of the identical "
            "configuration differing only in seed."
        ),
        (
            f"- effect / noise = **{effect / noise:.2f}×**. The launcher's own criterion for calling "
            f"the setup usable was effect > 3 × noise = {3 * noise:.4f} dB, which this does not "
            "meet: **TOO NOISY** at this run length."
        ),
        "",
        (
            "Final validation loss (mira's val loop, 512 samples — a different quantity from the PSNR "
            "above, on a different sample set):\n"
        ),
        "| arm | last val step | loss_total | loss_mae |",
        "|---|---|---|---|",
    ]
    for tag, label, _, _ in ARMS:
        last = curves[tag]["readings"][-1]
        stats.append(f"| {label} | {last['step']} | {last['loss_total']:.4f} | {last['loss_mae']:.4f} |")
    a_last = curves["A_baseline"]["readings"][-1]["loss_total"]
    b_last = curves["B_frozen_bneck"]["readings"][-1]["loss_total"]
    c_last = curves["C_baseline_seed2"]["readings"][-1]["loss_total"]
    stats.append(
        f"\nIn validation loss the seed gap is **larger** than the intervention: "
        f"|A−C| = {abs(a_last - c_last):.4f} against B−A = {b_last - a_last:.4f}. "
        "Same conclusion as the PSNR comparison, reached independently.\n"
    )
    return fig, "\n".join(stats)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "calibration_three_arm"))
