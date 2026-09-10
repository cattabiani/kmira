"""Figure: the three-arm calibration that decided how this benchmark had to be run.

Three runs of 15,299 steps each, before any experiment:

    A  baseline
    B  identical but with the bottleneck frozen at a random projection -- a published ~1.4 dB effect
    C  baseline again, different seed -- the noise floor

A−B says whether the benchmark can SEE a bottleneck-sized change. |A−C| says how big a gap has to
be before it means anything. The launcher demanded A−B > 3·|A−C| to call the setup usable, and it
did not clear that. This figure is why the protocol for everything after looks the way it does.

ONE PANEL, NOT TWO. What is plotted is mira's own validation loss (512 samples, parsed from the run
logs), which is the per-step record these arms kept; each arm's final dB goes in the legend beside
the arm it belongs to. These were the project's first runs and ran under the stock
`checkpoint_keep_recent: 1`, so one checkpoint per arm survived and there is no dB curve to plot
next to this one. Every run after them keeps its checkpoints and is scored densely, so this is the
only figure in the folder without one. Transforming the loss into something that rises would look
like PSNR without being it, so it is not done here.

The curves also settle the "had they converged?" question, and the answer is subtler than either
yes or no -- see the settling block in the stats this returns.

COLOUR FOLLOWS THE CONFIGURATION, not the run: A and C are the *same* configuration differing only
in seed, so they share a hue and are separated by marker and label. Two blue curves landing far
apart is the finding, and making them one colour is what says so.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from lib import (
    ANNEAL_TAG,
    DATA,
    INK,
    INK_SOFT,
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


def _slope_per_1k(readings) -> float:
    """Least-squares slope of loss against step, scaled to loss-per-1000-steps."""
    xs = [r["step"] for r in readings]
    ys = [r[METRIC] for r in readings]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs)
    return 1000 * num / den


def settling_stats(curves, scored) -> str:
    """Had these runs converged? The trailing slope says nearly, the wobble says you cannot tell,
    and the plateau run says it would not have mattered either way.

    RESULTS.md 0b quotes all three, so all three are computed here rather than eyeballed off the
    figure. The trap this guards against is the one the project actually fell into twice: a flat
    trailing curve is not evidence of convergence when the reading-to-reading spread is larger than
    the trend it is supposed to reveal.
    """
    lines = [
        "#### Had the calibration arms settled?\n",
        "| arm | slope, first half | slope, last 5 readings | spread of last 6 | last reading |",
        "|---|---|---|---|---|",
    ]
    for tag, label, _, _ in ARMS:
        rd = curves[tag]["readings"]
        half = len(rd) // 2
        ys = [r[METRIC] for r in rd[-6:]]
        lines.append(
            f"| {label} | {_slope_per_1k(rd[:half]):+.4f} | {_slope_per_1k(rd[-5:]):+.4f} | "
            f"{max(ys) - min(ys):.4f} | {rd[-1][METRIC]:.4f} |"
        )
    lines += [
        "",
        "Slopes are loss per 1,000 steps; negative is improving.\n",
        (
            "Each arm's trailing trend is roughly an order of magnitude shallower than its opening "
            "one, so on the curve alone these look converged. But the residual slope over the last "
            "five readings is smaller than the spread of the last six in every arm, so the trend is "
            "inside the noise and the curve cannot tell you whether it has stopped. All three ticked "
            "*up* at the final reading."
        ),
        "",
    ]
    pl = curves.get("plateau_baseline")
    if pl:
        rd = pl["readings"]
        band = [r[METRIC] for r in rd if 9000 <= r["step"] <= 20000]
        first, last = rd[0], rd[-1]
        # How much of the run's whole descent was still ahead of it while it sat in that flat band?
        remaining = max(band) - last[METRIC]
        total = first[METRIC] - last[METRIC]
        lines.append(
            f"And flatness there would have meant nothing anyway. The baseline run — A's "
            f"configuration taken to 272,000 steps — sits in the same flat band over steps 9k–20k "
            f"of its own trace "
            f"(loss {min(band):.4f}–{max(band):.4f}, spread {max(band) - min(band):.4f}) — and then "
            f"runs to step {last['step']:,}, ending at {last[METRIC]:.4f}. Sitting in that band it "
            f"still had {remaining:.4f} of loss to shed, **{remaining / total:.0%} of its entire "
            f"descent** from {first[METRIC]:.4f}."
        )
        # The same point in PSNR, which is the metric the page is about. Both readings come from
        # benchmark.jsonl so this sentence cannot drift from the scored record.
        early = scored.get("plateau-16000")
        final = scored.get(ANNEAL_TAG)
        if early and final:
            lines.append(
                f"\nIn the metric the page is actually about, that band is "
                f"{final['psnr'] - early['psnr']:.3f} dB short of where the run ends: "
                f"{early['psnr']:.3f} dB at step 16,000 against {final['psnr']:.3f} at "
                f"step {last['step']:,}.\n"
            )
    return "\n".join(lines)


def build():
    scored = {r["tag"]: r for r in load_rows()}
    meta = json.loads((DATA / "run_metadata.json").read_text())["runs"]
    curves = meta
    target = json.loads((DATA / "mira_bottleneck_ablation.json").read_text())
    expected = target["calibration_target"]["expected_drop_db"]

    a, b, c = (scored[t] for t, _, _, _ in ARMS)
    effect = a["psnr"] - b["psnr"]
    noise = abs(a["psnr"] - c["psnr"])
    step = int(a["checkpoint"].split("checkpoint-")[1].split("/")[0])
    n_frames = a["n_frames"]

    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.7))

    for tag, label, color, marker in ARMS:
        rec = curves.get(tag)
        if not rec:
            continue
        xs = [r["step"] for r in rec["readings"]]
        ys = [r[METRIC] for r in rec["readings"]]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=3.8,
            label=f"{label}   —   {scored[tag]['psnr']:.3f} dB at {step:,}",
            zorder=3,
        )
        ax.annotate(
            f"  {label.split('  ')[0]}",
            xy=(xs[-1], ys[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel(f"training step  (each arm ran to {step:,})")
    ax.set_ylabel("validation loss, mira's own val loop\n(512 samples)  ↓ better")
    ax.set_title(
        "Three calibration runs: two of them are the same config, and they land furthest apart",
        loc="left",
    )
    ax.set_xlim(-400, max(r["step"] for r in curves["A_baseline"]["readings"]) * 1.09)
    leg = ax.legend(loc="upper right", fontsize=8.5, title="arm   —   final PSNR, 2048 held-out frames")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(INK_SOFT)
    ax.annotate(
        "A and C are the SAME configuration — only the seed differs.\n"
        "In this metric they separate by MORE than the frozen bottleneck costs.",
        xy=(0.17, 0.62),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_SOFT,
        va="top",
    )
    thousands(ax)
    spines(ax)

    fig.suptitle(
        "Why every experiment here is long and paired",
        x=0.008,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.text(
        0.012,
        0.025,
        f"In PSNR: effect A−B = {effect:+.2f} dB, reproducing the published ≈{expected:.1f} dB — but "
        f"only {effect / noise:.2f}× the seed spread |A−C| = {noise:.2f} dB. The launcher required "
        f"{3 * noise:.2f} dB (3× noise) to call the setup usable.\nVERDICT at this length: TOO NOISY. "
        "Hence long runs, and arms paired on one warm start and one seed schedule so the spread "
        "cancels rather than being averaged over.",
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
    stats.append(settling_stats(curves, scored))
    return fig, "\n".join(stats)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "calibration_three_arm"))
