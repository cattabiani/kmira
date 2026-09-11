"""Figure: every reconstruction metric in the benchmark, per arm, as small multiples.

The point of showing all five is that they do not agree. PSNR separates the arms enormously; rFDD
clearly; P-DINO not at all. One panel per metric, each with its own y-scale -- which is what small
multiples are for, and the reason this is five panels rather than one chart with five y-axes.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from lib import (
    ARCHIVE_FIGURES,
    ARM_LABEL,
    ARMS,
    COLOR,
    HIGHER_IS_BETTER,
    INK_SOFT,
    METRIC_LABEL,
    apply_style,
    load_rows,
    matched_steps,
    save,
    series,
    spines,
    thousands,
)

METRICS = ("psnr", "ssim", "lpips", "p_dino", "r_fdd")
# P-DINO lives around 1e-4; plotting it raw fills the axis with exponents.
SCALE = {"p_dino": 1e5}
SCALE_NOTE = {"p_dino": " (×10⁻⁵)"}


def build():
    rows = load_rows()
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.0))
    flat = axes.flatten()

    stats = ["### All benchmark metrics, control vs learned_mix at matched steps\n"]
    for ax, metric in zip(flat, METRICS, strict=False):
        scale = SCALE.get(metric, 1.0)
        for arm in ARMS:
            pts = series(rows, arm, metric)
            if not pts:
                continue
            ax.plot(
                [s for s, _ in pts],
                [v * scale for _, v in pts],
                color=COLOR[arm],
                marker="o",
                markersize=2.4,
                label=ARM_LABEL[arm],
            )
        arrow = "↑" if HIGHER_IS_BETTER[metric] else "↓"
        ax.set_title(f"{METRIC_LABEL[metric]}{SCALE_NOTE.get(metric, '')}   {arrow} better", loc="left")
        thousands(ax)
        spines(ax)
        ax.tick_params(labelsize=7.5)

    # The 6th cell carries the shared legend and the finding, instead of a sixth chart.
    legend_ax = flat[5]
    legend_ax.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper left", fontsize=9, title="arm", alignment="left")

    last = matched_steps(rows, ("control", "learned_mix"))[-1]
    lines = [f"at matched step {last // 1000}k,", "learned_mix vs control:", ""]
    for metric in METRICS:
        c = dict(series(rows, "control", metric))[last]
        m = dict(series(rows, "learned_mix", metric))[last]
        # "Improvement" in the metric's own better direction, so the signs are comparable.
        rel = (m - c) / abs(c) * (1 if HIGHER_IS_BETTER[metric] else -1) * 100
        lines.append(f"  {METRIC_LABEL[metric].split(' ')[0]:<7}{rel:+6.1f}%")
        stats.append(
            f"- **{METRIC_LABEL[metric]}** @ {last}: control {c:.6g}, learned_mix {m:.6g} "
            f"→ {rel:+.1f}% in the better direction"
        )
    legend_ax.annotate(
        "\n".join(lines),
        xy=(0.02, 0.60),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_SOFT,
        va="top",
        family="monospace",
    )

    by_tag = {r["tag"]: r for r in rows}
    stats.append("\nBaseline reference values, for the same metrics:\n")
    stats.append("| metric | plateau-272000 | anneal-304000 |")
    stats.append("|---|---|---|")
    for metric in METRICS:
        stats.append(
            f"| {METRIC_LABEL[metric]} | {by_tag['plateau-272000'][metric]:.6g} | "
            f"{by_tag['anneal-304000'][metric]:.6g} |"
        )

    fig.suptitle(
        "The metrics disagree: PSNR separates the arms, P-DINO does not",
        x=0.008,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, "\n".join(stats) + "\n"


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "metric_panel", ARCHIVE_FIGURES))
    print(stats)
