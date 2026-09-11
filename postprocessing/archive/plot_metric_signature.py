"""Figure: WHICH metrics each intervention moves -- ours, beside mira's own layer ablation.

Two different interventions, so this compares SIGNATURES and never magnitudes:

* mira's: aggregating 7 blocks vs reading only the deepest. PSNR barely notices (1.0%); the Frechet
  distances move 42-47%. PSNR is the least layer-sensitive metric they report.
* ours: learned_mix vs control at a matched 200,000 steps. PSNR moves most (+11.7%); P-DINO does
  not move at all (-0.8%, and its sign flips across the run).

A gain concentrated in the metric that is least sensitive to layer choice is not what finding a
better latent looks like. That is the whole reason this figure exists.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import json

import matplotlib.pyplot as plt
from lib import (
    ARCHIVE_FIGURES,
    DATA,
    INK_MUTED,
    INK_SOFT,
    apply_style,
    load_rows,
    matched_steps,
    save,
    series,
    spines,
)

IMPROVED = "#2a78d6"  # blue: moved in the metric's better direction
REGRESSED = "#e34948"  # red: moved the wrong way. Polarity, so a diverging pair.

PRETTY = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "p_dino": "P-DINO",
    "r_fid": "rFID",
    "r_fvd": "rFVD",
    "r_fdd": "rFDD",
    "g_fid": "gFID",
    "g_fvd": "gFVD",
    "g_fdd": "gFDD",
}
ORDER = ["psnr", "ssim", "lpips", "p_dino", "r_fid", "r_fvd", "r_fdd", "g_fid", "g_fvd", "g_fdd"]


def _rel(better: float, worse: float, higher_is_better: bool) -> float:
    """Percent by which `better` beats `worse`, in the metric's own good direction."""
    return (better - worse) / abs(worse) * (1 if higher_is_better else -1) * 100


def build():
    mira = json.loads((DATA / "mira_layer_ablation.json").read_text())
    good = mira["higher_is_better"]
    multi, last = mira["rows"]["multi_layer_mean"], mira["rows"]["last_block_only"]
    mira_rel = {m: _rel(multi[m], last[m], good[m]) for m in ORDER}

    rows = load_rows()
    step = matched_steps(rows, ("control", "learned_mix"))[-1]
    ours_rel = {}
    for m in ("psnr", "ssim", "lpips", "p_dino", "r_fdd"):
        c = dict(series(rows, "control", m))[step]
        lm = dict(series(rows, "learned_mix", m))[step]
        ours_rel[m] = _rel(lm, c, good[m])

    apply_style()
    fig, (ax_m, ax_o) = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)

    def draw(ax, rel: dict[str, float], keys: list[str], title: str, sub: str):
        ys = range(len(keys))
        vals = [rel[k] for k in keys]
        ax.barh(
            list(ys),
            vals,
            height=0.56,
            color=[IMPROVED if v >= 0 else REGRESSED for v in vals],
            linewidth=0,
            zorder=3,
        )
        ax.axvline(0, color=INK_MUTED, linewidth=0.9, alpha=0.7, zorder=4)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([PRETTY[k] for k in keys])
        ax.invert_yaxis()
        for y, v in zip(ys, vals, strict=True):
            # Value labels always sit on the positive side of the zero rule. A negative bar's own
            # side is where the y-axis tick labels live, so labelling outward collides with them.
            ax.annotate(
                f"{v:+.1f}%",
                xy=(max(v, 0.0), y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8,
                color=REGRESSED if v < 0 else INK_SOFT,
            )
        ax.set_title(title, loc="left", pad=26)
        ax.annotate(sub, xy=(0, 1.012), xycoords="axes fraction", fontsize=8, color=INK_MUTED, va="bottom")
        ax.grid(axis="y", visible=False)
        spines(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)

    draw(
        ax_m,
        mira_rel,
        ORDER,
        "mira: 7 blocks vs deepest block only",
        "published, full video codec (appendix tab:exp-enc-layers)",
    )
    draw(
        ax_o,
        ours_rel,
        list(ours_rel),
        "kmira: learned_mix vs control",
        f"this rig, image-only, matched step {step // 1000}k",
    )
    ax_o.set_xlabel("gain of the less-restricted arm, relative to the more-restricted one (%)", fontsize=8.5)
    ax_m.set_xlabel("gain of the less-restricted arm, relative to the more-restricted one (%)", fontsize=8.5)

    fig.suptitle(
        "Same subsystem, opposite metric signatures",
        x=0.008,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "Two different interventions, so only the SHAPE is comparable, never the magnitudes. "
        "Absolute values differ between setups and so do some metric normalisations.",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    stats = [
        "### Metric signature: relative improvement in each metric's better direction\n",
        "| metric | mira: 7 blocks vs deepest only | kmira: learned_mix vs control |",
        "|---|---|---|",
    ]
    for m in ORDER:
        ours = f"{ours_rel[m]:+.1f}%" if m in ours_rel else "not measured here"
        stats.append(f"| {PRETTY[m]} | {mira_rel[m]:+.1f}% | {ours} |")
    stats.append(
        "\nmira's numbers are transcribed from `sections/appendix.tex`, table "
        "`tab:exp-enc-layers`; see `data/mira_layer_ablation.json` for provenance and "
        "the comparability warning.\n"
    )
    return fig, "\n".join(stats)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "metric_signature", ARCHIVE_FIGURES))
    print(stats)
