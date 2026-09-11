"""Figure: where each arm put its aggregation mass, on a shared DINOv3-block axis.

Reads data/layer_weights.json (written by extract_layer_weights.py), so it needs no checkpoints.

Shares, never raw magnitudes. The aggregation has an exact scaling symmetry with the bottleneck
projection -- scale the weights down and the projection up and the latent is unchanged -- and
weight decay resolves it arbitrarily, so only the normalised direction is identifiable. Reading a
raw magnitude here is how "92% on the non-stock blocks" once got written up as "92% on layer 0".
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import json

import matplotlib.pyplot as plt
from lib import (
    ARCHIVE_FIGURES,
    COLOR,
    DATA,
    INK_MUTED,
    INK_SOFT,
    RECESSIVE,
    apply_style,
    save,
    spines,
)

STOCK = (11, 13, 15, 17, 19, 21, 23)
DEPTH = 24
PANEL_NOTE = {
    "control": "weights frozen — this IS mira's stock formula",
    "learn7": "reads only mira's 7 blocks; weights learned",
    "learned_mix": "reads all 24 blocks; weights learned",
}


def build():
    blob = json.loads((DATA / "layer_weights.json").read_text())
    arms = [a for a in ("control", "learn7", "learned_mix") if a in blob["arms"]]
    apply_style()
    fig, axes = plt.subplots(len(arms), 1, figsize=(8.2, 2.35 * len(arms)), sharex=True)
    axes = [axes] if len(arms) == 1 else list(axes)

    stats = ["### Normalised |weight| share by DINOv3 block\n"]
    for ax, arm in zip(axes, arms, strict=True):
        rec = blob["arms"][arm]
        total = sum(abs(w) for w in rec["weights"]) or 1.0
        share = {b: 100 * abs(w) / total for b, w in zip(rec["blocks"], rec["weights"], strict=True)}
        top_block = max(share, key=share.get)

        # Emphasis, not a value-ramp: every bar is the same recessive grey and only the dominant
        # block takes the arm's colour. Colouring by magnitude would double-encode bar height.
        for b in range(DEPTH):
            if b not in share:
                continue
            ax.bar(
                b,
                share[b],
                width=0.68,
                color=COLOR[arm] if b == top_block else RECESSIVE,
                linewidth=0,
                zorder=3,
            )
        ax.annotate(
            f"L{top_block}  {share[top_block]:.1f}%",
            xy=(top_block, share[top_block]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=COLOR[arm],
        )

        unread = [b for b in range(DEPTH) if b not in share]
        if unread:
            for b in unread:
                ax.annotate(
                    "·",
                    xy=(b, 0),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK_MUTED,
                )

        ax.set_title(f"{arm} — {PANEL_NOTE[arm]}", loc="left", fontsize=9.5)
        ax.set_ylabel("share of |weight|")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75])
        ax.grid(axis="x", visible=False)
        spines(ax)

        off = sum(v for b, v in share.items() if b not in STOCK)
        l2 = sum(w * w for w in rec["weights"]) ** 0.5
        l2_init = sum(w * w for w in rec["init_weights"]) ** 0.5
        stats.append(
            f"\n**{arm}** @ step {rec['step']:,} (`{rec['checkpoint']}`) — "
            f"largest block L{top_block} at {share[top_block]:.1f}%; "
            f"{off:.1f}% of mass on the 17 non-stock blocks, {100 - off:.1f}% on mira's 7. "
            f"Raw vector L2 {l2:.4f}, from an init of {l2_init:.4f} "
            f"(unidentified scale — compare shares, not magnitudes).\n"
        )
        stats.append("| block | " + " | ".join(f"L{b}" for b in sorted(share)) + " |")
        stats.append("|---" * (len(share) + 1) + "|")
        stats.append("| share % | " + " | ".join(f"{share[b]:.1f}" for b in sorted(share)) + " |")

    axes[-1].set_xlabel("DINOv3-L block index   (0 = shallowest, 23 = deepest)", labelpad=26)
    axes[-1].set_xticks(range(DEPTH))
    axes[-1].set_xticklabels(
        [str(b) for b in range(DEPTH)],
        fontsize=7.5,
    )
    # Mark mira's own seven blocks under the axis, so "stock" is carried by position rather than by
    # spending a second colour on it.
    for b in STOCK:
        axes[-1].annotate(
            "▲",
            xy=(b, 0),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=5.5,
            color=INK_SOFT,
            annotation_clip=False,
        )
    fig.suptitle(
        "Each arm puts its mass on the shallowest block it is allowed to read",
        x=0.125,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.text(
        0.5,
        0.004,
        "▲ = one of mira's 7 aggregated blocks {11,13,15,17,19,21,23}      · = block this arm does not read",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
    )
    return fig, "\n".join(stats) + "\n"


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "layer_weight_shares", ARCHIVE_FIGURES))
