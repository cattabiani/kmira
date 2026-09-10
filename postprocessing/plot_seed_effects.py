"""Figure: the per-chunk data slice, not training dynamics, explains both "false plateaus".

Training here runs in hourly chunks of 8,000 steps, and each chunk resolves its own `run.seed`
as SEED_BASE + step/8000. mira's train loader reseeds from `run.seed` on every process start and
is not checkpointed, so a chunk's seed picks the entire 8,000-step stream that chunk trains on.
The seed is therefore an identity for a slice of data, and the same seed means the same slice in
every run that uses this schedule.

That makes a test possible. The plateau run and the warm-start arms hit the same seeds at
DIFFERENT step numbers and at very different training maturity. If a dip belongs to training
dynamics, it should track the step. If it belongs to the data, it should track the seed. Plot the
per-chunk PSNR increment against the seed and the answer is direct.

WHY THIS MATTERS BEYOND THE DIP. Section 0c originally read the plateau run's two flat stretches
as false plateaus in the optimisation -- the project's own cautionary tale about calling an elbow
too early. Both stretches, and both of the jumps that ended them, are seed-aligned and reproduce in
the control arm 100k+ steps away from where the plateau run saw them. The caution survives, but the
cause was the seed schedule.

Plateau is drawn as a gray reference rather than a fourth categorical hue: it is the baseline the
arms are read against, not another arm, and the palette's three slots are spoken for.
"""

from __future__ import annotations

import itertools
import json

import matplotlib.pyplot as plt
from lib import (
    COLOR,
    DATA,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    apply_style,
    load_rows,
    save,
    series,
    spines,
)

PLATEAU_C = "#6f6e69"  # gray: a reference series, deliberately not a categorical hue
RUNS = [
    ("plateau_baseline", "plateau", "plateau baseline", PLATEAU_C),
    ("warmstart_control", "control", "control", COLOR["control"]),
    ("warmstart_learn7", "learn7", "learn7", COLOR["learn7"]),
    ("warmstart_learned_mix", "learned_mix", "learned_mix", COLOR["learned_mix"]),
]
# The two stretches section 0c called false plateaus, and the two seeds that ended them.
POOR = [(36, 39), (47, 49)]
GOOD = [40, 50]
# The warm-start arms' first three chunks are their post-restart transient: the optimiser state was
# reset and the LR raised again, so they fall and climb back regardless of data. The plateau run
# never used these seeds, so there is no cross-run corroboration and they are NOT evidence about
# the data. Marked on the figure so the big seed-31 rebound cannot be misread as a seed effect.
RESTART = (29, 31)
# Where the runs had flattened and an elbow was actually being judged. Ranking a late chunk
# against the whole run buries it under the early steep ones.
LATE_FROM = 45


def seed_deltas() -> dict[str, dict[int, float]]:
    """``{run: {chunk seed: PSNR gained during that chunk}}``.

    A chunk trains up to step N and is scored there, so the increment scored AT step N is what that
    chunk bought, and the seed to attribute it to is the seed of the chunk that ended at N.
    """
    meta = json.loads((DATA / "run_metadata.json").read_text())["runs"]
    rows = load_rows()
    out: dict[str, dict[int, float]] = {}
    for run, prefix, _, _ in RUNS:
        # Chunk targets are recorded as `run.steps`, one past the step they stop at.
        seed_of = {c["steps"] - 1: c["seed"] for c in meta[run]["chunks"] if c.get("steps")}
        pts = series(rows, prefix)
        out[run] = {}
        for (_, prev), (step, cur) in itertools.pairwise(pts):
            seed = seed_of.get(step)
            if seed is not None:
                out[run][seed] = cur - prev
    return out


def build():
    deltas = seed_deltas()

    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 5.2))

    for lo, hi in POOR:
        ax.axvspan(lo - 0.45, hi + 0.45, color=GRID, zorder=0)
    for seed in GOOD:
        ax.axvline(seed, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax.axvspan(RESTART[0] - 0.5, RESTART[1] + 0.45, color="#f2e6dc", zorder=0)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=2)

    for run, _, label, color in RUNS:
        pts = sorted(deltas[run].items())
        if not pts:
            continue
        xs = [s for s, _ in pts]
        ys = [d for _, d in pts]
        ax.plot(xs, ys, color=color, marker="o", markersize=4.0, label=label, zorder=3)

    # Direct labels only where they can be placed without collision. control and learn7 track each
    # other so closely for the whole run that no position separates them -- which is itself worth
    # saying, so it is said instead of being worked around.
    ax.annotate(
        "plateau baseline",
        xy=(58, deltas["plateau_baseline"].get(58, 0)),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=PLATEAU_C,
        ha="center",
    )
    ax.annotate(
        "learned_mix",
        xy=(34, deltas["warmstart_learned_mix"].get(34, 0)),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR["learned_mix"],
    )
    ax.annotate(
        "control and learn7 lie on top of each other",
        xy=(44.5, -0.14),
        fontsize=8.5,
        fontweight="bold",
        color=COLOR["control"],
        ha="center",
        va="center",
    )

    ax.set_xlabel("chunk seed  (each seed is one fixed 8,000-step slice of the data stream)")
    ax.set_ylabel("PSNR gained during that chunk (dB)\n↑ the chunk helped")
    ax.set_title(
        "The same seeds help and hurt in every run, at completely different step numbers",
        loc="left",
    )
    ax.set_xlim(28.4, 63.5)
    leg = ax.legend(loc="upper right", fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(INK_SOFT)

    top = ax.get_ylim()[1]
    for lo, hi in POOR:
        ax.annotate(
            "0c: \u201cfalse plateau\u201d",
            xy=((lo + hi) / 2, top * 0.86),
            fontsize=7.8,
            color=INK_SOFT,
            ha="center",
            va="center",
        )
    ax.annotate(
        "warm-restart transient,\nnot data — the plateau run\nnever used these seeds",
        xy=(28.7, top * 0.92),
        fontsize=7.6,
        color="#a2603a",
        ha="left",
        va="top",
    )
    for seed in GOOD:
        ax.annotate(
            f"seed {seed}",
            xy=(seed, top),
            xytext=(0, -3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=INK,
            ha="center",
            va="top",
        )
    spines(ax)

    fig.suptitle(
        "Both \u201cfalse plateaus\u201d were the data schedule",
        x=0.008,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.945))
    fig.text(
        0.012,
        0.02,
        "A chunk's seed picks its whole 8,000-step stream, so the seed identifies a slice of data. "
        "The plateau run met these seeds at different steps and\ndifferent maturity from the "
        "warm-start arms, so a dip that tracks the seed rather than the step belongs to the data, "
        "not to the optimiser.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    return fig, stats_for(deltas)


def stats_for(deltas: dict[str, dict[int, float]]) -> str:
    lines = [
        "### Per-chunk seed effects\n",
        (
            "PSNR gained during the chunk trained with each seed. A chunk's seed picks its entire "
            "8,000-step stream, so the same seed is the same slice of data in every run using this "
            "schedule — and the runs met these seeds at different step numbers.\n"
        ),
        "| chunk seed | " + " | ".join(label for _, _, label, _ in RUNS) + " |",
        "|---|" + "---|" * len(RUNS),
    ]
    seeds = sorted({s for d in deltas.values() for s in d})
    for seed in seeds:
        cells = []
        for run, _, _, _ in RUNS:
            d = deltas[run].get(seed)
            cells.append(f"{d:+.3f}" if d is not None else "—")
        mark = ""
        if any(lo <= seed <= hi for lo, hi in POOR):
            mark = "  ←  read as a false plateau"
        elif seed in GOOD:
            mark = "  ←  the jump that ended it"
        lines.append(f"| {seed} | " + " | ".join(cells) + f" |{mark}")
    lines.append("")

    for seed in GOOD:
        vals = {label: deltas[run].get(seed) for run, _, label, _ in RUNS}
        got = {k: v for k, v in vals.items() if v is not None}
        # Two ranks, because they answer different questions. Over the whole run, the early steep
        # chunks dominate and a late-run standout looks unremarkable; restricted to the flat second
        # half, which is where the elbow was being judged, it is the standout.
        ranks, late = [], []
        for run, _, label, _ in RUNS:
            d = deltas[run]
            if seed not in d:
                continue
            ranks.append(f"{label} {sorted(d.values(), reverse=True).index(d[seed]) + 1}/{len(d)}")
            tail = {s: v for s, v in d.items() if s >= LATE_FROM}
            if seed in tail:
                pos = sorted(tail.values(), reverse=True).index(tail[seed]) + 1
                late.append(f"{label} {pos}/{len(tail)}")
        lines.append(
            f"- **seed {seed}**: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in got.items())
            + f" dB. Rank among that run's chunks: {', '.join(ranks)}"
            + (f"; among seeds {LATE_FROM}+ only: {', '.join(late)}." if late else ".")
        )
    poor_note = []
    for lo, hi in POOR:
        for run, _, label, _ in RUNS:
            vals = [deltas[run][s] for s in range(lo, hi + 1) if s in deltas[run]]
            if vals:
                poor_note.append(f"{label} {min(vals):+.3f}..{max(vals):+.3f}")
        lines.append(f"- **seeds {lo}–{hi}**: " + ", ".join(poor_note) + " dB.")
        poor_note = []
    lines.append(
        "\nThe plateau run spent its first seven chunks (steps 0–56,000) on seed 28 — the same "
        "slice replayed seven times, before the per-chunk seed schedule existed. Fresh seeds begin "
        "at its step 64,000, which is why it has no reading for seeds 29–35.\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    fig, stats = build()
    print(save(fig, "seed_effects"))
