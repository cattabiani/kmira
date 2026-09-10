"""Shared data loading, palette and matplotlib style for the result figures.

Two rules this module exists to enforce:

1. **Every number in every figure is read from a file, never typed in.** The metric source is
   ``codec/results/benchmark.jsonl``, written by ``kmira.benchmark.eval_codec`` at scoring time.
   The one exception is transcribed published numbers from mira's own paper, which live in
   ``data/mira_layer_ablation.json`` with their source path recorded in the file itself.
2. **Plotting never needs a checkpoint.** ``checkpoints/`` is gitignored, so anything reading a
   ``.pth`` belongs in an extraction step that writes small committed JSON under ``data/``. That
   keeps every ``plot_*.py`` runnable on a fresh clone.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK = REPO / "codec" / "results" / "benchmark.jsonl"
FIGURES = REPO / "postprocessing" / "figures"
DATA = REPO / "postprocessing" / "data"

# The three Experiment 1/2 arms, in the order they nest: control can do neither, learn7 can move
# weights but only over mira's 7 blocks, learned_mix can also reach the other 17.
ARMS = ("control", "learn7", "learned_mix")
ARM_LABEL = {
    "control": "control",
    "learn7": "learn7",
    "learned_mix": "learned_mix",
}

# Categorical slots 1-3 of the validated default palette. Validated as a set for this use:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
#   -> all checks PASS; worst all-pairs CVD deltaE 9.2, normal-vision 24.0.
# One WARN: aqua sits at 2.74:1 against the surface, below 3:1, which obligates "relief" -- so
# every series carries a visible direct label AND the numbers exist as a table view in stats.md.
# Colour follows the arm, never its rank, so these stay fixed across every figure.
COLOR = {"control": "#2a78d6", "learn7": "#eb6834", "learned_mix": "#1baf7a"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8981"
GRID = "#e8e7e3"
RECESSIVE = "#d6d5cf"  # de-emphasised bars, when one mark is highlighted

# Two baselines the arms are read against. Both come from benchmark.jsonl (tags plateau-272000 and
# anneal-304000); resolved at runtime by baselines() rather than hardcoded here.
PLATEAU_TAG = "plateau-272000"
ANNEAL_TAG = "anneal-304000"

# Whether higher is better, per metric key in benchmark.jsonl. Used to label panels and to give
# "improved" a defined direction instead of assuming it.
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "p_dino": False, "r_fdd": False}
METRIC_LABEL = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "p_dino": "P-DINO",
    "r_fdd": "rFDD",
}


def load_rows() -> list[dict]:
    """Every scored checkpoint, as written by eval_codec."""
    if not BENCHMARK.exists():
        raise FileNotFoundError(f"no benchmark at {BENCHMARK}")
    return [json.loads(line) for line in BENCHMARK.read_text().splitlines() if line.strip()]


def series(rows: list[dict], arm: str, metric: str = "psnr") -> list[tuple[int, float]]:
    """``[(step, value)]`` for one arm, sorted by step, deduplicated on step.

    Tag format is ``<arm>-<step>``. Matching on that prefix would make ``learned_mix-*`` a
    substring hazard for any future arm named with a shared prefix, so the step must parse as an
    integer and the arm must match exactly.
    """
    out: dict[int, float] = {}
    for row in rows:
        arm_part, _, step_part = row["tag"].rpartition("-")
        if arm_part == arm and step_part.isdigit():
            out[int(step_part)] = float(row[metric])
    return sorted(out.items())


def matched_steps(rows: list[dict], arms: tuple[str, ...], metric: str = "psnr") -> list[int]:
    """Steps at which EVERY named arm has been scored.

    Arm-to-arm numbers are only quotable at matched steps -- an arm scored further along than
    another is not a comparison, since all three arms were still improving when last measured.
    """
    per_arm = [{s for s, _ in series(rows, arm, metric)} for arm in arms]
    return sorted(set.intersection(*per_arm)) if per_arm else []


def baselines(rows: list[dict]) -> dict[str, float]:
    """The constant-LR plateau and the annealed baseline, read from the record.

    The plateau (24.747) is what a constant-LR warm start is read against; the annealed number
    (24.885) is shown only so it is not mistaken for a target. Neither is a ceiling -- the control
    passed both by step 200,000.
    """
    by_tag = {row["tag"]: float(row["psnr"]) for row in rows}
    return {"plateau": by_tag[PLATEAU_TAG], "anneal": by_tag[ANNEAL_TAG]}


def apply_style() -> None:
    """Thin marks, hairline solid grid, explicit light surface."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            # Never transparent: these render on README pages that may be dark, and a transparent
            # ground would borrow the host's background and strand the dark ink on it.
            "savefig.transparent": False,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # solid: dashing reads as "threshold" when it is just a grid
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "legend.frameon": False,
            "figure.dpi": 130,
        }
    )


def spines(ax, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def thousands(ax) -> None:
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")


def save(fig, name: str) -> pathlib.Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
