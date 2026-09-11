"""Regenerate every figure and rewrite stats.md.

    pixi run python postprocessing/make_all.py

stats.md is the numeric source of truth for RESULTS.md: every number quoted in the prose there
must appear here, and both are produced from codec/results/benchmark.jsonl plus the committed
data/*.json. Nothing in this folder is typed in from memory.

Some figures need inputs that are not in benchmark.jsonl, written by the extract_*.py scripts from
local checkpoints, logs or the DINO weights. Their outputs live in data/ and ARE committed, so this
script runs on a fresh clone. Re-run extract_layer_weights.py after training more steps.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# (module, figure basename) in the order they appear in RESULTS.md: the foundation the benchmark
# rests on first, then the experiments built on top of it.
FIGURES = [
    ("plot_baseline_v2", "baseline_v2_progress"),
]

# Figures of the RETIRED baseline's studies. Their scripts still run and their findings survive
# qualitatively, but every absolute number in them rests on a run that saw 53.4% of the training
# data, so they are not regenerated into the live page -- they belong to
# archive/RESULTS-legacy-baseline.md. Pass --archive to rebuild them.
ARCHIVED_FIGURES = [
    ("plot_baseline_elbow", "baseline_elbow"),
    ("plot_seed_effects", "seed_effects"),
    ("plot_trajectories", "arm_trajectories"),
    ("plot_decomposition", "gap_decomposition"),
    ("plot_metric_panel", "metric_panel"),
    ("plot_metric_signature", "metric_signature"),
    ("plot_layer_weights", "layer_weight_shares"),
]


def setup_table() -> str:
    """The built codec's shape and size. Stats-only -- there is no figure, but RESULTS.md quotes
    these numbers, so they have to be generated like every other number here."""
    blob = json.loads((HERE / "data" / "setup.json").read_text())
    v, e, p_, lat = blob["video"], blob["encoder"], blob["params"], blob["latent"]
    return "\n".join(
        [
            "### The benchmarked codec, as configured\n",
            "| | value |",
            "|---|---|",
            f"| frame | {v['height']}×{v['width']}×{v['channels']} at {v['fps']} fps |",
            f"| timesteps per sample | {v['timesteps']} (image-only) |",
            f"| feature extractor | `{e['rae_model']}`, frozen |",
            f"| aggregated blocks | {e['aggregation_layers']} |",
            f"| latent channels | {e['latent_dim']} |",
            (
                f"| bottleneck stride | {e['bottleneck_stride']} spatial, "
                f"{e['bottleneck_temporal_stride']} temporal |"
            ),
            (
                f"| decoder (Base, used) | width {blob['decoder_base']['vit_width']}, "
                f"depth {blob['decoder_base']['vit_depth']}, heads {blob['decoder_base']['vit_num_heads']} |"
            ),
            (
                f"| decoder (XL, mira's stock) | width {blob['decoder_xl']['vit_width']}, "
                f"depth {blob['decoder_xl']['vit_depth']}, heads {blob['decoder_xl']['vit_num_heads']} — "
                f"{blob['decoder_xl']['trainable_params']:,} trainable, which is what OOMs 12GB |"
            ),
            (
                f"| parameters | {p_['total']:,} total = {p_['trainable']:,} trainable "
                f"+ {p_['frozen']:,} frozen (the DINOv3-L backbone) |"
            ),
            f"| latent grid | {lat['grid'][0]}×{lat['grid'][1]} tokens |",
            (
                f"| compression | {lat['values_per_frame_pixels']:,} → {lat['values_per_frame_latent']:,} "
                f"values per frame = **{lat['reduction_x']:g}×**, spatial only |"
            ),
            (
                f"| optimiser | AdamW lr {blob['optim']['lr']:g}, betas {blob['optim']['betas']}, "
                f"weight decay {blob['optim']['weight_decay']:g} |"
            ),
            "",
            (
                "Source: `data/setup.json`, written by `extract_setup_facts.py` from `codec/configs/` and "
                "the constructed model.\n"
            ),
        ]
    )


def run_metadata_table() -> str:
    """What each run actually did, from the logs. Stats-only, no figure.

    Includes the paired design's central claim as a CHECK rather than an assertion: two arms
    compared against each other must have drawn the same data at the same steps.
    """
    blob = json.loads((HERE / "data" / "run_metadata.json").read_text())
    lines = [
        "### Training runs, from their logs\n",
        (
            "Captured by `extract_run_metadata.py`. `benchmark.jsonl` holds the scored PSNR rows; this "
            "holds the per-step validation readings and the schedule/seed each chunk actually used, "
            "which the logs are the only record of.\n"
        ),
        "| run | val readings | scored PSNR rows | chunks | seeds | LR schedule |",
        "|---|---|---|---|---|---|",
    ]
    import lib  # local import: lib pulls in matplotlib, only needed when this runs

    rows = lib.load_rows()
    for name, rec in blob["runs"].items():
        prefix = {
            "plateau_baseline": "plateau",
            "warmstart_control": "control",
            "warmstart_learn7": "learn7",
            "warmstart_learned_mix": "learned_mix",
            "ablation_baseline": "abl_baseline",
            "ablation_frozen_bneck": "abl_frozen",
        }.get(name)
        n_psnr = len(lib.series(rows, prefix)) if prefix else sum(1 for r in rows if r["tag"] == name)
        seeds = rec["seeds_unique"]
        span = f"{seeds[0]}–{seeds[-1]} ({len(seeds)})" if seeds else "—"
        lines.append(
            f"| `{name}` | {len(rec['readings'])} | {n_psnr} | {len(rec['chunks'])} | "
            f"{span} | {rec['schedule']} |"
        )
    lines.append("")
    lines.append(
        "**Paired-design check.** Arms compared against each other must have drawn the "
        "same data at the same steps. Seeds are derived as `SEED_BASE + step/CHUNK`, so a "
        "seed identifies a slice of the stream:\n"
    )
    for pair, rec in blob.get("pairing", {}).items():
        lines.append(
            f"- `{pair}`: {len(rec['shared_seeds'])} shared seeds, identical over the shared "
            f"range: **{rec['identical_over_shared_range']}**"
        )
    lines.append("")
    lines.append(
        "Note `plateau_baseline` (RESULTS.md calls it *the baseline run*) reads as mixed "
        "because `run_anneal.sh` continues into that run's own output directory, so the log "
        "holds both its constant-LR chunks and the cosine ones.\n"
    )
    return "\n".join(lines)


def protocol_evidence() -> str:
    """The two quantitative claims RESULTS.md section 2 makes about the protocol.

    Stats-only: the figures they came from belong to the archived page, but the claims are about
    the RIG rather than about any one study, so they stay on the live page -- and therefore have to
    stay generated. Both read committed inputs only.
    """
    import plot_calibration
    import plot_seed_effects

    lines = ["### Why comparisons here are paired\n"]

    # 1. How uneven is one 8,000-step data slice? Measured as the PSNR a chunk bought, per arm.
    deltas = plot_seed_effects.seed_deltas()
    # Read the spread off a PAIRED arm, where a chunk's gain or loss cannot be the intervention.
    # Seeds 29-31 are excluded: they are the warm-restart transient (optimiser reset, LR raised
    # again), which is a property of restarting, not of the data.
    lo, hi = plot_seed_effects.RESTART
    paired = {s: v for s, v in deltas["warmstart_control"].items() if not lo <= s <= hi}
    ranked = sorted(paired.items(), key=lambda kv: kv[1], reverse=True)
    best, worst = ranked[0], ranked[-3:]
    lines += [
        (
            "Each 8,000-step chunk draws its stream from a per-chunk seed, so a seed identifies a "
            "slice of training data. The slices are not equivalent.\n"
        ),
        (
            f"Measured on `warmstart_control`, a paired arm over {len(paired)} chunks, so a chunk's "
            f"gain or loss cannot be the intervention:\n"
        ),
        f"- best slice: **{best[1]:+.3f} dB** in one chunk (seed {best[0]})",
        (
            "- worst slices: "
            + ", ".join(f"**{v:+.3f} dB** (seed {s})" for s, v in worst)
        ),
        (
            "- and the ranking is a property of the slice, not of the run: the same seeds are best "
            "and worst in runs that met them at completely different steps."
        ),
        "",
    ]

    # 2. The early calibration's effect-vs-noise ratio, which is why unpaired was abandoned.
    _, cal = plot_calibration.build()
    ratio = [ln for ln in cal.splitlines() if "effect / noise" in ln]
    lines += [
        (
            "An early three-run study tried to resolve a published ~1.4 dB bottleneck effect with "
            "one run per arm, and could not:\n"
        ),
        *ratio,
        "",
        (
            "That study is not reported as a result -- one run per arm cannot estimate a spread, "
            "and all three stopped undertrained. It is reported here only as the reason the "
            "protocol is paired.\n"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    from lib import save

    sections: list[str] = [
        f"<!-- from make_all.py -->\n\n{setup_table()}",
        f"<!-- from make_all.py -->\n\n{run_metadata_table()}",
        f"<!-- from make_all.py -->\n\n{protocol_evidence()}",
    ]
    figures = FIGURES + (ARCHIVED_FIGURES if "--archive" in sys.argv else [])
    for module_name, basename in figures:
        module = importlib.import_module(module_name)
        fig, stats = module.build()
        path = save(fig, basename)
        print(f"  {path.relative_to(HERE.parent)}")
        sections.append(f"<!-- from {module_name}.py -->\n\n{stats}")

    stamp = dt.datetime.now(tz=dt.UTC).date().isoformat()
    out = HERE / "stats.md"
    out.write_text(
        "# Generated numbers\n\n"
        f"**Generated by `make_all.py` on {stamp}. Do not edit.**\n\n"
        "Every figure in `RESULTS.md` is drawn from these values, and every number quoted in that\n"
        "file's prose appears here. Sources: `codec/results/benchmark.jsonl` (written by\n"
        "`kmira.benchmark.eval_codec` at scoring time) and `data/*.json`.\n\n"
        "---\n\n" + "\n\n---\n\n".join(sections)
    )
    print(f"  {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
