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
    # plot_calibration is deliberately NOT here. Its three-run study cannot support the claim it
    # was meant to support (see RESULTS.md "Not on this page yet"), so the figure is not published
    # while the page has no calibration section. The script stays, ready for the paired
    # recalibration runs -- deleting it would throw away working code for a study that is queued.
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


def main() -> None:
    from lib import save

    sections: list[str] = [
        f"<!-- from make_all.py -->\n\n{setup_table()}",
        f"<!-- from make_all.py -->\n\n{run_metadata_table()}",
    ]
    for module_name, basename in FIGURES:
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
