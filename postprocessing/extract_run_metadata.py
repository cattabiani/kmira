"""Extraction step: everything the training logs already record, into committed JSON.

The logs under `checkpoints/calibration/*.log` hold far more than `benchmark.jsonl` does, and all
of it is gitignored and one `rm` from gone. For the plateau run alone that is **308 validation
readings** against 34 scored PSNR points -- roughly 9x the resolution -- plus one resolved-config
snapshot per chunk restart recording the seed and the LR schedule actually used.

Keeping it costs tens of kB. Not keeping it cost this project a figure already: the three
calibration arms can never be given a PSNR curve, because `checkpoint_keep_recent: 1` deleted the
weights and nothing else on disk can reconstruct the score.

What this captures, per run:

* every ``Validation at step`` reading -- step and the four loss terms
* every chunk's resolved ``run.seed``, ``run.steps`` and ``optim.scheduler`` block, so the schedule
  a run actually used is recorded rather than inferred from whichever launcher was in play (the
  calibration arms cosine-anneal while the plateau run holds constant LR, which is exactly the kind
  of difference that is invisible later and invalidates comparisons)

What it deliberately does NOT capture: PSNR. That only exists where a launcher scored a checkpoint
inline, and it lives in `codec/results/benchmark.jsonl`, which is already committed.

    pixi run python postprocessing/extract_run_metadata.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import DATA, REPO

LOGS = REPO / "checkpoints" / "calibration"

VAL = re.compile(
    r"Validation at step (\d+): "
    r"loss_mae=([\d.]+), loss_lpips_perceptual=([\d.]+), "
    r"loss_dino_latent_consistency=([\d.]+), loss_total=([\d.]+)"
)
NEW_CHUNK = re.compile(r"Training configuration:")
# Exact indent plus key, because `steps` and `decay_steps` both end in "steps" and `run.seed`
# must not be confused with anything else.
SCALARS = {
    "seed": re.compile(r"^  seed: (\d+)$"),
    "steps": re.compile(r"^  steps: (\d+)$"),
    "lr": re.compile(r"^    lr: ([\d.e+-]+)$"),
    "warmup_steps": re.compile(r"^    warmup_steps: (\d+)$"),
    "constant_steps": re.compile(r"^    constant_steps: (\d+)$"),
    "decay_steps": re.compile(r"^    decay_steps: (\d+)$"),
    "min_lr": re.compile(r"^    min_lr: ([\d.e+-]+)$"),
}


def parse(log: pathlib.Path) -> dict:
    chunks: list[dict] = []
    readings: dict[int, dict] = {}
    for raw in log.read_text(errors="ignore").splitlines():
        line = raw.split("] - ", 1)[-1] if "] - " in raw else raw
        if NEW_CHUNK.search(line):
            chunks.append({})
            continue
        if chunks:
            for key, pat in SCALARS.items():
                m = pat.match(line)
                if m and key not in chunks[-1]:
                    chunks[-1][key] = m.group(1)
        m = VAL.search(line)
        if m:
            # A restart re-validates a step already seen; the later reading wins.
            readings[int(m.group(1))] = {
                "step": int(m.group(1)),
                "loss_mae": float(m.group(2)),
                "loss_lpips_perceptual": float(m.group(3)),
                "loss_dino_latent_consistency": float(m.group(4)),
                "loss_total": float(m.group(5)),
            }
    for c in chunks:
        for k in ("seed", "steps", "warmup_steps", "constant_steps", "decay_steps"):
            if k in c:
                c[k] = int(c[k])
        for k in ("lr", "min_lr"):
            if k in c:
                c[k] = float(c[k])
    # A single boolean is wrong here: run_anneal.sh continues into the plateau run's own output
    # dir, so that log holds constant-LR chunks AND cosine ones. Report the mix.
    decays = [c.get("decay_steps") for c in chunks if "decay_steps" in c]
    if not decays:
        schedule = "unknown"
    elif all(d == 0 for d in decays):
        schedule = "constant"
    elif all(d and d > 0 for d in decays):
        schedule = "cosine"
    else:
        schedule = (
            f"mixed ({sum(1 for d in decays if d == 0)} constant, {sum(1 for d in decays if d)} cosine)"
        )

    # Seeds identify data, not chunks: the launchers derive seed = SEED_BASE + step/CHUNK, so a
    # given seed always means the same slice of the stream. Two arms sharing a seed SET are
    # therefore paired on data even if they were run in a different number of sessions.
    seeds = sorted({c["seed"] for c in chunks if "seed" in c})
    return {
        "log": str(log.relative_to(REPO)),
        "chunks": chunks,
        "seeds_unique": seeds,
        "seed_schedule": [c["seed"] for c in chunks if "seed" in c],
        "schedule": schedule,
        "readings": [readings[s] for s in sorted(readings)],
    }


def main() -> None:
    out: dict = {
        "_source": "extract_run_metadata.py, parsed from checkpoints/calibration/*.log",
        "_what": (
            "Per-run validation readings and the resolved seed / LR schedule of every chunk. "
            "The logs are gitignored; this is the committed copy."
        ),
        "_not_here": (
            "PSNR. It exists only where a launcher scored a checkpoint inline, in "
            "codec/results/benchmark.jsonl, which is already committed."
        ),
        "runs": {},
    }
    for log in sorted(LOGS.glob("*.log")):
        rec = parse(log)
        if not rec["readings"] and not rec["chunks"]:
            continue
        out["runs"][log.stem] = rec
        seeds = rec["seeds_unique"]
        span = f"{seeds[0]}..{seeds[-1]} ({len(seeds)})" if seeds else "-"
        print(
            f"  {log.stem:24s} {len(rec['readings']):>4} readings, "
            f"{len(rec['chunks']):>3} chunks, seeds {span}, {rec['schedule']}"
        )

    # The paired design's central claim, checked rather than asserted: arms compared against each
    # other must have drawn the same data at the same steps.
    pairs = [("warmstart_control", "warmstart_learned_mix"), ("warmstart_control", "warmstart_learn7")]
    out["pairing"] = {}
    for x, y in pairs:
        if x in out["runs"] and y in out["runs"]:
            sx, sy = set(out["runs"][x]["seeds_unique"]), set(out["runs"][y]["seeds_unique"])
            shared = sorted(sx & sy)
            out["pairing"][f"{x} vs {y}"] = {
                "shared_seeds": shared,
                "identical_over_shared_range": sorted(s for s in sx if s <= max(shared)) == shared
                and sorted(s for s in sy if s <= max(shared)) == shared,
            }
            ok = out["pairing"][f"{x} vs {y}"]["identical_over_shared_range"]
            print(
                f"  pairing {x.replace('warmstart_', '')} / {y.replace('warmstart_', '')}: "
                f"{len(shared)} shared seeds, identical over the shared range: {ok}"
            )

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "run_metadata.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(REPO)}  ({kb:.0f} kB)")


if __name__ == "__main__":
    main()
