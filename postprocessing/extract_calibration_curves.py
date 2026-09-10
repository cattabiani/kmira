"""Extraction step: the calibration arms' validation curves, parsed out of their training logs.

`benchmark.jsonl` holds exactly ONE scored row per calibration arm -- no intermediate checkpoints
were ever scored for A/B/C -- so the only per-step record of those runs is mira's own validation
loop, printed into `checkpoints/calibration/*.log`. Those logs are under a gitignored directory, so
this parses them into a few kB of committed `data/calibration_curves.json`.

NOTE what this is and is not. These are mira's VALIDATION LOSSES (512 samples, its own val loop),
not PSNR. PSNR comes from `eval_codec` on 2048 held-out frames and exists only at the final step.
The two are different quantities on different sample sets and must not be plotted on one axis or
described interchangeably.

    pixi run python postprocessing/extract_calibration_curves.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import DATA, REPO

ARMS = ("A_baseline", "B_frozen_bneck", "C_baseline_seed2")
LINE = re.compile(
    r"Validation at step (\d+): "
    r"loss_mae=([\d.]+), "
    r"loss_lpips_perceptual=([\d.]+), "
    r"loss_dino_latent_consistency=([\d.]+), "
    r"loss_total=([\d.]+)"
)


def main() -> None:
    out: dict = {
        "_source": "extract_calibration_curves.py, parsed from checkpoints/calibration/*.log",
        "_what": "mira's own validation losses (512 samples) during the three calibration runs",
        "_warning": (
            "Validation loss, NOT PSNR. PSNR for these arms exists only at the final step, in "
            "benchmark.jsonl, scored by eval_codec on 2048 frames. Different quantity, different "
            "sample set -- never put them on one axis."
        ),
        "arms": {},
    }
    for arm in ARMS:
        log = REPO / "checkpoints" / "calibration" / f"{arm}.log"
        if not log.exists():
            print(f"  {arm}: no log at {log}, skipping")
            continue
        rows = []
        for line in log.read_text(errors="ignore").splitlines():
            m = LINE.search(line)
            if m:
                rows.append(
                    {
                        "step": int(m.group(1)),
                        "loss_mae": float(m.group(2)),
                        "loss_lpips_perceptual": float(m.group(3)),
                        "loss_dino_latent_consistency": float(m.group(4)),
                        "loss_total": float(m.group(5)),
                    }
                )
        # A restarted run can log the same step twice; keep the last reading for each.
        dedup = {r["step"]: r for r in rows}
        out["arms"][arm] = {
            "log": str(log.relative_to(REPO)),
            "readings": [dedup[s] for s in sorted(dedup)],
        }
        print(f"  {arm}: {len(dedup)} validation readings, steps {min(dedup)}..{max(dedup)}")

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "calibration_curves.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
