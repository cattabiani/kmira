"""Extraction step: pull `encoder.layer_weights` out of each arm's latest checkpoint.

Separate from the plotting because `checkpoints/` is gitignored and each `.pth` is ~4.4GB. This
writes a few hundred bytes of JSON under `data/`, which IS committed, so `plot_layer_weights.py`
runs on a fresh clone with no checkpoints and no GPU.

    pixi run python postprocessing/extract_layer_weights.py

Re-run it after training more steps. The output records which checkpoint each vector came from, so
a stale figure is detectable rather than silent.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import ARMS, DATA, REPO

from kmira.codec.variants.learned_layer_mix import (
    DINO_L_DEPTH,
    STOCK_LAYERS,
    stock_equivalent_weights,
)


def latest_checkpoint(arm: str) -> pathlib.Path | None:
    d = REPO / "checkpoints" / "calibration" / f"warmstart_{arm}"
    found = sorted(d.glob("checkpoint-*/checkpoint.pth"), key=lambda p: int(p.parent.name.split("-")[1]))
    return found[-1] if found else None


def main() -> None:
    out = {"_source": "extract_layer_weights.py, from checkpoints/calibration/warmstart_*", "arms": {}}
    for arm in ARMS:
        ckpt = latest_checkpoint(arm)
        if ckpt is None:
            print(f"  {arm}: no checkpoint found, skipping")
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        key = "encoder.layer_weights"
        if key not in state:
            print(f"  {arm}: no {key} in {ckpt.parent.name}, skipping")
            continue
        w = state[key].float().tolist()

        # Which DINOv3 blocks the arm reads is a constructor argument, absent from the state_dict,
        # but the vector's length determines it: every arm reads either all 24 or exactly the
        # stock 7. Weights are indexed by POSITION in that set, so record the block numbers too --
        # position and block index only coincide for the all-24 arms.
        if len(w) == DINO_L_DEPTH:
            blocks = list(range(DINO_L_DEPTH))
        elif len(w) == len(STOCK_LAYERS):
            blocks = list(STOCK_LAYERS)
        else:
            print(f"  {arm}: unexpected width {len(w)}, skipping")
            continue

        out["arms"][arm] = {
            "checkpoint": str(ckpt.relative_to(REPO)),
            "step": int(ckpt.parent.name.split("-")[1]),
            "blocks": blocks,
            "weights": w,
            "init_weights": stock_equivalent_weights(blocks).tolist(),
        }
        print(f"  {arm}: step {out['arms'][arm]['step']}, {len(w)} weights over blocks {blocks}")

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "layer_weights.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
