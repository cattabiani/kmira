"""Summarise a learned-layer-mix warm-start run: PSNR trajectory, and what the weights actually did.

Run by run_learned_layer_mix_warmstart.sh; standalone-runnable at any time (including mid-run) to
read a partial result:

    ARMS=learned_mix pixi run python codec/scripts/report_layer_mix.py

The weights section is the part that does not need a control arm. The variant's PSNR after a warm
restart is confounded by the restart itself -- the baseline was annealed to its minimum LR and
finetune_from raises it again, which costs quality before it regains any. The 24 aggregation weights
are not confounded that way: they start at exactly the stock formula's values, and where they move
is a direct read on whether training wants a different layer combination at all. Weights that barely
move say the hand-picked formula was already at a local optimum, whatever PSNR is doing.
"""

from __future__ import annotations

import json
import os
import pathlib

import torch

from kmira.codec.variants.learned_layer_mix import (
    DINO_L_DEPTH,
    STOCK_LAYERS,
    stock_equivalent_weights,
)

# Two baseline numbers, and using the wrong one makes the whole run unreadable.
#
# CONSTANT_LR_PLATEAU is what a warm-started run at constant LR 1e-4 should be compared against: the
# equilibrium the baseline sat at under exactly that LR, reached at step 272,000 and flat there
# (24.736 -> 24.747 over the last 8,000 steps). LOCKED_BASELINE is the same model after cosine
# annealing 1e-4 -> 1e-6 over 32,000 steps, which bought +0.138 dB. That gain is a property of the
# LOW LR, not of a better parameter region: raise the LR back to 1e-4 and it is given straight back.
# A constant-LR run can therefore never reach 24.885, no matter how long it runs, and "it is still
# below 24.885" says nothing about the experiment. See AGENTS.md's gotcha on annealed vs
# constant-LR baselines.
CONSTANT_LR_PLATEAU = 24.747
LOCKED_BASELINE_PSNR = 24.885
ANNEAL_GAIN = LOCKED_BASELINE_PSNR - CONSTANT_LR_PLATEAU
BENCHMARK = pathlib.Path("codec/results/benchmark.jsonl")


def _trajectory(prefix: str) -> list[tuple[int, float]]:
    if not BENCHMARK.exists():
        return []
    rows = [json.loads(line) for line in BENCHMARK.read_text().splitlines() if line.strip()]
    out = [
        (int(r["tag"].split("-")[-1]), float(r["psnr"]))
        for r in rows
        if r["tag"].startswith(f"{prefix}-") and r["tag"].split("-")[-1].isdigit()
    ]
    return sorted(set(out))


def report_psnr(arms: list[str]) -> None:
    trajectories = {arm: _trajectory(arm) for arm in arms}
    if not any(trajectories.values()):
        print("no scored checkpoints yet")
        return

    print(f"baseline @ constant LR 1e-4 (step 272,000) : {CONSTANT_LR_PLATEAU:.3f} dB   <-- compare to THIS")
    print(f"baseline after annealing to 1e-6           : {LOCKED_BASELINE_PSNR:.3f} dB   (+{ANNEAL_GAIN:.3f}, LR effect only)")
    print()

    if len(arms) == 2 and all(trajectories.values()):
        control, learned = trajectories["control"], trajectories["learned_mix"]
        by_step = {s: p for s, p in control}
        print(f"{'step':>8}{'control':>10}{'learned_mix':>14}{'delta':>9}")
        for step, psnr in learned:
            if step in by_step:
                print(f"{step:>8}{by_step[step]:>10.3f}{psnr:>14.3f}{psnr - by_step[step]:>+9.3f}")
        if control and learned:
            print()
            print(
                f"final: control={control[-1][1]:.3f}  learned_mix={learned[-1][1]:.3f}  "
                f"delta={learned[-1][1] - control[-1][1]:+.3f} dB"
            )
            print()
            print("Paired (same seed, schedule, warm-start checkpoint, and pinned 7-layer loss), so")
            print("the delta is attributable to the aggregation change alone.")
        return

    for arm, traj in trajectories.items():
        if not traj:
            continue
        print(f"{arm}:")
        print(f"{'step':>8}{'PSNR':>10}{'vs plateau':>13}{'vs annealed':>14}")
        for step, psnr in traj:
            print(
                f"{step:>8}{psnr:>10.3f}"
                f"{psnr - CONSTANT_LR_PLATEAU:>+13.3f}{psnr - LOCKED_BASELINE_PSNR:>+14.3f}"
            )
        print()
    print("Read the 'vs plateau' column. This run is at constant LR 1e-4, so 24.747 is the")
    print("equilibrium it is heading for; 24.885 is unreachable at this LR by construction and the")
    print("last column is shown only so it is not mistaken for a target. The first readings should")
    print("FALL from 24.885 toward ~24.75 as the raised LR gives back the anneal -- that is the")
    print("restart settling, not the idea failing. What matters is where it settles relative to")
    print("+0.000, and beating the plateau needs an anneal at the end to become a headline number.")


def report_weights(checkpoint_dir: pathlib.Path) -> None:
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*/checkpoint.pth"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not checkpoints:
        print(f"\nno learned_mix checkpoints under {checkpoint_dir}")
        return

    latest = checkpoints[-1]
    state = torch.load(latest, map_location="cpu", weights_only=False)["state_dict"]
    key = "encoder.layer_weights"
    if key not in state:
        print(f"\n{latest.parent.name}: no {key} (frozen-control checkpoint?)")
        return

    learned = state[key].float()
    init = stock_equivalent_weights()
    moved = learned - init

    print()
    print("=" * 58)
    print(f" learned aggregation weights @ {latest.parent.name}")
    print("=" * 58)
    print(f"{'layer':>6}{'init':>10}{'learned':>10}{'moved':>10}   {'':<12}")
    for layer in range(DINO_L_DEPTH):
        mark = "stock" if layer in STOCK_LAYERS else ""
        print(f"{layer:>6}{init[layer]:>10.4f}{learned[layer]:>10.4f}{moved[layer]:>+10.4f}   {mark:<12}")

    off_stock = torch.tensor([moved[i] for i in range(DINO_L_DEPTH) if i not in STOCK_LAYERS])
    print()
    print(f"max |move|                : {moved.abs().max():.4f}")
    print(f"L2 |move|                 : {moved.norm():.4f}   (init L2 = {init.norm():.4f})")
    print(f"max |move| off stock layers: {off_stock.abs().max():.4f}")
    print(f"weight mass on the 17 previously-unused layers: {off_stock.abs().sum():.4f}")
    print()
    print("Interpretation: near-zero movement means the gradient does not want a different layer")
    print("combination, and the hand-picked formula was already near a local optimum. Substantial")
    print("mass appearing on previously-unused layers is the positive signal -- it means training")
    print("found information in layers the stock recipe discards.")


def main() -> None:
    arms = [a for a in os.environ.get("ARMS", "learned_mix").split(",") if a]
    report_psnr(arms)
    if "learned_mix" in arms:
        report_weights(
            pathlib.Path(
                os.environ.get("LEARNED_CKPT_DIR", "checkpoints/calibration/warmstart_learned_mix")
            )
        )


if __name__ == "__main__":
    main()
