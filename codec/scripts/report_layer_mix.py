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
# LOW LR, not of a better parameter region: raise the LR back to 1e-4 and it is given straight back,
# so "it is still below 24.885" early in a warm start says nothing about the experiment.
#
# Both numbers are stopping points, NOT asymptotes, and this comment used to claim otherwise. When
# Experiment 1's control was finally run to a matched 200,000 steps it passed both, finishing at
# 24.992 -- constant LR was still buying ~+0.03 dB per 8k steps out there. So prefer the PAIRED
# delta against `control` below over either constant; these two are context for reading a single
# arm, not a yardstick. See AGENTS.md's gotcha on annealed vs constant-LR baselines.
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
    print(
        f"baseline after annealing to 1e-6           : {LOCKED_BASELINE_PSNR:.3f} dB   (+{ANNEAL_GAIN:.3f}, LR effect only)"
    )
    print()

    # Any arm paired against `control` is reported as a delta at MATCHED STEPS, which is the number
    # this design supports. Not "gain over the 24.747 plateau": the control does not sit exactly on
    # the plateau (it finished Experiment 1 at 24.992, i.e. +0.245 above it), so the plateau version
    # silently credits the idea with whatever the continued baseline was doing anyway.
    others = [a for a in arms if a != "control" and trajectories.get(a)]
    if trajectories.get("control") and others:
        by_step = dict(trajectories["control"])
        header = f"{'step':>8}{'control':>10}"
        for arm in others:
            header += f"{arm:>14}{'delta':>9}"
        print(header)
        steps = sorted({s for arm in others for s, _ in trajectories[arm]} & set(by_step))
        for step in steps:
            row = f"{step:>8}{by_step[step]:>10.3f}"
            for arm in others:
                psnr = dict(trajectories[arm]).get(step)
                row += (
                    "             -        -"
                    if psnr is None
                    else f"{psnr:>14.3f}{psnr - by_step[step]:>+9.3f}"
                )
            print(row)

        print()
        final = f"final: control={trajectories['control'][-1][1]:.3f}"
        for arm in others:
            step, psnr = trajectories[arm][-1]
            matched = by_step.get(step)
            delta = f"{psnr - matched:+.3f} dB @ {step}" if matched else "no matched control step"
            final += f"  {arm}={psnr:.3f} (delta {delta})"
        print(final)
        print()
        print("Paired (same seed, schedule, warm-start checkpoint, and pinned 7-layer loss), so a")
        print("delta is attributable to which layer weights could move, and nothing else. Quote the")
        print("delta at a MATCHED step -- an arm scored further along than the control is not a")
        print("result, since both arms were still improving at 200,000.")
        if "learn7" in others and "learned_mix" in others:
            l7, lm = dict(trajectories["learn7"]), dict(trajectories["learned_mix"])
            shared = sorted(set(l7) & set(lm) & set(by_step))
            if shared:
                s = shared[-1]
                print()
                print(f"decomposition @ step {s}:")
                print(f"  freedom (learn7 - control)      : {l7[s] - by_step[s]:+.3f} dB")
                print(f"  reach   (learned_mix - learn7)  : {lm[s] - l7[s]:+.3f} dB")
                print(f"  total   (learned_mix - control) : {lm[s] - by_step[s]:+.3f} dB")
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
    print("Read the 'vs plateau' column. This run is at constant LR 1e-4, so 24.747 is the region")
    print("it is heading for; the first readings should FALL from 24.885 toward ~24.75 as the")
    print("raised LR gives back the anneal -- that is the restart settling, not the idea failing.")
    print("Neither constant is a ceiling: the control passed both by step 200,000, ending at")
    print("24.992. So these columns are for reading ONE arm in isolation. As soon as `control` has")
    print("been scored at the same steps, run with it in ARMS and quote the paired delta instead.")


def report_weights(checkpoint_dir: pathlib.Path) -> None:
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*/checkpoint.pth"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not checkpoints:
        print(f"\nno checkpoints under {checkpoint_dir}")
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
    print(f"max |move|                 : {moved.abs().max():.4f}")
    print(f"L2 |move|                  : {moved.norm():.4f}   (init L2 = {init.norm():.4f})")
    print(f"max |move| off stock layers: {off_stock.abs().max():.4f}")

    # SHARES, not absolute sums. The aggregation's overall scale is not identified: scaling
    # layer_weights down and the bottleneck projection up leaves the latent unchanged, and weight
    # decay resolves that symmetry arbitrarily (Experiment 1's vector collapsed from L2 1.1952 to
    # 0.0127 while its PSNR climbed 3 dB). An absolute "mass" figure therefore says nothing, and
    # reading one as a fraction is how "92% on the non-stock layers" first got written up as "92%
    # on layer 0" -- see experiments/2026-08-13-learned-layer-mix/NOTES.md.
    share = learned.abs() / learned.abs().sum()
    non_stock = sum(float(share[i]) for i in range(DINO_L_DEPTH) if i not in STOCK_LAYERS)
    top = sorted(range(DINO_L_DEPTH), key=lambda i: -float(share[i]))[:5]
    print()
    print(f"normalized |weight| share, stock 7 layers  : {100 * (1 - non_stock):5.1f}%")
    print(f"normalized |weight| share, other 17 layers : {100 * non_stock:5.1f}%")
    print("largest single layers: " + ", ".join(f"L{i} {100 * float(share[i]):.1f}%" for i in top))
    print()
    print("Interpretation: near-zero MOVEMENT means the gradient does not want a different layer")
    print("combination, and the hand-picked formula was already near a local optimum. Share moving")
    print("onto the other 17 layers is the positive signal -- it means training found information")
    print("in layers the stock recipe discards. Compare shares between arms, never raw magnitudes.")


def main() -> None:
    arms = [a for a in os.environ.get("ARMS", "learned_mix").split(",") if a]
    report_psnr(arms)
    # Both free-weight arms are worth a weight readout; `control` has nothing to say (frozen). For
    # learn7 the readout doubles as a check that the exclusion held: every non-stock layer must
    # still read exactly its init.
    for arm in ("learned_mix", "learn7"):
        if arm in arms:
            default = f"checkpoints/calibration/warmstart_{arm}"
            env = "LEARNED_CKPT_DIR" if arm == "learned_mix" else "LEARN7_CKPT_DIR"
            report_weights(pathlib.Path(os.environ.get(env, default)))


if __name__ == "__main__":
    main()
