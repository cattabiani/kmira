# Changelog

A log of what happened in this repo, in order. Written for a human (or a future agent session)
skimming for "what's the state of things and how did we get here" — not a commit-by-commit mirror
of `git log`. See `codec/README.md` for the codec work's current state and how to run things; this
file is the higher-level narrative across the whole project (the detailed reasoning trail lives in
commit messages, not duplicated into a file).

## 2026-08-10 -- 2026-08-13 — Baseline codec benchmark (milestone)

Built, from scratch, a benchmark that can answer "is this change to mira's codec bottleneck
better than the stock one?" on a single consumer GPU (RTX 4070 Ti, 12GB) — not to reproduce
mira's own codec quality, which needs a compute budget this project doesn't have.

Highlights, roughly in the order they happened:

- Set up the environment (`pixi`), got the two separately-gated downloads (DINOv3 weights from
  Meta, the `kyutai/rocket-science` dataset from HuggingFace), and verified mira's `VideoCodec`
  forward pass manually before automating anything.
- Discovered mira's stock XL decoder (~596M trainable) OOMs on a 12GB card under fp32 Adam.
  Briefly forked the trainer to force bf16, then found the real fix was switching to mira's
  smaller **Base** decoder config instead — which also turned out to be the only size affordable
  enough to train *near its own convergence ceiling* rather than comparing undertrained variants.
  The trainer fork was deleted once Base made it unnecessary; training now runs mira's
  `train_codec.py` completely unmodified, config-only.
- Built the metrics benchmark (`eval_codec.py`) and immediately caught a real bug via it: a
  `[-1, 1]` vs `[0, 1]` pixel-range mismatch that was silently destroying every reconstruction
  image looked at so far (PSNR was scoring *worse than a flat gray image*). Fixed, then trusted.
- Calibrated the benchmark itself before trusting it for real comparisons: reproduced the paper's
  known "frozen random bottleneck costs ~1.4dB" effect via a paired A/B/C run, and found and fixed
  three separate hidden biases in the eval sampling (streaming loader stuck on 3/17 matches,
  unequal per-match weighting, `max_clips` only ever sampling the opening minutes of a match).
- Trained the real baseline past a false plateau (a genuine +1.31dB jump after hours of apparent
  flatness) to the real elbow at 272k steps, then annealed the LR to recover schedule quality the
  constant-LR search phase had left on the table.
- **Locked reference baseline**: `checkpoints/calibration/plateau_baseline/checkpoint-304000`
  (304,000 steps, PSNR 24.88) — the fixed point every future bottleneck experiment compares against.

Also fixed along the way: a data-repetition bug (dataloader reseeded identically on every hourly
restart), an off-by-one causing an infinite loop, an orphaned-checkpoint scoring bug, and a
`torch.hub` flake root-caused to URL throttling rather than GitHub being down.

`mira` itself was never modified — every divergence from stock mira lived in config, or in small
forked files under `src/kmira/codec/variants/`.

## 2026-08-13 — Experiment 1: learned per-DINO-layer aggregation

First real idea tested against the locked baseline. Replaces mira's fixed
`mean(7 hand-picked layers) + features[-1]` DINOv3 feature aggregation with 24 learned scalar
weights (one per DINOv3-L layer), initialized to reproduce the stock formula exactly (verified
byte-identical against a real checkpoint, latent diff 1.4e-6) — so the new variant starts as a
no-op and only diverges as training moves the weights.

Had to explicitly decouple the DINO latent-consistency loss's layer set from the encoder's, since
mira's `bind_encoder_dino` normally derives one from the other — without this, exposing 24 tunable
layers would have silently swapped the *objective* too, making any PSNR delta unattributable to
the actual idea being tested.

Several supporting bugs fixed in the same pass: `eval_codec` not instantiating variants through
Hydra (would have failed only after paying for training hours), a `finetune_from` strict-loading
gap, a reintroduced fixed-seed data-repetition bug, a dropped RAEv2 noise regularizer in the
variant's forward pass, and the variant wastefully building three DINOv3-L backbones instead of one.

Warm-start-ready; comparison number to use is the constant-LR plateau (24.747), not the annealed
one (24.885), since the warm-start run doesn't anneal.

## 2026-08-26 -- 2026-09-05 — Experiment 1 result: the idea works

Ran the warm-started comparison across several sessions. `learned_mix` climbed from the 24.75dB
plateau to **27.429dB** by step 136,000 — 2.68dB above where it started, and 0.17dB from the
paper's own Base-decoder reference (27.6dB) — while the paired control (identical run, aggregation
weights frozen) never left the plateau's neighborhood, staying at 24.5-24.7dB throughout. That
divergence is the result: the gain is the learned aggregation itself, not an artifact of extra
training on a restarted optimizer.

The curve was not a smooth climb — a dip at 80k-96k steps briefly looked like a second plateau
before a jump to 104,000 resumed it, the same false-plateau shape the original baseline run hit
during its own search (2026-08-13). Kept running rather than stopped early, on the same reasoning.
Still climbing as of the last reading (136,000 steps); training continues, with an anneal planned
once several consecutive readings show near-zero movement. See `codec/README.md`'s "Current state"
and `codec/results/benchmark.jsonl` for the full trajectory.

## 2026-08-26 — Publishing pass

Removed every hardcoded `/home/katta`-style path in favor of `direnv` (`.envrc`) plus two env vars
(`RS_DINO_WEIGHTS_DIR`, `MIRA_TRAIN`), so the repo works for anyone with the same sibling-folder
layout instead of just this machine. Added `LICENSE` (Apache-2.0, matching mira's own, since this
project depends on and forks pieces of it) and `AGENTS.md` (orientation for a future agent session).
