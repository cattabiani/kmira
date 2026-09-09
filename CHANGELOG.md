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

## 2026-08-26 -- 2026-09-06 — Experiment 1 result: the idea works, with an open question

Ran the warm-started comparison across several sessions. `learned_mix` climbed from the 24.75dB
plateau to **27.905dB by step 200,000**, a within-setup gain of +3.16dB (this originally read
"past the paper's own Base-decoder reference (27.6dB)", which was a bad comparison: this rig is
image-only and reduced-scale, and its own faithful baseline sits at 24.75 where the paper's Base
decoder reaches 27.6, so the two numbers come from different setups),
while the paired control (identical run, aggregation weights frozen) never left the plateau's
neighborhood, staying at 24.5-24.7dB throughout. (Both halves of that sentence were superseded on
2026-09-09, below, once the control was run to a matched 200,000 steps: it finished at 24.992, and
the headline became the matched-step gap of +2.914dB.) Not a PSNR-only effect: SSIM, LPIPS, P-DINO and
rFDD all improved together, which rules out the aggregation gaming pixel error at perceptual
quality's expense. The control's flatness is what makes this attributable to the learned
aggregation itself, not extra training on a restarted optimizer.

The curve was not a smooth climb — two separate dip-then-jump stretches (16k-24k restart settling,
then 80k-96k followed by a jump at 104,000) — the same false-plateau shape the original baseline
run hit during its own search (2026-08-13). Kept running rather than stopped early, on the same
reasoning; still not fully flat at the last reading.

**What the model actually learned is not a reweighting of the paper's layers — it abandoned them.**
The 24 learned weights converged to ~92% of their (scale-normalized) mass on the 17 blocks the
paper's formula never reads, away from the mid/late blocks `{11,...,23}` that it does (this
originally read "~92% ... on DINOv3's *shallowest* block", which misattributed the figure: layer 0
is the largest single share at 45.7%, not 92%. Corrected 2026-09-09, below). Checked against the
paper's own text: that formula's rationale explicitly keeps a residual on the deepest block to
preserve semantics for the *world model* that later predicts in this latent, and the paper's one
layer-choice ablation (multi-layer vs. last-block-only) shows the multi-layer choice winning on
downstream world-model metrics too, not just reconstruction. Nothing in the paper tests the regime
this run landed in. So: a real, multi-metric win on the reconstruction benchmark, and a real open
question about whether this latent will serve the world model as well — untested here, since this
benchmark only measures reconstruction. See `codec/README.md`'s "Current state" and
`codec/results/benchmark.jsonl` for the full trajectory.

## 2026-09-08 -- 2026-09-09 — Experiment 1's control, run to a matched length

The one thing holding Experiment 1's claim back was that its control had been run to 56,000 steps
against the variant's 200,000 — so "the control stayed flat" described a quarter of the comparison,
and the control had never been observed through the region where the variant appeared to take off.
Ran it the rest of the way. Both arms now sit at 200,000 steps.

**The result holds, and the headline changes slightly.** `learned_mix` 27.905 dB against `control`
24.992 at matched steps: **+2.914 dB**, with SSIM (0.8587 vs 0.8105), LPIPS (0.0820 vs 0.1059) and
rFDD (0.6209 vs 0.6740) all moving with it. The control did not stay strictly inside its
pre-registered 24.5-24.7 band — it ends 0.245 dB above the 24.747 plateau it started from — which
is the third pre-registered outcome, whose rule is to report the arm-to-arm gap at matched steps
rather than the gain over the plateau. So the number to quote is +2.914 dB, not +3.16 dB. The
pre-registration said not to quietly requote, so this is the note saying so; the difference is
small, but it is the difference between a number the design supports and one it doesn't.

**What extending the control actually bought** was better than confirmation. The variant's jump at
104,000 had two candidate explanations — the aggregation, or the warm restart still unwinding — and
it turned out to be a third: **both arms show the same dip through 72k-96k and the same recovery at
104,000**, so that shape belongs to the shared per-chunk seed schedule and not to either arm. The
paired design absorbed it exactly as designed. The gap is untouched by it, widening at 23 of the 24
step-to-step transitions from +0.236 at 8,000 to +2.914 at 200,000 (the one exception is
96k->104k, where the control climbed out of the shared dip a reading before the variant). Both arms
were still improving at the end, the variant faster, so the gap is still widening but sharply
decelerating: +0.225 per 8k over the first 56,000 steps, +0.038 per 8k over the last 40,000.

**A gotcha in `AGENTS.md` turned out to be false, and was corrected.** It said a constant-LR warm
start "can never reach the annealed number no matter how long it runs". The control *is* the locked
baseline continued at constant LR, and it passed the annealed 24.885 and finished at 24.992. The
instinct behind the entry is still right for reading a short run — the restart gives the anneal
gain straight back and takes tens of thousands of steps to recover it — but "never" was too strong.
The corollary is about the baseline rather than the experiment: the plateau called at step 272,000
was a stopping point, not an asymptote, and constant LR was still buying roughly +0.03 dB per 8k
steps out at 200,000. Fourth time in this project a called plateau has had more in it.

**Also corrected**: the "~92% of normalized mass" figure had been written up as mass on *layer 0*.
It is the collective share of the 17 layers the stock formula never reads; layer 0 alone is 45.7%
of the L1-normalized mass (84.6% by energy), the largest single share. The conclusion — the learned
weights abandoned the paper's blocks rather than reweighting them — is unchanged, but the number
belonged to a different quantity.

The downstream question is untouched by any of this and remains the honest limit on the result:
this benchmark scores reconstruction only, and nothing here tests whether a layer-0-dominant latent
serves the world model that has to predict in it.

## 2026-09-09 — Experiment 2 scaffolding: `learn7`

Built (not yet run) the arm that decomposes Experiment 1's gain into *freedom* (weights allowed to
move) and *reach* (17 shallower layers becoming available). `learn7` is the existing machinery with
only the stock 7 weights trainable, so `learn7 - control` is the value of freedom and
`learned_mix - learn7` the value of reach.

The 17 excluded weights are frozen by a gradient hook plus exclusion from the optimizer's
parameter set, not merely initialised to zero: AdamW's `weight_decay=0.1` and gradient noise on a
zero-initialised trainable scalar would let them drift and quietly hand back the reach the arm
exists to withhold. A test asserts they are bit-identical after a real optimizer step.

Pre-registered outcomes were written before the scaffolding, in
`experiments/2026-09-08-decompose-layer-mix/NOTES.md`.

## 2026-08-26 — Publishing pass

Removed every hardcoded `/home/katta`-style path in favor of `direnv` (`.envrc`) plus two env vars
(`RS_DINO_WEIGHTS_DIR`, `MIRA_TRAIN`), so the repo works for anyone with the same sibling-folder
layout instead of just this machine. Added `LICENSE` (Apache-2.0, matching mira's own, since this
project depends on and forks pieces of it) and `AGENTS.md` (orientation for a future agent session).
