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
quality's expense. (Three of those four hold. **P-DINO does not** — corrected 2026-09-10 below:
it never separates the arms, so the claim was true against the plateau and false as an arm-to-arm
statement.) The control's flatness is what makes this attributable to the learned
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

## 2026-09-10 — learn7 becomes a layer selection, not a freezing mechanism

Simplification, before `learn7` has run a single step — which is the only free moment to change an
arm's parameterization.

The arm was built as "expose all 24 DINOv3 blocks, freeze 17 weights at zero", with a mask, frozen
buffers, and a substitution in the forward pass. It is now "read only the stock 7 blocks", which is
the same class as `learned_mix` with one config key: `expose_layers: [11,13,15,17,19,21,23]`. Its
weight vector is 7 long, all trainable.

The two are *exactly* equivalent, not approximately: a frozen term contributes `0.0 * f`, and
adding an exact zero to a float is exact in IEEE-754, so both spellings give a bit-identical
latent. Verified on the real model, not just in a unit test — `torch.equal` on the latent, max abs
diff 0.0, and identical consistency-loss targets.

Given equivalence, selection wins on the things that matter. It deletes the mask, the buffers, the
substitution, and three tests. More to the point it deletes an entire category of reasoning — does
the gradient reach these, does decoupled weight decay move them, is a mask enough — that this repo
got *wrong in writing* before it got right, twice over in one session. Machinery hard enough to
reason about that its own justification keeps coming out wrong is evidence against itself. It is
also cheaper: 17 fewer tensor multiply-adds and ~113MB fewer retained features per forward.

What is given up is pinning a weight at a *nonzero* value — holding mira's deep residual at 8/7
while the rest learns, say. No arm wants that, and if one ever does the mask returns in ~15 lines
from this history.

**The trap this nearly walked into**, worth knowing before touching the variant again: the exposure
is a constructor argument, NOT `config.encoder.aggregation_layers`. Both finished arms' saved
`codec_config.yaml` files record `aggregation_layers: [11,13,...,23]` while their checkpoints hold
24 weights, because the old class overrode that field internally. Deriving the exposure from the
config would have built a 7-weight encoder for those saved configs and failed the strict
`load_state_dict` against 24 saved weights — silently breaking re-scoring of all 50 checkpoints the
two finished arms produced, and only at scoring time, which is exactly the failure mode `AGENTS.md`
warns about. The argument defaults to all 24; both arms' step-200,000 checkpoints were re-loaded
after the change to confirm.

## 2026-09-10 — P-DINO doesn't separate the arms, and what that reframes

Went to check a claim rather than requote it, and it did not hold.

Every write-up of Experiment 1 said "SSIM, LPIPS, P-DINO and rFDD all improved together, which
rules out the aggregation trading perceptual quality for pixel error". Three of the four hold.
**P-DINO does not.** Across all 25 matched steps the sign of `learned_mix - control` flips six
times and the gap is ~1% either way, while within either arm P-DINO swings ~20% following the
shared dips — so the metric is dominated by training phase, not by which arm you are in. Both arms
improve P-DINO against the plateau's 10.0595e-5, but the *control* improves it slightly more
(9.6885 vs 9.7621e-5), so none of that is attributable to the aggregation. True against the
plateau, false arm-to-arm: exactly the confusion the matched-step rule exists to prevent.

So the +2.914 dB buys PSNR (+11.7%), LPIPS (+22.6%) and rFDD (+7.9%), and nothing measurable on
the benchmark's one paired DINOv3-feature perceptual distance. Not a degradation — an absence of
gain, on the metric closest to "is this semantically faithful".

**That matters because mira's own layer ablation has the opposite signature.** Dropping to the
deepest block alone costs 1.0% of PSNR but 44-47% of rFID/rFVD/rFDD and 14% of P-DINO — PSNR is
the *least* layer-sensitive metric in their table, by 14x to 47x. Our gain is concentrated in
precisely that metric. (This also corrects a claim made earlier in the day that the *downstream*
metrics were the sensitive ones: they move 6-14%, less than the reconstruction Fréchet distances.
PSNR is the outlier, not downstream.)

**The reframing.** mira's methodology says outright: *"We judge every codec by the world model
trained on it and select for an easy-to-generate latent, treating reconstruction quality as
secondary."* And the codec is confirmed reconstruction-only in training — `CodecLoss` is L1 +
LPIPS + DINO latent consistency, and nothing in `src/mira/codec/` references the world model. So
mira selects on downstream and trains on reconstruction, and the gap between the two is bridged
*by the layer set*.

Which means the fixed uniform mean over deep-ish blocks is not an untuned hyperparameter. It is a
**regularizer**: a hard constraint that discards shallow detail on purpose, encoding downstream
knowledge the loss cannot express. Fixing it is the mechanism, which is why nobody tuned it.
Learning the weights therefore does not improve a hyperparameter — it removes the constraint, and
the reconstruction objective immediately does what the constraint existed to prevent: goes as
shallow as it is permitted. Visible in the weights themselves — `learned_mix` (24 blocks readable)
puts 45.7% on layer 0; `learn7` (blocks 11-23 only) puts 72.5% on layer 11. Each dumps its mass on
the shallowest block it can reach, and both abandon the deep residual.

This turns "does the gain survive downstream?" from an open shrug into a directional prediction: it
predicts not. Still untestable in this rig, but it is a prediction now, and `learn7` landing near
`control` would be consistent with it.

## 2026-08-26 — Publishing pass

Removed every hardcoded `/home/katta`-style path in favor of `direnv` (`.envrc`) plus two env vars
(`RS_DINO_WEIGHTS_DIR`, `MIRA_TRAIN`), so the repo works for anyone with the same sibling-folder
layout instead of just this machine. Added `LICENSE` (Apache-2.0, matching mira's own, since this
project depends on and forks pieces of it) and `AGENTS.md` (orientation for a future agent session).
