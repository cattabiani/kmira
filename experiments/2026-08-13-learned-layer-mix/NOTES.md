# Experiment 1: learned per-DINO-layer aggregation

**Ran** 2026-08-13 (built) to 2026-09-09 (both arms complete at 200,000 steps). Backfilled from `CHANGELOG.md`, the
commit trail and `codec/README.md`; this folder convention was written down before it was used, so
Experiment 1 got its record after the fact rather than before the run.

## Hypothesis

mira's RAE encoder aggregates DINOv3-L features as `mean(layers 11,13,15,17,19,21,23) +
features[-1]`. That layer set is RAEv2's k=7 default (`mira/src/mira/codec/config.py`), adopted
unchanged, and the weighting within it is uniform by construction: never tuned, and not the
subject of the paper's own layer ablation (Appendix Table 22 varies *which* layers, not their
relative weight).

So: does learning the per-layer weights beat the fixed formula, at this rig's reduced scale?

## What changed

`src/kmira/codec/variants/learned_layer_mix.py`. All 24 DINOv3-L blocks exposed, one learned
scalar weight each (ELMo-style layer mixing), unconstrained rather than softmax-normalised, since
the stock formula's effective weights do not sum to 1 (the `+ features[-1]` term double-counts
layer 23).

Two things had to be true for the comparison to mean anything:

- **Initialised as an exact no-op.** Weights start at 0 on the 17 unused layers, 1/7 on the six
  non-last stock layers, 8/7 on layer 23. Verified against the real checkpoint: all 616 saved keys
  load byte-for-byte, latent difference 1.4e-6. Step 0 behaves identically to the locked baseline,
  so nothing that follows is an artifact of the modified encoder.
- **Objective held constant.** `CodecLoss.bind_encoder_dino` derives the DINO latent-consistency
  loss's layer set from the encoder's, so exposing 24 layers would have silently swapped a 7-layer
  objective for a 24-layer one, changing two things at once. The loss is pinned to the baseline's 7
  (`KMIRA_PIN_CONSISTENCY_LAYERS`, plus the encoder returning only those 7 as the loss's targets).
  The pairing goes through a non-strict `zip`, so a mismatch would misalign silently; checked
  numerically in `tests/test_learned_layer_mix.py` and against a real stock encoder (difference
  exactly 0.0).

## Arms

Both warm-started from the locked baseline `checkpoint-304000`, same seed schedule, same data per
chunk, `run_learned_layer_mix_warmstart.sh`:

- `learned_mix`: the 24 weights free to move.
- `control`: same class, same 24-layer exposure, same pinned loss, weights frozen at the
  stock-equivalent init. Absorbs the warm-restart penalty (optimizer reset and raised LR on a
  checkpoint annealed to its minimum), which costs quality regardless of the idea. Run to the same
  200,000 steps as the variant.

Comparison number is the constant-LR plateau, **24.747**, not the annealed 24.885: a warm start
cannot reach the annealed number no matter how long it runs.

## Result

Both arms have now been run to **200,000 steps**, so the comparison is at matched length.

| | `control` | `learned_mix` | gap |
|---|---|---|---|
| PSNR | 24.992 | 27.905 | **+2.914** |
| SSIM | 0.8105 | 0.8587 | +0.0482 |
| LPIPS | 0.1059 | 0.0820 | -0.0239 |
| P-DINO | 9.69e-05 | 9.76e-05 | +0.7e-06 (no separation — see below) |
| rFDD | 0.6740 | 0.6209 | -0.0531 |

**The headline is the arm-to-arm gap at matched steps, +2.914 dB — not +3.16 dB over the
plateau.** The control did not stay strictly inside its pre-registered 24.5-24.7 band: it ends at
24.992, i.e. +0.245 over the 24.747 plateau it warm-started from. That is the third pre-registered
outcome ("control climbs a little: report the gap at matched steps and use that as the result"),
so that is what is reported. The difference between the two framings is small (0.25 dB out of 3),
but the matched-step number is the one the design actually supports, and requoting the larger one
would be reading the pre-registration after the fact.

**What extending the control bought.** The open question was whether the variant's jump at 104,000
was the aggregation or the warm restart still unwinding. It was neither, in the sense that
mattered: **both arms trace the same dip-then-jump shape** — a dip through 72k-96k (control down to
23.774, variant down to 26.260) and a recovery at 104,000 — so that feature is a property of the
shared per-chunk seed schedule, not of the intervention. The paired design absorbed it exactly as
intended. The gap itself is unaffected by it: it widens at 23 of the 24 step-to-step transitions,
from +0.236 at 8,000 to +2.914 at 200,000, and the single exception is 96k->104k, where the control
happened to climb out of the shared dip one reading before the variant did.

So the attribution in this experiment now rests on a control observed over the variant's full
length, not on 56,000 steps of it.

Both arms were still climbing at the last reading, the variant faster (+0.065 per 8k over the last
40,000 steps, against the control's +0.028), so the gap is still widening but decelerating sharply:
+0.225 per 8k over the first 56,000 steps, +0.038 per 8k over the last 40,000.

Do not compare 27.905 to the paper's 27.6 Base-decoder row. Different setup: this rig is image-only
and reduced-scale, and its own faithful baseline sits at 24.747 where the paper's Base decoder
reaches 27.6.

## P-DINO does not separate the arms, and that is the interesting part

Worth stating separately because the first write-up of this result got it wrong. It claimed SSIM,
LPIPS, P-DINO and rFDD "all improved together, which rules out the aggregation trading perceptual
quality for pixel error". Three of those four hold. **P-DINO does not.**

Across all 25 matched steps the sign of `learned_mix - control` on P-DINO flips six times, and the
gap is ~1% either way, while *within* either arm P-DINO swings ~20% following the shared dips. So
the intervention does not move P-DINO at all — the metric is dominated by training phase, not by
which arm you are in:

| | PSNR | SSIM | LPIPS | rFDD | P-DINO |
|---|---|---|---|---|---|
| `learned_mix` vs `control` @200k | **+11.7%** | +5.9% | +22.6% | +7.9% | −0.8%, sign unstable |

Both arms *do* improve P-DINO against the 24.747 plateau's 10.0595e-5 (control 9.6885e-5,
`learned_mix` 9.7621e-5) — but the control improves it slightly **more**, so none of that gain is
attributable to the aggregation. The original claim was true against the plateau and false as an
arm-to-arm statement, which is exactly the confusion the matched-step rule exists to prevent.

What that leaves: the +2.914 dB buys pixel fidelity (PSNR), perceptual distance (LPIPS) and
distributional feature fidelity (rFDD), and buys **nothing measurable** on the one paired
DINOv3-feature perceptual distance in the benchmark. It is not a degradation. It is an absence of
gain, on the metric closest to "is this reconstruction semantically faithful".

Compare the direction of mira's own layer ablation (appendix, multi-layer vs last-block-only),
which has the opposite signature — dropping to the deepest block alone costs only 1.0% of PSNR but
44-47% of rFID/rFVD/rFDD and 14% of P-DINO. There, PSNR is the *least* sensitive metric to layer
choice by a factor of 14 to 47. Here, PSNR is the metric that moves most. A gain concentrated in
the least layer-sensitive metric is not what "found a better latent" looks like.

## Reframing: mira's uniform mean is a regularizer, not an untuned hyperparameter

The strongest reading of this experiment, and the one that survives the result.

mira's own methodology (`sections/6.experiments.tex`) states: *"We judge every codec by the world
model trained on it and select for an easy-to-generate latent, treating reconstruction quality as
secondary."* And the codec is confirmed reconstruction-only in training — `CodecLoss` is L1 + LPIPS
+ DINO latent consistency, `scripts/train_codec.py` imports nothing from the world model, and
`grep world_model src/mira/codec/` is empty. So mira **selects** on downstream and **trains** on
reconstruction.

That gap is bridged by the layer set. The fixed mean over deep-ish blocks is not a parameter nobody
tuned — it is a hard constraint that discards shallow detail *on purpose*, encoding downstream
knowledge the loss cannot express. Fixing it is the mechanism, which is why there was never a
reason to tune it.

So learning the weights does not improve a hyperparameter. It **removes a regularizer**, and the
reconstruction objective immediately does what the regularizer existed to prevent: it goes as
shallow as it is allowed to. That is visible directly in the weights — `learned_mix` (all 24
blocks readable) put 45.7% on layer 0, and `learn7` (blocks 11-23 only) put 72.5% on layer 11.
Each arm dumps its mass on the shallowest block available to it, and both abandon the deep
residual mira keeps deliberately.

This makes "does the gain survive downstream?" a directional prediction rather than an open shrug:
it predicts not, because the gain comes from undoing the constraint whose purpose was the
downstream latent. Untestable here, but it is a prediction, and `learn7` landing near `control`
would be consistent with it.

## Side finding: the control is also 200k more steps of the stock recipe

Worth separating out, because it is about the baseline rather than the idea. The control is the
locked baseline continued at constant LR for another 200,000 steps, and it ended at **24.992** —
above the 24.747 plateau it started from *and* above the annealed 24.885.

That contradicts the gotcha recorded in `AGENTS.md`, which said a constant-LR warm start "can never
reach the annealed number no matter how long it runs". Directionally the gotcha still holds and is
still the right instinct for reading a short run: the restart gives the anneal gain straight back
and spends tens of thousands of steps recovering it. But "never" was too strong, and it is now
measured. `AGENTS.md` has been corrected.

The practical consequence is for the baseline, not for this experiment: the plateau called at step
272,000 was not fully converged, and constant LR keeps buying roughly +0.03 dB per 8k steps out at
200k. That is small against a 2.9 dB gap and does not touch the comparison — both arms pay it
identically — but it means "the plateau" names a stopping point, not an asymptote. This is the
fourth time in this project a called plateau turned out to have more in it.

## What the weights actually learned

Not a reweighting of the paper's blocks. It abandoned them. At 200,000 steps **92.4% of the
normalized weight mass sits on the 17 layers the stock formula never reads**, leaving 7.6% on the
stock seven, and the single largest share is **layer 0**, DINOv3's shallowest block, at 45.7% of
the L1-normalized mass (84.6% by energy). Layers 3, 1 and 7 take most of the rest. Layer 23 — the
block the stock formula deliberately double-counts — went from the largest weight (1.1429) to
0.0003.

(An earlier version of this note, and of `codec/README.md`, reported the 92% figure as mass on
layer 0 specifically. That misattributed it: 92.4% is the collective share of the non-stock layers.
Layer 0 is the dominant single layer either way, so the conclusion is unchanged, but the number
belonged to a different quantity.)

Read the normalized direction, not raw magnitudes: the aggregation's scale and the bottleneck
projection's norm have an exact scaling symmetry that weight decay resolves arbitrarily, so only
relative weight is identified. The raw vector collapsed by roughly two orders of magnitude (L2 from
1.1952 to 0.0127) with the projection presumably absorbing the scale — which is that symmetry
being resolved, not a signal in itself.

## What this does not show

The benchmark scores reconstruction only. mira's stated reason for keeping a residual on the
deepest block is to preserve semantics for the **world model** that predicts in this latent, and
the paper's one relevant ablation favours depth on downstream metrics too. Nothing here tests that
regime. Shallow ViT features are close to an invertible encoding of the image, which is exactly
what a pixel-reconstruction objective rewards and not what a predictor needs, so there is a
specific reason to expect this gain may not transfer.

Treat it as a reconstruction result with an open question attached, not as an improvement to MIRA.

## Open questions

1. Does the gain survive downstream? Untestable in this rig, which has no world model and an
   image-only codec.
2. How much of the gain comes from *learning* the weights, and how much from *reaching shallow
   layers*? Undecomposed. A variant with the deep-block mass floored would separate them.
3. Would the effect appear from a cold start, or does it only help a model already sitting at an
   annealed minimum? Only warm starts have been run.
4. Closed by the extended control, so recorded here rather than left implied: the variant's jump at
   104,000 was not the idea and not the restart unwinding, but a shared feature of the seed
   schedule that both arms show.

## Files

- Variant and full rationale: `src/kmira/codec/variants/learned_layer_mix.py`
- Configs: `codec/configs/model/learned_layer_mix.yaml`, `..._control.yaml`
- Launcher: `codec/scripts/run_learned_layer_mix_warmstart.sh`
- Numbers: `codec/results/benchmark.jsonl`, tags `learned_mix-*` and `control-*`
- Weight trajectory report: `codec/scripts/report_layer_mix.py`
