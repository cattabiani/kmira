# Experiment 1: learned per-DINO-layer aggregation

**Ran** 2026-08-13 (built) to 2026-09-06 (last reading). Backfilled from `CHANGELOG.md`, the
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
  checkpoint annealed to its minimum), which costs quality regardless of the idea.

Comparison number is the constant-LR plateau, **24.747**, not the annealed 24.885: a warm start
cannot reach the annealed number no matter how long it runs.

## Result

`learned_mix` reached **27.905 dB at step 200,000, +3.16 dB over the plateau it started from**,
while `control` stayed at 24.5 to 24.7 throughout. SSIM, LPIPS, P-DINO and rFDD all improved in the
same direction, which rules out the aggregation trading perceptual quality for pixel error. The
flat control is what makes the gain attributable to the aggregation rather than to the restart.

Still climbing at the last reading (+0.07 per 8k steps). The curve was not smooth: two
dip-then-jump stretches (16k to 24k restart settling, then 80k to 96k with a jump at 104,000), the
same false-plateau shape the original baseline run showed.

Do not compare 27.905 to the paper's 27.6 Base-decoder row. Different setup: this rig is image-only
and reduced-scale, and its own faithful baseline sits at 24.747 where the paper's Base decoder
reaches 27.6.

## What the weights actually learned

Not a reweighting of the paper's blocks. About 92% of the normalized weight mass moved onto **layer
0**, DINOv3's shallowest block, away from the mid and late blocks the stock formula reads. Read the
normalized direction, not raw magnitudes: the aggregation's scale and the bottleneck projection's
norm have an exact scaling symmetry that weight decay resolves arbitrarily, so only relative weight
is identified.

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

## Files

- Variant and full rationale: `src/kmira/codec/variants/learned_layer_mix.py`
- Configs: `codec/configs/model/learned_layer_mix.yaml`, `..._control.yaml`
- Launcher: `codec/scripts/run_learned_layer_mix_warmstart.sh`
- Numbers: `codec/results/benchmark.jsonl`, tags `learned_mix-*` and `control-*`
- Weight trajectory report: `codec/scripts/report_layer_mix.py`
