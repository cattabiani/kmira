# A single-GPU reproduction of mira's video codec

**What this is.** [mira](https://github.com/kyutai-labs/mira) is a Rocket League world model. Its
perception front-end is a video codec — a frozen DINOv3-L/16 feature extractor, a learned linear
bottleneck, and a ViT decoder — and the world model predicts in the latent that codec produces.
This repository reproduces that codec at a scale that fits one consumer GPU, and uses it as a
benchmark for changes to the codec design.

**Status.** The benchmark and its training protocol are built and characterised. The reference
baseline is training now; section 3 is its live trajectory. Comparative studies are queued
(section 4).

**Every figure and number here is generated from the run record, not written.** The numbers are
quoted from [`stats.md`](stats.md); if the two disagree, `stats.md` is right.

```bash
pixi run python postprocessing/make_all.py
```

---

## 1. The codec

mira's architecture is kept exactly; only the parts that make it unaffordable on one 12GB card are
reduced. Full configuration table in [`stats.md`](stats.md).

| | this rig | mira |
|---|---|---|
| feature extractor | `dinov3_vitl16`, frozen | same |
| aggregated blocks | `[11, 13, 15, 17, 19, 21, 23]` | same (RAEv2's k=7 default) |
| decoder | Base — 768 wide, 12 deep | XL — 1152 wide, 28 deep |
| trainable parameters | 114,188,320 | 596,129,440 |
| frame | 288×512, 20 fps | same |
| timesteps per sample | 1 (image-only) | 40 |
| compression | 96×, spatial only | 192×, spatial + 2× temporal |

Two reductions matter for reading anything below.

**The Base decoder.** XL does not fit in 12GB under fp32 Adam. Base is one of the three sizes mira
themselves ablate, so it is a published operating point rather than an arbitrary shrink, and it is
the only size cheap enough to train deep into diminishing returns here — which matters more than
absolute quality, because a gap measured between two undertrained models tells you which trains
faster, not which is better. Both are mira's own configs, so this is a config change and not a fork: training runs
`mira/scripts/train_codec.py` unmodified.

**Image-only.** `timesteps: 1`, no temporal stride. This is the single biggest reason no absolute
number here is comparable with mira's published values.

## 2. Protocol

**Metrics.** Every checkpoint is scored on 2,048 held-out frames with a fixed evaluation seed:
PSNR, SSIM, LPIPS, P-DINO (paired DINOv3-B feature distance) and rFDD (reconstruction Fréchet DINO
distance). Training additionally logs mira's own validation loss — four terms, every 1,000 steps,
on a separate 512-sample split.

**Runs are chunked and scored inline.** Training advances in 8,000-step chunks; each chunk ends in
a checkpoint, a validation reading and a scoring pass. A run is therefore a curve rather than an
endpoint, and can be stopped and resumed on any chunk boundary because the learning rate is held
constant.

**The recipe is locked** — constant LR 1e-4, 8,000-step chunks, and a data seed fixed by the
absolute step. Every arm runs it and is read at matched steps. Pre-registered in
[`experiments/2026-09-10-paired-recalibration/NOTES.md`](../experiments/2026-09-10-paired-recalibration/NOTES.md).

**Runs stop on a step budget.** Constant LR here has never produced a detectable plateau, so the
length is a decision rather than a measurement. Consecutive chunks train on different data, so
progress is read as a trailing trend, never off a single chunk.

## 3. The reference baseline

![baseline_v2 progress](figures/baseline_v2_progress.png)

A cold start under the protocol above: constant LR 1e-4 after a 1,000-step warmup to 264,000
steps, then a 32,000-step cosine decay to min_lr. The anneal is worth **+0.2498 dB**
(24.6624 → **24.9122 dB**), and that endpoint is the fixed reference every arm is read against.
Numbers throughout are quoted from [`stats.md`](stats.md), which carries its generation date.

What the six panels show:

- **PSNR and its per-chunk increment.** The increment panel carries the trend-versus-noise
  readout, which is the one to read for whether the run is still improving.
- **mira's four validation loss terms**, each on its own axis — they span three orders of
  magnitude, so a shared axis would flatten three of them into a line. Each panel reports its own
  trailing slope, because `loss_total` is dominated by `loss_lpips_perceptual` and hides the rest.

Two caveats that are properties of the instrumentation, not of the model:

- **`loss_dino_latent_consistency` has reached mira's four-decimal logging resolution.** Its flat
  line is rounding, not convergence; nothing about that term's ceiling can be read from this log.
- **It was still improving when the constant-LR phase was stopped**, at **+0.019 dB per 10,000
  steps** against a between-reading spread of **0.010 dB**. The stopping point is a budget, not a
  ceiling, so the reference is *trained to this recipe* rather than *trained to convergence* — which
  is why every arm is read at matched steps against it.

## 4. Does the benchmark resolve a known effect?

![baseline and the frozen-bottleneck ablation](figures/arms.png)

The open question for any benchmark is whether it can separate arms at all. The test is mira's own
frozen-bottleneck ablation: replace the trained linear bottleneck with a random frozen one and
train everything else identically. mira report roughly **1.4 dB** for it.

Both arms run the locked recipe from the same seed base, so at any step they have trained on the
same data and the gap is attributable to the bottleneck. Over the four matched readings so far, the
frozen arm sits **1.4470 to 1.7352 dB** below the baseline — the effect is resolved, at a size
consistent with the published one.

Two limits on how far that goes. The ablation has run **32,000** of the baseline's **296,000** steps,
so this is an early-training comparison and the gap may still move. And a single pair cannot
separate the intervention from run-to-run spread; the second-seed baseline, which measures that
spread directly, has not run yet.

## 5. In progress

Queued under the same recipe: the second-seed baseline, the rest of the frozen ablation, and the
learned layer-aggregation studies. Earlier studies that ran against a superseded baseline are
archived at [`archive/RESULTS-legacy-baseline.md`](archive/RESULTS-legacy-baseline.md); their
magnitudes do not carry over and nothing here depends on them.
