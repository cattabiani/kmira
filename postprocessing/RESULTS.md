# A single-GPU reproduction of mira's video codec

**What this is.** [mira](https://github.com/kyutai-labs/mira) is a Rocket League world model. Its
perception front-end is a video codec — a frozen DINOv3-L/16 feature extractor, a learned linear
bottleneck, and a ViT decoder — and the world model predicts in the latent that codec produces.
This repository reproduces that codec at a scale that fits one consumer GPU, and uses it as a
benchmark for changes to the codec design.

**Status.** The benchmark and its training protocol are built and characterised. The reference
baseline is training now; section 3 is its live trajectory. Comparative studies are queued
(section 4). Earlier results exist but rest on a superseded baseline — see section 5.

**Every figure and number here is generated from the run record, not written.** The numbers are
quoted from [`stats.md`](stats.md); if the two disagree, `stats.md` is right. The exception is
section 5, which describes superseded work and cites its own notes.

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
the only size cheap enough to train to its elbow here — which matters more than absolute quality,
because a gap measured between two undertrained models tells you which trains faster, not which is
better. Both are mira's own configs, so this is a config change and not a fork: training runs
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

**Comparisons are paired, and this is not optional.** Each chunk draws its training stream from a
per-chunk seed, so a seed identifies a slice of training data — and the slices are far from
equivalent. Measured over 21 chunks of a paired arm, where a chunk's gain cannot be the
intervention, the best slice is worth **+0.795 dB** in a single chunk while the worst three cost
**0.18–0.40 dB**. The ranking is a property of the slice: the same seeds come out best and worst in
runs that met them at completely different points in training.

An unpaired comparison at this scale therefore measures the data draw as much as the intervention.
An early three-run study confirmed this the expensive way, resolving a published ~1.4 dB
bottleneck effect at only **1.27×** the run-to-run spread — against the 3× its own criterion
demanded.

The protocol that follows from this: arms are **warm-started from one checkpoint**, run on an
**identical per-chunk seed schedule**, and read at **matched steps**. The unevenness then lands on
both arms and cancels.

**Calling an elbow needs the increment, not the curve.** Constant-LR training here has twice
produced multi-hour flat stretches that resumed improving by over 1 dB. The per-chunk increment is
plotted alongside every trajectory for this reason, and an elbow requires several trailing
increments near zero.

## 3. The reference baseline

![baseline_v2 progress](figures/baseline_v2_progress.png)

A cold start under the protocol above: constant LR 1e-4 after a 1,000-step warmup, no anneal yet.
It is **still training**, so the prose here stays qualitative and the numbers live in
[`stats.md`](stats.md), which carries its generation date.

What the six panels show:

- **PSNR and its per-chunk increment.** The increment panel is the one to read for the elbow.
- **mira's four validation loss terms**, each on its own axis — they span three orders of
  magnitude, so a shared axis would flatten three of them into a line. Each panel reports its own
  trailing slope, because `loss_total` is dominated by `loss_lpips_perceptual` and hides the rest.

Two caveats that are properties of the instrumentation, not of the model:

- **`loss_dino_latent_consistency` has reached mira's four-decimal logging resolution.** Its flat
  line is rounding, not convergence; nothing about that term's ceiling can be read from this log.
- **The elbow is unknown and may land late.** The increments are falling quickly, but that is also
  what both false plateaus looked like.

## 4. Queued

| study | what it establishes |
|---|---|
| baseline elbow + anneal | the fixed reference point every comparison is read against |
| frozen-bottleneck ablation | whether the benchmark resolves a known ~1.4 dB effect, paired |
| second-seed baseline | the run-to-run spread at the operating point |
| learned layer aggregation | Experiments 1 and 2 (see section 5) |

Until the middle two land, this benchmark has **no demonstration that it can resolve an effect of
the size it is asked to judge**, and that limitation is load-bearing for everything else.

## 5. Superseded results

An earlier set of studies, on a previous baseline, is archived at
[`archive/RESULTS-legacy-baseline.md`](archive/RESULTS-legacy-baseline.md). They found that
replacing mira's fixed 7-layer aggregation with 24 learned per-layer weights improves
reconstruction substantially against its paired control; that the gain comes mostly from gaining
access to shallower DINOv3 blocks rather than from the weights being free; that each variant puts
its mass on the shallowest block it is permitted to read; and that P-DINO does not separate the
arms at all.

Those comparisons were paired, so their **directions stand**. Their **magnitudes do not**: the
baseline they were measured against had, for its first 56,000 steps, trained on roughly half the
available data because of a data-loader seeding fault, leaving it undertrained by about 1.9 dB at
that point — details and measurements in
[`../experiments/2026-09-10-paired-recalibration/NOTES.md`](../experiments/2026-09-10-paired-recalibration/NOTES.md).
The studies are being redone on the baseline in section 3.

## 6. Not comparable with the paper

Every absolute value here is within-setup only. This rig is image-only, reduced-scale, and trained
on one consumer GPU for hours rather than a cluster for days. Published numbers appear in this
repository only as *relative* gaps between two of mira's own rows, transcribed with provenance in
[`data/mira_layer_ablation.json`](data/mira_layer_ablation.json) and
[`data/mira_bottleneck_ablation.json`](data/mira_bottleneck_ablation.json).
