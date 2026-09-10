# Results, in pictures

The codec benchmark's numbers live in `codec/results/benchmark.jsonl`, one row per scored
checkpoint. That file is the record but it is not readable — this page is the readable version.

**Every figure and every number here is generated, not written.** The figures come from the
`plot_*.py` scripts beside this file; the numbers are quoted from
[`stats.md`](stats.md), which `make_all.py` regenerates from `benchmark.jsonl` and the committed
`data/*.json`. If a number here disagrees with `stats.md`, `stats.md` is right. Regenerate with:

```bash
pixi run python postprocessing/make_all.py
```

Results are split into **established** (paired, matched-step, complete) and **provisional** (arms
still running, or interpretation rather than measurement). The split is the point of the page.

---

## Established

### 1. Learning the DINO layer weights gains +2.914 dB over its paired control

![PSNR trajectories for all three arms](figures/arm_trajectories.png)

Three arms, all warm-started from the same locked baseline (`checkpoint-304000`), same per-chunk
seed schedule, same 7-layer consistency loss. They differ only in which DINOv3 blocks the
aggregation reads and whether its weights can move.

At a matched 200,000 steps: **`learned_mix` 27.905 dB against `control` 24.992 dB**.

Quote that gap, not "+3.16 dB over the plateau". The control does not sit on the 24.747 plateau it
started from — it ends 0.245 above it, and above the annealed 24.885 too. The matched-step
arm-to-arm gap is the number this design supports.

Two things visible in the figure that are easy to miss in a table:

- **Both arms dip through 72k–96k and recover at 104k.** That shape belongs to the shared seed
  schedule, not to the intervention. It is why the control had to be run to full length: before it
  was, the variant's jump at 104k looked like it might be the idea.
- **The control keeps creeping up.** Constant LR was still buying roughly +0.03 dB per 8k steps out
  at 200k, so the plateau called at step 272,000 was a stopping point, not an asymptote.

### 2. The gap widens monotonically, and it is mostly *reach* rather than *freedom*

![Total gap and its decomposition](figures/gap_decomposition.png)

The gap grows from +0.236 dB at 8k to +2.914 dB at 200k, **widening at 23 of 24 step-to-step
transitions**. The one exception is 96k→104k, where the control climbed out of the shared dip a
reading before the variant.

The right panel splits it, which needs all three arms scored at the same step:

- **freedom** = `learn7 − control`: the weights are allowed to move, over mira's own 7 blocks
- **reach** = `learned_mix − learn7`: the other 17, mostly shallower, blocks become available

At 64k, freedom is **+0.367 dB — 20% of the gap**, down from 57% at 8k. See the provisional section
below: `learn7` is only scored to 64,000 steps.

### 3. The metrics disagree, and P-DINO does not separate the arms at all

![All five benchmark metrics per arm](figures/metric_panel.png)

At matched 200k, `learned_mix` over `control`: PSNR **+11.7%**, LPIPS **+22.5%**, SSIM **+5.9%**,
rFDD **+7.9%** — and P-DINO **−0.8%**, i.e. very slightly *worse*.

Look at the P-DINO panel: the two arms lie on top of each other for the whole run. The sign of the
difference flips six times, the gap is about 1% either way, and *within* either arm P-DINO swings
~20% following the shared dips. The metric is dominated by training phase, not by which arm you are
in.

Both arms do improve P-DINO against the plateau's `0.000100595` — but the control improves it
slightly more (`9.68847e-05` against `learned_mix`'s `9.76208e-05`), so none of that is
attributable to the aggregation. The claim was true against the plateau and false arm-to-arm.

So the +2.914 dB buys pixel fidelity, perceptual distance and distributional feature fidelity, and
buys **nothing measurable** on the benchmark's one paired DINOv3-feature perceptual distance. That
is an absence of gain, not a degradation.

### 4. Each arm puts its mass on the shallowest block it is allowed to read

![Normalised weight share per DINOv3 block, per arm](figures/layer_weight_shares.png)

This is the mechanism, and it is the most legible result here.

| arm | can read | largest block | share |
|---|---|---|---|
| `control` | all 24, weights frozen | **L23** — the deepest | 57.1% |
| `learn7` | blocks 11–23 only | **L11** — shallowest it can reach | 76.6% |
| `learned_mix` | all 24 | **L0** — shallowest of all | 45.7% |

The top panel *is* mira's stock formula, drawn: the mean over seven blocks plus a residual on the
deepest, which puts 57.1% of the mass on L23. Each arm that is allowed to move its weights then
dumps mass on the shallowest block available to it, and abandons that deep residual —
`learned_mix` moves 92.4% of its mass onto the 17 blocks mira never reads, leaving 7.6% on mira's
seven.

Read shares, never raw magnitudes: the aggregation has an exact scaling symmetry with the
bottleneck projection, resolved arbitrarily by weight decay, so only the normalised direction is
identifiable. `learned_mix`'s raw weight vector collapsed from an init of L2 1.1952 to **0.0127**
while its PSNR climbed 3 dB (`learn7`: 0.0564), with the bottleneck projection absorbing the
scale. The collapse is that symmetry being resolved, not a signal.

### 5. The gain is concentrated in the metric least sensitive to layer choice

![Metric signature: ours beside mira's own layer ablation](figures/metric_signature.png)

mira's own layer ablation (published, transcribed with provenance in
[`data/mira_layer_ablation.json`](data/mira_layer_ablation.json)) moves in the opposite shape.
Reading seven blocks instead of the deepest one alone gains them **+1.0% PSNR** and **+0.5% SSIM**
— but **+30.8% rFID, +29.8% rFVD, +32.0% rFDD**. The pixel and structural metrics are the two
*least* layer-sensitive things in their table, by 30–60× against the Fréchet distances.

Ours is the reverse: PSNR moves **most** (+11.7%) and the DINO-feature metrics lag (rFDD +7.9%,
P-DINO −0.8%).

Their generation metrics sit in between (gFID +12.3%, gFVD +5.4%, gFDD +11.3%) — so it is *not*
that downstream metrics are the sensitive ones. PSNR is simply the outlier.

Two comparisons of two different interventions, so only the **shape** transfers, never the
magnitudes — different setups, and some metric normalisations differ too (mira reports P-DINO
around 0.021 where this rig reports ~9.7e-05 for the same kind of quantity). What transfers is
that a gain concentrated in the least layer-sensitive metric is not what finding a better latent
looks like.

---

## Provisional

Real measurements, but not yet quotable as results.

### `learn7` is only a third of the way

It is scored to 64,000 steps against the other arms' 200,000. The freedom/reach split above is
therefore provisional in both directions: `learned_mix` had banked only +1.797 dB of its eventual
+2.914 dB at 64k — under two-thirds — so this is the part of the curve that turned out least
representative. The shared 72k–96k dip is also immediately ahead of where `learn7` currently sits.

Directionally the pre-registered hypothesis is holding (freedom's share falls at every reading),
but the number to report is the split at a matched 200k, and that does not exist yet.

### Nothing here tests the world model

The benchmark scores reconstruction only. Whether a layer-0-dominant latent serves the world model
that has to predict in it is untested, and there is a specific reason to expect trouble — see
`experiments/2026-08-13-learned-layer-mix/NOTES.md` for the argument, and
`experiments/2026-09-10-latent-predictability/NOTES.md` for the pre-registered experiment that
would measure the part of it reachable without a world model.

### The "regularizer" reading is an argument, not a measurement

The interpretation that mira's fixed layer set is a *regularizer* — encoding downstream knowledge
its reconstruction loss cannot express, so learning the weights removes a constraint rather than
tuning a hyperparameter — is consistent with everything on this page and with mira's own stated
methodology. It is not established by anything on this page. Experiment 4 is what would test it.

---

## Not comparable with the paper

Every absolute number here is within-setup only. This rig is image-only and reduced-scale on one
consumer GPU; its own faithful reproduction of mira's stock codec plateaus at 24.747 dB where the
paper's Base decoder reaches 27.6. So `learned_mix`'s 27.905 is **not** "past the paper's Base
decoder", and the figures above are never to be read against the paper's tables. The one place
published numbers appear is figure 5, where only the relative shape is used.
