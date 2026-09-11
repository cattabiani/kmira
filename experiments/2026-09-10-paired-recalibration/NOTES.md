# Recalibration: a new baseline, the ablation, and the seed

**Status: pre-registered, not run. This is the project's highest-priority compute.** Written
before either run exists, so the design and the falsification conditions are on record rather than
fitted to whatever comes out.

## Why this is priority one

The benchmark's trust rests on the original A/B/C calibration, and that study establishes almost
nothing:

- **One run per arm.** A − B = +1.4489 dB against a seed spread |A−C| = 1.142 dB, so the effect is
  1.27× the noise where the launcher's own criterion demanded 3×. Verdict at the time: TOO NOISY.
  The point estimate's closeness to mira's published ≈1.4 dB carries almost no information.
- **All three stopped undertrained.** 15,299 steps, with 39% of the loss descent and 4.762 dB of
  PSNR still ahead of them (the baseline run later needed 272,000). A gap between two undertrained
  models mixes how fast each trains with how good each gets, so this was never mira's ablation at
  mira's operating point.
- **Unpaired.** Section 0d of `postprocessing/RESULTS.md` later showed the per-chunk data slices
  are wildly uneven — seed 40's chunk is the best in every run that met it, seeds 36–39 the worst.
  Unpaired short runs therefore measure the seeds as much as the intervention.

So the two numbers the whole benchmark leans on — *can it resolve a known bottleneck-sized effect*
and *how big is the seed spread* — are both unmeasured. Everything else in this repo is a
comparison whose credibility depends on them. That is why this outranks Experiments 3 and 4.

## Design: three runs

Cold-started under one protocol: constant LR 1e-4 after a 1,000-step warmup, hourly chunks of
8,000 steps, one checkpoint and one 2048-frame scoring per chunk, **no anneal**, and a per-chunk
seed from the very first step. Nothing about the schedule differs between them, so the comparisons
are clean.

| arm | model | seed base | role |
|---|---|---|---|
| `baseline_v2` *(alias `abl_baseline`)* | `baseline_image_base` | 1028 | **the baseline** |
| `abl_frozen` | `calib_frozen_bottleneck` | 1028 | ablation, *paired* against `baseline_v2` |
| `baseline_v2_s2` | `baseline_image_base` | 2028 | seed replicate of `baseline_v2` |

**A second, free read on `baseline_v2_s2`, pre-registered before it runs.** `baseline_v2`'s
per-chunk increment collapsed to +0.029 dB at step 104,000 and the next chunk returned +0.199 — a
false elbow, and the first on this rig that the data-repetition bug cannot explain. Two candidate
causes: the *slice* (seed 1041 is simply a poor 8,000 steps) or the *step* (something about
training dynamics around 104k). `abl_frozen` cannot separate them, because it shares seed base
1028 and therefore meets seed 1041 at step 104,000 too — step and seed are perfectly confounded.
`baseline_v2_s2` is base 2028, so it meets *different* seeds at the *same* steps:

- **`baseline_v2_s2` also dips near 104,000** → the cause is the step, and it is a property of this
  training setup rather than of the data.
- **It does not** → the cause is the slice, which is the same mechanism the archived section 0c
  documented, now reproduced on clean data.

Either way, record it. This costs nothing extra: the run is already required for the seed spread.

**`baseline_v2` and `abl_frozen` share seed base 1028 deliberately.** They are then paired chunk
for chunk, so `baseline_v2 − abl_frozen` is the frozen-bottleneck ablation with the seed spread
cancelled — the thing A/B could not resolve.

**The seed replicate needs its own run (`baseline_v2_s2`), which was not the original plan.** The
plan was to read `baseline_v2` against the existing baseline run, whose seed base is 28. That is
dead: see "The legacy baseline is retired" below. The two runs differ by far more than a seed, so
the comparison measures the bug, not the seed.

## The legacy baseline is retired (2026-09-11)

The original baseline run's first seven chunks all ran `run.seed=28` — the data-repetition bug.
A chunk consumes ~41% of the stream and the seed fixes the shard traversal order, so those seven
chunks replayed the **same 53.4% of the training data** seven times and **46.6% was never seen**
until step 56,000. `baseline_v2` had a per-chunk seed from step 0 and reached **100% coverage**
over the same seven chunks (measured by replaying the loader's own `_my_shards` / `rng` logic over
the real index).

The cost, at matched steps, accumulates inside the replay window and is then carried:

| step | legacy baseline | `baseline_v2` | gap | Δgap |
|---|---|---|---|---|
| 8,000 *(both clean)* | 19.414 | 19.485 | +0.071 | |
| 16,000 *(replay begins)* | 20.123 | 20.299 | +0.176 | +0.105 |
| 32,000 | 20.829 | 21.956 | +1.127 | +0.920 |
| 56,000 *(replay ends)* | 21.270 | 23.138 | +1.868 | +0.127 |
| 88,000 | 21.731 | 23.795 | +2.065 | +0.033 |

**+1.797 dB accumulated over the six replay chunks; +0.196 over the next four.** A ~9× difference
in rate, which is the signature of a coverage deficit rather than a per-chunk sampling fluke — and
it explains the persistence that ordinary seed variance could not.

Consequences:

- The legacy run's **training speed, its 272,000-step elbow and its 24.747 dB plateau are artifacts
  of the bug**, not properties of this rig. None of them should be quoted as what the rig reaches
  or how long it takes. `run_plateau.sh baseline` now refuses to run.
- Its checkpoints stay on disk, with `RETIRED.md` beside them. `checkpoint-304000` is the
  warm-start origin of the Experiment 1/2 arms, and `learn7` still has chunks to run from it.
  Those results are **paired** — every arm shares that origin and seed schedule — so they remain
  valid as *arm-to-arm* comparisons. What is invalid is the absolute baseline level.
- `baseline_v2` reached 23.869 dB at 96,000, which the legacy run did not reach until roughly
  200,000. So the elbow is likely to land much earlier and higher, and the "this rig plateaus at
  24.747 where the paper's Base reaches 27.6" line in `postprocessing/RESULTS.md` is probably
  understating the rig.

### What actually trains in `abl_frozen`

Only the decoder. The encoder is a frozen DINOv3-L/16 backbone (frozen in stock mira too) followed
by the bottleneck projection, and this arm freezes that projection at its random initialisation —
`requires_grad_(False)` in `src/kmira/codec/variants/frozen_bottleneck.py`, nothing else changed.

The projection is `Conv2d(1024 → 32, kernel 2, stride 2, bias)` = **131,104 parameters**, so
trainable drops 114,188,320 → 114,057,216: the ablation removes **0.115%** of the trainable
parameters. mira report it costing 1.4 dB. That ratio is the whole reason it is a good calibration
target — a large effect from a tiny, unambiguous change.

## Hypotheses, and what falsifies them

1. **The ablation resolves once paired and run long.** `abl_baseline − abl_frozen` settles to a
   stable gap several times the residual chunk-to-chunk wobble. *Falsified* if the paired gap stays
   inside the wobble, which would mean the benchmark cannot see a bottleneck-sized effect even
   under the good protocol — and would put every result in this repo in question.
2. **The gap is in the neighbourhood of mira's 1.4 dB.** Not equal: different scale, image-only,
   Base decoder. *Falsified* if it comes out several times larger or smaller, which would say the
   reduced rig distorts the effect rather than shrinking with it.
3. **The seed spread is usually small, with occasional large draws.** The 1.142 dB came from two
   runs, which cannot estimate a spread at all. *Falsified* if `baseline_v2` and `baseline_v2_s2`
   end more than ~0.5 dB apart at a matched step, which would mean unpaired comparison is hopeless
   here at any length and every future study must be paired.

   Early evidence for the "occasional large draw" shape, from the first chunk of `abl_baseline`
   (2026-09-10): at comparable steps A (seed 28, cosine) sits at 0.7143 val loss, the baseline run
   (seed 28, constant) at 0.7069 and `abl_baseline` (seed 1029, constant) at 0.7078 — while C (seed
   1234, cosine) sits at 0.5782. Three cluster within 0.008 and C is 0.13 away, including a run
   with a *different* seed landing with the pack. So the original |A−C| looks like one unusual draw
   rather than a typical spread. Suggestive only: A and C ran cosine where the other two are
   constant-LR.

   Investigated and ruled out as causes of the A/C gap (2026-09-10), so this is not an artifact:
   the two configs are identical apart from `seed` and `output_dir`; mira's validation loader is
   hardcoded `seed=37`, so val losses are computed on the same data regardless of `run.seed`; the
   dataset index predates every run and is unchanged; and the eval-sampling biases could only move
   `eval_codec`'s PSNR, while mira's independent val loop shows the same gap. C separates from A
   within 1,500 steps and holds it, which is an init-and-ordering effect from the start.

## Watch-outs

- **This watch-out fired.** It was written as "the seed comparison against the legacy run is
  confounded by early data coverage, and a third baseline run may be needed". It was confounded,
  the confound was the dominant term, and the third run (`baseline_v2_s2`) is needed. Measured
  rather than guessed: 53.4% coverage against 100%, worth ~1.9 dB.
- **Do not call the ablation early.** This project has hit apparent plateaus twice, both traced to
  the seed schedule (0d). Pairing removes that for the *gap*, but each arm's own curve will still
  show the shape.
- **Read the gap, not the levels.** Both arms are cold-started, so their absolute dB pass through
  the same shared dips.
- **No anneal.** Deliberate: the anneal buys ~+0.138 dB from the low LR alone and is not a
  different region of parameter space, so it adds ~4.4h per arm and changes no comparison here. The
  locked baseline keeps its anneal because it is a warm-start origin; these two are not.
- Both arms keep `checkpoint_keep_recent=6` (~28GB each) and are scored every chunk at 2048 frames,
  so the dB curve exists this time — the thing A/B/C permanently lack.

## Running it

One GPU, so the arms run sequentially. Slots accumulate, so any number of hours can be added later.

```bash
bash codec/scripts/run_plateau.sh 8                  # baseline_v2 (default), currently running
bash codec/scripts/run_plateau.sh 8 abl_frozen       # the ablation, paired with it
bash codec/scripts/run_plateau.sh 8 baseline_v2_s2   # the seed replicate
```

Cost per arm, at the measured 0.45 s/step plus 372s scoring per chunk:

| target | chunks | per arm | three arms |
|---|---|---|---|
| 96,000 (gap should be visible) | 12 | ~13.3h | ~40h |
| 200,000 (matches the warm-start arms) | 25 | ~27.7h | ~83h |
| 272,000 (the legacy elbow, for reference only) | 34 | ~37.6h | ~113h |

`baseline_v2` is already past 96,000, so only the two new arms owe that.

**Recommended: bring `abl_frozen` to 96,000 first (~13h), then `baseline_v2_s2` (~13h).** The gap is paired, so it
should separate well before convergence; extending to 272,000 is only needed if hypothesis 1 looks
marginal or if the seed comparison (which does need a matched elbow) becomes the priority.

Note this queues behind the `learn7` arm currently training toward 200,000.

## What this unblocks

`postprocessing/RESULTS.md` currently has **no calibration section** — it was removed rather than
left standing on A/B/C, which cannot support it. When these two runs land, the foundation gets its
first section back, and for the first time the sentence "this benchmark can resolve a
bottleneck-sized effect" will be a measurement.
