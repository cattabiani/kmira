# Recalibration: the ablation and the seed, paired and run long

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

## Design: two runs, both questions

Two cold-started runs under the **baseline run's exact protocol**: constant LR 1e-4 after a
1,000-step warmup, hourly chunks of 8,000 steps, one checkpoint and one 2048-frame scoring per
chunk, **no anneal**. Nothing about the schedule differs between them or from the baseline run, so
the comparisons are clean.

| arm | model | seed base | what it measures |
|---|---|---|---|
| `abl_baseline` | `baseline_image_base` | 1028 | — |
| `abl_frozen` | `calib_frozen_bottleneck` | 1028 | ablation, *paired* against `abl_baseline` |
| *(existing)* `plateau_baseline` | `baseline_image_base` | 28 | seed replicate for `abl_baseline` |

**The two new arms share seed base 1028 deliberately.** They are then paired chunk for chunk, so
`abl_baseline − abl_frozen` is the frozen-bottleneck ablation with the seed spread cancelled — the
thing A/B could not resolve. And because the existing baseline run used base 28, `abl_baseline`
doubles as an independent-seed replicate of it. One pair of runs answers both questions.

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
   runs, which cannot estimate a spread at all. *Falsified* if `abl_baseline` and
   `plateau_baseline` end more than ~0.5 dB apart at a matched step, which would mean unpaired
   comparison is hopeless here at any length and every future study must be paired.

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

- **The seed comparison is confounded, and by how much is not knowable in advance.** The existing
  baseline run trained its first seven chunks (steps 0–56,000) on seed 28 repeatedly — the
  data-repetition bug — while `abl_baseline` will get seven distinct chunks. So `abl_baseline`
  minus `plateau_baseline` mixes the seed with early data coverage. If they land close, the bound
  is still useful; if they land far apart, the cause is ambiguous and a third baseline run would be
  needed to separate it. Say so rather than reporting it as a clean seed effect.
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
bash codec/scripts/run_plateau.sh 8 abl_frozen      # 8h slot, repeat to target
bash codec/scripts/run_plateau.sh 8 abl_baseline    # 8h slot, repeat to target
```

Cost per arm, at the measured 0.45 s/step plus 372s scoring per chunk:

| target | chunks | per arm | both arms |
|---|---|---|---|
| 96,000 (gap should be visible) | 12 | ~13.3h | ~26.6h |
| 200,000 (matches the warm-start arms) | 25 | ~27.7h | ~55.3h |
| 272,000 (matches the baseline run's elbow) | 34 | ~37.6h | ~75.2h |

**Recommended: run both to 96,000 first (~27h total), then decide.** The gap is paired, so it
should separate well before convergence; extending to 272,000 is only needed if hypothesis 1 looks
marginal or if the seed comparison (which does need a matched elbow) becomes the priority.

Note this queues behind the `learn7` arm currently training toward 200,000.

## What this unblocks

`postprocessing/RESULTS.md` currently has **no calibration section** — it was removed rather than
left standing on A/B/C, which cannot support it. When these two runs land, the foundation gets its
first section back, and for the first time the sentence "this benchmark can resolve a
bottleneck-sized effect" will be a measurement.
