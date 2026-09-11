# Codec benchmark

Everything for benchmarking and improving mira's codec (frozen DINOv3-L/16 encoder -> bottleneck ->
ViT decoder) lives here: configs, training/launch scripts, and results. Shared infrastructure that
any future part of this project would also need (raw datasets, weight caches, model checkpoints)
stays at the repo root (`../data/`, `../checkpoints/`) rather than nested here.

The goal is **not** to reproduce mira's codec quality — that needs a compute budget we don't have.
It is to build a benchmark that can reliably answer *"is this change to the bottleneck better than
mira's stock one?"* on a single consumer GPU.

**For history and reasoning** — why the Base decoder over XL, every bug found and how, the
calibration methodology, the full experiment trajectory — read `git log` (commit messages carry
this in detail) or `../CHANGELOG.md` for the narrative version. This file stays a current-state
reference; `../AGENTS.md` carries the non-obvious facts worth not re-learning the hard way.

## Layout

**There is no forked trainer.** Training runs mira's own `scripts/train_codec.py` unmodified; our
entire divergence from mira is configuration and small standalone patch modules under
`../src/kmira/` (see `../AGENTS.md`).

- `configs/kmira_train_codec.yaml` — the top-level training config, passed to mira's script via
  `--config-dir=codec/configs --config-name=kmira_train_codec`. Deliberately *not* named
  `train_codec.yaml`, which would ambiguously shadow mira's own config of that name.
- `configs/model/` — Hydra "model" group members, selected with `model=<name>`:
  - `baseline_image_base.yaml` — **the benchmark baseline**: mira's codec, image-only, Base decoder.
  - `baseline_image.yaml` — same but the stock XL decoder (kept for reference; too slow to iterate).
  - `calib_frozen_bottleneck.yaml` — the calibration downgrade (a frozen random bottleneck,
    targeting the paper's Table 4 "-1.4 dB" row).
  - `learned_layer_mix.yaml` / `learned_layer_mix_control.yaml` — Experiment 1 (below).
  - `learned_layer_mix_learn7.yaml` — Experiment 2's `learn7` arm: the same class as
    `learned_layer_mix.yaml` reading only mira's stock 7 blocks (`expose_layers`), i.e. freedom
    without reach.
- `configs/baseline.yaml` — standalone config for the visualization script (not a training entry).
- `scripts/run_calibration.sh` — the three-arm calibration run.
- `scripts/run_plateau.sh` / `run_anneal.sh` — constant-LR plateau search, then cosine anneal.
- `scripts/run_learned_layer_mix_warmstart.sh` — Experiment 1's launcher.
- `scripts/report_layer_mix.py` — prints Experiment 1's PSNR trajectory and learned-weight movement.
- `results/` — `benchmark.jsonl` (the accumulating metrics record) and reconstruction PNGs.
- `../src/kmira/codec/variants/` — real code forks, one file per idea.
- `../src/kmira/benchmark/` — `eval_codec.py` (metrics) and `visualize_reconstruction.py` (a
  before/after image for one frame).

## Reference numbers to compare against

From the paper (full video codec, fully trained). Our image-only, reduced-scale setup will not match
these absolutely — the *relative* effects are what we calibrate against:

| Source | PSNR | SSIM | LPIPS | P-DINO | rFDD |
|---|---|---|---|---|---|
| Baseline codec (Table 3) | 29.7 | 0.891 | 0.051 | 0.021 | 0.17 |
| Decoder Large (Table 7) | 29.3 | 0.882 | 0.055 | 0.022 | 0.17 |
| Decoder Base (Table 7) | 27.6 | 0.842 | 0.082 | 0.029 | 0.27 |

Known bottleneck effects, used as **calibration targets** — if our benchmark cannot resolve a gap
of roughly this size, it cannot be trusted to judge a new bottleneck either. Note these are
converged numbers, so a short calibration run is not the same measurement:

| Bottleneck (Table 4) | PSNR | LPIPS |
|---|---|---|
| Learned convolution (mira's default) | 29.7 | 0.051 |
| Random frozen projection | 28.3 | 0.067 |
| PCA + pooling | 28.4 | 0.068 |
| Pooling only | 29.2 | 0.055 |

Appendix Table 22 additionally ablates *which* DINO layers are aggregated — the subsystem Experiment
1 targets.

## Current state

**The reference baseline was retired on 2026-09-11, and no absolute dB here should be quoted.**
`checkpoints/calibration/plateau_baseline/checkpoint-304000` trained its first seven hourly chunks
on a fixed `run.seed`, so it saw 53.4% of the dataset for 56,000 steps and never reached the rest
(reproduce with `scripts/measure_data_coverage.py`). Its training speed, its 272,000-step elbow and
its plateau are artifacts of that. `run_plateau.sh baseline` refuses to extend it; `RETIRED.md`
sits beside the checkpoints.

**The new baseline is `baseline_v2`** (`checkpoints/calibration/ablation_baseline`, tags
`abl_baseline-*`), cold-started with a per-chunk seed from step 0 and therefore 100% coverage.
Still training. Its live trajectory is the one figure in `../postprocessing/RESULTS.md`.

**Queued**, in order: finish `baseline_v2` to its elbow and anneal; then `abl_frozen` (the
frozen-bottleneck ablation, sharing seed base 1028 so it is paired with `baseline_v2` on both
initialisation and data) and `baseline_v2_s2` (seed base 2028, the run-to-run spread); then
Experiments 1 and 2 redone. Costs and scope cuts are in `../AGENTS.md`; the design is pre-registered
in `../experiments/2026-09-10-paired-recalibration/NOTES.md`.

**Experiment 1** (learned per-DINO-layer aggregation,
`src/kmira/codec/variants/learned_layer_mix.py`) and **Experiment 2** (its freedom/reach
decomposition) are **superseded, not discarded**. Their arms were paired against each other — same
warm-start origin, same seed schedule, matched steps — so their *directions* hold: learned per-layer
weights beat their control substantially, and most of that came from reaching shallower DINOv3
blocks rather than from the weights being free. Their *magnitudes* rest on the retired baseline and
are being remeasured. The write-up, figures and generated numbers are archived whole at
`../postprocessing/archive/`.

  **What changed, and it is the part that survives**: the learned weights didn't reweight the
  paper's 7 blocks — they abandoned them, moving most of their normalized mass onto the 17 layers
  the stock formula never reads, with DINOv3's shallowest block taking the largest single share,
  while the deepest block the stock formula deliberately double-counts collapsed to near zero. Read
  the *normalized* direction, not the raw magnitudes — the aggregation's overall scale and the
  bottleneck projection's norm have an exact scaling symmetry that weight decay resolves
  arbitrarily, so only relative weight is identified.

  **P-DINO never separated the arms**, which is the finding that makes this a reconstruction result
  with an open question rather than an improvement: the gain concentrated in PSNR, the metric mira's
  own layer ablation shows to be *least* sensitive to layer choice, while the benchmark's one paired
  DINO-feature perceptual distance was indifferent between them.

  **Compare within this rig, not against the paper.** The Base-decoder row in the table above is not
  a like-for-like target: this setup is image-only and reduced-scale. The defensible claim is always
  a delta against a baseline measured here with its paired control, never crossing a number produced
  by a different setup.

## Running things

**Baseline and calibration runs.** All three use one protocol — constant LR after a short warmup,
hourly 8,000-step chunks, a checkpoint and a 2,048-frame scoring per chunk, a fresh seed per chunk —
so they are directly comparable. Slots accumulate: re-run with more hours to extend an arm.

```bash
bash codec/scripts/run_plateau.sh 8                  # baseline_v2, the reference baseline
bash codec/scripts/run_plateau.sh 8 abl_frozen       # frozen-bottleneck ablation, paired with it
bash codec/scripts/run_plateau.sh 8 baseline_v2_s2   # second seed, for the run-to-run spread
bash codec/scripts/run_anneal.sh                     # cosine-decay from wherever a run stopped
```

Runs in the **foreground** — deliberately not detached, so an arm lives only as long as its terminal
and leaves no orphaned process. Interrupted runs resume from the last checkpoint.

`run_calibration.sh` ran an earlier three-arm study (baseline, frozen bottleneck, baseline again at
another seed) which came back **TOO NOISY**: one run per arm cannot estimate a spread, and all three
stopped undertrained. That result is why the protocol above is paired, and the paired replacement is
pre-registered in `../experiments/2026-09-10-paired-recalibration/NOTES.md`.

**Scoring a checkpoint directly:**

```bash
python -m kmira.benchmark.eval_codec \
  --checkpoint checkpoints/<run>/checkpoint-<N>/checkpoint.pth \
  --n-frames 2048 --tag my-variant
```

Appends a row to `results/benchmark.jsonl`. `--n-frames 256` is a fast check, but ignore rFDD at
that size (it fits a 768x768 covariance and is rank-deficient below ~2048 frames).

**Experiment 1** (learned layer mix, warm-started from a baseline checkpoint — superseded, see
"Current state"; it will be re-run from `baseline_v2`'s annealed checkpoint):

```bash
bash codec/scripts/run_learned_layer_mix_warmstart.sh 8 learned_mix          # 8h more, variant only
bash codec/scripts/run_learned_layer_mix_warmstart.sh 8 control,learned_mix  # + paired control
```

`HOURS` is relative to each arm's own current checkpoint, not an absolute target — an arm at step
136,000 given `8` trains to 200,000. Prints a PSNR trajectory and the learned per-layer weights
(`report_layer_mix.py`) at the end.
