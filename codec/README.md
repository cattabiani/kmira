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
    reproducing the paper's Table 4 "-1.4 dB" row).
  - `learned_layer_mix.yaml` / `learned_layer_mix_control.yaml` — Experiment 1 (below).
  - `learned_layer_mix_learn7.yaml` — Experiment 2's `learn7` arm: freedom without reach.
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

Known bottleneck effects, used as **calibration targets** — if our benchmark cannot reproduce a gap
of roughly this size, it cannot be trusted to judge a new bottleneck either:

| Bottleneck (Table 4) | PSNR | LPIPS |
|---|---|---|
| Learned convolution (mira's default) | 29.7 | 0.051 |
| Random frozen projection | 28.3 | 0.067 |
| PCA + pooling | 28.4 | 0.068 |
| Pooling only | 29.2 | 0.055 |

Appendix Table 22 additionally ablates *which* DINO layers are aggregated — the subsystem Experiment
1 targets.

## Current state

- **Locked reference baseline**: `checkpoints/calibration/plateau_baseline/checkpoint-304000`
  (304,000 steps: 272,000-step constant-LR plateau + 32,000-step cosine anneal). PSNR 24.885
  (`benchmark.jsonl` tag `anneal-304000`). The constant-LR plateau itself, **24.747** at step
  272,000, is the number to compare a warm-started (constant-LR) run against — see the gotcha in
  `../AGENTS.md` about why those two numbers answer different questions.
- **Experiment 1** (learned per-DINO-layer aggregation,
  `src/kmira/codec/variants/learned_layer_mix.py`): **complete, both arms at 200,000 steps.**
  `learned_mix` 27.905 dB against `control` 24.992 dB — a paired, matched-step gain of
  **+2.914 dB**, and not a PSNR-only effect: SSIM (0.8587 vs 0.8105), LPIPS (0.0820 vs 0.1059) and
  rFDD (0.6209 vs 0.6740) all move together with it.

  Quote the arm-to-arm gap, not "+3.16 dB over the plateau". The control ends 0.245 dB above the
  24.747 plateau it warm-started from rather than dead flat, so the pre-registered rule applies
  (`experiments/2026-08-13-learned-layer-mix/NOTES.md`) and the matched-step gap is the result.

  The control's extension also closed the open question about the variant's jump at 104,000: both
  arms show the same dip through 72k-96k and the same recovery at 104,000, so that shape belongs to
  the shared seed schedule, not the intervention. The gap widens at 23 of 24 transitions, from
  +0.236 at 8,000 to +2.914 at 200,000, still widening at the end but decelerating (+0.038/8k over
  the last 40,000 steps, against +0.225/8k over the first 56,000). Check
  `results/benchmark.jsonl` (tags `learned_mix-*` / `control-*`) for the numbers themselves.

  **Compare within this rig, not against the paper.** The 27.6 Base-decoder row in the table above
  is not a like-for-like target: this setup is image-only and reduced-scale, and its own faithful
  baseline sits at 24.747 where the paper's Base decoder reaches 27.6. The defensible claim is the
  delta against the locked baseline with its paired control, not crossing a number produced by a
  different setup.

  **What changed**: the learned weights didn't reweight the paper's 7 blocks — they abandoned them.
  At 200,000 steps **92.4% of the normalized mass sits on the 17 layers the stock formula never
  reads**, 7.6% on the stock seven, with **layer 0** — the shallowest DINOv3 block — the largest
  single share at 45.7% (84.6% by energy). Layer 23, which the stock formula deliberately
  double-counts, fell from 1.1429 to 0.0003. Read the *normalized* direction, not the raw
  magnitudes — the aggregation's overall scale and the bottleneck projection's norm have an exact
  scaling symmetry that weight decay resolves arbitrarily, so only relative weight is identified.
  (Earlier revisions of this file reported the 92% figure as mass on layer 0; it is the collective
  share of the non-stock layers. Layer 0 dominates either way.)

  **Caveat before treating this as final**: the paper's own DINOv3-L layer choice (`sections/4.method.tex`)
  keeps a residual on the deepest selected block specifically to retain semantics that shallow
  features lack, reasoning that matters for the *world model* which later predicts forward in this
  latent, not just for reconstruction. Their one layer-choice ablation (multi-layer vs. last-block-only,
  `sections/appendix.tex`) shows the multi-layer choice winning on downstream world-model metrics
  (gFID/gFVD/gFDD), not just reconstruction — real evidence that latent semantics affect the world
  model, not just paranoia. Nothing in the paper tests the regime our result landed in (layer 0
  dominant, deep residual nearly gone), so "beats the paper on reconstruction" and "will be a good
  latent for the world model" are separate claims — only the first has evidence behind it so far.

## Running things

**Calibration** (reproduces the paper's known frozen-bottleneck effect, to trust the benchmark
itself before using it):

```bash
bash codec/scripts/run_calibration.sh          # all three arms, ~2.2h each
bash codec/scripts/run_calibration.sh 4000     # shorter probe, ~35 min each
```

Runs in the **foreground** — deliberately not detached, so it lives only as long as its terminal and
leaves no orphaned process. Interrupted runs resume from the last checkpoint. Each arm scores its
final checkpoint automatically and the script prints a signal-vs-noise verdict.

**Plateau search + anneal** (constant LR located a stopping point at step 272,000; a further cosine
anneal recovered +0.13 dB — this is how the locked baseline above was produced):

```bash
bash codec/scripts/run_plateau.sh          # ~8h per slot, extend by re-running with more hours
bash codec/scripts/run_anneal.sh           # cosine-decay from wherever the plateau stopped
```

**Scoring a checkpoint directly:**

```bash
python -m kmira.benchmark.eval_codec \
  --checkpoint checkpoints/<run>/checkpoint-<N>/checkpoint.pth \
  --n-frames 2048 --tag my-variant
```

Appends a row to `results/benchmark.jsonl`. `--n-frames 256` is a fast check, but ignore rFDD at
that size (it fits a 768x768 covariance and is rank-deficient below ~2048 frames).

**Experiment 1** (learned layer mix, warm-started from the locked baseline):

```bash
bash codec/scripts/run_learned_layer_mix_warmstart.sh 8 learned_mix          # 8h more, variant only
bash codec/scripts/run_learned_layer_mix_warmstart.sh 8 control,learned_mix  # + paired control
```

`HOURS` is relative to each arm's own current checkpoint, not an absolute target — an arm at step
136,000 given `8` trains to 200,000. Prints a PSNR trajectory and the learned per-layer weights
(`report_layer_mix.py`) at the end.
