# For an agent picking this up cold

If you're reading this without prior context on the session, here's the gist.

## What this project is

`kmira` is a research fork built on top of [`mira`](https://github.com/mira-wm/mira) — MIRA
(Multiplayer Interactive World Models with Representation Autoencoders), a real-time multiplayer
world model for 2v2 Rocket League from a technical report at `../mira/technical_report/`. `mira`
lives untouched as a sibling folder and is installed editable as a dependency; it is the
ground-truth baseline and is **never modified here**. When an idea needs to change a specific
piece (e.g. the codec's bottleneck), we fork just that file into `src/kmira/codec/variants/`
rather than vendoring the whole codec, so diffs against mira stay small and explicit.

Currently the entire focus is **the codec** — the video representation autoencoder (frozen
DINOv3-L feature extractor -> learned bottleneck -> ViT decoder) that MIRA's world model runs its
predictions in. Not the world model itself yet, though the layout (`codec/`, with other parts
getting their own sibling subfolder later) anticipates that.

**Read `README.md` first, then `codec/README.md`.** The latter is the detailed, numbered account
of every step taken so far — environment setup, dataset/weight gating, bugs found and fixed, the
calibration methodology, and the reasoning behind every non-obvious decision (e.g. why the Base
decoder instead of XL, why batch size doesn't buy throughput on this GPU). `CHANGELOG.md` is the
higher-level narrative version of the same history — read that instead if you just want the gist
without the full step-by-step. Don't duplicate content between `codec/README.md` and
`CHANGELOG.md`; add new work to `codec/README.md`'s numbered steps and summarize the milestone in
`CHANGELOG.md` when it's done.

## The goal

**Not** to reproduce mira's own codec quality — that needs a compute budget this project doesn't
have (mira trains 250k+ steps on 8 GPUs; this runs on one consumer 4070 Ti). The goal is a
benchmark that reliably answers *"is this change to the codec better than mira's stock one?"* at a
much smaller, single-GPU scale, using paired A/B comparisons against a locked baseline checkpoint.

## Where things stand (as of the last commit)

- A calibrated, plateaued, annealed **baseline codec** is locked at
  `checkpoints/calibration/plateau_baseline/checkpoint-304000` (304k steps, PSNR 24.88). This is
  the fixed comparison point for every future variant.
- **Experiment 1** (learned per-DINO-layer aggregation, replacing mira's fixed 7-layer mean) is
  built and warm-start-ready; see `codec/README.md` steps 16-21 and
  `src/kmira/codec/variants/learned_layer_mix.py`. Check `codec/results/benchmark.jsonl` for
  whatever the latest recorded comparison numbers are — don't assume the outcome from this file.
- Hardcoded paths were removed in favor of `direnv` (`.envrc`) + two env vars
  (`RS_DINO_WEIGHTS_DIR`, `MIRA_TRAIN`) so the repo isn't tied to one machine.

## Conventions worth preserving

- **No forked trainer.** Training runs mira's own `scripts/train_codec.py` unmodified; every
  divergence from mira is Hydra config (`codec/configs/`) or a small standalone module under
  `src/kmira/` (e.g. `lr_resume_override.py`, `pin_consistency_loss_layers.py`), not a copy-edit of
  mira's own code.
- **Verify before trusting.** Recurring pattern throughout the history: don't accept a metric or a
  result at face value — check it against a trivial baseline (a flat-gray-image PSNR caught the
  `[-1,1]` range bug), reproduce a known effect from the paper to calibrate the benchmark itself
  (the frozen-bottleneck A/B/C), and verify claims about the pipeline (resume, determinism,
  sampling coverage) by measurement, not by reading the code and assuming it does what it says.
- **One experiment, one control.** New codec ideas get a dedicated variant file under
  `src/kmira/codec/variants/`, initialized to reproduce the stock behavior exactly where possible
  (see `learned_layer_mix.py`'s byte-identical-at-init check), so any measured difference is
  attributable to the idea and not to an incidental change riding along with it.
- `checkpoints/` and `data/` are gitignored and local-only, shared across every part of this
  project (not nested under `codec/`). Don't expect them to be present after a fresh clone —
  `codec/README.md` has the steps to regenerate them (gated downloads + training).

## Setup, if you need to run anything

```bash
pixi run setup      # GPU
pixi run setup-cpu  # CPU only
pixi run test
```

Needs two separately-gated downloads (DINOv3 weights from Meta, `kyutai/rocket-science` from
HuggingFace) — see `README.md`'s Setup section, it's easy to conflate the two gates.
