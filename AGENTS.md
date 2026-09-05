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

**Read `README.md` first, then `codec/README.md`.** The latter is a current-state reference —
layout, how to run things, the locked baseline, where each experiment stands — not a history. For
*why* things are the way they are, `git log` carries it (commit messages are written with that
detail) and `CHANGELOG.md` is the narrative summary. Don't let `codec/README.md` grow back into a
changelog: new work updates its "Current state" section and gets a real commit message; the
reasoning trail lives in the commit, not in prose duplicated across files.

## The goal

**Not** to reproduce mira's own codec quality — that needs a compute budget this project doesn't
have (mira trains 250k+ steps on 8 GPUs; this runs on one consumer 4070 Ti). The goal is a
benchmark that reliably answers *"is this change to the codec better than mira's stock one?"* at a
much smaller, single-GPU scale, using paired A/B comparisons against a locked baseline checkpoint.

## Where things stand (as of the last commit)

- A calibrated, plateaued, annealed **baseline codec** is locked at
  `checkpoints/calibration/plateau_baseline/checkpoint-304000` (304k steps, PSNR 24.88). This is
  the fixed comparison point for every future variant.
- **Experiment 1** (learned per-DINO-layer aggregation, replacing mira's fixed 7-layer mean) works:
  warm-started from the locked baseline, reached 27.429 dB by step 136,000 (+2.68 over the plateau
  it started from) while a paired frozen-weight control stayed flat. Still running. See
  `codec/README.md`'s "Current state" and `codec/results/benchmark.jsonl` (tags `learned_mix-*` /
  `control-*`) for the latest numbers — don't assume the outcome from this file.
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
  see `README.md`'s Setup section (gated downloads) and `codec/README.md` (training).

## Gotchas that will bite you again if forgotten

Non-obvious facts about mira and this environment, each found the hard way once already. Re-finding
any of these costs real time or a real bug, so they live here rather than only in git history.

- **`VideoCodec`'s video tensors are in `[-1, 1]`, not `[0, 1]`.** Every mira metric and
  visualization utility expects `[0, 1]` — convert with `(x + 1) / 2` before scoring. Getting this
  wrong doesn't error, it silently zeroes every negative pixel (a real PSNR once scored *worse than
  a flat gray image* because of this).
- **mira's train loader reseeds from `run.seed` on every process start and is not checkpointed.** A
  script that restarts training in chunks (to checkpoint/score hourly, say) with a fixed seed
  replays the *identical* data stream on every restart. Vary the seed per chunk (see
  `run_plateau.sh`, `run_learned_layer_mix_warmstart.sh`).
- **`CodecLoss.bind_encoder_dino` derives the DINO latent-consistency loss's layer set from the
  encoder's own layers.** Changing which DINO layers the encoder reads silently changes the
  training *objective* too, not just the aggregation — unless pinned
  (`src/kmira/pin_consistency_loss_layers.py`). The pairing between the encoder's returned features
  and the loss's targets goes through a **non-strict** `zip`, so a mismatch misaligns layers
  silently instead of raising; verify numerically, not by inspection.
- **`VideoCodec.load_from_checkpoint` ignores `_target_` and always rebuilds a stock `VideoCodec`.**
  Loading a variant checkpoint through it fails the strict `load_state_dict` on any extra
  parameters — after paying for the training time, not before. Instantiate through Hydra instead
  (`load_codec_respecting_target` in `eval_codec.py`).
- **`torch.hub` does an unconditional GitHub `urlopen` even when the repo is already cached
  locally**, and its `except URLError` doesn't catch `RemoteDisconnected` — one dropped connection
  crashes model construction outright. `src/kmira/torch_hub_offline.py` patches it to resolve from
  the cache first.
- **`/tmp` is tmpfs (RAM-backed), and this machine has only ~512MB of swap.** A memory spike (e.g.
  writing a multi-GB checkpoint there) hard-locks the machine instead of degrading gracefully.
  Never write checkpoints or large scratch files under `/tmp`.
- **Annealing a converged constant-LR run to a low LR buys real dB that constant LR alone never
  reaches — but that gain is a property of the low LR, not a better region of parameter space.** A
  warm start (`finetune_from`) resets the optimizer and raises the LR back up, so it can never reach
  the annealed number no matter how long it runs. Compare a constant-LR warm-start against the
  pre-anneal plateau, not the post-anneal one.
- **Calling a plateau/elbow needs several trailing readings, not one flat stretch.** This project
  has hit real false plateaus more than once — a multi-hour flat stretch followed by a further jump
  of over 1 dB. Stopping on the first flat reading has already cost real signal here.

## Setup, if you need to run anything

```bash
pixi run setup      # GPU
pixi run setup-cpu  # CPU only
pixi run test
```

Needs two separately-gated downloads (DINOv3 weights from Meta, `kyutai/rocket-science` from
HuggingFace) — see `README.md`'s Setup section, it's easy to conflate the two gates.
