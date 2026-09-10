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

**Read `README.md` first, then `postprocessing/RESULTS.md` for what the results actually are, then
`codec/README.md`.** The latter is a current-state reference —
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
- **Experiment 1** (learned per-DINO-layer aggregation, replacing mira's fixed 7-layer mean) is
  **complete and it works**: both arms warm-started from the locked baseline and run to 200,000
  steps, `learned_mix` 27.905 dB against `control` 24.992 — a paired, matched-step **+2.914 dB**.
  Quote that gap, not "+3.16 over the plateau": the control ends 0.245 above the plateau rather
  than flat, which is the pre-registered "climbs a little" branch, and its rule is to report the
  matched-step gap. Do not compare either number to the paper's 27.6 Base-decoder row: different
  setup, and this rig's own faithful baseline sits at 24.75 where the paper's reaches 27.6.
  Extending the control also settled what it was run for — the variant's "takeoff at 104,000" is a
  shared artifact of the seed schedule, since both arms dip through 72k-96k and recover at 104,000,
  and the gap itself widens at 23 of 24 transitions independently of it. What remains open is
  downstream, not attribution: the learned weights didn't reweight the paper's layers, they
  abandoned them (92.4% of normalized mass on the 17 non-stock layers, 45.7% on layer 0 alone); the
  paper's own reasoning for its layer choice is about preserving semantics *for the world model*,
  and its one relevant ablation favors depth there too — so this is a demonstrated reconstruction
  win with an open question about the downstream latent, not a settled improvement to MIRA. Two
  things sharpen that: **P-DINO never separates the arms** (the gain is concentrated in PSNR, which
  mira's own layer ablation shows is the metric *least* sensitive to layer choice, by 14x-47x), and
  mira's stated methodology is to *select* codecs on downstream metrics while *training* them on
  reconstruction — so the fixed layer set is best read as a **regularizer** encoding what the loss
  cannot express, not as an untuned hyperparameter. On that reading, learning the weights removes a
  constraint rather than tuning one, and the gain is predicted not to transfer. See
  `codec/README.md`'s "Current state" and `codec/results/benchmark.jsonl` (tags `learned_mix-*` /
  `control-*`) for the numbers — don't assume the outcome from this file.
- Hardcoded paths were removed in favor of `direnv` (`.envrc`) + two env vars
  (`RS_DINO_WEIGHTS_DIR`, `MIRA_TRAIN`) so the repo isn't tied to one machine.

## Next, in order

Step 1 (run the control to a matched length) is **done** — both arms sit at 200,000 steps and
Experiment 1's attribution now rests on a full-length control. What follows is what is left.

### 1. Experiment 2, `learn7` (priority; see `experiments/2026-09-08-decompose-layer-mix/NOTES.md`)

Experiment 1 changed two things at once: the weights became *free*, and 17 shallower layers became
*reachable*. The result — 92.4% of the mass landing on layers the stock formula never reads —
points hard at reach, but that is inference, not measurement. `learn7` measures it: the same
machinery with only the stock 7 weights trainable. `learn7` minus `control` is the value of
freedom; `learned_mix` minus `learn7` is the value of reach.

This matters more than tidiness, because reach is exactly what is in tension with mira's semantic
rationale. If freedom alone recovers most of the gain, there is a version of this result that is
compatible with the paper's layer choice instead of opposed to it.

Scaffolding is in place. `learn7` is not a new class or a freezing mechanism — it is
`VideoCodecLearnedLayerMix` with `expose_layers: [11,13,15,17,19,21,23]`, so it simply never reads
the shallow blocks and its weight vector is 7 long rather than 24. Config at
`codec/configs/model/learned_layer_mix_learn7.yaml`, a `learn7` arm in the launcher, tests pinning
the exposure/zero-weight equivalence. What is left is the compute:

```bash
bash codec/scripts/run_learned_layer_mix_warmstart.sh 1 learn7    # 8k steps, ~1h + ~6min scoring
```

HOURS is relative and per arm: each invocation runs that many hours more from wherever the arm
currently sits. Repeat to ~200,000 to match the other two arms, roughly 25 hourly chunks. There is
no shortcut to a shorter run here — the gap between the existing arms was still widening at 200k,
so a `learn7` stopped early would understate whichever component it measures.

**Pre-registered outcomes** are in that NOTES.md and were written before the run. Read them there
rather than deciding after the fact.

### 2. Experiment 4, latent predictability (see `experiments/2026-09-10-latent-predictability/NOTES.md`)

**Cheapest experiment on the list and the only one that addresses the open question**, so it ranks
above Experiment 3 despite being written later. Nothing is retrained: the three arms stay frozen
and a small next-step predictor is fitted on the latents they already produce, as evaluation
apparatus. An afternoon, against ~25 h of GPU for a training arm.

It measures the property the reframing above turns on — whether the layer-0 latent is
disproportionately harder to predict than mira's deep-ish one. Read `FVU` only alongside PSNR: a
codec that encodes nothing is perfectly predictable, so the result is a point in
`(PSNR, FVU)` space and the question is whether `learned_mix` moved along a frontier or off it.

### 3. Experiment 3, cold start (see `experiments/2026-09-08-cold-start-layer-mix/NOTES.md`)

Its comparison arm already exists, since the locked baseline is itself a cold-start run under the
same protocol. ~35 h of GPU for the plateau alone, which is why it sits behind the probe.

## Conventions worth preserving

- **Figures are generated, prose quotes them.** `postprocessing/` reads
  `codec/results/benchmark.jsonl` and emits both the figures and `stats.md`; `RESULTS.md` quotes
  only what `stats.md` contains, and `stats.md` wins any disagreement. This exists because two
  claims in this repo's write-ups were wrong in ways a plot would have caught at once — a metric
  that never separated the arms, and a weight share attributed to the wrong quantity. Don't
  hand-type a number into a results document; regenerate and quote.
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
- **Prefer not doing a thing over doing it and disabling it.** `learn7` withholds the shallow
  layers by not reading them, rather than reading all 24 and freezing 17 weights at zero. The two
  are bit-identical, but the second needs a mask, frozen buffers, and a correct argument about what
  AdamW's decoupled weight decay does to a held weight — an argument this repo got wrong once
  before it got it right. Machinery that is hard to reason about is evidence against itself.
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
- **Annealing a converged constant-LR run to a low LR buys real dB, and a warm start gives it
  straight back.** A warm start (`finetune_from`) resets the optimizer and raises the LR again, so
  it drops below the annealed number and spends tens of thousands of steps recovering. Compare a
  constant-LR warm start against the pre-anneal plateau (24.747), not the post-anneal one (24.885).
  This entry used to say a warm start "can never reach the annealed number no matter how long it
  runs" — **that was too strong and is now measured false**: Experiment 1's control, which is the
  locked baseline continued at constant LR, passed 24.885 and finished at 24.992 by step 200,000.
  The rule of thumb still holds for reading a short run; the "never" did not. Corollary worth
  keeping: the plateau called at 272,000 was a stopping point, not an asymptote — constant LR was
  still buying ~+0.03 dB per 8k steps out at 200k.
- **`auto_weight` (on in every arm) rescales each perceptual term every step by the ratio of its
  gradient norm to the L1 anchor's, at the decoder's last layer** (VQ-GAN style, `codec/loss.py`).
  It is a per-step normalization, not a schedule and not a learned parameter, so it holds the loss
  mix roughly constant rather than letting it drift as a variant changes the latent. Worth knowing
  before suspecting it of confounding an arm-to-arm comparison: the factors are logged as
  `loss_*_auto_w` if you ever want to check rather than assume.
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
