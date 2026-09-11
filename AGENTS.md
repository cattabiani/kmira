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

**Scope the experiment to the resources, not the ambition.** Every reduction here — the Base
decoder instead of XL, image-only instead of temporal, one dataset — exists so that what remains
can be run *properly*: to its elbow, paired, and long enough that the gap is not a difference in
training speed. A smaller experiment carried to completion answers a question; a bigger one stopped
early answers none. When a proposal does not fit the GPU, cut the scope before cutting the
protocol.

## Plan of record: rebuild on the clean baseline (2026-09-11)

The retired baseline invalidates every *absolute* number in the repo, so the studies are being
redone on `baseline_v2`. Priced from the live run's measured 1.20 h per 8k chunk (cold) and 1.00 h
(warm start), the full programme is **~154 h if `baseline_v2` elbows at 200,000, ~186 h at
272,000** -- 6.5-8 days of continuous GPU. The elbow is genuinely unknown and may be *later* than
the legacy 272,000, since a run seeing 100% of the data has more to learn, not less. The increment
panel decides it, not a guess.

Run in three phases so that value lands early rather than all at the end:

**Phase 1 -- get a trustworthy baseline (~19 h).** Continue `baseline_v2` to its elbow, then anneal
32,000 steps. Everything else is blocked on this, and on its own it already replaces the single
most load-bearing number in the repo.

**Phase 2 -- the calibration that never existed (~60 h).** `abl_frozen` (paired, seed base 1028)
and `baseline_v2_s2` (seed base 2028), both to a matched step. This is the first time the benchmark
will have a measured effect-vs-noise ratio at its own operating point.

**Phase 3 -- redo Experiments 1 and 2 (~75 h).** `control`, `learned_mix`, then `learn7`, warm
started from Phase 1's checkpoint, 200,000 steps each.

Scope cuts available if time is short, in the order to take them:

- **Drop or defer `learn7` (-25 h).** It only serves Experiment 2's freedom/reach split. The
  headline Experiment 1 result needs `control` and `learned_mix` only, and the split is already
  known directionally (reach dominates).
- **Stop Phase 2's arms short of the elbow (-20 h or so).** The ablation gap is paired, so it
  should read stably well before convergence; say at what step it was read.
- **Skip the anneal (-4.8 h)** if no headline "rig quality" number is wanted. Warm-start arms reset
  the optimizer and raise the LR, handing the anneal's ~+0.138 dB straight back, so they can start
  from the pre-anneal elbow checkpoint and lose nothing.

**Shelved: `learn7` at 96,000.** It was training toward a matched 200,000 against arms warm started
from the retired baseline. Finishing it would spend ~13 h to produce a number on a foundation that
is being replaced. Its scored rows stay in `benchmark.jsonl` and its findings stay in
`postprocessing/RESULTS.md` as provisional; it restarts from scratch in Phase 3.

## Where things stand (as of the last commit)

- **HIGHEST PRIORITY COMPUTE: the paired recalibration.** The benchmark has no working
  demonstration that it can resolve an effect of the size it is asked to judge. The original
  A/B/C study cannot provide one -- one run per arm, all three stopped undertrained at 15,299
  steps, unpaired, effect 1.27x the seed spread against a required 3x. Two cold-started runs under
  the baseline run's exact protocol replace it, and they answer both open questions at once:
  `abl_frozen` and `abl_baseline` share seed base 1028 so the ablation is paired, and
  `abl_baseline` doubles as an independent-seed replicate of the existing baseline run (base 28).
  Pre-registered in `experiments/2026-09-10-paired-recalibration/NOTES.md`; run with
  `bash codec/scripts/run_plateau.sh <hours> abl_frozen|abl_baseline`. ~13.3h per arm to 96,000
  steps, ~37.6h to the baseline's 272,000. This outranks Experiments 3 and 4 -- every result in the
  repo is a comparison whose credibility depends on these two numbers.

- **The baseline was replaced on 2026-09-11.** `checkpoints/calibration/plateau_baseline`
  (`checkpoint-304000`, 304k steps, PSNR 24.88) is **RETIRED** -- see `RETIRED.md` beside it. Its
  first seven chunks all ran `run.seed=28`, so steps 0-56,000 replayed the same **53.4%** of the
  training data seven times and **46.6% was never seen**. Cost, measured against a clean run at
  matched steps: **+1.87 dB by step 56,000**, accumulated inside the replay window and then
  carried. Its training speed, its 272,000-step elbow and its 24.747 plateau are artifacts of that
  bug -- never quote them as what this rig reaches or how long it takes.
  `run_plateau.sh baseline` refuses to run.
- The new baseline is **`baseline_v2`** (`checkpoints/calibration/ablation_baseline`, tags
  `abl_baseline-*` -- the run was launched under the older name and the tag must stay continuous).
  Per-chunk seeds from step 0, 100% data coverage, seed base 1028. Still training.
- The retired checkpoints stay on disk because `checkpoint-304000` is the warm-start origin of the
  Experiment 1/2 arms and `learn7` still has chunks to run from it. Those results stay valid as
  **paired, arm-to-arm** comparisons -- every arm shares that origin and seed schedule. What is
  invalid is the absolute baseline level they sit on.
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
  `[-1,1]` range bug), try to recover a known effect from the paper to calibrate the benchmark
  itself (the frozen-bottleneck A/B/C, which came back TOO NOISY -- a calibration that fails is
  still calibration, and it is what set the paired protocol), and verify claims about the pipeline (resume, determinism,
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
- `checkpoints/` and the dataset `/data/` are gitignored and local-only, shared across every part
  of this project (not nested under `codec/`). Don't expect them to be present after a fresh clone —
  see `README.md`'s Setup section (gated downloads) and `codec/README.md` (training). Note the
  dataset rule is *anchored* (`/data/`): `postprocessing/data/` is small generated JSON and IS
  committed, because it is what lets the figures rebuild without a GPU. An unanchored `data/` rule
  silently swallowed it once, which made the folder's "runs on a fresh clone" claim false.

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
  **"Replays the same data" badly understates the damage, and this cost the project a month.** The
  seed fixes each worker's shard traversal ORDER, and a chunk ends before it has walked all of it,
  so a fixed seed pins the same prefix forever and the rest of the dataset is *never reached*.
  Measured with `codec/scripts/measure_data_coverage.py` (which replays the loader's own
  `_my_shards`/`rng` logic): the first baseline's seven fixed-seed chunks saw **53.4%** of the
  training data and never touched the other **46.6%**, where per-chunk seeds reach **100%** over the
  same span. It was ~1.9 dB behind a clean run by step 56,000 and that whole baseline had to be
  retired. If a run's seed schedule is ever in doubt, run that script before trusting the run.
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
- **The three calibration arms are cosine-annealed, the plateau run is not.**
  `run_calibration.sh` sets `decay_steps = steps - warmup`, so each arm runs a full cosine to
  `min_lr` over its own 15,300 steps and ends annealed; `run_plateau.sh` sets `decay_steps=0` and
  holds 1e-4. Their readings are therefore NOT comparable at equal step counts, however tempting it
  is — A_baseline's 20.105 at 15,299 and the plateau run's 20.123 at 16,000 look like a seed
  replicate of the same configuration and are not one.
- **Score every checkpoint inline, and capture the log's metadata before the log is gone.** Every
  launcher except the original `run_calibration.sh` chunks hourly and scores each checkpoint, which
  is why the plateau and warm-start arms have 25-34 PSNR readings and the calibration arms have
  one. `run_calibration.sh` now keeps its checkpoints through the run, scores them all, then
  deletes them (keep-then-score rather than chunk-and-score, because it runs a cosine schedule and
  resuming one of those is its own hazard). Separately, the logs hold ~10x the resolution of the
  scored rows plus each chunk's resolved seed and LR schedule, and `checkpoints/` is gitignored --
  so the launchers now call `postprocessing/extract_run_metadata.py` at the end of a session to
  persist it. A few hundred kB of committed JSON against data that is otherwise unrecoverable.
- **`checkpoint_keep_recent` decides whether a run's trajectory can EVER be scored.** It defaults
  to 1 in `codec/configs/kmira_train_codec.yaml`, so intermediate checkpoints are deleted as
  training advances and a run scored only at the end can never be given a curve afterwards — the
  three calibration arms are permanently one point each for exactly this reason. If a run's shape
  might matter later, either raise `checkpoint_keep_recent` or score inline per chunk the way the
  plateau and warm-start launchers do. Disk is the trade: each checkpoint is ~4.4GB.
- **The per-chunk seed is an identity for a slice of data, and the slices are very uneven.** A
  chunk resolves `run.seed = 28 + step/8000`, and mira's loader reseeds per process start, so that
  seed selects the whole 8,000-step stream the chunk trains on. Measured across all four runs
  (`postprocessing/plot_seed_effects.py`, figure `seed_effects.png`): seed 40's chunk is the single
  best in every run that met it (+1.308 dB in the plateau run, +0.795 control), seed 50's is the
  best of every chunk from seed 45 on, and seeds 36-39 and 47-49 are the worst in every run — at
  step numbers 100,000+ apart. **Both "false plateaus" in the baseline run were these slices, not
  the optimiser.** Consequences: an elbow judged on trailing readings is confounded by which seeds
  those readings landed on (compare like seeds, or average over a run of them), and this unevenness
  is the reason pairing works at all — it lands on both arms and cancels.
- **NEVER edit a launcher script that is currently running.** bash reads a script lazily, by byte
  offset, so an in-place rewrite (anything that truncates and rewrites the same inode -- `Write`,
  `sed -i`, Python `write_text`) makes the running shell resume mid-file at the wrong offset and
  execute garbage. Write the new version to a temp file and `mv` it into place: the rename swaps
  the inode and the running process keeps reading the old one. Check with `ls -i` before and after.
- **Piping a launcher into `head` does not "just show the banner" -- it starts training.** These
  scripts run in the foreground and begin their first chunk immediately; `head` closing the pipe
  only sends SIGPIPE afterwards, and `tee` to the logfile can swallow even that. This actually
  launched an unwanted run once, which then had to be killed and its output directory cleaned up.
  To inspect a launcher, read it (`sed -n`, `grep`) or check its syntax (`bash -n`). To exercise
  argument handling, use a path that exits before training, such as an invalid argument.
- **mira's validation loader is hardcoded `seed=37`, independent of `run.seed`** (`train_codec.py`,
  the `val_loader = create_loader(...)` call). So validation losses ARE comparable across runs with
  different seeds -- worth knowing before spending time on the hypothesis that a cross-run val-loss
  gap is a sampling artifact. It is not; it is the model.
- **mira logs validation losses to four decimal places.** A term small enough to sit on that grid
  stops being a curve and becomes a staircase, and its trailing slope is then rounding rather than
  training. `loss_dino_latent_consistency` runs ~1e-4 here and is already pinned at `0.0001`, so
  nothing about its convergence can be read from the log. Say so on the figure rather than letting
  a flat line be read as a ceiling.
- **One run per arm cannot estimate a spread, so a matching point estimate is not a reproduction.**
  The first calibration measured A-B = +1.4489 dB against mira's published ~1.4 and it was tempting
  to call the effect reproduced. It is not evidence: the seed spread |A-C| was 1.142 dB, so the
  effect could not be distinguished from a substantially different one, and both arms stopped
  undertrained where mira's numbers are converged. Report what survives (B fell below *both*
  baseline draws, so the sign and rough scale hold) and not the coincidence.
- **Calling a plateau/elbow needs several trailing readings, not one flat stretch.** This project
  has hit real false plateaus more than once — a multi-hour flat stretch followed by a further jump
  of over 1 dB. Stopping on the first flat reading has already cost real signal here. The usable
  test is a ratio, not an eye: compare the trailing trend against the reading-to-reading spread over
  the same window. The calibration arms' trailing slope was ~10x shallower than their opening one
  and would pass any visual flatness check, but it was *smaller than the spread of their last six
  readings* — so the curve carried no information about whether the descent had stopped, and in fact
  39% of the run's total loss descent (4.76 dB of PSNR) was still ahead of it. Flat is not
  converged; flat plus a trend larger than the noise is.

## Setup, if you need to run anything

```bash
pixi run setup      # GPU
pixi run setup-cpu  # CPU only
pixi run test
```

Needs two separately-gated downloads (DINOv3 weights from Meta, `kyutai/rocket-science` from
HuggingFace) — see `README.md`'s Setup section, it's easy to conflate the two gates.
