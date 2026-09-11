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

## Where things stand, and what is next

**The baseline was replaced on 2026-09-11, and that invalidates every absolute number in the
repo.** `checkpoints/calibration/plateau_baseline` (`checkpoint-304000`) is **RETIRED** -- see
`RETIRED.md` beside it. Its first seven chunks all ran `run.seed=28`, so it trained on 53.4% of the
data for 56,000 steps and never saw the other 46.6%; measured at matched steps it was **1.868 dB**
behind a clean run by the end of that. Its training speed, its 272,000-step elbow and its 24.747
plateau are artifacts. `run_plateau.sh baseline` refuses to run. Reproduce the coverage with
`codec/scripts/measure_data_coverage.py`.

The new baseline is **`baseline_v2`** (`checkpoints/calibration/ablation_baseline`, tags
`abl_baseline-*` -- launched under the older name, and the tag must stay continuous). Per-chunk
seeds from step 0, 100% coverage, seed base 1028. **Still training.**

The retired checkpoints stay on disk: `checkpoint-304000` is the warm-start origin of the
Experiment 1/2 arms, whose results remain valid as **paired, arm-to-arm** comparisons -- every arm
shares that origin and seed schedule. Only the absolute level they sit on is void. Those studies
and their figures are archived under `postprocessing/archive/`; their surviving findings are
summarised in `postprocessing/RESULTS.md` section 5.

### The rebuild, in three phases

Priced from the live run's measured 1.20 h per 8k chunk (cold) and 1.00 h (warm): **~154 h if
`baseline_v2` elbows at 200,000, ~186 h at 272,000.** The elbow is unknown and may land *later*
than the legacy 272,000 -- a run seeing 100% of the data has more to learn, not less. The increment
panel decides it. Pre-registered in `experiments/2026-09-10-paired-recalibration/NOTES.md`.

1. **A trustworthy baseline (~19 h).** Finish `baseline_v2` to its elbow, then anneal 32,000 steps.
   Everything is blocked on this.
2. **The calibration that never existed (~60 h).** `abl_frozen` (seed base 1028, so it is paired
   with `baseline_v2` chunk for chunk) and `baseline_v2_s2` (seed base 2028, the run-to-run
   spread). Until these land, the benchmark has **no demonstration that it can resolve an effect of
   the size it is asked to judge**, and that limitation is load-bearing for everything else.
3. **Experiments 1 and 2 redone (~75 h).** `control`, `learned_mix`, then `learn7`, warm started
   from phase 1.

```bash
bash codec/scripts/run_plateau.sh <hours>                  # baseline_v2
bash codec/scripts/run_plateau.sh <hours> abl_frozen
bash codec/scripts/run_plateau.sh <hours> baseline_v2_s2
```

Scope cuts if time is short, in order: drop `learn7` (-25 h; Experiment 1's headline needs only
`control` and `learned_mix`); stop phase 2 short of the elbow (-20 h; the gap is paired so it reads
early, but say at what step); skip the anneal (-4.8 h; warm-start arms reset the optimiser and hand
its ~+0.138 dB straight back, so they can start from the pre-anneal checkpoint).

**Shelved: `learn7` at 96,000.** It was heading for a matched 200,000 against arms warm started
from the retired baseline. Its rows stay in `benchmark.jsonl`; it restarts in phase 3.

### What the archived studies found, and what is still open

Experiment 1 replaced mira's fixed 7-layer mean with 24 learned per-layer weights and beat its
paired control substantially; Experiment 2 attributed most of that to *reach* (shallower blocks
becoming available) rather than *freedom* (the weights moving). Directions stand, magnitudes do
not. The open question is downstream and unchanged by the retirement: the learned weights did not
reweight mira's layers, they **abandoned** them, and mira's rationale for that layer set is about
preserving semantics *for the world model*. **P-DINO never separated the arms**, and mira's stated
methodology is to *select* codecs on world-model metrics while *training* them on reconstruction --
so the fixed layer set reads as a **regularizer** encoding what the loss cannot express. On that
reading the gain is predicted not to transfer downstream.

### Off the critical path

- **Experiment 4, latent predictability** (`experiments/2026-09-10-latent-predictability/NOTES.md`)
  -- the only queued work that addresses the open question above, and it retrains nothing: the
  existing frozen arms plus a small next-step predictor as evaluation apparatus. An afternoon
  against ~25 h for a training arm, and it can run on the archived arms because it is a paired
  comparison between them. Read `FVU` only alongside PSNR: a codec that encodes nothing is
  perfectly predictable.
- **Experiment 3, cold start** (`experiments/2026-09-08-cold-start-layer-mix/NOTES.md`) -- needs
  its own full plateau run, so it sits behind everything above.

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
  it drops below the annealed number and spends tens of thousands of steps recovering. **Read a
  constant-LR warm start against the pre-anneal plateau, never the post-anneal number.** Two
  corrections this entry has already needed, both worth keeping: it once said a warm start "can
  never reach the annealed number no matter how long it runs", which measurement falsified -- a
  control arm passed it and kept going; and the elbow that plateau was called at was a stopping
  point rather than an asymptote, with constant LR still buying ~+0.03 dB per 8k steps well beyond
  it. (The specific dB in the original entry came from the retired baseline and have been removed;
  the mechanism is what transfers.)
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
- **Calling a plateau/elbow needs several trailing readings, not one flat stretch.** Three
  occurrences so far, the most recent on the clean baseline where the seed schedule cannot be
  blamed: its per-chunk increment fell to **+0.029 dB** at step 104,000 and the next chunk returned
  **+0.199 dB**. Stopping on the first flat reading has already cost real signal here. The usable
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
