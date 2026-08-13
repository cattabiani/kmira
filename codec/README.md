# Codec benchmark

Everything for benchmarking and improving mira's codec (frozen DINOv3-L/16 encoder -> bottleneck ->
ViT decoder) lives here: configs, training/launch scripts, and results. Shared infrastructure that
any future part of this project would also need (raw datasets, weight caches, model checkpoints)
stays at the repo root (`../data/`, `../checkpoints/`) rather than nested here.

The goal is **not** to reproduce mira's codec quality — that needs a compute budget we don't have.
It is to build a benchmark that can reliably answer *"is this change to the bottleneck better than
mira's stock one?"* on a single consumer GPU.

## Layout

**There is no forked trainer.** Training runs mira's own `scripts/train_codec.py` unmodified; our
entire divergence from mira is configuration (see step 11).

- `configs/kmira_train_codec.yaml` — the top-level training config, passed to mira's script via
  `--config-dir=codec/configs --config-name=kmira_train_codec`. Deliberately *not* named
  `train_codec.yaml`, which would ambiguously shadow mira's own config of that name.
- `configs/model/` — Hydra "model" group members, selected with `model=<name>`:
  - `baseline_image_base.yaml` — **the benchmark baseline**: mira's codec, image-only, Base decoder.
  - `baseline_image.yaml` — same but the stock XL decoder (kept for reference; too slow to iterate).
  - `calib_frozen_bottleneck.yaml` — the calibration downgrade (step 12).
- `configs/baseline.yaml` — standalone config for the visualization script (not a training entry).
- `scripts/run_calibration.sh` — the three-arm calibration run.
- `results/` — `benchmark.jsonl` (the accumulating metrics record) and reconstruction PNGs.
- `../src/kmira/codec/variants/` — real code forks, one file per idea.
- `../src/kmira/benchmark/` — `eval_codec.py` (metrics) and `visualize_reconstruction.py` (a
  before/after image for one frame).

## Steps taken so far

1. **DINOv3 weights.** Two gated checkpoints are needed: `dinov3_vitl16_...` (the codec's frozen
   encoder) and `dinov3_vitb16_...` (used by mira's eval metrics, `DinoForMetrics`). Both come from
   Meta's own gated download page, **not** the HuggingFace `facebook/dinov3-*` repos — those only
   ship `model.safetensors` (transformers format), not the `.pth` files mira's `torch.hub`-based
   loader expects. Placed at `../data/dino_weights/`, pointed to via `RS_DINO_WEIGHTS_DIR` (set
   automatically by `../.envrc` via direnv).

2. **Environment.** `pixi` (not a Python package — a standalone conda+PyPI env/task manager)
   installed via the official installer script; `pixi run setup` builds the env with mira installed
   editable from `../../mira`.

3. **Verified the stock forward pass manually** before automating anything: loaded `baseline.yaml`
   via Hydra, instantiated `mira.codec.VideoCodec`, ran a real forward pass on GPU. Output shapes
   match the config (latent `(1,1,32,9,16)` given `latent_dim=32` and `bottleneck.stride=2` on top
   of DINO's `patch_size=16`).

4. **Built `visualize_reconstruction.py`** for a fast visual check. An untrained decoder produces
   pure noise — expected, not a bug, since only the DINO encoder is pretrained; the bottleneck and
   decoder start from random weights.

5. **Got real data.** `kyutai/rocket-science` (mira's Rocket League dataset) is gated on HuggingFace
   *separately* from the DINOv3 weights — different org, different approval, easy to conflate.
   Pulled a slice via `RocketScienceDataset.from_hub(shards=N)` and wrote **restricted** local
   `index.json` files (`../data/rocket_science/{train,test}/`) listing only the downloaded shards.
   The raw snapshot's `index.json` references the *entire* split (2,821 shards / 15,769 matches,
   ~7.6TB), so pointing a trainer at it directly fails with `FileNotFoundError` on shards we never
   downloaded.

6. **Briefly forked the trainer, then deleted the fork** (see step 11 for the deletion). mira's
   `train_codec.py` hardcodes fp32 `AdamW` with no config hook for precision, and with the stock
   **XL** decoder (~596M trainable) that OOMs on a 12GB card even at `batch_size=1` — the crash
   lands *inside the optimizer step*, because fp32 Adam holds four copies of every trainable
   parameter (weights, gradient, `exp_avg`, `exp_avg_sq`) before any activation memory. A thin fork
   (`train_codec_bf16.py`) cast the model to bfloat16 first, halving all four. That fork produced
   every result in steps 7-10 and was then removed once the Base decoder made it unnecessary.

7. **First real training run**: 2000 steps, `batch_size=1`, 3 shards (18 matches), XL decoder. Loss
   1.26 -> 0.18; reconstruction went from noise to coarse but correct structure. Confirms the
   pipeline learns end to end on real data.

8. **Scaled the dataset up**: 30 train shards / 3 test shards (~85GB, one-time). That is
   **179 matches / 6.2M usable frames** for train and **17 matches / 596k frames** for test — far
   more than any run we can afford consumes, so data volume is not a constraint and no further
   shards are needed. An initial plan to run 45,000 XL steps overnight was **discarded** after the
   measurements in step 10.

9. **Finished the metrics benchmark** (`../src/kmira/benchmark/eval_codec.py`) and scored the
   2000-step checkpoint. Doing this *before* a long run paid for itself immediately: it exposed a
   bug that had been silently corrupting every image we had looked at.

   **The [-1, 1] range bug.** `VideoCodec.normalize_video` applies `(x - 0.5) / 0.5`, so
   `input_video` and `output_video` are in **[-1, 1]**, not [0, 1]. Both our scripts assumed [0, 1]
   and clamped to it — which zeroes every negative pixel, and the mean pixel is about -0.45, so most
   of the image was destroyed. Symptoms: PSNR 4.9 dB (*worse than predicting a flat gray image*,
   which scores 9.1) and reconstructions that looked implausibly dark and over-saturated. Every mira
   metric class expects [0, 1], so the fix is one `(x + 1) / 2` before scoring — `to_unit_range()`.

   The giveaway was checking against trivial baselines rather than accepting the number: a trained
   model losing to a per-image-mean predictor means a pipeline bug, not a quality problem.

   After the fix the same checkpoint scores **PSNR 18.5, SSIM 0.51, LPIPS 0.53** — believable for
   2000 steps against the paper's fully-trained 29.7 / 0.891 / 0.051.

   Two design choices aimed at trustworthy comparisons: the eval set is **fixed, not sampled**
   (seeded shuffle, fixed frame count), so a baseline-vs-variant difference carries no
   eval-sampling noise; and results **append to `results/benchmark.jsonl`** so runs accumulate into
   a diffable record. Caveat in the code: **rFDD fits a 768x768 covariance**, so below ~2048 frames
   it is rank-deficient and meaningless — the other four metrics are stable at a few hundred.

10. **Measured what scale is actually affordable**, instead of guessing at an 8h run. All figures
    single-GPU (RTX 4070 Ti, 12GB), bf16, synthetic data to isolate compute from the dataloader:

    | Decoder | Trainable | s/step | frames/s | Peak VRAM |
    |---|---|---|---|---|
    | XL `28x1152` (mira default) | 596M | 0.597 @ b2 | 3.3 | 6.3 GB |
    | Large `24x1024` | 410M | 0.711 @ b4 | 5.6 | 6.9 GB |
    | **Base `12x768`** | 114M | 0.233 @ b2 | **8.6** | 3.2 GB |
    | Base @ batch 8 | 114M | 0.947 | 8.4 | 9.0 GB |

    Three findings:

    - **Compute-bound, not data-bound.** Synthetic XL/b2 = 0.597 s/step vs 0.590 in a real run, so
      the dataloader contributes nothing and decoder savings translate 1:1.
    - **Batch size does not buy throughput** (8.6 -> 8.4 frames/s from batch 2 to 8). The GPU is
      already saturated by a single frame through DINO-L plus the decoder. Batch size is a
      gradient-stability knob only.
    - **~60% of Base's step time is fixed cost** — DINOv3-L runs *twice* (encoding, then re-encoding
      the reconstruction for the consistency loss) plus VGG for LPIPS. None of that shrinks with the
      decoder, so Base is already near the floor; going smaller would buy little.

    **Decision: Base decoder.** Not because it is better — the paper's Table 7 says Large ≈ XL and
    Base is clearly worse in absolute quality — but because at our budget it is the only size that
    can get *near its own ceiling*. Undertraining actively corrupts a comparison: it measures which
    variant converges fastest rather than which ends up best. The risk in the other direction is
    that Base's lower capacity becomes the binding constraint and *masks* latent improvements; that
    is exactly what the calibration run below is designed to detect.

11. **Deleted the trainer fork.** Its sole justification was XL's memory footprint, which the Base
    decoder removes. Measured, Base decoder, batch 4:

    | Strategy | s/step | frames/s | Peak VRAM |
    |---|---|---|---|
    | **fp32 weights + bf16 autocast** (mira's own) | 0.478 | 8.4 | 7.2 GB |
    | bf16 cast (the fork) | 0.451 | 8.9 | 5.1 GB |

    fp32 costs ~6% throughput and caps us at batch 4 (batch 8 OOMs) — and that cap is free, since
    batch size buys no throughput anyway. What it buys back matters more than 6%: **pure-bf16 master
    weights silently drop any Adam update smaller than one ULP of the weight**, which bites hardest
    in late convergence, exactly where sub-dB differences between bottlenecks live.

    Verified mira's unmodified trainer end to end on our image-only Base config, *including* the
    `_visualize()` validation path that had never run at `timesteps=1`: 0.47 s/step at batch 4,
    peak **9.6GB of 12.3GB** with EMA enabled and the desktop running. So there is no forked
    training code — only config. (`timesteps=1` was never the obstacle: mira's trainer had already
    got through model build, dataloader, forward, loss and backward with it back in step 6; only
    fp32 Adam stopped it.)

12. **Built the calibration arm** (`../src/kmira/codec/variants/frozen_bottleneck.py`) — the first
    *real* code fork, and deliberately a downgrade rather than an idea. It subclasses `VideoCodec`
    and freezes `encoder.rae_projection` at its random initialisation, reproducing Table 4's "random
    frozen projection" row (-1.4 dB PSNR at full scale). Verified it differs from the baseline in
    exactly one respect: same `Conv2d(1024 -> 32, k=2, s=2)`, `requires_grad=False`, 114.1M vs
    114.2M trainable parameters — the 131k difference is the projection itself.

13. **Hardened the setup against an inconclusive night.** A full 30-step rehearsal of all three arms
    ran end to end, and the checks below were done before committing to a long run:

    - **A negative `decay_steps` crash.** A fixed 1000-step warmup exceeds a short probe's total
      steps, leaving `decay_steps < 0` and tripping an assertion in mira's LR scheduler. Warmup is
      now 10% of the run, capped at 1000.
    - **No crash recovery.** `checkpoint_every=100%` means the *only* checkpoint is the final one,
      so the README's "interrupted runs resume" claim was false. Now 10%: a crash costs ~12 min.
      Disk usage is unchanged (`checkpoint_keep_recent=1` deletes the previous one); only write
      volume grows, to ~132GB across three arms — under 0.2% of a consumer SSD's rated endurance.
    - **Resume verified, not assumed**: interrupted a run, restarted it, confirmed it logged
      `Auto-resuming` and continued from step 52 of a checkpoint at 50 rather than restarting.
    - **Eval determinism verified**: scoring the same checkpoint twice returns bit-identical numbers
      to nine decimal places. Without this, every paired comparison would be silently contaminated.
    - **Eval sampling had three separate biases**, each found by measuring rather than assuming:
      1. Streaming `create_loader`, 2048 frames covered only **3 of 17** matches — one match yields
         tens of thousands of clips, so the stream never escapes the first match in each shard.
         Neither shuffling nor a 40x larger shuffle buffer changed it.
      2. Equal clips per match **over-weights short matches**: lengths run 7,120-10,480 frames
         (1.47x), so a fixed share samples a short match ~1.2x more densely than a long one.
         Allocation is now proportional to match length (largest-remainder, sums exactly).
      3. `max_clips` **takes the first N clips and breaks**, so it was scoring only the opening
         minutes of every match. Clip ids are now drawn at random across the whole match.

      Verified after the fix: all 17 matches contribute, allocation/length is constant to 4 decimals,
      sampled ids span 20..8058 of 8160 within a match, two builds are byte-identical, and a
      different seed gives different frames. Yield is 99% (a few ids fall in excluded replays).
      Costs ~5 min per scoring pass.
    - **Verdict logic**: a *reversed* result (frozen bottleneck beating the learned one) is now
      reported as `REVERSED` rather than mislabelled "too noisy" — it is a real and informative
      outcome, indicating we measured on the slope rather than the plateau.

14. **Ran the calibration (2026-08-11, 5.6h).** The effect reproduced — but the runs were not
    converged, so the noise floor it reported is not the one that matters. See *Calibration outcome*
    below for the numbers. The decisive context arrived from mira's own
    `configs/train_codec.yaml`: **mira trains this codec for 250,001 steps**, so our 15,300-step arms
    were 6% of the reference schedule. Any talk of a plateau at that length was wishful.

15. **Plateau probe** (`scripts/run_plateau.sh`) — one baseline run at **constant LR**, extended in
    **time slots**: the argument is hours of training, and re-running adds another slot to the same
    run (`bash scripts/run_plateau.sh 8`, then `... 16`, and so on). The machine is dedicated to this
    now, so the budget is 1-2 days rather than one night; for scale, mira's full 250,001 steps is
    ~31h at our measured 0.45 s/step, so a full-length reference run is actually in reach. Rationale, since it looks odd next to mira's cosine default: under cosine decay
    the loss flattens because the LR anneals to ~0, and the shape of that flattening moves with the
    run length you chose, so "where does it flatten" has no stable answer. Holding the LR fixed makes
    a flat curve a statement about the model. The number it yields is therefore a *location*, not a
    quality result — constant LR settles higher and noisier than an annealed run — and the protocol
    we adopt afterwards is "elbow-many steps, cosine decay put back".

    Constant LR is what makes slots possible at all: **every checkpoint is a valid model**, so
    stopping is free. Under cosine decay this breaks twice over — a checkpoint from the middle of a
    decaying schedule has not settled, and *extending* a finished cosine run is useless because the
    LR has already annealed to ~0 and nothing moves. With a flat LR the run is a ratchet: stop,
    inspect, resume, indefinitely. mira's trainer auto-resumes from `output_dir`, so a slot only has
    to raise the step target; the script reads the completed count off the newest checkpoint.

    **The hour is the unit.** The argument is whole hours; N hours is exactly N chunks; and each
    chunk trains for an hour, then checkpoints and scores on its last step. At the measured
    0.45 s/step an hour is **8,000 steps**. Fractional and non-numeric arguments are rejected rather
    than silently rounded, and a slot can only ever stop on a chunk boundary — which keeps a whole
    class of questions from arising instead of having to be handled.

    Validation runs more often than that: every **1,000 steps**, 8 per hour, ~7.5 min apart. It costs
    ~18s, so 4% overhead buys 8x the resolution for locating the elbow, where scoring at 6 min per
    point would not. The interval must **divide** the chunk exactly, because `plateau_report.py`
    joins scored checkpoints to validation readings *on step number* — if the hourly checkpoint step
    were not also a validation step, every scored point would carry an empty loss cell. 1000 divides
    8000; 1333 (a literal 10 minutes) does not, which is why the interval is round rather than exact.

    Scoring inline rather than batching it at the end means an interruption loses at most the current
    hour; everything earlier is already durable in `results/benchmark.jsonl`. mira's trainer
    auto-resumes from `output_dir`, so a chunk is just another invocation with a higher step target —
    interleaving needs no fork of mira. Process restart costs ~10s per chunk, under 0.5%.

    **A rolling window of the last 6 hourly checkpoints is kept** (`keep_recent=6`), ~28GB flat
    whether the slot is 12h or 48h. A checkpoint directory is 4.7GB — `checkpoint.pth` (weights +
    EMA, 1.6GB) plus `training_state.pth` (optimiser moments, scheduler, loader position, 3.0GB).

    The window is sized by the **elbow detector's lag**, not by taste. Its criterion is trailing over
    2h, so when it reports an elbow at hour X the model actually wanted lies in hours X-2..X; six
    hours covers that with margin. This is why the elbow is checked after *every chunk* rather than
    only at slot end — a rolling window is useless if the detection is noticed after the checkpoint
    has scrolled out of it.

    Keeping *all* of them (146GB at 31h) was considered and dropped. Past the elbow window an old
    checkpoint only helps in narrow cases: re-scoring without retraining if the **eval** changes,
    falling back if the newest is corrupted mid-write, or branching a run from the middle. None of
    them help against a **training** bug, which sends you back to step 0 whatever is on disk — so
    the distinction that matters is eval-bug vs training-bug, not disk size.

    Scoring costs **6.2 min** for 2048 frames (measured 371.3/369.8/369.1s across the three
    calibration arms — stable to within 2s). That is what sets the chunk at an hour: scoring every
    30min would be 21% overhead, hourly is 11%. So a 12h slot is ~13.3h wall clock, which the script
    prints up front. For scale, mira's full 250,001 steps is 31 hours.

    | slot | steps | val readings | scored points | wall clock |
    |---|---|---|---|---|
    | 12h | 96,000 | 96 | 12 | 13.3h |
    | 24h | 192,000 | 192 | 24 | 26.5h |
    | 48h | 384,000 | 384 | 48 | 53.1h |

    Two things still worth knowing: `val_first` is on only for the first chunk, since later ones
    would re-validate the step they resume from at 18s a time for a reading already in the log; and
    the log is appended (`tee -a`), never truncated — earlier slots' validation history *is* the
    curve. Scoring skips anything already in `benchmark.jsonl`, so re-running never re-scores, and
    a resume that re-runs the steps after the last checkpoint logs their validations twice, so the
    report keeps the last reading per step (the live trajectory, not the discarded one).

    **The hidden data-repetition bug.** Discovered while paused after the first real slot (56,000
    steps, 7 hourly chunks): mira's train loader reseeds from `cfg.run.seed` on every process start
    (`training_loader.py`, `rng = random.Random(self.seed + rank*1024 + worker_id)`), and the
    dataloader is **not** among the checkpointed components — only `optimizer`, `lr_scheduler`, and
    the two latent EMAs are registered (`train_codec.py`). Every hourly restart therefore
    reconstructed the loader with the *identical* seed and replayed the *identical* ~32k-sample
    stream from the beginning. Those first 56,000 steps were not 56,000 steps of coverage across the
    6.2M-frame training pool — closer to seven epochs over one chunk's worth of data. This is
    specific to the hourly-restart scheme; mira's own continuous 250,001-step runs never restart the
    process, so the bug does not exist there.

    The giveaway was the validation curve, not a crash: every 8-reading block between chunk
    boundaries showed the *same shape* (rise, flat, rise, rise, then a steep four-reading drop) —
    the model re-encountering the same batches at the same relative position each hour, not noise.
    Validation itself was unaffected: its loader uses a hardcoded `seed=37` independent of
    `run.seed`, so it was always a fixed, non-repeating readout, and the PSNR numbers already
    recorded are real.

    Fix: **vary `run.seed` per chunk**, derived from the absolute step
    (`HOUR_INDEX = NEXT / CHUNK`, `seed = 28 + HOUR_INDEX`) rather than a counter, so it stays
    reproducible across separate script invocations. This is safe because a resumed chunk overwrites
    the seed-initialized weights with the checkpoint immediately after init — the seed only ever
    determines which data (and dropout draws) that chunk sees, never what it resumes from. The
    56,000 steps already trained are not being discarded (PSNR/validation-loss readings there are
    genuine), but the training-loss curve's shape below the elbow should be re-read with this in
    mind, and the sawtooth should be gone in future chunks.

    **The orphaned-checkpoint scoring bug.** A recurring `torch.hub` GitHub flake (see below) meant
    a chunk could train successfully but fail during the *scoring* call that followed it. The
    original loop scored "whatever step training just reached" once, at the end of that same
    iteration — so if scoring failed and the script was simply re-run, the new invocation resumed
    straight into training the *next* chunk. `NEXT` is always strictly greater than `DONE`, so
    nothing in the loop ever revisited the checkpoint left unscored by the previous attempt.
    `checkpoint-208000` sat trained-but-unscored through an entire extra invocation this way.

    Fixed by moving scoring to the *top* of each iteration — score whatever checkpoint we're
    currently standing on (`DONE`) before training toward the next one — plus one explicit call
    after the loop for the slot's final chunk, since no further iteration exists to score it. Now a
    resume always checks the checkpoint it's resuming from, not just the one about to be produced.

    **The torch.hub GitHub flake itself.** `torch.hub._parse_repo_info` does an *unconditional*
    `urlopen` to GitHub to resolve the default branch on every single load, even though the repo is
    already cached locally (`~/.cache/torch/hub/facebookresearch_dinov3_main`) — the cache is only
    consulted *after* this call succeeds or its `except URLError` fires. That fallback doesn't catch
    `http.client.RemoteDisconnected`, which was observed escaping uncaught straight out of `urlopen`,
    so one dropped connection crashes model construction outright despite a perfectly good cache
    sitting right there. This is a gap in `torch.hub` itself, not mira's code or ours, and the call
    site (`mira/src/mira/codec/dino.py`) is out of reach without forking mira. Both the training call
    and the scoring call now retry (5 attempts, exponential backoff 10/20/40/80s) since both are safe
    to retry blindly: training auto-resumes from the last checkpoint, and scoring writes nothing to
    `benchmark.jsonl` until it succeeds. An initial 3x15s budget proved too thin for a real flaky
    patch and was widened after it burned through all three attempts while network checks moments
    later showed the exact failing URL responding normally.

    **The off-by-one that ate the first run.** mira's loop is `range(start_step, cfg.run.steps)`, so
    asking for 8,000 steps ends at iteration **7,999** and the final checkpoint is `checkpoint-7999`.
    (This is why mira's own config says `steps: 250_001`, not `250_000` — the convention was there to
    be copied and was not.) Two failures at once: the chunk loop's condition `current_step < 8000`
    was never satisfied, so it resumed, trained a single step, re-saved the same checkpoint and
    repeated — 82 times before it was noticed; and 7,999 is not a multiple of the 1,000-step
    validation interval, so that checkpoint had no reading to join against. Three fixes:

    - `run.steps = NEXT + 1`, matching mira's convention.
    - **Targets live on the chunk grid** (multiples of 8,000) rather than being computed as offsets
      from wherever the last checkpoint sits. An off-grid checkpoint is then absorbed by a single
      chunk instead of shifting every future one off the validation grid permanently.
    - **A no-progress guard**: if a chunk ends no further along than it started, abort. This is what
      turns a one-line mistake into a caught error rather than a silently burned slot.

    `scripts/plateau_report.py` joins the validation curve to the scored metrics and calls the elbow.
    It is standalone, so the curve can be inspected mid-flight. The elbow criterion is expressed in
    **hours** ("no improvement beyond noise across 2h") and converted to a reading count using the
    spacing actually found in the log, so it does not silently change meaning when the validation
    cadence changes — "5 readings" meant 50 minutes at one point in this project's history and 5
    hours at another. The window is *trailing*, so the reported elbow lags by up to 2h: on a
    synthetic curve flattening at step 80,000 it reports 95,000. A leading window would instead call
    an elbow early on any temporary flat spot, which is the worse error when the number is being used
    to choose a training length. Treat it as an upper bound and read the printed curve for the shape.

    Considered and dropped: shrinking the run to fit alongside a game on the same GPU. Measured
    batch 4 at 0.446 s/step vs batch 2 at 0.239 s/step — i.e. per *sample* it is a wash, so batch
    size is nearly free to trade for memory. Moot now (there is a second machine for that), but it
    confirms we are compute-bound, and it recorded that mira's trainer already runs **bf16 autocast
    internally** (`_autocast`, scripts/train_codec.py) — our "fp32" runs never were fp32.

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

Appendix Table 22 additionally ablates *which* DINO layers are aggregated — the subsystem between
DINO and the bottleneck.

## Our results so far

| Tag | Decoder | Steps | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| smoke-2000steps-XL | XL | 2,000 @ b1 | 18.5 | 0.51 | 0.53 |
| A_baseline | Base | 15,300 @ b4 | 20.10 | 0.593 | 0.406 |
| B_frozen_bneck | Base | 15,300 @ b4 | 18.66 | 0.557 | 0.486 |
| C_baseline_seed2 | Base | 15,300 @ b4 | 21.25 | 0.616 | 0.356 |

Real results land in `results/benchmark.jsonl`. `results/benchmark_prelim.jsonl` holds pre-calibration
rows (the smoke test and a 30-step plumbing rehearsal) — kept for history, not comparable to
anything, since they predate the eval-coverage fix in step 13.

## Calibrating the benchmark

The training length is **measured, not guessed**. Three arms, run by
`scripts/run_calibration.sh`:

| Arm | Model | Seed | Purpose |
|---|---|---|---|
| A | `baseline_image_base` | 28 | reference |
| B | `calib_frozen_bottleneck` | 28 | a **known** -1.4 dB effect (Table 4) |
| C | `baseline_image_base` | 1234 | the **noise floor** |

A and B share a seed, so they are a *paired* comparison — identical initialisation and data order,
with the architecture as the only difference. C re-runs A with a different seed; the A-vs-C spread
is the smallest difference the benchmark can distinguish from luck. Both numbers are needed:
A-vs-B answers *"can we see a large effect?"*, A-vs-C answers *"how small an effect can we trust?"*.

The per-run validation curve (logged every 5% of steps) shows where the loss flattens, which is what
sets the protocol length for every future experiment. If the curve is still steep at the end, runs
are resumable and can be extended rather than restarted.

### Calibration outcome (run 2026-08-11, 5.6h, 15,300 steps/arm)

| | |
|---|---|
| signal, A - B (paired) | **+1.45 dB** — paper's Table 4 gap is 1.4 dB |
| noise, \|A - C\| (unpaired) | **1.14 dB** |
| ratio | 1.3x — below the 3x bar the script asks for |

The **effect size reproduces the paper almost exactly**, so the pipeline does measure the thing it
claims to. That is the result worth keeping.

The noise number, however, should not be read as a noise floor. Two identical configs at different
seeds landed 1.14 dB apart, their validation curves separated by step 1k and never reconverged, and
both were still drifting down at 15,300 steps (A: 0.696 -> 0.671 over the last five readings). At 6%
of mira's 250,001-step schedule these runs were **on the slope, not the plateau** — the regime where
seed spread is largest. So 1.14 dB measures the spread of unconverged runs, which is not the quantity
any future experiment needs.

An earlier draft of this section argued that A-C is *unpaired* and therefore merely an upper bound on
the paired noise our real experiments would see. That is true but was doing too much work: it is an
assumption, not a measurement, and it was being used to rescue a result the data does not support.
The honest position is that the noise floor is **unknown until we train to convergence**, which is
what step 15 goes after.

### Finding the plateau

```bash
bash scripts/run_plateau.sh          # ~8h, 72,000 steps, constant LR
bash scripts/run_plateau.sh 20000    # ~2.3h, a shorter look
```

Prints validation loss vs step with per-reading deltas, and the last step still improving by more
than 0.002 per five readings — the elbow estimate. Then score the retained checkpoints (the script
prints the loop) for PSNR vs steps on the real metric.

Two outcomes, both actionable:

- **Curve flattens well before 72k** — that length becomes the protocol, and each variant costs
  proportionally less than a night.
- **Still climbing at 72k** — variants are expensive at full resolution, and the reserve lever below
  becomes the way to buy the time back.

Held in reserve if this proves too slow: **halving resolution** (288x512 -> 144x256) cuts patches
per frame 4x and, unlike shrinking the decoder, also cuts the fixed DINO+VGG cost. Not used yet
because it stacks a second scale-down and costs us the calibration anchor: the 1.4 dB Table 4 gap we
just reproduced is a full-resolution number, so at half resolution we would have to re-run the frozen
bottleneck arm to re-establish it before trusting any variant result.

## Scoring a checkpoint

```bash
python -m kmira.benchmark.eval_codec \
  --checkpoint checkpoints/<run>/checkpoint-<N>/checkpoint.pth \
  --n-frames 2048 --tag my-variant
```

Appends a row to `results/benchmark.jsonl`. `--n-frames 256` is a fast check, but ignore rFDD at
that size.

16. **The torch.hub GitHub flake, diagnosed and removed rather than retried around.** Both the
    plateau and anneal scripts crashed repeatedly with `RemoteDisconnected` inside DINOv3 loading.
    Traced to `torch.hub._parse_repo_info`: it does an *unconditional* `urlopen` to GitHub to resolve
    the default branch on every single model construction, before ever checking whether the repo is
    already cached (`~/.cache/torch/hub/facebookresearch_dinov3_main`) — and its `except URLError`
    fallback doesn't catch `http.client.RemoteDisconnected`, so one dropped connection kills the
    process outright. Measured directly: plain `curl` to the exact URL succeeded 12/12, but Python's
    `urlopen` (what torch.hub actually uses) failed 3/10 — GitHub throttling `urllib`'s bare User-Agent,
    not an outage. Our chunked design calls this path twice an hour (train + score) where a
    continuous run calls it once, so we didn't cause the flakiness, we multiplied exposure to it
    ~60x over a long run.

    Fixed with two small patches applied from launcher scripts, neither touching mira:
    `src/kmira/torch_hub_offline.py` resolves the ref from the cached directory name instead of
    asking GitHub, and `src/kmira/lr_resume_override.py` (needed for the anneal below) lets a
    resumed run's `warmup_steps`/`constant_steps`/`decay_steps`/`min_lr` win instead of being
    silently overwritten by the checkpoint's old values — torch's default `LRScheduler.load_state_dict`
    is a blind `self.__dict__.update(...)`. Both were verified against the real trainer (not just
    unit tests) before being trusted: the hub patch was proven by hard-wiring `urlopen` to always
    raise `RemoteDisconnected` and confirming both DINO loads still succeed; the schedule patch by
    resuming a real copied checkpoint for 50 real steps and reading the *actual applied LR* back out
    of the resulting `training_state.pth` (landed exactly on `min_lr`, not silently flat). The
    orphaned-checkpoint scoring bug this flake exposed — a chunk could train successfully but the
    *scoring* call could fail, and a naive resume would jump straight to the next chunk without ever
    retrying the one left unscored — is fixed in `run_plateau.sh`/`run_anneal.sh` by scoring whatever
    checkpoint we're standing on at the *top* of each loop iteration, not the one just produced at
    the bottom.

17. **Found the elbow and annealed — this is the locked baseline.** The plateau run (constant LR)
    stopped improving meaningfully around **272,000 steps** (109% of mira's own 250,001-step
    schedule) — PSNR gains had fallen from +0.71 dB in the first hour to +0.01-0.06 dB/hour, after
    two genuine false plateaus (jumps of +0.17 and +1.31 dB following multi-hour flat stretches)
    had already taught us not to trust a short flat run. A 4h anneal from that checkpoint (`warmup=1000,
    constant=271000, decay=32000, min_lr=1e-6`) recovered a further **+0.13 dB** (24.75 -> 24.88),
    confirming the constant-LR number was leaving real quality on the table, as expected.

    **This is the reference baseline for every future experiment:**

    | | |
    |---|---|
    | Checkpoint | `checkpoints/calibration/plateau_baseline/checkpoint-304000/` |
    | PSNR / SSIM / LPIPS | 24.88 / see `benchmark.jsonl` tag `anneal-304000` |
    | Locked recipe | `warmup_steps=1000 constant_steps=271000 decay_steps=32000 min_lr=1e-6` |
    | Total steps | 304,000 |

    Any bottleneck variant compared **from scratch** against this baseline must use the identical
    recipe (same total steps, same warmup/constant/decay split, paired seed) — exactly the discipline
    the A/B/C calibration already validated, just at the real length instead of 15,300 steps.
    Variants compared **warm-started** from this checkpoint need their own baseline-continued control
    run alongside them (same starting checkpoint, same extra steps) — see the discussion above on
    why warm-started and from-scratch results are different questions and are not directly comparable
    to each other or to this number.

    For reference against the paper's own Base-decoder ablation (Table 7, 27.6 dB — the config that
    actually matches ours, not the headline 29.7 which is a different decoder): a 2.7 dB gap remains,
    almost certainly structural (their full corpus and temporal setup vs. our 30 shards / image-only)
    rather than something more training would close, given the plateau's late-stage gains were
    already down to hundredths of a dB per hour.

18. **Experiment 1: learned per-layer DINO aggregation** (`src/kmira/codec/variants/learned_layer_mix.py`).
    mira's stock aggregation is `mean(7 hand-picked layers) + features[-1]`, a fixed point chosen
    once in the paper. This variant exposes all 24 DINOv3-L layers and learns one scalar weight each
    (ELMo-style layer mixing), initialised to reproduce the stock formula exactly -- 0 on the 17
    unused layers, 1/7 on six stock layers, 8/7 on layer 23 (its mean share plus the separate
    `+ features[-1]` term). Free weights, not softmax: the stock formula's effective weights sum to
    2, which a normalised mixture could not represent. Costs no extra DINO compute -- layer 23 *is*
    the final block, so the same forward pass already computes all 24 intermediates.

    Three things had to be got right before this was runnable, each caught by checking rather than
    assuming:

    - **The consistency loss silently changes with the layer count.** `CodecLoss.bind_encoder_dino`
      derives `DinoPerceptualLoss`'s layer set from `encoder.rae_dino.layers` and averages per-layer
      MSE over exactly those. A 24-layer encoder therefore trains against a 24-layer consistency
      loss where a stock one uses 7 -- a *different objective*, not just a different aggregation.
      Racing the variant against `baseline_image_base` would have differed in two ways at once and
      no PSNR delta could have been attributed to the idea. First fix was a paired control
      (`VideoCodecFixedLayerMix`, the identical class with `layer_weights.requires_grad_(False)`),
      which costs a second arm; **superseded in step 20** by pinning the loss instead. Verified along
      the way: both arms expose 24 layers, both reproduce the stock latent at init to 1.4e-6, and
      they differ by exactly 24 trainable parameters (114,188,320 vs 114,188,344).
    - **`load_from_checkpoint` ignores `_target_`.** It hardcodes `VideoCodec(config, ...)`, so
      scoring a variant checkpoint would build a *stock* encoder and then fail the strict
      `load_state_dict` on the extra `encoder.layer_weights` -- an arm would train for hours and only
      then fail at scoring. `eval_codec` now instantiates the saved `model.architecture` node through
      Hydra (`load_codec_respecting_target`), which respects `_target_`; stock checkpoints are
      unaffected since theirs names `mira.codec.VideoCodec`. Verified by round-tripping a real
      variant checkpoint to disk and back.
    - **`finetune_from` is strict.** Warm-starting from the stock baseline needs to tolerate exactly
      one absent key. `src/kmira/finetune_allow_new_params.py` asserts the missing set is *exactly*
      what the caller declared before loading, rather than passing a blanket `strict=False` that
      would swallow a genuine future mismatch. Its own failure path was tested deliberately -- which
      is how a bug in its idempotency guard was found (a marker check meant a second call with
      different keys silently kept the first call's).

19. **Crashed the machine, and why.** A smoke test hard-froze the PC. Post-mortem: `/tmp` is
    **tmpfs -- RAM-backed** (16GB), and the smoke tests were writing 4.7GB checkpoints
    (1.6GB weights + 3.0GB training state) into `/tmp/.../scratchpad`, i.e. straight into RAM, on
    top of the trainer's own ~6-8GB, against 30GB total with only **512MB of swap** -- so a spike
    hard-locks instead of degrading. The journal simply stops mid-run with no OOM or Xid message,
    the signature of a lockup where nothing gets flushed. Two consequences, both permanent rules:
    **never write checkpoints under `/tmp`** (use a gitignored dir on the real disk), and prefer
    construction patterns that do not transiently hold several copies of a 300M-parameter backbone.
    The variant originally built *three* DINOv3-L backbones (~1.2GB each) to keep one, by
    constructing then replacing at both the encoder and codec level; it now hands `RAEEncoder` an
    all-layers config so the backbone is built correctly the first time, and upgrades that instance
    in place. Peak RSS for a full build: 2.96GB. The `torch.stack` in the aggregation was likewise
    replaced with an in-place accumulation, dropping a redundant ~113MB copy of every layer.

20. **Decoupled the consistency loss from the encoder's layer set, so one arm suffices.** Step 18
    handled the objective confound by pairing the variant against a frozen-weight control — correct,
    but it doubles the cost of every experiment that changes the encoder's layer exposure, and it
    treats a fixable coupling as a fact of nature. `src/kmira/pin_consistency_loss_layers.py` patches
    `CodecLoss.bind_encoder_dino` to build `DinoPerceptualLoss` over a *fixed* layer set regardless
    of what the encoder reads, and the variant encoder returns only those 7 layers as
    `dino_features`. The aggregation still sees all 24; the objective is held at the baseline's.
    Applied via `KMIRA_PIN_CONSISTENCY_LAYERS` in the launcher, unset by default, so stock runs are
    untouched.

    **The pairing is a silent footgun and was checked numerically, not by inspection.**
    `DinoPerceptualLoss.forward` does `for p, t in zip(pred_features, target_features)` — a
    **non-strict** `zip`. If the encoder's returned targets and the pinned loss's layers ever
    disagree, layers misalign (layer 11's prediction scored against layer 0's target) or the list
    silently truncates; nothing raises, and training would quietly optimise the wrong thing. Two
    checks: `tests/test_learned_layer_mix.py` asserts with `torch.equal` that layer *i* of a
    24-layer `get_intermediate_layers` read is the *same tensor* as the matching entry of a native
    7-layer read (the premise both the aggregation and the loss claims rest on — and not something
    to assume about an API whose `n=` argument changes meaning between an int and a tuple), and a
    one-off end-to-end comparison built a stock encoder plus its own unpinned `CodecLoss` alongside
    the variant plus the pinned one and ran both on identical input:

    ```
    targets: identical tensors at every position
    latent max abs diff      : 1.311e-06
    stock   consistency loss : 0.0000017442
    variant consistency loss : 0.0000017442
    abs diff                 : 0.000e+00
    ```

    So the variant can now be run **alone** and compared against the locked baseline's 24.885 dB.
    The control arm survives as optional (`run_learned_layer_mix_warmstart.sh 4 control,learned_mix`)
    for the one thing a single arm still cannot separate: the **warm restart itself**. The baseline
    was annealed to its minimum LR, and `finetune_from` resets the optimizer and re-applies warmup +
    constant LR — leaving an annealed minimum at a raised LR costs quality before it regains any, so
    a first-hour dip below 24.885 is expected *whether or not the idea works* and must not be read
    as a negative result.

    The control-free signal is the **weights themselves**, printed by
    `codec/scripts/report_layer_mix.py`. They start at exactly the stock formula's values, so where
    they move is a direct read on whether the gradient wants a different layer combination at all —
    unaffected by the LR restart. Barely-moved weights say the hand-picked formula was already near
    a local optimum, whatever PSNR happens to be doing that hour; mass appearing on the 17
    previously-unused layers is the positive signal.

    Also fixed here: the variant's `forward` had dropped stock `RAEEncoder.forward`'s RAEv2 noise
    regulariser. Inert at our `noise_tau: 0.0`, but a silent divergence from stock the day any
    experiment turns it on. Restored verbatim.

21. **The warm-start comparison number is 24.747, not 24.885 — and "recovering to convergence" is
    the wrong mental model.** Asked how long a warm-started run needs to shake off the anneal and
    re-converge, the premise turned out to be the thing to fix. The two baseline numbers are:

    | | LR | PSNR |
    |---|---|---|
    | plateau, step 272,000 | constant 1e-4 | **24.747** |
    | annealed, step 304,000 | cosine 1e-4 -> 1e-6 over 32k | **24.885** |

    The +0.138 dB is a **property of the low LR**, not a better region of parameter space that
    training found and can find again. `finetune_from` resets the optimizer and re-applies
    warmup + constant 1e-4, which hands that 0.138 dB straight back. So a constant-LR run can never
    reach 24.885 *however long it runs* — and there is no "re-convergence" to wait for, because
    24.747 is precisely where the baseline had already converged at 1e-4 (24.736 -> 24.747 over its
    final 8,000 steps). The restart does not knock the model off a plateau it must climb back; it
    drops it onto one it is already sitting on.

    Practical consequences:

    - **Compare against 24.747.** Both it and the variant's number are then constant-LR-1e-4
      equilibria, which is an apples-to-apples comparison. `report_layer_mix.py` prints both columns
      with the annealed one labelled unreachable, so it cannot be mistaken for a target.
    - **The settling is short, not hours.** It is a step change in LR plus zeroed Adam moments, not a
      re-traversal of the 272k-step training curve. Unmeasured so far, so the first hour's
      validation curve (8 readings at `val_every=1000`) is what establishes it; expect PSNR to fall
      from 24.885 toward ~24.75 and flatten.
    - **Beating the plateau is the result; annealing turns it into a headline number.** If the
      variant settles meaningfully above 24.747, re-anneal with `run_anneal.sh`'s recipe to get a
      number directly comparable to 24.885.
    - **The noise floor at convergence is still unknown** (step 13's 1.14 dB measured unconverged
      runs and does not apply). The variant is a *paired* warm start — same checkpoint, same seed,
      identical at step 0 — so it is far tighter than that, but a small positive delta should be
      confirmed with the control arm rather than believed.

    Also fixed here: this runner had a **fixed** `run.seed` across chunks, reintroducing step 14's
    data-repetition bug (mira reseeds its train loader per process start and does not checkpoint it,
    so every hourly chunk would replay the identical ~32k-sample stream). Now derived from the chunk
    index, which keeps the two arms paired — each arm's chunk N draws the same data — while still
    advancing the stream.

## Running a training session

```bash
bash codec/scripts/run_calibration.sh          # all three arms, ~2.2h each
bash codec/scripts/run_calibration.sh 4000     # shorter probe, ~35 min each
```

Runs in the **foreground** — deliberately not detached, so it lives only as long as its terminal and
leaves no orphaned process. Leave the terminal open; output prints live and each arm also writes
`checkpoints/calibration/<arm>.log`. Interrupted runs resume from the last checkpoint, so re-running
the script continues rather than restarting.

Each arm scores its final checkpoint automatically, and the script prints a signal-vs-noise verdict
at the end.

Timing at the measured 0.47 s/step (batch 4): training is `steps x 0.47s`, plus ~7 min of validation
(20 x ~21s) and ~3 min of scoring per arm.

| Steps/arm | Per arm | All three |
|---|---|---|
| 15,300 (default) | ~2.2h | **~6.6h** |
| 8,000 | ~1.2h | ~3.5h |
| 4,000 | ~35min | ~1.8h |
