#!/usr/bin/env bash
# Train the baseline for a SLOT OF TIME, scoring as it goes. Run it again to add another slot;
# slots accumulate into one continuous run.
#
#   bash codec/scripts/run_plateau.sh 12     # 12h of training (plus ~11% scoring)
#   bash codec/scripts/run_plateau.sh 1      # one hour
#   bash codec/scripts/run_plateau.sh        # default 8 hours
#
# STRUCTURE: an outer loop over hourly CHUNKS. Each chunk trains for an hour, validates, checkpoints
# and scores -- all on the same step -- then moves on. An interruption loses at most the current
# hour; every completed hour is already durable in results/benchmark.jsonl. mira's trainer
# auto-resumes from output_dir, so each chunk is just another invocation with a higher step target;
# no fork of mira is needed to interleave scoring. Restart costs ~10s per chunk, under 0.5%.
#
# The hour is the unit, which is what keeps this simple: a slot can only end on a chunk boundary,
# which is always a checkpoint AND a validation step. Validation runs more often than that (every
# 15min, as a divisor of the hour) because it is cheap and the elbow is easier to locate with more
# points; scoring stays hourly because it is not cheap.
#
# A ROLLING WINDOW of the last KEEP_RECENT hourly checkpoints is kept (4.7GB each, so ~28GB flat,
# whether the run is 12h or 48h). The window is sized by the elbow detector's lag, not by taste: its
# criterion is trailing over 2h, so when it reports an elbow at hour X the model actually wanted is
# somewhere in hours X-2..X. Six hours of history covers that with margin.
#
# Keeping ALL of them (146GB at 31h) was considered and dropped: past the elbow window, an old
# checkpoint only helps if the EVAL changes and we want to re-score without retraining, if the
# newest is corrupted mid-write, or if we want to branch a run from the middle -- all rare, and none
# of them help against a TRAINING bug, which sends you back to step 0 whatever is on disk.
#
# This only works because the elbow is checked after EVERY chunk (see the report call in the loop):
# a window is useless if you notice the elbow after it has already scrolled out of it.
KEEP_RECENT=6
#
# WHY SLOTS WORK HERE: the LR is CONSTANT (after a short warmup), so there is no schedule to
# interrupt -- every checkpoint is a valid model and stopping is free. Under mira's default cosine
# decay this would not hold: a checkpoint from the middle of a decaying schedule has not settled,
# and extending a finished cosine run is useless because the LR has already annealed to ~0.
# Constant LR also makes "where does the curve flatten" a well-posed question, since the flattening
# cannot be an artifact of the schedule running out.
#
# The number this produces is a LOCATION, not a quality result: constant LR settles higher and
# noisier than an annealed run. The protocol we adopt afterwards is "elbow-many steps, with cosine
# decay put back", which lands a little better than anything seen here.
#
# For scale: mira trains this codec for 250,001 steps ~= 31h at our measured rate.
#
# Runs FOREGROUND -- no nohup/disown, so it lives only as long as this terminal. Leave it open.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Whole hours only. The chunk loop trains an hour at a time, so anything finer would round anyway;
# rejecting it up front beats silently doing something other than what was asked.
HOURS="${1:-8}"
case "$HOURS" in
  ''|*[!0-9]*) echo "usage: $0 <whole hours>   (got '$HOURS')" >&2; exit 1 ;;
esac
[ "$HOURS" -lt 1 ] && { echo "usage: $0 <whole hours>, minimum 1" >&2; exit 1; }
WARMUP=1000                          # only applies to the very first chunk; later ones are past it
NAME=plateau_baseline
OUT="$PWD/checkpoints/calibration/$NAME"
LOG="checkpoints/calibration/${NAME}.log"
# A launcher around mira's unmodified trainer, not a fork: it patches torch.hub to resolve the
# DINOv3 repo ref from the local cache instead of asking GitHub on every single model construction,
# then runs mira's script as __main__. See src/kmira/torch_hub_offline.py.
MIRA_TRAIN="$PWD/codec/scripts/train_codec_offline_hub.py"

# Cadences are WALL-CLOCK, not a fraction of the run: a percentage would mean a short slot and a
# long slot have different spacing, so their curves could not be laid end to end. mira's
# periodic_event takes a plain int as "every N steps", so this is just a conversion.
SEC_PER_STEP=0.45                    # measured: batch 4, Base decoder, 288x512, compile off
SCORE_SECONDS=372                    # measured: 2048 frames, consistent to within 2s across runs

# ONE HOUR IS THE UNIT. A chunk is an hour; a slot is N hours; checkpointing and scoring happen once
# per chunk. Every alignment question this script used to answer -- does a slot end on a step with a
# validation reading, does a short slot round to zero work -- stops existing rather than being
# handled.
CHUNK=8000                           # one hour at 0.45 s/step

# Validation every 1000 steps: 8 per hour, ~7.5 min apart. It must DIVIDE the chunk exactly, because
# plateau_report.py joins scored checkpoints to validation readings ON STEP NUMBER -- if the hourly
# checkpoint step is not also a validation step, every scored point gets an empty loss cell. 1000
# divides 8000; 1333 (a literal 10 minutes) does not, which is why the interval is round rather than
# exact. A validation costs ~18s, so 8/hour is ~4% overhead -- cheap enough to buy resolution on the
# elbow, where scoring at 6min per point is not.
VAL_EVERY=1000
VAL_PER_HOUR=$((CHUNK / VAL_EVERY))

if command -v pixi >/dev/null 2>&1; then PIXI=pixi; else PIXI="$HOME/.pixi/bin/pixi"; fi

export RS_DINO_WEIGHTS_DIR="$PWD/data/dino_weights"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT" codec/results

current_step () {
  if compgen -G "$OUT/checkpoint-*" > /dev/null; then
    basename "$(ls -d "$OUT"/checkpoint-*/ | sort -V | tail -1)" | sed 's#checkpoint-##;s#/##'
  else
    echo 0
  fi
}

# Score one checkpoint, retrying through the torch.hub GitHub flake (see the training-call comment
# below for what it is). 5 attempts with exponential backoff (10/20/40/80s, ~150s of sleep plus
# attempt time) covers a longer flaky patch than the original 3x15s did -- that budget turned out
# too thin for a real outage, not just a single dropped connection.
score_checkpoint () {
  local step="$1"
  if grep -q "\"tag\": \"plateau-$step\"" codec/results/benchmark.jsonl 2>/dev/null; then
    echo "--- step $step already scored, skipping"
    return 0
  fi
  echo "--- scoring step $step (~$((SCORE_SECONDS / 60))min) ---"
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python -m kmira.benchmark.eval_codec \
      --checkpoint "$OUT/checkpoint-$step/checkpoint.pth" --n-frames 2048 \
      --tag "plateau-$step" 2>&1 | tail -8; then
      return 0
    fi
    if [ "$attempt" -lt 5 ]; then
      local wait=$((10 * 2 ** (attempt - 1)))
      echo "WARN: scoring step $step failed (attempt $attempt/5) -- retrying in ${wait}s"
      sleep "$wait"
    fi
  done
  echo "ABORT: scoring step $step failed 5 times in a row -- stopping rather than looping forever." >&2
  exit 1
}

# Targets live on the CHUNK GRID (multiples of CHUNK), not relative to wherever the last checkpoint
# happens to sit. If a checkpoint is ever slightly off-grid -- as happened when an off-by-one in
# run.steps left a checkpoint-7999 -- planning relative to it would propagate that offset to every
# future checkpoint, and every one of them would miss the validation grid and join to nothing.
# Snapping to the nearest grid point instead absorbs the offset in a single chunk.
DONE="$(current_step)"
N_CHUNKS="$HOURS"
GRID=$(python3 -c "print(round($DONE / $CHUNK) * $CHUNK)")
TOTAL=$(( GRID + N_CHUNKS * CHUNK ))
WALL=$(python3 -c "print(f'{($TOTAL - $DONE) * $SEC_PER_STEP / 3600 + $N_CHUNKS * ($SCORE_SECONDS + 10) / 3600:.1f}')")

echo "=================================================================="
echo " PLATEAU PROBE -- slot of ${HOURS}h training"
echo "   already done : $DONE steps"
echo "   this slot    : +$((TOTAL - DONE)) steps -> $TOTAL total, in $N_CHUNKS chunks of $CHUNK"
echo "   per hour     : $CHUNK steps, $VAL_PER_HOUR validations (every $VAL_EVERY), one checkpoint, one scoring (~$((SCORE_SECONDS / 60))min)"
echo "   wall clock   : ~${WALL}h including scoring"
echo "   checkpoints  : rolling last $KEEP_RECENT (~$((KEEP_RECENT * 47 / 10))GB, flat regardless of slot length)"
echo "   started      : $(date '+%H:%M:%S')"
echo "=================================================================="

# Flat footprint, but check anyway: dying 30h in on a full disk is a bad way to find out.
NEED_GB=$((KEEP_RECENT * 47 / 10 + 20))
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -d ' G')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "ABORT: need ~${NEED_GB}GB (${KEEP_RECENT} checkpoints + headroom), only ${FREE_GB}GB free." >&2
  exit 1
fi

CHUNK_N=0
while [ "$(current_step)" -lt "$TOTAL" ]; do
  DONE="$(current_step)"

  # Score whatever checkpoint we're currently standing on, BEFORE training the next chunk. This
  # matters on resume: NEXT (below) is always strictly greater than DONE, so a scheme that only
  # scores "whatever step training just reached" can never go back and score DONE if a PREVIOUS
  # invocation trained it but then crashed -- or exhausted its retries -- before scoring succeeded.
  # That orphaned checkpoint-208000 for a full extra invocation: this run resumed straight into
  # training the 208000->216000 chunk without ever retrying the scoring that had failed earlier.
  if [ "$DONE" -gt 0 ]; then
    score_checkpoint "$DONE"
  fi

  # Next grid point above where we are, so an off-grid checkpoint is corrected on the first chunk
  # rather than carried forever.
  NEXT=$(python3 -c "print(round($DONE / $CHUNK) * $CHUNK + $CHUNK)")
  [ "$NEXT" -gt "$TOTAL" ] && NEXT="$TOTAL"
  CHUNK_N=$((CHUNK_N + 1))

  # A DIFFERENT run.seed per chunk. Discovered after the fact: mira's train loader reseeds from
  # cfg.run.seed on every process start (src/mira/data/training_loader.py, rng = seed + rank*1024 +
  # worker_id) and is NOT among the checkpointed components -- only optimizer/scheduler/latent-EMA
  # are registered. With a fixed seed, every hourly restart replayed the IDENTICAL ~32k-sample
  # stream from the beginning, so chunks 1-7 of the first real run each trained on the same slice of
  # data rather than fresh data -- not 56,000 steps of coverage, closer to 7 epochs over one chunk's
  # worth. The sawtooth in the validation curve is that repetition, not noise. Deriving the seed from
  # the absolute step (HOUR_INDEX = NEXT/CHUNK) keeps it reproducible across separate script
  # invocations, unlike a counter that would reset to 1 every time. Model weights are unaffected: a
  # resumed chunk immediately overwrites the seed-initialized weights from the checkpoint, so this
  # only changes which data (and dropout draws) that chunk sees, never what it resumes from.
  HOUR_INDEX=$((NEXT / CHUNK))
  SEED=$((28 + HOUR_INDEX))

  # Progress within THIS slot, not the overall run -- chunks left * (train + score + restart) is a
  # clean estimate since every chunk in a slot is the same size (CHUNK steps).
  PCT=$(python3 -c "print(round(100 * ($CHUNK_N - 1) / $N_CHUNKS))")
  REMAINING_CHUNKS=$((N_CHUNKS - CHUNK_N + 1))
  REMAINING_SEC=$(python3 -c "print(round($REMAINING_CHUNKS * ($CHUNK * $SEC_PER_STEP + $SCORE_SECONDS + 10)))")
  REMAINING_H=$(python3 -c "print(f'{$REMAINING_SEC / 3600:.1f}')")
  ETA="$(date -d "+${REMAINING_SEC} seconds" '+%H:%M %a')"

  MIRA_PCT=$(python3 -c "print(round(100 * $NEXT / 250001))")

  echo ""
  echo "--- chunk $CHUNK_N/$N_CHUNKS: $DONE -> $NEXT   $(date '+%H:%M:%S') ---"
  echo "    slot progress: ${PCT}%   ~${REMAINING_H}h remaining   ETA ${ETA}   (${NEXT} steps, ${MIRA_PCT}% of mira's 250,001)"

  # run.steps is NEXT+1, following mira's own convention (its config says steps: 250_001, not
  # 250_000). The loop is `range(start_step, steps)`, so the last iteration -- and therefore the
  # final checkpoint -- lands on steps-1. Asking for exactly NEXT would checkpoint at NEXT-1, which
  # is both an infinite loop here (current_step never reaches the target, so the chunk repeats
  # forever training one step at a time) and a broken join, since NEXT-1 is not a validation step.
  #
  # decay_steps=0 disables cosine decay entirely (mira's train_codec.yaml documents this).
  # checkpoint_every=CHUNK gives exactly one checkpoint per chunk, at its end.
  # val_first only makes sense on the very first chunk; later ones would re-validate the step they
  # resume from, at 18s a time, for a reading already in the log.
  # Retried with exponential backoff: torch.hub._parse_repo_info does an unconditional urlopen to
  # GitHub to resolve the default branch, EVERY load, regardless of the repo already being cached
  # locally -- and its except URLError fallback-to-cache does not catch http.client.RemoteDisconnected
  # (observed escaping uncaught from urlopen), so a single dropped connection crashes the whole chunk
  # despite a perfectly good local cache sitting right there. This is torch.hub's own bug, not mira's
  # or ours, and out of reach without forking mira (the call site is inside its dino.py). A transient
  # connection reset is safe to just retry: training auto-resumes from the last checkpoint, so a
  # retry after a crash before any steps ran (as observed) simply starts the chunk over.
  #
  # 5 attempts / 10-20-40-80s backoff, not the original 3x15s: that budget (~45s of coverage) proved
  # too thin for a real flaky patch, which is what actually happened -- three straight failures with
  # network confirmed fine moments after the script gave up.
  TRAIN_OK=0
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python "$MIRA_TRAIN" \
      --config-dir="$PWD/codec/configs" \
      --config-name=kmira_train_codec \
      model=baseline_image_base \
      dataset.train_index="$PWD/data/rocket_science/train" \
      dataset.test_index="$PWD/data/rocket_science/test" \
      run.output_dir="$OUT" \
      run.seed="$SEED" \
      run.steps="$((NEXT + 1))" \
      run.batch_size=4 \
      run.compile=false \
      run.checkpoint_every="$CHUNK" \
      run.checkpoint_keep_permanent_every=-1 \
      run.checkpoint_keep_recent="$KEEP_RECENT" \
      validation.val_every="$VAL_EVERY" \
      validation.val_n_samples=512 \
      validation.val_first=$([ "$DONE" -eq 0 ] && echo true || echo false) \
      optim.scheduler.warmup_steps=$WARMUP \
      optim.scheduler.constant_steps=$((TOTAL - WARMUP)) \
      optim.scheduler.decay_steps=0 \
      wandb.mode=disabled \
      2>&1 | tee -a "$LOG"; then
      TRAIN_OK=1
      break
    fi
    if [ "$attempt" -lt 5 ]; then
      wait=$((10 * 2 ** (attempt - 1)))
      echo "WARN: training chunk failed (attempt $attempt/5) -- retrying in ${wait}s" | tee -a "$LOG"
      sleep "$wait"
    fi
  done
  if [ "$TRAIN_OK" -ne 1 ]; then
    echo "ABORT: training chunk failed 5 times in a row -- this is more than the known flake, stopping." >&2
    exit 1
  fi

  STEP="$(current_step)"
  # If a chunk did not advance the step counter, stop. Without this, any mismatch between what we
  # ask for and where mira actually checkpoints becomes an infinite loop that quietly burns the
  # whole slot -- which is exactly what an off-by-one in run.steps did on the first real run.
  if [ "$STEP" -le "$DONE" ]; then
    echo "ABORT: chunk finished at step $STEP, no further than it started ($DONE)." >&2
    echo "       Expected a checkpoint at $NEXT. Not looping." >&2
    exit 1
  fi

  # Note: STEP is NOT scored here. Scoring for the checkpoint a chunk just produced happens at the
  # TOP of the next loop iteration (or, for the slot's final chunk, in the call right after the loop
  # below) -- see the comment there for why.

  # Check for the elbow after every chunk, not just at the end of the slot. Retention is a rolling
  # 6h window, so a detection noticed only at slot end could refer to a checkpoint already deleted.
  # Just the verdict here; the full curve is printed once the slot finishes.
  "$PIXI" run python codec/scripts/plateau_report.py "$LOG" 2>&1 | tail -4
done

# The loop's "score what we're standing on" runs at the top of each iteration; there is no
# iteration after the slot's final chunk, so its checkpoint needs an explicit call here.
score_checkpoint "$(current_step)"

echo ""
echo "=================================================================="
echo " slot done at $(date '+%H:%M:%S') -- $TOTAL steps total"
echo "=================================================================="
"$PIXI" run python codec/scripts/plateau_report.py "$LOG"

echo ""
echo "No elbow yet? Add another slot -- it picks up exactly where this one stopped:"
echo "  bash codec/scripts/run_plateau.sh <hours>"
