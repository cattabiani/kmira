#!/usr/bin/env bash
# Anneal a plateaued arm: cosine-decay the LR to min_lr from wherever run_plateau.sh stopped,
# producing the actual reference number (constant LR alone leaves real dB on the table) and locking
# the recipe -- warmup + constant(to elbow) + decay(this) -- that every future bottleneck variant
# should also use, so comparisons stay apples-to-apples.
#
#   bash codec/scripts/run_anneal.sh                      # anneal baseline_v2 over 4h (~32,000 steps)
#   bash codec/scripts/run_anneal.sh 6                    # a longer anneal
#   bash codec/scripts/run_anneal.sh 4 abl_frozen         # anneal a different arm
#
# ARMS. This takes the SAME arm names as run_plateau.sh and resolves each to the same output dir,
# model, tag and seed base, so an anneal continues that arm's own run rather than a hardcoded one.
# It used to be pinned to NAME=plateau_baseline with seed base 28 -- the retired pre-bugfix run --
# which meant running it after any other arm silently extended the retired one.
#
# THE DECAY WINDOW IS FIXED ONCE, AT THE FIRST LAUNCH, AND PERSISTED TO $OUT/anneal_window.env.
# decay_steps and constant_steps jointly define WHERE the decay starts, so recomputing them against
# a step count that has grown would push the decay start later on every resume and the LR would
# never actually come down. Holding them constant within one invocation is not enough: the anneal
# spans several hours and gets interrupted, and each relaunch would re-base off the newer
# checkpoint. The window therefore lives in a file next to the checkpoints, and a relaunch reuses it
# verbatim -- including the original HOURS, so passing a different number on a resume is refused
# rather than quietly honoured. Delete that file to start a fresh anneal from the current step.
set -euo pipefail

cd "$(dirname "$0")/../.."

HOURS="${1:-4}"
case "$HOURS" in
  ''|*[!0-9]*) echo "usage: $0 <whole hours> [arm]   (got '$HOURS')" >&2; exit 1 ;;
esac
[ "$HOURS" -lt 1 ] && { echo "usage: $0 <whole hours>, minimum 1" >&2; exit 1; }

# Arm resolution, kept deliberately identical to run_plateau.sh's case block. If you add an arm
# there, add it here too, or that arm can be trained but never annealed.
ARM="${2:-baseline_v2}"
case "$ARM" in
  baseline_v2|abl_baseline)
    NAME=ablation_baseline      MODEL=baseline_image_base     TAG=abl_baseline    SEED_BASE=1028 ;;
  baseline_v2_s2)
    NAME=baseline_v2_seed2      MODEL=baseline_image_base     TAG=baseline_v2_s2  SEED_BASE=2028 ;;
  abl_frozen)
    NAME=ablation_frozen_bneck  MODEL=calib_frozen_bottleneck TAG=abl_frozen      SEED_BASE=1028 ;;
  baseline)
    if [ "${KMIRA_ALLOW_LEGACY_BASELINE:-0}" != "1" ]; then
      echo "REFUSING: 'baseline' is the retired pre-bugfix run (saw 53.4% of the data)." >&2
      echo "          Use 'baseline_v2' for the baseline, or 'baseline_v2_s2' for its seed branch." >&2
      echo "          Set KMIRA_ALLOW_LEGACY_BASELINE=1 only to re-score its existing checkpoints." >&2
      exit 1
    fi
    NAME=plateau_baseline       MODEL=baseline_image_base     TAG=plateau         SEED_BASE=28 ;;
  *) echo "usage: $0 <whole hours> [baseline_v2|baseline_v2_s2|abl_frozen]   (got arm '$ARM')" >&2; exit 1 ;;
esac

WARMUP=1000                          # matches every plateau chunk; irrelevant once past it anyway
OUT="$PWD/checkpoints/calibration/$NAME"
LOG="checkpoints/calibration/${NAME}.log"
WINDOW="$OUT/anneal_window.env"
SEC_PER_STEP=0.45
SCORE_SECONDS=372
CHUNK=8000                            # one hour, same grid as run_plateau.sh
VAL_EVERY=1000
# Two patches, neither touching mira: the torch.hub GitHub call short-circuited to the local cache
# (see src/kmira/torch_hub_offline.py), and the LR scheduler allowed to pick up decay_steps>0 on
# resume instead of having it silently overwritten by the checkpoint's old decay_steps=0 (see
# src/kmira/lr_resume_override.py -- confirmed with a standalone scheduler test before trusting it
# against 34 hours of training: resuming at step 10 with a freshly-configured 8-step decay produced
# a clean cosine 1e-4 -> 1e-6, exactly as specified, not silently flat).
HUB_SCRIPT="$PWD/codec/scripts/train_codec_anneal_hub.py"

if command -v pixi >/dev/null 2>&1; then PIXI=pixi; else PIXI="$HOME/.pixi/bin/pixi"; fi

export MIRA_TRAIN="${MIRA_TRAIN:?set by ../.envrc via direnv, or export it by hand}"
export RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$HOME/projects/shared/dino_weights}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

current_step () {
  if compgen -G "$OUT/checkpoint-*" > /dev/null; then
    basename "$(ls -d "$OUT"/checkpoint-*/ | sort -V | tail -1)" | sed 's#checkpoint-##;s#/##'
  else
    echo 0
  fi
}

score_checkpoint () {
  local step="$1"
  # Skip on the CHECKPOINT PATH, not the tag. The step the anneal starts from was already scored
  # during the constant-LR run, under that run's own tag; matching on the path reuses that row
  # instead of spending another six GPU-minutes re-measuring an identical checkpoint. Safe because
  # the anneal only ever moves forward past BASE_STEP, so no step is ever scored under two
  # different sets of weights.
  if grep -q "$OUT/checkpoint-$step/checkpoint.pth" codec/results/benchmark.jsonl 2>/dev/null; then
    echo "--- step $step already scored, skipping"
    return 0
  fi
  echo "--- scoring step $step (~$((SCORE_SECONDS / 60))min) ---"
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python -m kmira.benchmark.eval_codec \
      --checkpoint "$OUT/checkpoint-$step/checkpoint.pth" --n-frames 2048 \
      --tag "${TAG}-anneal-$step" 2>&1 | tail -8; then
      return 0
    fi
    if [ "$attempt" -lt 5 ]; then
      local wait=$((10 * 2 ** (attempt - 1)))
      echo "WARN: scoring step $step failed (attempt $attempt/5) -- retrying in ${wait}s"
      sleep "$wait"
    fi
  done
  echo "ABORT: scoring step $step failed 5 times in a row." >&2
  exit 1
}

# Drop the resume-only half of every checkpoint except the newest. Safe by construction: mira reads
# `training_state.pth` only in `continue_from`, which resolves the LATEST checkpoint, and that one
# is always kept. `checkpoint.pth` -- what eval_codec and `finetune_from` read -- is never touched.
prune_training_state () {
  local newest
  newest="$(ls -d "$OUT"/checkpoint-*/ 2>/dev/null | sort -V | tail -1)"
  [ -z "$newest" ] && return 0
  local freed=0 d
  for d in "$OUT"/checkpoint-*/; do
    [ "$d" = "$newest" ] && continue
    if [ -f "$d/training_state.pth" ]; then
      freed=$((freed + $(stat -c%s "$d/training_state.pth") / 1073741824))
      rm -f "$d/training_state.pth"
    fi
  done
  [ "$freed" -gt 0 ] && echo "--- pruned ~${freed}GiB of resume-only state from older checkpoints"
  return 0
}

# ---- fix the decay window ONCE, and keep it across resumes ---------------------------------------
MIN_LR=1e-6
if [ -f "$WINDOW" ]; then
  # shellcheck disable=SC1090
  . "$WINDOW"
  if [ "$HOURS" != "$ANNEAL_HOURS" ]; then
    echo "REFUSING: this arm is already annealing over ${ANNEAL_HOURS}h (window fixed at step" >&2
    echo "          $BASE_STEP, target $TOTAL). Changing the length mid-decay would move where the" >&2
    echo "          decay started and invalidate the schedule." >&2
    echo "          Re-run as: bash $0 $ANNEAL_HOURS $ARM" >&2
    echo "          To abandon this anneal and start a new one from the current step, delete" >&2
    echo "          $WINDOW first." >&2
    exit 1
  fi
  echo "--- resuming the anneal window fixed at step $BASE_STEP (target $TOTAL)"
else
  BASE_STEP="$(current_step)"
  if [ "$BASE_STEP" -eq 0 ]; then
    echo "ABORT: no checkpoint found in $OUT -- run_plateau.sh hasn't produced one yet." >&2
    exit 1
  fi
  ANNEAL_HOURS="$HOURS"
  ANNEAL_STEPS=$((HOURS * CHUNK))
  TOTAL=$((BASE_STEP + ANNEAL_STEPS))
  CONSTANT_STEPS=$((BASE_STEP - WARMUP))   # decay begins exactly at BASE_STEP
  DECAY_STEPS="$ANNEAL_STEPS"
  mkdir -p "$OUT"
  cat > "$WINDOW" <<EOF
# Written by run_anneal.sh on $(date -Is). The decay window for $ARM, fixed at first launch.
# Delete this file only to abandon the anneal and start a new one from the current step.
ANNEAL_HOURS=$ANNEAL_HOURS
BASE_STEP=$BASE_STEP
ANNEAL_STEPS=$ANNEAL_STEPS
TOTAL=$TOTAL
CONSTANT_STEPS=$CONSTANT_STEPS
DECAY_STEPS=$DECAY_STEPS
EOF
  echo "--- fixed the anneal window at step $BASE_STEP and wrote $WINDOW"
fi

echo "=================================================================="
echo " ANNEAL [$ARM] -- from step $BASE_STEP, decaying to min_lr over ${ANNEAL_HOURS}h (${ANNEAL_STEPS} steps)"
echo "   model         : $MODEL, seeds $((SEED_BASE + 1))+, tags ${TAG}-anneal-*"
echo "   base (elbow)  : $BASE_STEP steps, constant LR (that run's own summary is in the log above)"
echo "   target        : $TOTAL steps"
echo "   schedule      : warmup=$WARMUP constant=$CONSTANT_STEPS decay=$DECAY_STEPS min_lr=$MIN_LR"
echo "   checkpoints   : one per chunk, KEPT (~1.6GiB each after pruning resume-only state)"
echo "   started       : $(date '+%H:%M:%S')"
echo "=================================================================="

CHUNK_N=0
N_CHUNKS="$ANNEAL_HOURS"
while [ "$(current_step)" -lt "$TOTAL" ]; do
  DONE="$(current_step)"
  score_checkpoint "$DONE"   # score whatever we're standing on before training further (see
                              # run_plateau.sh's captain's-log entry for why this ordering matters)

  NEXT=$((DONE + CHUNK)); [ "$NEXT" -gt "$TOTAL" ] && NEXT="$TOTAL"
  CHUNK_N=$((CHUNK_N + 1))

  echo ""
  echo "--- anneal chunk $CHUNK_N/$N_CHUNKS: $DONE -> $NEXT   $(date '+%H:%M:%S') ---"

  # The same per-chunk seed schedule the plateau run used, continued unbroken: seeds are keyed to
  # the absolute step, so the anneal picks up the next slices rather than replaying earlier ones.
  SEED=$((SEED_BASE + NEXT / CHUNK))

  TRAIN_OK=0
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python "$HUB_SCRIPT" \
      --config-dir="$PWD/codec/configs" \
      --config-name=kmira_train_codec \
      model="$MODEL" \
      dataset.train_index="$PWD/data/rocket_science/train" \
      dataset.test_index="$PWD/data/rocket_science/test" \
      run.output_dir="$OUT" \
      run.seed="$SEED" \
      run.steps="$((NEXT + 1))" \
      run.batch_size=4 \
      run.compile=false \
      run.checkpoint_every="$CHUNK" \
      run.checkpoint_keep_permanent_every="$CHUNK" \
      validation.val_every="$VAL_EVERY" \
      validation.val_n_samples=512 \
      validation.val_first=false \
      optim.scheduler.warmup_steps=$WARMUP \
      optim.scheduler.constant_steps="$CONSTANT_STEPS" \
      optim.scheduler.decay_steps="$DECAY_STEPS" \
      optim.scheduler.min_lr="$MIN_LR" \
      wandb.mode=disabled \
      2>&1 | tee -a "$LOG"; then
      TRAIN_OK=1
      break
    fi
    if [ "$attempt" -lt 5 ]; then
      wait=$((10 * 2 ** (attempt - 1)))
      echo "WARN: anneal chunk failed (attempt $attempt/5) -- retrying in ${wait}s" | tee -a "$LOG"
      sleep "$wait"
    fi
  done
  if [ "$TRAIN_OK" -ne 1 ]; then
    echo "ABORT: anneal chunk failed 5 times in a row." >&2
    exit 1
  fi

  STEP="$(current_step)"
  if [ "$STEP" -le "$DONE" ]; then
    echo "ABORT: chunk finished at step $STEP, no further than it started ($DONE)." >&2
    exit 1
  fi
  prune_training_state
done

score_checkpoint "$(current_step)"
prune_training_state

echo ""
echo "=================================================================="
echo " anneal done at $(date '+%H:%M:%S') -- $TOTAL steps total"
echo "=================================================================="
"$PIXI" run python -c "
import json, pathlib, re
rows = [json.loads(l) for l in pathlib.Path('codec/results/benchmark.jsonl').read_text().splitlines() if l.strip()]
# Select on the checkpoint path, not the tag: tags are per-arm and one of them deliberately does not
# match its own run directory, so a prefix test can pair one run's numbers with another's.
mine = [r for r in rows if re.search(r'/$NAME/checkpoint-(\d+)/', r.get('checkpoint', ''))]
step = lambda r: int(re.search(r'/checkpoint-(\d+)/', r['checkpoint']).group(1))
mine.sort(key=step)
const = [r for r in mine if step(r) <= $BASE_STEP]
anneal = [r for r in mine if step(r) > $BASE_STEP]
if const:
    print(f'last constant-LR            step={step(const[-1]):>7}  psnr={const[-1][\"psnr\"]:.4f}')
for r in anneal:
    print(f'anneal                      step={step(r):>7}  psnr={r[\"psnr\"]:.4f}')
if const and anneal:
    print(f'\nanneal gained {anneal[-1][\"psnr\"] - const[-1][\"psnr\"]:+.4f} dB over the constant-LR endpoint')
"
echo ""
echo "This is the FINAL reference for arm '$ARM'. Its checkpoint and this recipe (warmup=$WARMUP"
echo "constant=$CONSTANT_STEPS decay=$DECAY_STEPS min_lr=$MIN_LR) are what future bottleneck"
echo "variants should match for a fair from-scratch comparison."
