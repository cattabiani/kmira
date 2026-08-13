#!/usr/bin/env bash
# Anneal the plateaued baseline: cosine-decay the LR to min_lr from wherever run_plateau.sh stopped,
# producing the actual reference number (constant LR alone leaves real dB on the table) and locking
# the recipe -- warmup + constant(to elbow) + decay(this) -- that every future bottleneck variant
# should also use, so comparisons stay apples-to-apples.
#
#   bash codec/scripts/run_anneal.sh          # anneal over 4h (~32,000 steps)
#   bash codec/scripts/run_anneal.sh 2        # a shorter anneal
#
# The decay window is fixed ONCE at the checkpoint this script finds on first launch, then reused
# unchanged on every resumed chunk. This is NOT the plateau script's chunk logic (which recomputed
# constant_steps every chunk, which was harmless there since decay_steps=0 made constant_steps'
# exact value irrelevant -- see codec/README.md). Here decay_steps and constant_steps jointly define
# WHERE the decay starts, so recomputing them relative to a growing step count would keep pushing
# the decay start later on every resume and never actually anneal.
set -euo pipefail

cd "$(dirname "$0")/../.."

HOURS="${1:-4}"
case "$HOURS" in
  ''|*[!0-9]*) echo "usage: $0 <whole hours>   (got '$HOURS')" >&2; exit 1 ;;
esac
[ "$HOURS" -lt 1 ] && { echo "usage: $0 <whole hours>, minimum 1" >&2; exit 1; }

WARMUP=1000                          # matches every plateau chunk; irrelevant once past it anyway
NAME=plateau_baseline                # SAME output_dir as run_plateau.sh -- this continues that run
OUT="$PWD/checkpoints/calibration/$NAME"
LOG="checkpoints/calibration/${NAME}.log"
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
MIRA_TRAIN="$PWD/codec/scripts/train_codec_anneal_hub.py"

if command -v pixi >/dev/null 2>&1; then PIXI=pixi; else PIXI="$HOME/.pixi/bin/pixi"; fi

export RS_DINO_WEIGHTS_DIR="$PWD/data/dino_weights"
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
  if grep -q "\"tag\": \"anneal-$step\"" codec/results/benchmark.jsonl 2>/dev/null; then
    echo "--- step $step already scored, skipping"
    return 0
  fi
  echo "--- scoring step $step (~$((SCORE_SECONDS / 60))min) ---"
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python -m kmira.benchmark.eval_codec \
      --checkpoint "$OUT/checkpoint-$step/checkpoint.pth" --n-frames 2048 \
      --tag "anneal-$step" 2>&1 | tail -8; then
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

# ---- fix the decay window ONCE, from wherever the plateau run currently stands -------------------
BASE_STEP="$(current_step)"
if [ "$BASE_STEP" -eq 0 ]; then
  echo "ABORT: no checkpoint found in $OUT -- run_plateau.sh hasn't produced one yet." >&2
  exit 1
fi
ANNEAL_STEPS=$((HOURS * CHUNK))
TOTAL=$((BASE_STEP + ANNEAL_STEPS))
CONSTANT_STEPS=$((BASE_STEP - WARMUP))   # decay begins exactly at BASE_STEP
DECAY_STEPS="$ANNEAL_STEPS"
MIN_LR=1e-6

echo "=================================================================="
echo " ANNEAL -- from step $BASE_STEP, decaying to min_lr over ${HOURS}h (${ANNEAL_STEPS} steps)"
echo "   base (elbow)  : $BASE_STEP steps, constant LR (that run's own summary is in the log above)"
echo "   target        : $TOTAL steps"
echo "   schedule      : warmup=$WARMUP constant=$CONSTANT_STEPS decay=$DECAY_STEPS min_lr=$MIN_LR"
echo "   started       : $(date '+%H:%M:%S')"
echo "=================================================================="

CHUNK_N=0
N_CHUNKS="$HOURS"
while [ "$(current_step)" -lt "$TOTAL" ]; do
  DONE="$(current_step)"
  score_checkpoint "$DONE"   # score whatever we're standing on before training further (see
                              # run_plateau.sh's captain's-log entry for why this ordering matters)

  NEXT=$((DONE + CHUNK)); [ "$NEXT" -gt "$TOTAL" ] && NEXT="$TOTAL"
  CHUNK_N=$((CHUNK_N + 1))

  echo ""
  echo "--- anneal chunk $CHUNK_N/$N_CHUNKS: $DONE -> $NEXT   $(date '+%H:%M:%S') ---"

  TRAIN_OK=0
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python "$MIRA_TRAIN" \
      --config-dir="$PWD/codec/configs" \
      --config-name=kmira_train_codec \
      model=baseline_image_base \
      dataset.train_index="$PWD/data/rocket_science/train" \
      dataset.test_index="$PWD/data/rocket_science/test" \
      run.output_dir="$OUT" \
      run.seed=$((28 + NEXT / CHUNK)) \
      run.steps="$((NEXT + 1))" \
      run.batch_size=4 \
      run.compile=false \
      run.checkpoint_every="$CHUNK" \
      run.checkpoint_keep_permanent_every=-1 \
      run.checkpoint_keep_recent=6 \
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
done

score_checkpoint "$(current_step)"

echo ""
echo "=================================================================="
echo " anneal done at $(date '+%H:%M:%S') -- $TOTAL steps total"
echo "=================================================================="
$HOME/.pixi/bin/pixi run python -c "
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path('codec/results/benchmark.jsonl').read_text().splitlines() if l.strip()]
plateau = [r for r in rows if r['tag'].startswith('plateau-')]
anneal = [r for r in rows if r['tag'].startswith('anneal-')]
plateau.sort(key=lambda r: int(r['tag'].split('-')[1]))
anneal.sort(key=lambda r: int(r['tag'].split('-')[1]))
if plateau:
    p = plateau[-1]
    print(f\"last plateau (constant LR)  step={p['tag'].split('-')[1]:>7}  psnr={p['psnr']:.2f}\")
for r in anneal:
    print(f\"anneal                      step={r['tag'].split('-')[1]:>7}  psnr={r['psnr']:.2f}\")
"
echo ""
echo "This is the FINAL baseline reference. Its checkpoint and this recipe (warmup=$WARMUP"
echo "constant=$CONSTANT_STEPS decay=$DECAY_STEPS min_lr=$MIN_LR) are what future bottleneck"
echo "variants should match for a fair from-scratch comparison."
