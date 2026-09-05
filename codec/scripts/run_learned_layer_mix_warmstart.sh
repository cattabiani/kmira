#!/usr/bin/env bash
# Warm-started experiment: does a learned per-DINO-layer combination beat the stock fixed formula,
# starting from the locked baseline (checkpoint-304000) rather than from scratch?
#
#   bash codec/scripts/run_learned_layer_mix_warmstart.sh 1                   # 1h MORE, variant only
#   bash codec/scripts/run_learned_layer_mix_warmstart.sh 4                   # 4h MORE, variant only
#   bash codec/scripts/run_learned_layer_mix_warmstart.sh 4 control,learned_mix  # + paired control
#
# HOURS IS RELATIVE, NOT ABSOLUTE: each arm resumes from wherever its own checkpoint directory
# currently sits and trains HOURS more from there -- an arm at step 64000 given HOURS=7 runs to
# 120000 (64000 + 7*8000), not to 56000. The two arms can be at different step counts (e.g. after
# running them for different total hours across sessions) and each still gets exactly HOURS more.
#
# DEFAULT IS ONE ARM: `learned_mix` (VideoCodecLearnedLayerMix). The cheap first pass -- run the
# idea, look at what it does, decide whether it earns a controlled comparison.
#
# WHAT ONE ARM CAN AND CANNOT TELL YOU
# ------------------------------------
# The confound that originally forced a control is gone. mira ties the DINO latent-consistency
# loss's layer set to the encoder's (CodecLoss.bind_encoder_dino), so a 24-layer encoder would have
# silently trained against a 24-layer objective where the baseline used 7 -- two changes at once.
# KMIRA_PIN_CONSISTENCY_LAYERS below pins that loss to the baseline's 7 regardless of what the
# encoder reads, and the encoder returns exactly those 7 as the loss's targets. The objective is now
# byte-identical to the baseline's; the ONLY difference is that the aggregation weights can move.
#
# What one arm still cannot separate is the warm-restart itself. The baseline was ANNEALED to its
# minimum LR, and finetune_from resets the optimizer and re-applies warmup + constant LR. Leaving an
# annealed minimum at a raised LR costs quality before it regains any, so this arm's PSNR is
# expected to dip below 24.885 first and climb back -- with or without a working idea. Do not read
# "below the baseline after 1h" as "the idea failed".
#
# The control-free signals worth reading, both printed at the end:
#   1. the learned weights themselves -- if after an hour they have barely left their
#      stock-equivalent init, the gradient is telling you the fixed formula was already near a local
#      optimum, and no amount of further training changes that verdict;
#   2. PSNR *recovering past* 24.885 quickly -- that would be positive despite the restart penalty,
#      since the restart only ever works against it.
# If either signal is interesting, re-run with `control,learned_mix` for the paired number. The
# control is VideoCodecFixedLayerMix: same class, same 24-layer exposure, same pinned loss, same
# seed and schedule, weights frozen -- so it absorbs exactly the restart penalty and nothing else.
#
# `learned_mix`'s weights are initialised to reproduce the stock formula EXACTLY (see
# src/kmira/codec/variants/learned_layer_mix.py), verified against the real checkpoint: loading it
# reproduces all 616 saved keys byte-for-byte, and the one new key (encoder.layer_weights) is left at
# that stock-equivalent init. Step 0 therefore behaves identically to the locked baseline.
set -euo pipefail

cd "$(dirname "$0")/../.."

HOURS="${1:-4}"
ARMS="${2:-learned_mix}"
case "$HOURS" in
  ''|*[!0-9]*) echo "usage: $0 <whole hours MORE, per arm> [arms]   (got '$HOURS')" >&2; exit 1 ;;
esac
[ "$HOURS" -lt 1 ] && { echo "usage: $0 <whole hours MORE, per arm>, minimum 1" >&2; exit 1; }

for arm in ${ARMS//,/ }; do
  case "$arm" in
    control|learned_mix) ;;
    *) echo "usage: arms must be 'control', 'learned_mix', or 'control,learned_mix' (got '$arm')" >&2; exit 1 ;;
  esac
done

BASELINE_CKPT="$PWD/checkpoints/calibration/plateau_baseline/checkpoint-304000/checkpoint.pth"
if [ ! -f "$BASELINE_CKPT" ]; then
  echo "ABORT: locked baseline not found at $BASELINE_CKPT" >&2
  exit 1
fi

# Per-CHUNK seed, derived from the chunk index. mira reseeds its train loader from cfg.run.seed at
# every process start and does not checkpoint the loader, so a fixed seed makes every hourly chunk
# replay the identical ~32k-sample stream -- the silent bug that invalidated 56k steps of coverage in
# the plateau run (README step 14). Deriving it from the chunk index instead keeps both arms PAIRED
# (each arm's chunk N uses the same seed, hence the same data) while still advancing the stream.
SEED_BASE=28
CHUNK=8000                           # one hour at 0.45 s/step, same grid as the other scripts
VAL_EVERY=1000
SCORE_SECONDS=372
WARMUP=200                           # short: fine-tuning a converged model, not training from 0
HUB_SCRIPT="$PWD/codec/scripts/train_codec_finetune_hub.py"

if command -v pixi >/dev/null 2>&1; then PIXI=pixi; else PIXI="$HOME/.pixi/bin/pixi"; fi

export MIRA_TRAIN="${MIRA_TRAIN:?set by ../.envrc via direnv, or export it by hand}"
export RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$HOME/projects/shared/dino_weights}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Hold the DINO latent-consistency loss at the baseline's 7 layers even though the encoder now reads
# all 24, so the objective is unchanged and only the aggregation differs. Must match STOCK_LAYERS in
# src/kmira/codec/variants/learned_layer_mix.py, which is what the encoder hands back as targets --
# mira pairs the two positionally with a NON-strict zip, so a mismatch misaligns layers silently
# rather than raising. tests/test_learned_layer_mix.py checks the pairing numerically.
export KMIRA_PIN_CONSISTENCY_LAYERS=11,13,15,17,19,21,23

current_step () {
  local out="$1"
  if compgen -G "$out/checkpoint-*" > /dev/null; then
    basename "$(ls -d "$out"/checkpoint-*/ | sort -V | tail -1)" | sed 's#checkpoint-##;s#/##'
  else
    echo 0
  fi
}

score_checkpoint () {
  local out="$1" tag_prefix="$2" step="$3"
  if [ "$step" -eq 0 ]; then return 0; fi
  if grep -q "\"tag\": \"${tag_prefix}-${step}\"" codec/results/benchmark.jsonl 2>/dev/null; then
    echo "--- step $step already scored, skipping"
    return 0
  fi
  echo "--- scoring step $step (~$((SCORE_SECONDS / 60))min) ---"
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$PIXI" run python -m kmira.benchmark.eval_codec \
      --checkpoint "$out/checkpoint-$step/checkpoint.pth" --n-frames 2048 \
      --tag "${tag_prefix}-${step}" 2>&1 | tail -8; then
      return 0
    fi
    if [ "$attempt" -lt 5 ]; then
      local wait=$((10 * 2 ** (attempt - 1)))
      echo "WARN: scoring failed (attempt $attempt/5) -- retrying in ${wait}s"
      sleep "$wait"
    fi
  done
  echo "ABORT: scoring step $step failed 5 times in a row." >&2
  exit 1
}

# Runs one arm HOURS more, from wherever it currently sits: chunked, scored before each further
# chunk (see run_plateau.sh's captain's-log entry for why that ordering -- not after -- avoids
# orphaning a trained checkpoint if scoring fails), retried through the torch.hub flake.
run_arm () {
  local name="$1" model="$2" new_keys="$3" tag_prefix="$4"
  local out="$PWD/checkpoints/calibration/warmstart_${name}"
  local log="checkpoints/calibration/warmstart_${name}.log"
  mkdir -p "$out"

  # Relative to THIS arm's own current step, not a shared/global target -- so two arms at different
  # step counts (e.g. run for different total hours across sessions) each get exactly HOURS more.
  local start_step; start_step="$(current_step "$out")"
  local total=$((start_step + HOURS * CHUNK))

  echo ""
  echo "=================================================================="
  echo " ARM $name   model=$model   +${HOURS}h: $start_step -> $total steps"
  echo "=================================================================="

  local chunk_n=0 n_chunks="$HOURS"
  while [ "$(current_step "$out")" -lt "$total" ]; do
    local done_step; done_step="$(current_step "$out")"
    score_checkpoint "$out" "$tag_prefix" "$done_step"

    local next=$((done_step + CHUNK)); [ "$next" -gt "$total" ] && next="$total"
    chunk_n=$((chunk_n + 1))
    local seed=$((SEED_BASE + done_step / CHUNK))

    echo ""
    echo "--- $name chunk $chunk_n/$n_chunks: $done_step -> $next   seed=$seed   $(date '+%H:%M:%S') ---"

    # finetune_from only on this arm's very first chunk (done_step==0): it resets step/optimizer to
    # 0, so passing it again on a later chunk would erase this run's own progress and restart from
    # the baseline every time. Later chunks rely on ordinary output_dir auto-resume (mira's _resume,
    # same as run_plateau.sh/run_anneal.sh), triggered by leaving finetune_from unset.
    local finetune_arg=""
    if [ "$done_step" -eq 0 ]; then
      finetune_arg="run.finetune_from=$BASELINE_CKPT"
    fi

    local train_ok=0
    for attempt in 1 2 3 4 5; do
      if KMIRA_FINETUNE_NEW_KEYS="$new_keys" "$PIXI" run python "$HUB_SCRIPT" \
        --config-dir="$PWD/codec/configs" \
        --config-name=kmira_train_codec \
        model="$model" \
        dataset.train_index="$PWD/data/rocket_science/train" \
        dataset.test_index="$PWD/data/rocket_science/test" \
        run.output_dir="$out" \
        run.seed="$seed" \
        run.steps="$((next + 1))" \
        run.batch_size=4 \
        run.compile=false \
        run.checkpoint_every="$CHUNK" \
        run.checkpoint_keep_permanent_every=-1 \
        run.checkpoint_keep_recent=6 \
        validation.val_every="$VAL_EVERY" \
        validation.val_n_samples=512 \
        validation.val_first=$([ "$done_step" -eq 0 ] && echo true || echo false) \
        optim.scheduler.warmup_steps=$WARMUP \
        optim.scheduler.constant_steps=$((total - WARMUP)) \
        optim.scheduler.decay_steps=0 \
        wandb.mode=disabled \
        $finetune_arg \
        2>&1 | tee -a "$log"; then
        train_ok=1
        break
      fi
      if [ "$attempt" -lt 5 ]; then
        local wait=$((10 * 2 ** (attempt - 1)))
        echo "WARN: $name chunk failed (attempt $attempt/5) -- retrying in ${wait}s" | tee -a "$log"
        sleep "$wait"
      fi
    done
    if [ "$train_ok" -ne 1 ]; then
      echo "ABORT: $name chunk failed 5 times in a row." >&2
      exit 1
    fi

    local step; step="$(current_step "$out")"
    if [ "$step" -le "$done_step" ]; then
      echo "ABORT: $name chunk finished at step $step, no further than it started ($done_step)." >&2
      exit 1
    fi
  done

  score_checkpoint "$out" "$tag_prefix" "$(current_step "$out")"
}

START=$(date +%s)

# Both arms declare encoder.layer_weights as the expected-new key: the control has that parameter
# too (frozen), so both load the same stock baseline checkpoint the same way.
for arm in ${ARMS//,/ }; do
  case "$arm" in
    control)     run_arm control     learned_layer_mix_control "encoder.layer_weights" control ;;
    learned_mix) run_arm learned_mix learned_layer_mix         "encoder.layer_weights" learned_mix ;;
  esac
done

echo ""
echo "=================================================================="
echo " done in $(( ($(date +%s) - START) / 60 )) min"
echo "=================================================================="
LEARNED_CKPT_DIR="$PWD/checkpoints/calibration/warmstart_learned_mix"
ARMS="$ARMS" LEARNED_CKPT_DIR="$LEARNED_CKPT_DIR" "$PIXI" run python codec/scripts/report_layer_mix.py
