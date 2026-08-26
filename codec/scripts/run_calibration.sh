#!/usr/bin/env bash
# Calibrate the benchmark: three runs that together answer "can this setup detect a bottleneck
# change, and how long must a run be?"  Everything after tonight depends on these numbers.
#
#   bash codec/scripts/run_calibration.sh          # ~6h  (default 15,300 steps/run)
#   bash codec/scripts/run_calibration.sh 4000     # ~1.6h, a shorter probe
#
#   A  baseline                  reference arm
#   B  frozen random bottleneck  a known -1.4 dB effect (paper Table 4) -- the calibration
#   C  baseline, different seed  the NOISE FLOOR: any gap smaller than |A-C| is luck, not signal
#
# Ordered so the headline (A vs B) lands first; if you stop early you still have the effect size.
# Runs FOREGROUND on purpose -- no nohup/disown, so it lives only as long as this terminal and
# leaves nothing orphaned. Leave the terminal open. Each run resumes from its own last checkpoint
# if interrupted, so re-running this script continues rather than restarting.
#
# Training uses mira's OWN scripts/train_codec.py, unmodified -- our only divergence is config.
set -euo pipefail

cd "$(dirname "$0")/../.."

STEPS="${1:-15300}"                 # 15,300 steps @ 0.47 s/step ~= 2h per run
# Warmup is 10% of the run, capped at mira's 1000. Scaling it matters because a fixed 1000 would
# exceed a short probe's total steps and leave decay_steps negative (assertion in mira's scheduler).
WARMUP=$(( STEPS / 10 )); [ "$WARMUP" -gt 1000 ] && WARMUP=1000; [ "$WARMUP" -lt 1 ] && WARMUP=1
DECAY=$((STEPS - WARMUP))
MIRA_TRAIN="${MIRA_TRAIN:?set by ../.envrc via direnv, or export it by hand}"

# pixi is on PATH in an interactive shell (installer appends to ~/.bashrc) but not necessarily here.
if command -v pixi >/dev/null 2>&1; then PIXI=pixi; else PIXI="$HOME/.pixi/bin/pixi"; fi

export RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$HOME/projects/shared/dino_weights}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p checkpoints/calibration codec/results

run_arm () {
  local name="$1" model="$2" seed="$3"
  local out="$PWD/checkpoints/calibration/$name"

  echo ""
  echo "=================================================================="
  echo " ARM $name   model=$model  seed=$seed  steps=$STEPS"
  echo " started $(date '+%H:%M:%S')"
  echo "=================================================================="

  # --config-name selects OUR top-level config (mira's is also called train_codec.yaml, so ours is
  # deliberately named differently to avoid an ambiguous shadow on the search path).
  "$PIXI" run python "$MIRA_TRAIN" \
    --config-dir="$PWD/codec/configs" \
    --config-name=kmira_train_codec \
    model="$model" \
    dataset.train_index="$PWD/data/rocket_science/train" \
    dataset.test_index="$PWD/data/rocket_science/test" \
    run.output_dir="$out" \
    run.seed="$seed" \
    run.steps="$STEPS" \
    run.batch_size=4 \
    run.compile=false \
    run.checkpoint_every=10% \
    validation.val_every=5% \
    validation.val_n_samples=512 \
    optim.scheduler.warmup_steps=$WARMUP \
    optim.scheduler.decay_steps=$DECAY \
    wandb.mode=disabled \
    2>&1 | tee "checkpoints/calibration/${name}.log"

  # Score the final checkpoint with the paper-comparable metrics.
  local ckpt
  ckpt="$(ls -d "$out"/checkpoint-*/ | sort -V | tail -1)checkpoint.pth"
  echo "--- scoring $ckpt ---"
  "$PIXI" run python -m kmira.benchmark.eval_codec \
    --checkpoint "$ckpt" --n-frames 2048 --tag "$name" \
    2>&1 | tail -12
}

START=$(date +%s)

run_arm A_baseline        baseline_image_base      28
run_arm B_frozen_bneck    calib_frozen_bottleneck  28     # same seed as A: paired, so only the
                                                          # architecture differs between the two
run_arm C_baseline_seed2  baseline_image_base      1234

echo ""
echo "=================================================================="
echo " done in $(( ($(date +%s) - START) / 60 )) min"
echo "=================================================================="
echo ""
echo "Results table:"
"$PIXI" run python -c "
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path('codec/results/benchmark.jsonl').read_text().splitlines() if l.strip()]
rows = [r for r in rows if r['tag'].startswith(('A_','B_','C_'))]
print(f\"{'arm':<18}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'P-DINO':>9}{'rFDD':>8}\")
for r in rows:
    print(f\"{r['tag']:<18}{r['psnr']:>8.2f}{r['ssim']:>8.3f}{r['lpips']:>8.3f}{r['p_dino']:>9.4f}{r['r_fdd']:>8.2f}\")
if len(rows) >= 3:
    a, b, c = rows[-3], rows[-2], rows[-1]
    signal, noise = a['psnr'] - b['psnr'], abs(a['psnr'] - c['psnr'])
    print(f\"\nsignal  (A - B, expect ~+1.4 dB): {signal:+.2f} dB\")
    print(f\"noise   (|A - C|, seed spread)  : {noise:.2f} dB\")
    if noise > 1e-6:
        print(f\"ratio   signal/noise            : {signal/noise:+.1f}x\")
    if signal < 0:
        # The frozen bottleneck BEAT the learned one. Not noise -- a reversed effect, most likely
        # because at this training length the decoder adapts faster to a fixed projection than the
        # two learn jointly. It means we are measuring convergence speed, not final quality.
        verdict = 'REVERSED -- effect has the wrong sign; almost certainly still on the slope, not the plateau. Train longer.'
    elif signal > 3 * noise:
        verdict = 'USABLE -- the benchmark resolves a bottleneck-sized change.'
    else:
        verdict = 'TOO NOISY -- signal not clear of the seed spread. Needs longer runs or a bigger decoder.'
    print('\nVERDICT:', verdict)
    print('\nAlso read the validation curves in checkpoints/calibration/*.log: if loss is still')
    print('dropping steeply at the end, this protocol is too short whatever the verdict says.')
"
