#!/usr/bin/env bash
# Print a progress line every N seconds while a training chunk runs, so you can tell at a glance
# whether it is worth waiting for the next checkpoint or safe to kill the run now.
#
#   bash codec/scripts/progress_watcher.sh <log> <from_step> <to_step> <chunk_n> <n_chunks> [interval_s]
#
# Started in the background by run_plateau.sh / run_anneal.sh around each chunk and killed when the
# chunk ends. It only READS the log and appends its own lines; it never touches training.
#
# WHY THIS EXISTS AS A WATCHER rather than a config change: mira's trainer logs a step line every
# ~2,160 steps (about 15 minutes on this rig), which is coarser than the decision it is being used
# for -- "can I kill this now?" -- and the answer to that question is a wall-clock ETA the trainer
# never prints. Lowering mira's own log_every would mean patching mira. Reading its log from outside
# costs nothing and works for both launchers.
#
# The rate is MEASURED, not assumed: seconds-per-step comes from this chunk's own progress so far,
# so the ETA self-corrects if the GPU is being shared or thermally throttled, and no hardcoded
# SEC_PER_STEP can go stale.
set -uo pipefail

LOG="$1"; FROM="$2"; TO="$3"; CHUNK_N="$4"; N_CHUNKS="$5"; INTERVAL="${6:-600}"

START=$(date +%s)

# Exit PROMPTLY when the launcher kills us at the end of a chunk. A bare `sleep "$INTERVAL"` would
# hold the signal until the sleep returned -- up to a full interval (ten minutes in production) of a
# ticker outliving its chunk and reporting on a run that had already stopped. Backgrounding the
# sleep and `wait`ing on it lets the trap run the moment the signal arrives, and the trap takes the
# sleep down with it so nothing is orphaned.
SLEEP_PID=""
trap 'if [ -n "$SLEEP_PID" ]; then kill "$SLEEP_PID" 2>/dev/null; fi; exit 0' TERM INT HUP

# Thousands separators without relying on the locale (LC_NUMERIC is not set on this box).
commas () { printf '%s' "$1" | sed -e :a -e 's/\(.*[0-9]\)\([0-9]\{3\}\)/\1,\2/;ta'; }

hms () {  # seconds -> "1h23m" / "23m" / "45s"
  local s="$1"
  if   [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60))
  elif [ "$s" -ge 60 ];   then printf '%dm' $((s / 60))
  else                         printf '%ds' "$s"
  fi
}

while :; do
  sleep "$INTERVAL" & SLEEP_PID=$!
  wait "$SLEEP_PID" 2>/dev/null
  SLEEP_PID=""

  NOW=$(date +%s)
  ELAPSED=$((NOW - START))

  # Latest step the trainer has reported. Tail a bounded slice: these logs reach hundreds of MB and
  # the progress bars dominate them, so grepping the whole file every 10 minutes would be wasteful.
  STEP="$(tail -c 2000000 "$LOG" 2>/dev/null | grep -oE 'INFO\] - Step [0-9]+:' | tail -1 | grep -oE '[0-9]+')"

  if [ -z "${STEP:-}" ] || [ "$STEP" -le "$FROM" ]; then
    # No step line for this chunk yet -- normal for the first ~15 minutes, and after a resume while
    # the model and the DINOv3 backbone are still loading.
    printf -- '--- [chunk %s/%s] %s into the chunk, no step line yet (target %s)\n' \
      "$CHUNK_N" "$N_CHUNKS" "$(hms "$ELAPSED")" "$(commas "$TO")" | tee -a "$LOG"
    continue
  fi

  DONE_STEPS=$((STEP - FROM))
  LEFT=$((TO - STEP))
  SPAN=$((TO - FROM))
  PCT=$((100 * DONE_STEPS / SPAN))

  if [ "$LEFT" -le 0 ]; then
    printf -- '--- [chunk %s/%s] step %s of %s -- at target, checkpointing and scoring next\n' \
      "$CHUNK_N" "$N_CHUNKS" "$(commas "$STEP")" "$(commas "$TO")" | tee -a "$LOG"
    continue
  fi

  # Measured rate over this chunk so far, in milliseconds per step to keep integer arithmetic honest
  # at ~450 ms/step.
  MS_PER_STEP=$((ELAPSED * 1000 / DONE_STEPS))
  ETA=$((LEFT * MS_PER_STEP / 1000))

  printf -- '--- [chunk %s/%s] step %s of %s (%s%%) | %s since checkpoint-%s | ~%s to checkpoint-%s\n' \
    "$CHUNK_N" "$N_CHUNKS" "$(commas "$STEP")" "$(commas "$TO")" "$PCT" \
    "$(hms "$ELAPSED")" "$FROM" "$(hms "$ETA")" "$TO" | tee -a "$LOG"
done
