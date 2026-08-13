"""Join the training log's validation curve with the scored eval metrics and locate the elbow.

    python codec/scripts/plateau_report.py checkpoints/calibration/plateau_baseline.log

Writes codec/results/plateau_curve.csv. Run by run_plateau.sh, but standalone on purpose: the run is
resumable and stoppable (constant LR means every checkpoint is a valid model), so it is useful to
look at the curve mid-flight without disturbing anything.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import pairwise
from pathlib import Path

RESULTS = Path("codec/results/benchmark.jsonl")
CURVE_CSV = Path("codec/results/plateau_curve.csv")

# An improvement smaller than this, sustained across WINDOW_HOURS of training, counts as flat.
# 0.002 total loss is roughly the reading-to-reading jitter seen in the calibration arms, so it is a
# noise threshold rather than a chosen tolerance.
#
# The window is expressed in HOURS, not in readings, and converted using the actual spacing found in
# the log. Otherwise the criterion silently changes meaning whenever the validation cadence changes
# -- "5 readings" meant 50 minutes at one point in this project's history and 5 hours at another.
#
# The window is TRAILING, so the reported elbow lags the true one by up to WINDOW_HOURS: it cannot
# report flatness until the whole window sits inside the flat region. A leading window would instead
# call an elbow early on any temporary flat spot, which is the worse error when the number is used
# to choose a training length.
FLAT_THRESHOLD = 0.002
WINDOW_HOURS = 2
STEPS_PER_HOUR = 8000  # 0.45 s/step, matching run_plateau.sh's chunk


def read_val_curve(log_path: Path) -> list[tuple[int, float]]:
    """Validation loss by step, de-duplicated.

    A resumed run restarts from its last checkpoint and re-runs the steps after it, so any
    validation in that window is logged twice: once on the trajectory that was discarded, once on
    the live one. Keeping the LAST reading per step keeps the live one. Without this the curve would
    show phantom non-monotonic jumps at exactly the steps where a restart happened.
    """
    text = log_path.read_text()
    pairs = re.findall(r"Validation at step (\d+):.*?loss_total=([\d.]+)", text)
    by_step = {int(step): float(loss) for step, loss in pairs}  # later wins
    return sorted(by_step.items())


def read_scored() -> dict[int, dict]:
    if not RESULTS.exists():
        return {}
    scored = {}
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["tag"].startswith("plateau-"):
            scored[int(row["tag"].removeprefix("plateau-"))] = row
    return scored


def main() -> None:
    log_path = Path(sys.argv[1])
    val = read_val_curve(log_path)
    if not val:
        sys.exit(f"no validation readings found in {log_path}")
    scored = read_scored()

    csv = ["step,val_loss,psnr,ssim,lpips,p_dino,r_fdd"]
    print(f"{'step':>8}{'val_loss':>10}{'delta':>9}{'PSNR':>8}{'dPSNR':>8}{'LPIPS':>8}")
    prev_loss: float | None = None
    prev_psnr: float | None = None
    for step, loss in val:
        row = scored.get(step)
        d_loss = "" if prev_loss is None else f"{loss - prev_loss:+.4f}"
        psnr_s = f"{row['psnr']:.2f}" if row else "-"
        lpips_s = f"{row['lpips']:.3f}" if row else "-"
        d_psnr = f"{row['psnr'] - prev_psnr:+.2f}" if (row and prev_psnr is not None) else ""
        print(f"{step:>8}{loss:>10.4f}{d_loss:>9}{psnr_s:>8}{d_psnr:>8}{lpips_s:>8}")
        if row:
            csv.append(
                f"{step},{loss:.4f},{row['psnr']:.4f},{row['ssim']:.4f},"
                f"{row['lpips']:.4f},{row['p_dino']:.6f},{row['r_fdd']:.4f}"
            )
            prev_psnr = row["psnr"]
        prev_loss = loss

    CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    CURVE_CSV.write_text("\n".join(csv) + "\n")
    print(f"\nWrote {CURVE_CSV}")

    # Convert the window from hours to readings using the spacing actually present in the log, so
    # the criterion keeps meaning the same thing if the validation cadence is ever changed.
    deltas = sorted(b - a for (a, _), (b, _) in pairwise(val))
    spacing = deltas[len(deltas) // 2] if deltas else STEPS_PER_HOUR
    window = max(2, round(WINDOW_HOURS * STEPS_PER_HOUR / spacing))

    if len(val) <= window + 1:
        print(f"too few validation readings to call an elbow (need > {window + 1}, have {len(val)})")
        return

    improving = [
        val[i][0] for i in range(window, len(val)) if val[i - window][1] - val[i][1] > FLAT_THRESHOLD
    ]
    last_improving = improving[-1] if improving else None
    print(
        f"last step still improving >{FLAT_THRESHOLD} per {WINDOW_HOURS}h "
        f"({window} readings): {last_improving or 'none'}"
    )

    final_step = val[-1][0]
    if last_improving is None:
        print("=> FLAT THROUGHOUT. Suspicious rather than good news: check the run actually trained")
        print("   (compare the first and last readings) before believing it converged instantly.")
    elif last_improving >= val[-window][0]:
        print(f"=> STILL ON THE SLOPE at {final_step} steps ({final_step / STEPS_PER_HOUR:.0f}h).")
        print("   Add another slot to extend it: bash codec/scripts/run_plateau.sh <hours>")
    else:
        print(f"=> ELBOW near step {last_improving}. Use that as the protocol length, with cosine")
        print("   decay put back (decay_steps = steps - warmup): an annealed run of that length")
        print("   lands a little better than this constant-LR curve shows.")


if __name__ == "__main__":
    main()
