"""Free a finished arm's checkpoints, but only where the numbers are already safely extracted.

A checkpoint is 4.4GiB (1.56 weights + 2.83 resume state) and one 200,000-step arm keeps 25 of
them. That is the right footprint WHILE an arm is live -- it is what lets any step be re-scored
when a metric changes. It is the wrong footprint once the arm is settled.

THE SAFETY RULE, and the reason this is a script rather than an `rm`: a checkpoint is only
deletable once everything derived from it is already committed. Specifically, for each step this
checks that

  * its metrics are in `codec/results/benchmark.jsonl` (the dB), and
  * the run's validation readings and per-chunk seeds are in
    `postprocessing/data/run_metadata.json` (the training curve and the schedule).

Both files are committed and stay forever; the weights are the only reproducible-by-retraining
part, and even that costs ~30h. Anything not yet captured is REFUSED, not deleted.

Always kept, never offered for deletion:
  * the newest checkpoint      -- the resume point; deleting it ends the run
  * anything named by --keep   -- e.g. an annealed checkpoint used as a warm-start origin

Dry run by default. Prints exactly what it would remove and what that frees.

    pixi run python codec/scripts/prune_arm_checkpoints.py ablation_baseline
    pixi run python codec/scripts/prune_arm_checkpoints.py ablation_baseline --delete
    pixi run python codec/scripts/prune_arm_checkpoints.py warmstart_control --delete --keep 200000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "checkpoints" / "calibration"
BENCHMARK = REPO / "codec" / "results" / "benchmark.jsonl"
METADATA = REPO / "postprocessing" / "data" / "run_metadata.json"

# run directory -> the tag prefix its scored rows use in benchmark.jsonl
TAGS = {
    "plateau_baseline": "plateau",
    "ablation_baseline": "abl_baseline",
    "ablation_frozen_bneck": "abl_frozen",
    "baseline_v2_seed2": "baseline_v2_s2",
    "warmstart_control": "control",
    "warmstart_learn7": "learn7",
    "warmstart_learned_mix": "learned_mix",
}

GIB = 1024**3


def dir_size(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def scored_steps(tag: str) -> set[int]:
    """Steps whose metrics are already in the committed record."""
    if not BENCHMARK.exists():
        return set()
    out = set()
    for line in BENCHMARK.read_text().splitlines():
        if not line.strip():
            continue
        arm, _, step = json.loads(line)["tag"].rpartition("-")
        if arm == tag and step.isdigit():
            out.add(int(step))
    return out


def metadata_ok(run: str) -> tuple[bool, str]:
    """Is the run's log already distilled into committed JSON?"""
    if not METADATA.exists():
        return False, "postprocessing/data/run_metadata.json is missing"
    blob = json.loads(METADATA.read_text())["runs"]
    if run not in blob:
        return False, f"{run} has no entry -- run postprocessing/extract_run_metadata.py first"
    rec = blob[run]
    if not rec.get("readings") or not rec.get("chunks"):
        return False, f"{run}'s entry has no readings/chunks -- re-run the extractor"
    return True, f"{len(rec['readings'])} readings, {len(rec['chunks'])} chunks captured"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help=f"run directory under {RUNS.relative_to(REPO)}")
    ap.add_argument("--tag", help="benchmark.jsonl tag prefix (inferred for known runs)")
    ap.add_argument("--keep", type=int, nargs="*", default=[], help="step(s) to keep besides the newest")
    ap.add_argument("--delete", action="store_true", help="actually delete (still asks for confirmation)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    out = RUNS / args.run
    if not out.is_dir():
        raise SystemExit(f"no such run: {out}")
    tag = args.tag or TAGS.get(args.run)
    if not tag:
        raise SystemExit(f"unknown run '{args.run}'; pass --tag with its benchmark.jsonl prefix")

    ok, why = metadata_ok(args.run)
    print(f"metadata : {'OK' if ok else 'MISSING'} -- {why}")
    if not ok:
        raise SystemExit("REFUSING: the training curve is not committed yet. Nothing deleted.")

    ckpts = sorted(
        ((int(p.name.split("-")[1]), p) for p in out.glob("checkpoint-*") if p.is_dir()),
        key=lambda kv: kv[0],
    )
    if not ckpts:
        raise SystemExit(f"no checkpoints in {out}")
    scored = scored_steps(tag)
    newest = ckpts[-1][0]
    keep = {newest, *args.keep}

    print(f"scored   : {len(scored)} steps carry metrics under tag '{tag}-*'\n")
    print(f"{'step':>9}  {'size':>9}  {'scored':>6}  action")
    doomed, freed, blocked = [], 0, []
    for step, path in ckpts:
        size = dir_size(path)
        is_scored = step in scored
        if step in keep:
            action = "KEEP (newest)" if step == newest else "KEEP (--keep)"
        elif not is_scored:
            action = "KEEP -- not scored, would lose its dB"
            blocked.append(step)
        else:
            action = "delete"
            doomed.append((step, path))
            freed += size
        print(f"{step:>9,}  {size / GIB:>8.2f}G  {'yes' if is_scored else 'NO':>6}  {action}")

    print(f"\nwould free {freed / GIB:.1f} GiB across {len(doomed)} checkpoints")
    if blocked:
        print(f"REFUSED for {len(blocked)} unscored step(s): {blocked}")
        print("  score them first (kmira.benchmark.eval_codec) or they are gone for good.")
    if not doomed:
        return
    if not args.delete:
        print("\ndry run -- nothing deleted. Re-run with --delete to proceed.")
        return

    if not args.yes:
        print(f"\nAbout to permanently delete {len(doomed)} checkpoints from {out}.")
        if input("Type the run name to confirm: ").strip() != args.run:
            raise SystemExit("not confirmed; nothing deleted")
    for step, path in doomed:
        shutil.rmtree(path)
        print(f"  removed checkpoint-{step}")
    print(f"freed {freed / GIB:.1f} GiB. Metrics and curves are untouched in "
          f"{BENCHMARK.relative_to(REPO)} and {METADATA.relative_to(REPO)}.")


if __name__ == "__main__":
    sys.exit(main())
