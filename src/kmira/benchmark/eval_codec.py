"""Standalone codec reconstruction benchmark: DINOv3 -> bottleneck -> decoder -> frame.

Scores a trained codec checkpoint on held-out clips with the reconstruction-side metrics mira's
paper reports for codec ablations (Tables 3-7): PSNR, SSIM, LPIPS, P-DINO and rFDD. World-model-side
metrics (gFID/gFVD/gFDD, ARR) are out of scope -- they require training a world model on top of the
codec, which we deliberately skip for fast iteration.

Every metric class is imported from mira rather than reimplemented. Usage:

    python -m kmira.benchmark.eval_codec --checkpoint checkpoints/<run>/checkpoint-N/checkpoint.pth

Results print as a table and are appended to a JSON-lines file (``--out``) so runs accumulate and
can be diffed against each other -- that file is the actual benchmark record.

Two deliberate choices that matter for comparing runs:

* **The eval set is a fixed, uniform random sample of the whole test split.** Three biases had to
  be designed out, each measured rather than assumed:

  1. *Streaming the training loader covered 3 of 17 matches.* ``create_loader`` reads clips in shard
     order, and one match yields tens of thousands of clips, so 2048 frames never escapes the first
     match in each shard. Neither shuffling nor a 40x larger shuffle buffer changed it.
  2. *Equal clips per match over-weights short matches.* Matches run 7,120-10,480 frames (1.47x), so
     a fixed per-match share samples a short match ~1.2x more densely than average. Allocation is
     therefore proportional to match length (:func:`allocate_clips_per_match`).
  3. *``max_clips`` takes the first N clips and stops* (``_select_plan`` breaks out of the loop), so
     it would have scored only the opening minutes of every match. Clip ids are drawn at random from
     the whole match instead.

  Determinism is preserved throughout — sorted match order and a seeded RNG — and verified: two
  builds of the eval set are byte-identical, while a different ``--seed`` yields different frames.
* **P-DINO is defined here, not imported.** The paper names the metric but mira's code exposes it
  only as a training loss. We follow that loss's convention (channel-wise L2 normalisation, then
  MSE) but read the last layer of DINOv3-**B** via ``DinoForMetrics``, the backbone mira puts in its
  metrics module. Absolute values are therefore not guaranteed to match the paper's; they are
  consistent across our own runs, which is what a variant comparison needs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mira.codec import VideoCodec
from mira.data.batch import VideoActionBatch
from mira.data.dataset import RocketScienceDataset
from mira.training.metrics.image_metrics import (
    DinoForMetrics,
    DistributedLPIPS,
    DistributedSSIM,
    OnlineGaussian,
    PSNRMetric,
    frechet_distance,
)
from mira.world_model.actions_config import ActionConfig, ActionTensors
from torch import Tensor

from kmira.torch_hub_offline import use_cached_hub_repos

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_EVAL_INDEX = REPO_ROOT / "data" / "rocket_science" / "test"
DEFAULT_OUT = REPO_ROOT / "codec" / "results" / "benchmark.jsonl"

DINO_METRIC_DIM = 768  # DINOv3-B/16 feature width, for the rFDD Gaussian


def to_unit_range(x: Tensor) -> Tensor:
    """Map the codec's [-1, 1] pixels to the [0, 1] every mira metric class expects.

    ``VideoCodec.normalize_video`` applies ``(x - 0.5) / 0.5``, so both ``input_video`` and
    ``output_video`` come out in [-1, 1]. Feeding those straight to the metrics (or clamping them
    to [0, 1], which silently zeroes every negative pixel) makes the scores meaningless.
    """
    return ((x + 1) / 2).clamp(0, 1)


# rFDD fits a 768x768 covariance; well under a few thousand frames it is rank-deficient and the
# number is meaningless. PSNR/SSIM/LPIPS/P-DINO are per-frame averages and settle much sooner.
RFDD_MIN_FRAMES = 2048

# exclude_replays drops some sampled clip ids, so ask for this multiple and trim back.
REPLAY_OVERSAMPLE = 1.35


def allocate_clips_per_match(match_frames: dict[str, int], target_clips: int) -> dict[str, int]:
    """Split ``target_clips`` across matches **in proportion to each match's length**.

    Equal-per-match allocation would be biased: matches here run 7,120-10,480 frames (1.47x), so a
    fixed share per match samples a short match ~1.2x more densely than average and a long one ~0.8x,
    over-weighting whatever is peculiar to short games. Proportional allocation makes the eval set a
    uniform sample of the underlying frame population instead.

    Uses the largest-remainder method so the parts sum to exactly ``target_clips``.
    """
    total = sum(match_frames.values())
    exact = {m: target_clips * n / total for m, n in match_frames.items()}
    alloc = {m: int(x) for m, x in exact.items()}
    shortfall = target_clips - sum(alloc.values())
    # Hand the leftover clips to the largest fractional parts.
    for match_id in sorted(exact, key=lambda m: exact[m] - alloc[m], reverse=True)[:shortfall]:
        alloc[match_id] += 1
    return alloc


def iter_eval_batches(index_path: Path, n_frames: int, batch_size: int, timesteps: int, fps: int, seed: int):
    """Yield ``VideoActionBatch`` forming a uniform random sample of the whole test split.

    Two sources of bias are avoided deliberately:

    * **Across matches** — clips are allocated in proportion to match length (see
      :func:`allocate_clips_per_match`), not equally.
    * **Within a match** — clip ids are drawn at random rather than via ``max_clips``, which
      ``_select_plan`` implements as "take the first N and break". That would have scored only the
      opening couple of minutes of every match.

    Deterministic: matches are visited in sorted order and every draw comes from a seeded RNG.
    ``exclude_replays`` drops a fraction of the sampled ids, so slightly fewer frames than
    ``n_frames`` may come back; the true count is recorded in the results row.

    Actions are dummies -- ``VideoCodec.forward`` only ever reads ``batch.video``.
    """
    ds = RocketScienceDataset.from_local(index_path)
    match_ids = sorted(ds.match_ids())
    # Frames per perspective; we read one perspective, so this is the clip pool per match.
    match_frames = {m: sum(ds.matches[m].chunk_frames) for m in match_ids}
    alloc = allocate_clips_per_match(match_frames, max(1, n_frames // timesteps))
    rng = random.Random(seed)

    def make_batch(frames: list[Tensor]) -> VideoActionBatch:
        return VideoActionBatch(
            video=torch.stack(frames),  # (B, T, C, H, W) uint8, native resolution
            actions=ActionTensors(config=ActionConfig(valid_keys=["_unused"]), batch_size=len(frames)),
        )

    buffer: list[Tensor] = []
    for match_id in match_ids:
        want = alloc[match_id]
        if want == 0:
            continue
        n_available = max(1, match_frames[match_id] // timesteps)
        # Oversample, because exclude_replays silently drops some of the ids we ask for.
        n_draw = min(n_available, math.ceil(want * REPLAY_OVERSAMPLE))
        clip_ids = sorted(rng.sample(range(n_available), n_draw))

        clips = ds.load_match(
            match_id,
            clip_len=timesteps,
            target_fps=fps,
            clip_ids=clip_ids,
            perspective=0,
            seed=seed,
            exclude_replays=True,  # mira's convention for evaluation
        )
        # Trim to the allocation by random subset, never by truncation -- the ids are sorted, so
        # taking the first `want` would reintroduce exactly the chronological bias we just removed.
        if len(clips) > want:
            clips = rng.sample(clips, want)

        for clip in clips:
            assert clip.frames is not None
            buffer.append(clip.frames[0])  # (T, C, H, W), perspective 0
            if len(buffer) == batch_size:
                yield make_batch(buffer)
                buffer = []
    if buffer:
        yield make_batch(buffer)


@dataclass
class ReconstructionScores:
    psnr: float
    ssim: float
    lpips: float
    p_dino: float
    r_fdd: float
    n_frames: int


def build_metrics(device: str | torch.device) -> dict:
    return {
        "psnr": PSNRMetric(device=device),
        "ssim": DistributedSSIM(device=device),
        "lpips": DistributedLPIPS(device=device),
        # Shared by P-DINO (per-frame perceptual distance) and rFDD (Frechet distance between the
        # real and reconstructed feature distributions).
        "dino": DinoForMetrics(model_size="base").to(device).eval(),
        "real_gaussian": OnlineGaussian(dim=DINO_METRIC_DIM).to(device),
        "recon_gaussian": OnlineGaussian(dim=DINO_METRIC_DIM).to(device),
        "_p_dino_sum": 0.0,
        "_p_dino_n": 0,
        "_frames": 0,
    }


@torch.no_grad()
def update_metrics(metrics: dict, real: Tensor, recon: Tensor) -> None:
    """real, recon: [B, T, C, H, W] float in [0, 1]."""
    metrics["psnr"].update(recon, real)
    metrics["ssim"].update(recon, real)
    metrics["lpips"].update(recon, real)

    real_feat = metrics["dino"].dino_forward(real)  # (b, t, c, h, w)
    recon_feat = metrics["dino"].dino_forward(recon)

    # P-DINO, following mira's DinoLoss convention: normalise per channel, then MSE.
    real_n = torch.nn.functional.normalize(real_feat, dim=2, eps=1e-6)
    recon_n = torch.nn.functional.normalize(recon_feat, dim=2, eps=1e-6)
    metrics["_p_dino_sum"] += float(torch.nn.functional.mse_loss(recon_n, real_n))
    metrics["_p_dino_n"] += 1

    # One pooled feature vector per clip for the Frechet distance (mean over time and space).
    metrics["real_gaussian"].update(real_feat.mean(dim=(1, 3, 4)))
    metrics["recon_gaussian"].update(recon_feat.mean(dim=(1, 3, 4)))

    metrics["_frames"] += real.shape[0] * real.shape[1]


def compute_and_reset(metrics: dict) -> ReconstructionScores:
    real_mean, real_cov = metrics["real_gaussian"].compute()
    recon_mean, recon_cov = metrics["recon_gaussian"].compute()
    scores = ReconstructionScores(
        psnr=float(metrics["psnr"].compute_and_reset()),
        ssim=float(metrics["ssim"].compute_and_reset()),
        lpips=float(metrics["lpips"].compute_and_reset()),
        p_dino=metrics["_p_dino_sum"] / max(metrics["_p_dino_n"], 1),
        r_fdd=float(frechet_distance(real_mean, real_cov, recon_mean, recon_cov)),
        n_frames=metrics["_frames"],
    )
    metrics["real_gaussian"].reset()
    metrics["recon_gaussian"].reset()
    metrics["_p_dino_sum"] = 0.0
    metrics["_p_dino_n"] = 0
    metrics["_frames"] = 0
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a checkpoint.pth")
    parser.add_argument("--eval-index", type=Path, default=DEFAULT_EVAL_INDEX)
    parser.add_argument("--n-frames", type=int, default=2048, help="Frames to score (fixed set, not sampled)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=37, help="Fixes which frames make up the eval set")
    parser.add_argument("--tag", default=None, help="Label for this row in the results file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Both the codec's encoder and DinoForMetrics build DINOv3 via torch.hub, which otherwise
    # contacts GitHub on every construction even though the repo is cached. See the module docstring.
    use_cached_hub_repos()

    t0 = time.time()
    model = VideoCodec.load_from_checkpoint(args.checkpoint, device=args.device).eval()
    video_cfg = model.config.encoder.video

    metrics = build_metrics(args.device)

    for batch in iter_eval_batches(
        args.eval_index,
        n_frames=args.n_frames,
        batch_size=args.batch_size,
        timesteps=video_cfg.timesteps,
        fps=video_cfg.fps,
        seed=args.seed,
    ):
        batch = batch.to(args.device)
        with torch.no_grad():
            out = model(batch)
        update_metrics(
            metrics, to_unit_range(out.input_video.float()), to_unit_range(out.output_video.float())
        )

    scores = compute_and_reset(metrics)

    row = {
        "tag": args.tag or args.checkpoint.parent.name,
        "checkpoint": str(args.checkpoint),
        "eval_index": str(args.eval_index),
        # Recorded because they define *which* frames were scored: rows are only comparable to each
        # other when these match.
        "eval_seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_seconds": round(time.time() - t0, 1),
        **asdict(scores),
    }

    print(f"\n{'metric':<10}{'value':>12}   (higher better: PSNR, SSIM; lower better: rest)")
    print("-" * 46)
    for k in ("psnr", "ssim", "lpips", "p_dino", "r_fdd"):
        print(f"{k:<10}{row[k]:>12.4f}")
    print(f"{'frames':<10}{row['n_frames']:>12}")

    if scores.n_frames < RFDD_MIN_FRAMES:
        print(
            f"\nWARNING: rFDD fitted a {DINO_METRIC_DIM}x{DINO_METRIC_DIM} covariance from only "
            f"{scores.n_frames} frames; treat it as unreliable below ~{RFDD_MIN_FRAMES}. "
            "The other four metrics are fine at this count."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
