"""Run a single image through a codec and save an input/output side-by-side PNG.

Manual sanity check, not a metric: lets you *look* at what the codec does to one picture before
building anything more automated. Requires RS_DINO_WEIGHTS_DIR (see README.md).

    python -m kmira.benchmark.visualize_reconstruction --image path/to/pic.jpg
    python -m kmira.benchmark.visualize_reconstruction  # uses a generated placeholder image
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionConfig, ActionTensors
from PIL import Image

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "codec" / "configs"


def load_image(path: Path | None, height: int, width: int) -> torch.Tensor:
    """Return a (3, H, W) uint8 tensor, resized/center-cropped to (height, width)."""
    if path is None:
        # Simple placeholder with visible structure (gradient + checkerboard), so a broken
        # reconstruction is obvious even without a real photo on hand.
        import numpy as np

        y, x = np.mgrid[0:height, 0:width]
        checker = ((x // 16 + y // 16) % 2) * 255
        gradient = (x / width * 255).astype("uint8")
        arr = np.stack([checker, gradient, (checker + gradient) // 2], axis=0).astype("uint8")
        return torch.from_numpy(arr)

    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.BICUBIC)
    arr = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1)  # (3, H, W)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "baseline.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained VideoCodec checkpoint; omit to use randomly-initialized weights",
    )
    parser.add_argument(
        "--image", type=Path, default=None, help="Image to reconstruct; omit for a generated placeholder"
    )
    parser.add_argument("--out", type=Path, default=Path("codec/results/reconstruction.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is not None:
        from mira.codec import VideoCodec

        model = VideoCodec.load_from_checkpoint(args.checkpoint, device=args.device).eval()
        video_cfg = model.config.encoder.video
        height, width = video_cfg.height, video_cfg.width
    else:
        with initialize_config_dir(config_dir=str(args.config.parent), version_base=None):
            cfg = compose(config_name=args.config.stem)

        video_cfg = cfg.architecture.config.encoder.video
        height, width = video_cfg.height, video_cfg.width

        model = instantiate(cfg.architecture).eval().to(args.device)

    frame = load_image(args.image, height, width)  # (3, H, W) uint8
    video = frame[None, None].to(args.device)  # (B=1, T=1, C, H, W)
    actions = ActionTensors(config=ActionConfig(valid_keys=["_unused"]), batch_size=1).to(args.device)
    batch = VideoActionBatch(video=video, actions=actions)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        out = model(batch)

    # The codec works in [-1, 1] (VideoCodec.normalize_video applies (x - 0.5) / 0.5), so map back
    # to [0, 1] for display -- clamping [-1, 1] straight to [0, 1] would black out every dark pixel.
    input_frame = ((out.input_video[0, 0].float() + 1) / 2).clamp(0, 1)  # (C, H, W)
    output_frame = ((out.output_video[0, 0].float() + 1) / 2).clamp(0, 1)

    side_by_side = torch.cat([input_frame, output_frame], dim=2)  # concat along width
    arr = (side_by_side.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(args.out)
    print(f"Saved input | output side-by-side to {args.out}")


if __name__ == "__main__":
    main()
