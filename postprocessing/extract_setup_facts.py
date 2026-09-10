"""Extraction step: the baseline codec's shape and size, read from the configs and the built model.

Writes data/setup.json so the "Foundation" section of RESULTS.md quotes generated numbers rather
than remembered ones -- the parameter counts in particular need the model actually built, and the
compression ratio is arithmetic on config fields that is easy to get wrong by hand (mira's own
192x figure includes a 2x temporal stride this image-only rig does not have).

Needs the DINOv3 weights (RS_DINO_WEIGHTS_DIR) because it constructs the codec to count
parameters. Its output is committed, so no plot_*.py depends on this having been run.

    pixi run python postprocessing/extract_setup_facts.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import DATA, REPO

from kmira.torch_hub_offline import use_cached_hub_repos

CONFIGS = REPO / "codec" / "configs"


def decoder_shape(model_name: str, *, count_params: bool = False) -> dict:
    """Decoder geometry, and optionally its trainable parameter count.

    Counting means building the whole codec, which for the XL config is the thing that OOMs a 12GB
    card under fp32 Adam -- fine on CPU, and worth doing so the size claim in RESULTS.md is
    generated rather than remembered.
    """
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="kmira_train_codec", overrides=[f"model={model_name}", "run.compile=false"])
    dec = cfg.model.architecture.config.decoder
    out = {"vit_width": dec.vit_width, "vit_depth": dec.vit_depth, "vit_num_heads": dec.vit_num_heads}
    if count_params:
        codec = instantiate(cfg.model.architecture)
        out["trainable_params"] = sum(p.numel() for p in codec.parameters() if p.requires_grad)
        del codec
    return out


def main() -> None:
    use_cached_hub_repos()
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(
            config_name="kmira_train_codec",
            overrides=["model=baseline_image_base", "run.compile=false"],
        )
    enc = cfg.model.architecture.config.encoder
    vid = enc.video

    codec = instantiate(cfg.model.architecture)
    total = sum(p.numel() for p in codec.parameters())
    trainable = sum(p.numel() for p in codec.parameters() if p.requires_grad)

    # Compression is spatial only here: DINOv3 patchifies by patch_size, the bottleneck strides by
    # `stride` again, and timesteps=1 / temporal_stride=1 means no temporal reduction at all.
    patch = cfg.model.architecture.config.decoder.patch_size
    stride = enc.bottleneck.stride
    grid_h, grid_w = vid.height // patch // stride, vid.width // patch // stride
    per_frame = vid.height * vid.width * vid.channels
    latent = grid_h * grid_w * enc.latent_dim

    out = {
        "_source": "extract_setup_facts.py, from codec/configs/ and the constructed model",
        "video": {
            "height": vid.height,
            "width": vid.width,
            "channels": vid.channels,
            "timesteps": vid.timesteps,
            "fps": vid.fps,
        },
        "encoder": {
            "rae_model": enc.rae_model,
            "aggregation_layers": list(enc.aggregation_layers),
            "latent_dim": enc.latent_dim,
            "bottleneck_stride": stride,
            "bottleneck_temporal_stride": enc.bottleneck.temporal_stride,
        },
        "decoder_base": decoder_shape("baseline_image_base"),
        "decoder_xl": decoder_shape("baseline_image", count_params=True),
        "params": {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "_note": "frozen is the DINOv3-L backbone; trainable is the bottleneck plus decoder",
        },
        "latent": {
            "grid": [grid_h, grid_w],
            "values_per_frame_pixels": per_frame,
            "values_per_frame_latent": latent,
            "reduction_x": round(per_frame / latent, 1),
            "_note": "spatial only -- timesteps=1 and temporal_stride=1, so no temporal compression",
        },
        "optim": {
            "lr": cfg.optim.optimizer.lr,
            "betas": list(cfg.optim.optimizer.betas),
            "weight_decay": cfg.optim.optimizer.weight_decay,
            "min_lr": cfg.optim.scheduler.min_lr,
        },
    }
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "setup.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["params"], indent=2))
    print(json.dumps(out["latent"], indent=2))
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
