"""Smoke test: baseline.yaml parses into mira's VideoCodecConfig without instantiating DINO.

Only checks config plumbing (Hydra/OmegaConf -> pydantic), not the actual model build, so it
doesn't need DINOv3 weights or a GPU.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = Path(__file__).parent.parent / "codec" / "configs"


def test_baseline_config_instantiates_video_codec_config() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="baseline")
    codec_config = instantiate(cfg.architecture.config)
    assert codec_config.encoder.latent_dim == 32
    assert codec_config.encoder.video.timesteps == 1
    assert codec_config.encoder.bottleneck.temporal_stride == 1
