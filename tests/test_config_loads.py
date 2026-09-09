"""Smoke tests: the configs parse into mira's VideoCodecConfig without instantiating DINO.

Only checks config plumbing (Hydra/OmegaConf -> pydantic) and that each `_target_` names something
importable -- not the actual model build, so no DINOv3 weights or GPU are needed. Cheap, and the
failures it catches (a typo'd `_target_`, a model config that no longer composes) would otherwise
surface only after an hour of GPU time in an unattended overnight run.
"""

import importlib
from pathlib import Path

import pytest
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


MODEL_CONFIGS = sorted(p.stem for p in (CONFIG_DIR / "model").glob("*.yaml"))


@pytest.mark.parametrize("model", MODEL_CONFIGS)
def test_model_config_composes_and_target_is_importable(model: str) -> None:
    """Every `model=<name>` the launch scripts can select must compose and name a real class."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="kmira_train_codec", overrides=[f"model={model}"])

    module_name, _, class_name = cfg.model.architecture._target_.rpartition(".")
    assert hasattr(importlib.import_module(module_name), class_name), (
        f"{model}: architecture._target_ points at a missing {class_name}"
    )

    codec_config = instantiate(cfg.model.architecture.config)
    assert codec_config.encoder.latent_dim == 32
    assert codec_config.encoder.video.timesteps == 1


def test_layer_mix_arms_differ_only_in_target_and_exposure() -> None:
    """The three Experiment 1/2 arms must be identical apart from the arm's own definition.

    They are separate YAML files, so a change to one can silently drift from the others -- and a
    drifted arm produces a number that looks fine and compares nothing. The variant docstrings and
    both NOTES.md claim these differ only in which weights can move and which layers they read;
    this is that claim, checked.
    """
    arms = ["learned_layer_mix", "learned_layer_mix_control", "learned_layer_mix_learn7"]
    configs = {}
    for arm in arms:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name="kmira_train_codec", overrides=[f"model={arm}"])
        configs[arm] = cfg.model

    reference = configs["learned_layer_mix"]
    for arm in arms[1:]:
        assert configs[arm].loss == reference.loss, f"{arm}'s loss config drifted"
        assert configs[arm].architecture.config == reference.architecture.config, (
            f"{arm}'s codec config drifted from learned_layer_mix's"
        )

    # control differs by class (weights frozen); learn7 differs by exposure (reach withheld).
    assert configs["learned_layer_mix_control"].architecture._target_ != reference.architecture._target_
    assert configs["learned_layer_mix_learn7"].architecture._target_ == reference.architecture._target_
    assert list(configs["learned_layer_mix_learn7"].architecture.expose_layers) == [
        11,
        13,
        15,
        17,
        19,
        21,
        23,
    ]
    # The other two must NOT pin an exposure: they default to all 24, and that default is what
    # keeps already-written checkpoint configs loading (see VideoCodecLearnedLayerMix's docstring).
    for arm in ("learned_layer_mix", "learned_layer_mix_control"):
        assert "expose_layers" not in configs[arm].architecture, (
            f"{arm} pins an exposure; it must inherit the all-24 default"
        )
