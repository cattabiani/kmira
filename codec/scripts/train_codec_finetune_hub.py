"""Run mira's unmodified ``scripts/train_codec.py`` with three patches applied, none touching mira:

1. the torch.hub GitHub call short-circuited to the local cache (see torch_hub_offline.py)
2. ``run.finetune_from`` allowed to load a checkpoint into a model with declared EXTRA parameters
   the checkpoint doesn't have, instead of mira's strict load_state_dict raising (see
   finetune_allow_new_params.py)
3. optionally, the DINO latent-consistency loss pinned to a fixed layer set instead of inheriting
   the encoder's, so an experiment can change the encoder's layer exposure without also changing
   what it is trained against (see pin_consistency_loss_layers.py)

Both (2) and (3) are configured by env var rather than an argparse flag, since every other override
here already flows through Hydra's own ``key=value`` CLI and neither is one of mira's config keys:

    KMIRA_FINETUNE_NEW_KEYS=encoder.layer_weights \\
    KMIRA_PIN_CONSISTENCY_LAYERS=11,13,15,17,19,21,23 \\
      python codec/scripts/train_codec_finetune_hub.py --config-dir=... --config-name=... key=value

Unset ``KMIRA_PIN_CONSISTENCY_LAYERS`` leaves mira's own encoder-derived behaviour untouched.
"""

from __future__ import annotations

import os
import runpy
import sys

from kmira.finetune_allow_new_params import allow_new_params_on_finetune
from kmira.pin_consistency_loss_layers import pin_consistency_loss_layers
from kmira.torch_hub_offline import use_cached_hub_repos

MIRA_TRAIN = "/home/katta/projects/mira/scripts/train_codec.py"


def main() -> None:
    use_cached_hub_repos()
    raw = os.environ.get("KMIRA_FINETUNE_NEW_KEYS", "")
    expected_new_keys = frozenset(k for k in raw.split(",") if k)
    allow_new_params_on_finetune(expected_new_keys)

    pinned = os.environ.get("KMIRA_PIN_CONSISTENCY_LAYERS", "")
    if pinned:
        pin_consistency_loss_layers(tuple(int(i) for i in pinned.split(",")))

    sys.argv = [MIRA_TRAIN, *sys.argv[1:]]
    runpy.run_path(MIRA_TRAIN, run_name="__main__")


if __name__ == "__main__":
    main()
