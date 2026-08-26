"""Run mira's unmodified ``scripts/train_codec.py`` with the torch.hub GitHub call patched out.

This is a launcher, not a fork: it applies ``kmira.torch_hub_offline`` (see that module for why the
patch exists) and then executes mira's trainer as ``__main__`` with the arguments it was given, so
Hydra sees exactly the command line it would have seen otherwise.

    python codec/scripts/train_codec_offline_hub.py --config-dir=... --config-name=... key=value
"""

from __future__ import annotations

import os
import runpy
import sys

from kmira.torch_hub_offline import use_cached_hub_repos

MIRA_TRAIN = os.environ["MIRA_TRAIN"]


def main() -> None:
    use_cached_hub_repos()
    # Hydra reads sys.argv; make argv[0] the trainer's own path so its usage/error messages and
    # Hydra's own run-dir naming are what they would be if it had been invoked directly.
    sys.argv = [MIRA_TRAIN, *sys.argv[1:]]
    runpy.run_path(MIRA_TRAIN, run_name="__main__")


if __name__ == "__main__":
    main()
