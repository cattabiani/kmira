"""Run mira's unmodified ``scripts/train_codec.py`` with two patches applied, neither touching mira:

1. the torch.hub GitHub call short-circuited to the local cache (see torch_hub_offline.py)
2. the LR scheduler allowed to pick up NEW warmup/constant/decay/min_lr on resume instead of having
   them silently overwritten by the checkpoint's old (constant-LR) values (see lr_resume_override.py)

    python codec/scripts/train_codec_anneal_hub.py --config-dir=... --config-name=... key=value
"""

from __future__ import annotations

import os
import runpy
import sys

from kmira.lr_resume_override import use_new_schedule_on_resume
from kmira.torch_hub_offline import use_cached_hub_repos

MIRA_TRAIN = os.environ["MIRA_TRAIN"]


def main() -> None:
    use_cached_hub_repos()
    use_new_schedule_on_resume()
    sys.argv = [MIRA_TRAIN, *sys.argv[1:]]
    runpy.run_path(MIRA_TRAIN, run_name="__main__")


if __name__ == "__main__":
    main()
