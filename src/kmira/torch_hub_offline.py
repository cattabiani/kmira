"""Stop ``torch.hub`` from phoning GitHub when the repo it wants is already cached.

``torch.hub.load("facebookresearch/dinov3", ...)`` -- the call mira's ``codec/dino.py`` and
``metrics/image_metrics.py`` both make -- passes a repo string with no ``:ref``. torch.hub therefore
has to work out whether the default branch is ``main`` or ``master``, and does so in
``_parse_repo_info`` with an unconditional ``urlopen("https://github.com/<owner>/<name>/tree/main/")``
**before** any cache lookup happens. So a fully cached repo still requires GitHub to be reachable on
every single model construction.

Worse, that call's failure handling is incomplete: it catches ``URLError`` (and falls back to the
cache, which is exactly what we want) but not ``http.client.RemoteDisconnected``, which is an
``HTTPException``. A dropped connection therefore escapes uncaught and kills the process outright,
with a perfectly good cache sitting on disk. That is what repeatedly killed hourly training chunks
and scoring runs -- see codec/README.md.

The frozen DINOv3 weights are local (``RS_DINO_WEIGHTS_DIR``); only the *architecture code* comes
from the hub, and it is cached at ``~/.cache/torch/hub/facebookresearch_dinov3_main``. This patch
resolves the ref from that cache directly and only falls through to the original network path when
nothing is cached, so a first-time setup still works exactly as before.

Applied by ``codec/scripts/train_codec_offline_hub.py`` (which then runs mira's unmodified trainer)
and by ``kmira.benchmark.eval_codec``. mira itself is not modified.
"""

from __future__ import annotations

from pathlib import Path

import torch.hub

_original_parse_repo_info = torch.hub._parse_repo_info


def _parse_repo_info_cache_first(github: str) -> tuple[str, str, str]:
    """Resolve owner/name/ref, preferring a cached checkout over asking GitHub."""
    if ":" not in github:
        owner, _, name = github.partition("/")
        hub_dir = Path(torch.hub.get_dir())
        for ref in ("main", "master"):
            if (hub_dir / f"{owner}_{name}_{ref}").is_dir():
                return owner, name, ref
    # Nothing cached (or an explicit ref was given): let torch.hub do its normal thing, including
    # the download it would need anyway.
    return _original_parse_repo_info(github)


def use_cached_hub_repos() -> None:
    """Patch ``torch.hub`` in this process. Idempotent."""
    if torch.hub._parse_repo_info is not _parse_repo_info_cache_first:
        torch.hub._parse_repo_info = _parse_repo_info_cache_first
