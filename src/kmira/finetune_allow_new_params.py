"""Let ``run.finetune_from`` load a checkpoint into a model with EXTRA parameters the checkpoint
doesn't have, instead of mira's strict ``load_state_dict`` raising.

``CheckpointManager.finetune_from`` calls ``self._components["model"].load_state_dict(ckpt["state_dict"])``
with no ``strict=`` argument, i.e. strict matching. Warm-starting
:class:`kmira.codec.variants.learned_layer_mix.VideoCodecLearnedLayerMix` from a stock ``VideoCodec``
checkpoint needs to tolerate exactly one gap: ``encoder.layer_weights`` (24 scalars), which the
variant adds and the baseline's checkpoint naturally never had. Every other key -- including all 368
keys of the frozen DINO backbone -- matches exactly, since the underlying transformer is
architecturally identical regardless of how many intermediate layers a forward pass chooses to read
out of it (see ``learned_layer_mix.py``'s module docstring).

This does NOT just pass ``strict=False`` and hope: it asserts the missing-key set is *exactly* what
the caller declared expected, and that there are no unexpected extra/missing keys, before loading.
An unrelated shape mismatch introduced by a future change should still be a loud error, not silently
tolerated the way bare ``strict=False`` would.
"""

from __future__ import annotations

import torch
from mira.training.checkpoint_manager import CheckpointManager, resolve_checkpoint

_original_finetune_from = CheckpointManager.finetune_from


def allow_new_params_on_finetune(expected_new_keys: frozenset[str]) -> None:
    """Patch ``CheckpointManager.finetune_from`` in this process.

    Always reassigns rather than skipping on a "some patch already applied" check:
    ``expected_new_keys`` is a per-call parameter, and an identity/marker-based idempotency guard
    would silently keep an EARLIER call's keys if this were called twice with different arguments --
    a real bug caught by testing the failure path deliberately before trusting this module.
    """

    def _finetune_from_allow_new_params(self: CheckpointManager, checkpoint) -> dict:
        src = resolve_checkpoint(checkpoint)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        model = self._components["model"]
        target_keys = set(model.state_dict().keys())
        checkpoint_keys = set(ckpt["state_dict"].keys())

        missing = target_keys - checkpoint_keys
        unexpected = checkpoint_keys - target_keys
        if missing != expected_new_keys:
            raise RuntimeError(
                f"finetune_from: missing keys {missing} do not match the declared "
                f"expected_new_keys {expected_new_keys} -- refusing to load_state_dict(strict=False) "
                f"blindly over an unexpected mismatch."
            )
        if unexpected:
            raise RuntimeError(f"finetune_from: unexpected keys in checkpoint not in model: {unexpected}")

        model.load_state_dict(ckpt["state_dict"], strict=False)
        return {k: v for k, v in ckpt.items() if k != "state_dict"}

    CheckpointManager.finetune_from = _finetune_from_allow_new_params
