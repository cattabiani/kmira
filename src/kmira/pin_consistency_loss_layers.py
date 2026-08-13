"""Hold the DINO latent-consistency loss at a fixed layer set, independent of the encoder's.

mira couples the two: ``CodecLoss.bind_encoder_dino(encoder.rae_dino)`` builds the loss's
``DinoPerceptualLoss`` with ``layer_indices=dino.layers``, so *how many layers the encoder reads*
silently decides *what objective the model is trained against*. That coupling is fine for stock
mira, where the encoder's layer set is the recipe. It is a confound for any experiment that changes
the encoder's layer exposure on purpose -- e.g.
:mod:`kmira.codec.variants.learned_layer_mix`, whose whole point is to read all 24 DINOv3 layers
instead of the hand-picked 7. Left alone, such a run differs from the baseline in two ways at once
(aggregation *and* objective) and no quality delta can be attributed to the intended change.

This patch decouples them: the loss is pinned to ``layers`` regardless of what the encoder exposes.

The encoder must cooperate. ``DinoPerceptualLoss.forward`` pairs its own per-layer predictions with
the encoder-supplied ``dino_features`` by position::

    for p, t in zip(pred_features, target_features):   # NOT strict

so the encoder has to return features for exactly ``layers``, in the same order. A mismatch would
misalign layers (comparing layer 11's prediction against layer 0's target) or silently truncate --
never raise. The pairing is therefore checked numerically rather than trusted: see
``tests/test_learned_layer_mix.py`` and README step 20.
"""

from __future__ import annotations

from mira.codec.dino import DinoModel, DinoPerceptualLoss
from mira.codec.loss import CodecLoss


def pin_consistency_loss_layers(layers: tuple[int, ...]) -> None:
    """Patch ``CodecLoss.bind_encoder_dino`` in this process to always use ``layers``.

    Reassigns unconditionally rather than guarding on a "already patched" marker: ``layers`` is a
    per-call argument, so a marker-based guard would silently keep an earlier call's layer set (the
    same bug already found and fixed in :mod:`kmira.finetune_allow_new_params`).
    """
    if not layers:
        raise ValueError("pin_consistency_loss_layers: empty layer set")

    def _bind_encoder_dino_pinned(self: CodecLoss, dino: DinoModel) -> None:
        if self.weights.loss_dino_latent_consistency <= 0:
            return
        # The pinned layers must be readable from the backbone the encoder actually built; if the
        # encoder exposes a tuple, they must be a subset of it, or the encoder cannot hand back
        # matching targets.
        if isinstance(dino.layers, tuple) and not set(layers) <= set(dino.layers):
            raise ValueError(
                f"pin_consistency_loss_layers: pinned layers {layers} are not a subset of the "
                f"encoder's exposed layers {dino.layers}; the encoder cannot supply targets for them."
            )
        self.dino_latent_consistency_loss = DinoPerceptualLoss(
            dino_model=dino.dino_model_name,
            preloaded_dino_module=dino.dino_model,  # shared backbone, no second copy loaded
            layer_indices=layers,
            last_layer_only=False,
            compile=False,
            normalize=True,
        ).to(next(dino.parameters()).device)

    CodecLoss.bind_encoder_dino = _bind_encoder_dino_pinned
