"""Learned per-layer combination of DINOv3's intermediate features, instead of a fixed subset.

mira's stock aggregation is ``mean(features at 7 hand-picked layers) + features[-1]`` (RAEv2's own
default: layers 11,13,15,17,19,21,23 out of DINOv3-L/16's 24 transformer blocks) -- a single fixed
point in the space of "how much does each layer contribute", chosen once in the paper and never
revisited per-run. This variant asks that as a question instead of an assumption: expose all 24
layers and one learned scalar weight per layer (ELMo-style layer mixing, Peters et al. 2018), and
let training find the combination.

Free (unconstrained) weights, not a softmax-normalised mixture: the stock formula's *effective*
per-layer weights don't sum to 1 (the ``+ features[-1]`` term double-counts layer 23), so a
normalisation constraint couldn't even represent the point we want to start from. AdamW's existing
``weight_decay=0.1`` regularises these scalars the same as every other parameter -- no extra
machinery needed.

Cost: nothing extra from DINO. Layer 23 -- the last of the stock 7 -- *is* DINOv3-L's true final
block (depth=24, 0-indexed), so requesting all 24 intermediate outputs instead of 7 requires exactly
the same forward pass; only 24 more small tensors are kept from a pass that already computes them.

**Initialised to reproduce the locked baseline's aggregation exactly**: weight 0 on the 17 unused
layers, 1/7 on the six non-last stock layers, and 1/7 + 1 = 8/7 on layer 23 (the mean's own share,
plus the stock formula's extra addition). At construction the latent this encoder produces is
numerically identical to ``checkpoint-304000``'s -- verified both in
``tests/test_learned_layer_mix.py`` and against the real checkpoint. So a warm start begins from
exactly the baseline's behaviour, and anything that moves afterward comes from the layer weights
leaving that point.

**The training objective is deliberately held at the baseline's.** ``CodecLoss.bind_encoder_dino``
derives the DINO latent-consistency loss's layer set from the encoder's ``rae_dino.layers`` and
averages its per-layer MSE over exactly those, so naively exposing 24 layers would silently swap the
baseline's 7-layer consistency loss for a 24-layer one -- changing the *objective* at the same time
as the aggregation, and making any PSNR difference unattributable. Two coordinated pieces prevent
that: this encoder returns only the :data:`STOCK_LAYERS` features as ``dino_features`` (the loss's
targets), and :func:`kmira.pin_consistency_loss_layers.pin_consistency_loss_layers` pins the loss's
own layer set to the same 7. The aggregation still reads all 24 layers; only the loss is held fixed.

Both pieces must agree, and nothing makes them: the loss zips its per-layer predictions against
``dino_features`` positionally (``mira/codec/dino.py``'s ``DinoPerceptualLoss.forward``) and that
``zip`` is not strict, so a mismatch would misalign layers *silently* rather than raise. Checked two
ways rather than by inspection: ``tests/test_learned_layer_mix.py`` asserts (with ``torch.equal``)
that layer ``i`` of a 24-layer read is the same tensor as the matching entry of a native 7-layer
read -- the property both claims rest on -- and a one-off comparison against a real stock encoder
plus its own unpinned ``CodecLoss`` gave a consistency-loss difference of exactly 0.0 (README step
20). So a warm start begins from the baseline's behaviour *and* its objective.
"""

from __future__ import annotations

import torch
from einops import rearrange
from mira.codec import VideoCodec
from mira.codec.config import VideoCodecConfig
from mira.codec.rae_encoder import RAEEncoder, RAEEncoderOutputs
from torch import Tensor, nn

# RAEv2's own default 7-layer selection (mira/src/mira/codec/config.py's docstring), and the
# DINOv3-L/16 depth it's drawn from. This variant is scoped to that backbone; asserted in __init__
# rather than silently mis-aggregating if the config ever points at a different DINO variant.
STOCK_LAYERS = (11, 13, 15, 17, 19, 21, 23)
DINO_L_DEPTH = 24


def stock_equivalent_weights() -> Tensor:
    """The per-layer scalars that reproduce ``mean(stock layers) + features[-1]`` exactly."""
    w = torch.zeros(DINO_L_DEPTH)
    for layer in STOCK_LAYERS:
        w[layer] += 1 / len(STOCK_LAYERS)
    w[STOCK_LAYERS[-1]] += 1.0  # the stock formula's separate "+ features[-1]" term
    return w


def all_layers_config(config: VideoCodecConfig) -> VideoCodecConfig:
    """``config`` with the encoder set to read every DINOv3 layer rather than the stock 7.

    Handing this to ``RAEEncoder`` means its ``DinoModel`` is built correctly the *first* time. The
    obvious alternative -- let it build the stock 7-layer backbone and then overwrite ``rae_dino``
    with a 24-layer one -- constructs and discards a whole DINOv3-L (~300M params, ~1.2GB fp32) per
    replacement, and stacking that pattern at both the encoder and codec level built *three*
    backbones to keep one. On a 30GB machine with 512MB of swap, that transient is not a rounding
    error; see codec/README.md.
    """
    encoder = config.encoder.model_copy(update={"aggregation_layers": list(range(DINO_L_DEPTH))})
    return config.model_copy(update={"encoder": encoder})


class LearnedLayerMixEncoder(RAEEncoder):
    """`RAEEncoder` with a learned scalar weight per DINOv3 layer instead of the fixed formula.

    Deliberately defines no ``__init__``: it is never constructed directly. ``VideoCodec.__init__``
    hardcodes ``RAEEncoder``, so :class:`VideoCodecLearnedLayerMix` lets it build a stock encoder
    from an all-layers config -- identical module tree, identical frozen backbone -- and then
    re-points that instance's class here and attaches ``layer_weights``. Only ``forward`` differs,
    so there is nothing an ``__init__`` would need to do that hasn't already been done.
    """

    def effective_layer_weights(self) -> Tensor:
        """The weights the aggregation actually uses.

        Identity on ``layer_weights`` unless a trainable subset was requested (see
        :class:`VideoCodecLearn7LayerMix`), in which case the excluded entries are read from a
        constant instead, so no gradient reaches them and no optimizer state can move what the
        aggregation actually uses.

        Substituting here rather than masking the gradient is what makes the exclusion airtight.
        AdamW's *decoupled* ``weight_decay=0.1`` updates a parameter from its own value even when
        its gradient is exactly zero, so a gradient hook would let an excluded weight drift off its
        init and back into the latent. Excluded entries of ``layer_weights`` do still drift under
        that decay; they are simply never read.
        """
        mask = getattr(self, "layer_weight_trainable_mask", None)
        if mask is None:
            return self.layer_weights
        return torch.where(mask, self.layer_weights, self.frozen_layer_weights)

    def forward(self, video: Tensor) -> RAEEncoderOutputs:
        video = (video + 1) / 2  # VideoCodec normalizes to [-1, 1]; DinoModel expects [0, 1].

        with torch.no_grad():
            features = self.rae_dino.dino_forward(video)  # 24 x (B, T, dino_dim, H, W)

        # Accumulate rather than torch.stack + einsum: stacking allocates a full second copy of
        # every layer (24 x ~4.7MB per layer at batch 4, so ~113MB) purely to reduce it away again.
        # Summing in place needs only the running total and one temporary. The features themselves
        # must stay alive regardless -- dino_forward returns them all at once, and they are handed
        # back as `dino_features` for the consistency loss to use as targets.
        agg = sum(w * f for w, f in zip(self.effective_layer_weights().unbind(), features, strict=True))

        if isinstance(self.rae_projection, nn.Conv3d):
            x = rearrange(agg, "b t c h w -> b c t h w")
            z = self.rae_projection(x)
            z = rearrange(z, "b c t h w -> b t c h w")
        elif isinstance(self.rae_projection, nn.Conv2d):
            b, t = agg.shape[:2]
            x = rearrange(agg, "b t c h w -> (b t) c h w")
            z = self.rae_projection(x)
            z = rearrange(z, "(b t) c h w -> b t c h w", b=b, t=t)
        else:
            raise TypeError(f"Unexpected bottleneck projection: {type(self.rae_projection)}")

        # RAEv2-style noise regulariser, carried over verbatim from RAEEncoder.forward. Inert at our
        # noise_tau=0.0, but omitting it would make this variant quietly diverge from stock the day
        # any experiment turns it on.
        if self.training and self.config.bottleneck.noise_tau > 0:
            sigma = (
                torch.rand(z.shape[0], z.shape[1], 1, 1, 1, device=z.device, dtype=z.dtype)
                * self.config.bottleneck.noise_tau
            )
            z = z + sigma * torch.randn_like(z)

        # Only the stock 7 go back as `dino_features`. That tuple is used for exactly one thing --
        # the latent-consistency loss's targets (mira/codec/loss.py: `real_lc`) -- so returning the
        # stock subset holds that loss at the baseline's 7-layer objective while the aggregation
        # above still reads all 24. Must stay in lockstep with pin_consistency_loss_layers().
        return RAEEncoderOutputs(z=z, dino_features=tuple(features[i] for i in STOCK_LAYERS))


class VideoCodecLearnedLayerMix(VideoCodec):
    """`VideoCodec` using `LearnedLayerMixEncoder` instead of the stock fixed-formula aggregation."""

    def __init__(self, config: VideoCodecConfig, require_dino_weights: bool = True) -> None:
        assert config.encoder.rae_model == "dinov3_vitl16", (
            f"LearnedLayerMix assumes DINOv3-L/16's {DINO_L_DEPTH} layers, got {config.encoder.rae_model!r}"
        )
        # Build through mira's own VideoCodec.__init__ (so its encoder/decoder shape checks still
        # run, and stay in sync if mira ever changes them), but with a config whose encoder already
        # reads all 24 layers -- so the frozen backbone it builds is the one we want and nothing is
        # constructed twice.
        super().__init__(all_layers_config(config), require_dino_weights=require_dino_weights)
        # Upgrade that encoder in place rather than replacing the object: same modules, same frozen
        # backbone, same bottleneck projection and its initialisation. Only the aggregation step
        # differs, which is `forward` plus one new parameter. nn.Module.__setattr__ registers the
        # Parameter normally, so it appears in state_dict as `encoder.layer_weights`.
        self.encoder.__class__ = LearnedLayerMixEncoder
        self.encoder.layer_weights = nn.Parameter(stock_equivalent_weights())


class VideoCodecFixedLayerMix(VideoCodecLearnedLayerMix):
    """The PAIRED CONTROL for :class:`VideoCodecLearnedLayerMix`: identical, but weights frozen.

    Optional, not required for a first look -- with the consistency loss pinned (see the module
    docstring) this control is behaviourally the locked baseline continued, so the variant can be
    run alone and compared against the baseline's own 24.885 dB.

    What it buys when you do run it is the one thing a single arm cannot separate: the warm restart.
    ``finetune_from`` resets the optimizer and re-applies warmup + constant LR to a checkpoint that
    was *annealed to its minimum*, which costs quality before it regains any. Both arms pay that
    cost identically, so the control absorbs it and the arm-to-arm delta isolates the aggregation.

    Frozen weights rather than a stock ``VideoCodec`` so the two arms stay byte-identical in
    architecture, layer exposure and objective, differing only in whether the 24 aggregation weights
    can move -- the same trick ``frozen_bottleneck.py`` used for the calibration arm.
    """

    def __init__(self, config: VideoCodecConfig, require_dino_weights: bool = True) -> None:
        super().__init__(config, require_dino_weights=require_dino_weights)
        self.encoder.layer_weights.requires_grad_(False)


class VideoCodecLearn7LayerMix(VideoCodecLearnedLayerMix):
    """EXPERIMENT 2's arm: learned weights, but only over mira's own 7 layers.

    Experiment 1 changed two things at once relative to stock mira -- the per-layer weights became
    *free*, and 17 additional DINOv3 layers (almost all shallower than anything the stock formula
    reads) became *reachable*. Its result, with 92% of the learned mass landing on those 17, points
    hard at reach, but pointing is not measuring. This arm holds reach at the stock formula's and
    varies only freedom, so ``learn7 - control`` is the value of freedom alone and
    ``learned_mix - learn7`` the value of reach. See
    ``experiments/2026-09-08-decompose-layer-mix/NOTES.md`` for the pre-registered outcomes.

    Which one wins matters beyond bookkeeping: mira keeps a residual on the deepest block
    specifically to preserve semantics for the world model that predicts in this latent. A gain that
    needs the shallow layers is in tension with that rationale; a gain that does not may be
    compatible with it, which would be the far more useful result.

    The 17 excluded weights are **genuinely** non-trainable, not merely initialised to zero. They
    are substituted out of the forward pass by
    :meth:`LearnedLayerMixEncoder.effective_layer_weights`, so the value the aggregation multiplies
    the features by is a constant no gradient reaches and no optimizer state can move. Note what
    that does and does not promise: AdamW's decoupled weight decay still touches every entry of the
    ``layer_weights`` tensor each step, so the excluded entries drift *in storage*. They are inert
    -- nothing reads them -- and for this arm's actual configuration they do not even drift, since
    they start at exactly 0.0 and ``p -= lr * wd * p`` leaves 0.0 at 0.0.

    Substitution rather than a gradient mask is the whole point. Masking the gradient is the obvious
    implementation and it silently fails: decoupled decay moves a parameter from its own value with
    no gradient involved, so masked weights would creep off their init and back into the latent,
    handing this arm exactly the reach it exists to withhold -- invisibly, since the run would still
    train and still produce a number. ``tests/test_learned_layer_mix.py`` pins both the bit-exact
    invariant for this configuration and the general substitution guarantee where decay does bite.

    The mask and the frozen values are NON-PERSISTENT buffers, so ``state_dict`` still contains
    exactly ``encoder.layer_weights`` and this arm warm-starts from the locked baseline through the
    same ``KMIRA_FINETUNE_NEW_KEYS=encoder.layer_weights`` path as the other two.
    """

    def __init__(
        self,
        config: VideoCodecConfig,
        require_dino_weights: bool = True,
        trainable_layers: tuple[int, ...] = STOCK_LAYERS,
    ) -> None:
        super().__init__(config, require_dino_weights=require_dino_weights)
        mask = torch.zeros(DINO_L_DEPTH, dtype=torch.bool)
        for layer in trainable_layers:
            assert 0 <= layer < DINO_L_DEPTH, f"layer {layer} outside DINOv3-L's {DINO_L_DEPTH}"
            mask[layer] = True
        # Non-persistent: derived from the config, never loaded or saved, so the checkpoint surface
        # stays identical to the other two arms.
        self.encoder.register_buffer("layer_weight_trainable_mask", mask, persistent=False)
        self.encoder.register_buffer("frozen_layer_weights", stock_equivalent_weights(), persistent=False)
