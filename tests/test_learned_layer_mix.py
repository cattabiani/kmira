"""LearnedLayerMixEncoder's initial weights must reproduce the stock aggregation exactly.

This is the correctness property the whole warm-start plan depends on: if the encoder's output at
construction differs from the stock formula's, "warm-started from the locked baseline" would not
actually mean what it claims to. Runs on CPU with random (untrained) DINO weights -- the backbone's
actual pretrained values don't matter for this check, only that both aggregations are fed the same
intermediate features and combine them the same way.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kmira.codec.variants.learned_layer_mix import (
    DINO_L_DEPTH,
    STOCK_LAYERS,
    LearnedLayerMixEncoder,
    stock_equivalent_weights,
)


def test_stock_equivalent_weights_sum_matches_formula() -> None:
    w = stock_equivalent_weights()
    assert w.shape == (DINO_L_DEPTH,)
    # 6 layers at 1/7 plus 1 layer at 1/7 + 1 == 2 total, exactly the stock formula's weight mass
    # (mean's 7 terms of 1/7 each = 1, plus the separate "+ features[-1]" term = 1 more).
    assert torch.isclose(w.sum(), torch.tensor(2.0))
    for layer in range(DINO_L_DEPTH):
        if layer not in STOCK_LAYERS:
            assert w[layer] == 0.0
    assert torch.isclose(w[STOCK_LAYERS[-1]], torch.tensor(1 / len(STOCK_LAYERS) + 1.0))
    for layer in STOCK_LAYERS[:-1]:
        assert torch.isclose(w[layer], torch.tensor(1 / len(STOCK_LAYERS)))


def test_learned_mix_reproduces_stock_aggregation_at_init() -> None:
    """Feed synthetic per-layer features through both formulas; outputs must match exactly."""
    torch.manual_seed(0)
    dino_dim = 8
    features = [torch.randn(1, 1, dino_dim, 3, 3) for _ in range(DINO_L_DEPTH)]

    # Stock: mean(features at STOCK_LAYERS) + features[-1] (mira/src/mira/codec/rae_encoder.py),
    # where "features[-1]" is the last of the SELECTED layers, i.e. index STOCK_LAYERS[-1].
    stock_selected = [features[i] for i in STOCK_LAYERS]
    stock_agg = torch.stack(stock_selected, dim=0).mean(dim=0) + stock_selected[-1]

    # Learned mix at its stock-equivalent initialization, over ALL 24 layers.
    weights = stock_equivalent_weights()
    stacked_all = torch.stack(features, dim=0)
    learned_agg = torch.einsum("l,l...->...", weights, stacked_all)

    assert torch.allclose(stock_agg, learned_agg, atol=1e-6)


@pytest.mark.slow
def test_layer_index_means_the_same_thing_at_7_and_24_layers() -> None:
    """Slicing STOCK_LAYERS out of a 24-layer read must equal a native 7-layer read.

    This is what makes the whole experiment interpretable, and it is not obvious enough to assume.
    Two separate claims ride on it:

    * the *aggregation* is initialised to reproduce the baseline -- true only if layer ``i`` of a
      24-layer ``get_intermediate_layers(n=(0..23))`` is the same tensor as the corresponding entry
      of a 7-layer ``n=(11,13,...,23)`` read, rather than, say, an offset or a re-normalisation;
    * the *consistency loss* is held at the baseline's objective -- the encoder hands back
      ``features[i] for i in STOCK_LAYERS`` as targets while the pinned ``DinoPerceptualLoss``
      computes its predictions from a native 7-layer read. mira zips those two positionally with a
      non-strict ``zip``, so a misalignment here would silently train against the wrong targets
      instead of raising.

    Uses one randomly-initialised backbone shared by both wrappers (``preloaded_dino_module``), so
    the check costs a single DINOv3-L and needs no pretrained weights on disk.
    """
    from mira.codec.dino import DinoModel

    from kmira.torch_hub_offline import use_cached_hub_repos

    use_cached_hub_repos()
    seven = DinoModel(
        "dinov3_vitl16",
        last_layer_only=False,
        layer_indices=STOCK_LAYERS,
        compile=False,
        require_pretrained=False,
    )
    twenty_four = DinoModel(
        "dinov3_vitl16",
        last_layer_only=False,
        layer_indices=tuple(range(DINO_L_DEPTH)),
        compile=False,
        require_pretrained=False,
        preloaded_dino_module=seven.dino_model,  # same weights, so any difference is indexing
    )

    torch.manual_seed(0)
    video = torch.rand(1, 1, 3, 64, 64)  # DinoModel expects [0, 1]
    with torch.no_grad():
        stock_features = seven.dino_forward(video)
        all_features = twenty_four.dino_forward(video)

    assert len(stock_features) == len(STOCK_LAYERS)
    assert len(all_features) == DINO_L_DEPTH
    for position, layer in enumerate(STOCK_LAYERS):
        assert torch.equal(all_features[layer], stock_features[position]), (
            f"layer {layer} differs between a 24-layer and a 7-layer read (position {position})"
        )


class _BareMixEncoder(LearnedLayerMixEncoder):
    """Just the layer-weight machinery, with no DINOv3 backbone or bottleneck attached.

    ``VideoCodecLearn7LayerMix`` normally acquires these attributes by having ``VideoCodec`` build a
    real encoder and then re-pointing its class, which needs a 300M-parameter backbone and gated
    weights on disk. The freezing property under test involves none of that -- it is entirely about
    ``effective_layer_weights`` and the optimizer -- so this stands the same machinery up directly.
    """

    def __init__(self, trainable_layers: tuple[int, ...]) -> None:
        nn.Module.__init__(self)
        self.layer_weights = nn.Parameter(stock_equivalent_weights())
        mask = torch.zeros(DINO_L_DEPTH, dtype=torch.bool)
        for layer in trainable_layers:
            mask[layer] = True
        self.register_buffer("layer_weight_trainable_mask", mask, persistent=False)
        self.register_buffer("frozen_layer_weights", stock_equivalent_weights(), persistent=False)


def test_learn7_freezes_non_stock_layers() -> None:
    """The 17 excluded weights must be UNCHANGED after a real optimizer step, not just zero at init.

    This is the property the whole of Experiment 2 rests on. ``learn7`` exists to measure freedom
    *without* reach, so if the excluded weights can drift at all the arm quietly regains the reach it
    was built to withhold and measures nothing -- and it would do so invisibly, since the run would
    still train and still produce a number.

    Uses the real optimizer configuration, ``AdamW(weight_decay=0.1)``, and drives a real gradient
    into the trainable entries so that "nothing moved" cannot pass vacuously. See
    ``test_zero_init_alone_does_not_freeze_a_layer_weight`` for the failure this is guarding
    against, which is a *gradient* leak rather than a weight-decay one.
    """
    torch.manual_seed(0)
    encoder = _BareMixEncoder(STOCK_LAYERS)
    opt = torch.optim.AdamW([encoder.layer_weights], lr=1e-2, weight_decay=0.1)

    before = encoder.layer_weights.detach().clone()
    for _ in range(5):
        opt.zero_grad()
        # Any scalar objective that depends on the effective weights; the point is that a real
        # gradient reaches the trainable entries, so "nothing moved" cannot pass vacuously.
        (encoder.effective_layer_weights() * torch.randn(DINO_L_DEPTH)).sum().backward()
        opt.step()
    after = encoder.layer_weights.detach()

    effective = encoder.effective_layer_weights().detach()
    for layer in range(DINO_L_DEPTH):
        if layer in STOCK_LAYERS:
            continue
        # Bit-exact in storage AND in what the aggregation uses. Storage stays exact here only
        # because these entries init at 0.0 and decoupled decay leaves 0.0 at 0.0; the general
        # guarantee is the effective one -- see the next test.
        assert after[layer] == before[layer], f"excluded layer {layer} moved"
        assert effective[layer] == before[layer], f"excluded layer {layer} reached the aggregation"
        assert encoder.layer_weights.grad[layer] == 0.0, f"gradient reached excluded layer {layer}"

    moved = [layer for layer in STOCK_LAYERS if after[layer] != before[layer]]
    assert len(moved) == len(STOCK_LAYERS), f"only {moved} of the stock layers moved"


def test_zero_init_alone_does_not_freeze_a_layer_weight() -> None:
    """The hazard the mechanism exists for: a zero-initialised weight left TRAINABLE drifts.

    Pre-registered in ``experiments/2026-09-08-decompose-layer-mix/NOTES.md`` as the thing that must
    not happen, and worth a test because the intuition runs the other way -- a weight sitting at 0.0
    looks inert. It is not. The aggregation is ``sum(w_i * f_i)``, so the gradient with respect to
    ``w_i`` is ``f_i``, the layer's own features, which is nonzero *regardless of ``w_i`` being 0*.
    Excluded weights left trainable therefore receive real gradients on every step and walk away
    from zero, putting the shallow layers back into the latent and silently handing ``learn7`` the
    reach it exists to withhold.

    This is the measurement behind that claim, so the docstrings can assert it rather than reason
    about it.
    """
    torch.manual_seed(0)
    weights = nn.Parameter(stock_equivalent_weights())
    opt = torch.optim.AdamW([weights], lr=1e-2, weight_decay=0.1)
    excluded = [layer for layer in range(DINO_L_DEPTH) if layer not in STOCK_LAYERS]

    for _ in range(200):
        opt.zero_grad()
        (weights * torch.randn(DINO_L_DEPTH)).sum().backward()
        opt.step()

    assert weights.detach()[excluded].abs().max() > 0.1, (
        "a zero-initialised trainable layer weight was expected to drift; if this now holds still, "
        "the reasoning in VideoCodecLearn7LayerMix's docstring needs revisiting"
    )


def test_a_gradient_mask_would_also_have_held_at_a_zero_init() -> None:
    """Records what substitution is and is NOT bought with, so the choice is not oversold.

    For ``learn7``'s actual configuration a gradient mask would have been correct too: with the
    gradient zeroed, AdamW leaves an entry at exactly 0.0, because its Adam step is zero and its
    decoupled decay term ``p -= lr * wd * p`` also vanishes at ``p == 0``. Substitution is preferred
    for two narrower reasons -- it holds for any frozen value, not only 0.0, and it is structural
    rather than depending on a hook firing inside mira's unmodified training loop -- and this test
    exists so a future reader is not told the stronger, false story that a mask would have leaked
    here.
    """
    torch.manual_seed(0)
    weights = nn.Parameter(stock_equivalent_weights())
    opt = torch.optim.AdamW([weights], lr=1e-2, weight_decay=0.1)
    trainable = torch.zeros(DINO_L_DEPTH, dtype=torch.bool)
    for layer in STOCK_LAYERS:
        trainable[layer] = True

    for _ in range(200):
        opt.zero_grad()
        (weights * torch.randn(DINO_L_DEPTH)).sum().backward()
        weights.grad[~trainable] = 0.0
        opt.step()

    assert torch.equal(weights.detach()[~trainable], torch.zeros(int((~trainable).sum())))


def test_substitution_also_holds_a_NONZERO_frozen_value() -> None:
    """Where substitution is strictly stronger than a gradient mask: a frozen value that is not 0.

    AdamW's decoupled decay moves a parameter from its own value even at zero gradient, so a masked
    weight frozen at 8/7 WOULD shrink away. Substitution does not care: the aggregation reads the
    constant in ``frozen_layer_weights``, so the effective value is exact no matter what the stored
    entry does. ``learn7`` never exercises this -- all its excluded entries are 0.0 -- but the
    option is what makes the mechanism reusable for a future arm that freezes a real weight.
    """
    torch.manual_seed(0)
    frozen_layer = STOCK_LAYERS[-1]
    trainable = tuple(layer for layer in range(DINO_L_DEPTH) if layer != frozen_layer)
    encoder = _BareMixEncoder(trainable)
    opt = torch.optim.AdamW([encoder.layer_weights], lr=1e-2, weight_decay=0.1)

    before = float(stock_equivalent_weights()[frozen_layer])
    assert before != 0.0, "this test is only meaningful on a non-zero frozen init"

    for _ in range(5):
        opt.zero_grad()
        (encoder.effective_layer_weights() * torch.randn(DINO_L_DEPTH)).sum().backward()
        opt.step()

    # The value the aggregation uses is exactly the init, to the bit, after five decayed steps.
    assert float(encoder.effective_layer_weights()[frozen_layer].detach()) == before
    assert float(encoder.layer_weights.grad[frozen_layer]) == 0.0
    # ...even though the raw storage HAS drifted, which is the whole difference from a mask.
    assert float(encoder.layer_weights[frozen_layer].detach()) != before


def test_effective_weights_are_identity_without_a_mask() -> None:
    """No mask means no behaviour change: `learned_mix` and `control` must be untouched by this."""
    encoder = _BareMixEncoder(tuple(range(DINO_L_DEPTH)))
    del encoder.layer_weight_trainable_mask
    assert torch.equal(encoder.effective_layer_weights(), encoder.layer_weights)
