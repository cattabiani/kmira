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

from kmira.codec.variants.learned_layer_mix import (
    DINO_L_DEPTH,
    STOCK_LAYERS,
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
