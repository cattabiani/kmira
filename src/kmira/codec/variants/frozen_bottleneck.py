"""Calibration variant: the bottleneck projection is random and never trained.

This is a deliberate *downgrade*, not an idea — it reproduces the "random frozen projection" row of
mira's paper Table 4, which scores 1.4 dB PSNR below the learned convolution (28.3 vs 29.7).

Its job is to tell us whether our scaled-down benchmark (Base decoder, image-only, a few hours on
one consumer GPU) can still resolve a bottleneck-sized effect. If training this against the stock
baseline recovers a clear gap, the benchmark can be trusted to judge a *new* bottleneck design. If
the gap vanishes into the noise, the benchmark is blind and any result it produced about a real idea
would be meaningless — better to learn that from one deliberate test than after weeks of ambiguity.

The only change is `requires_grad_(False)` on the projection: same architecture, same random
initialisation (mira's `init_weights`), same everything else. So a difference in the metrics is
attributable to the bottleneck being learned or not, and nothing else.
"""

from __future__ import annotations

from mira.codec import VideoCodec
from mira.codec.config import VideoCodecConfig


class VideoCodecFrozenBottleneck(VideoCodec):
    """`VideoCodec` whose `encoder.rae_projection` stays at its random initialisation."""

    def __init__(self, config: VideoCodecConfig, require_dino_weights: bool = True) -> None:
        super().__init__(config, require_dino_weights=require_dino_weights)
        # RAEEncoder has already built and initialised the projection; freezing it here leaves the
        # random weights in place for the whole run. The decoder still trains normally, so this
        # isolates "is the bottleneck learned?" from decoder capacity.
        self.encoder.rae_projection.requires_grad_(False)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
