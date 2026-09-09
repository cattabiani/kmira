# Experiment 2: decompose the Experiment 1 gain

**Status: built, not run.** The hypothesis and falsification conditions below were written before
the scaffolding existed and are unchanged since; only the "Implementation" and "Design" sections
have been updated, to record what was built and to fold in Experiment 1's final numbers.

## The confound

Experiment 1 changed two things at once relative to stock mira:

1. **Freedom**: the per-layer weights became learnable instead of fixed.
2. **Reach**: 17 additional DINOv3 layers became available, almost all of them shallower than
   anything the stock formula reads.

The no-op initialisation makes the *starting point* identical to the baseline, which is what makes
the gain (finally +2.914 dB against the paired control at a matched 200,000 steps) attributable to
the intervention. It does not separate the two components of that
intervention. Since the learned weights ended with 92.4% of their normalized mass on the 17 layers
the stock formula never reads (45.7% on layer 0 alone), reach is the obvious suspect, but "obvious"
is not measured.

This matters beyond tidiness. mira keeps a residual on the deepest block to preserve semantics for
the world model. If the gain needs the shallow layers, it is in direct tension with that rationale.
If it does not, there may be a variant that is both better and compatible with it, which is a far
more useful result.

## Hypothesis

The gain is mostly reach, not freedom. Learning the weights *within* mira's own 7-layer set will
recover only a small part of the +2.914 dB.

## Design

One new arm, `learn7`: the Experiment 1 machinery with the 17 non-stock layer weights held at zero
and the 7 stock weights free. Warm-started from `checkpoint-304000`, identical protocol, seed
schedule and chunking to the existing arms, so the three are directly comparable:

| arm | reach | freedom | status |
|---|---|---|---|
| `control` | 24 layers exposed, weights frozen at stock-equivalent | none | measured, **24.992 at 200k** |
| `learn7` | stock 7 only | free | **this experiment**, built, 0 steps |
| `learned_mix` | all 24 | free | measured, **27.905 at 200k** |

Read `learn7` against `control` at a matched step, never against the 24.747 plateau: the control
finished Experiment 1 at 24.992, i.e. 0.245 above the plateau, so the plateau version would credit
`learn7` with whatever the continued baseline was doing anyway.

`learn7` minus `control` is the value of freedom alone. `learned_mix` minus `learn7` is the value
of reach.

In every arm the only frozen component is the DINOv3 backbone (`mira/src/mira/codec/dino.py`,
frozen by design in the RAE and in stock mira). The bottleneck and decoder train throughout, so
each arm's system adapts to whatever latent its aggregation produces; the arms differ only in which
layer weights are trainable.

## Implementation (built 2026-09-09, simplified 2026-09-10)

`learn7` is not a freezing mechanism. It is the same class as `learned_mix` reading a different set
of DINOv3 blocks:

```yaml
_target_: kmira.codec.variants.learned_layer_mix.VideoCodecLearnedLayerMix
expose_layers: [11, 13, 15, 17, 19, 21, 23]
```

Its weight vector is 7 long, all 7 trainable, initialised to `[1/7]*6 + [8/7]`. The shallow layers
are not withheld from the optimizer — they are simply never read, so there is nothing to withhold.

- `src/kmira/codec/variants/learned_layer_mix.py` — `expose_layers` on
  `VideoCodecLearnedLayerMix`, defaulting to all 24.
- `codec/configs/model/learned_layer_mix_learn7.yaml` — differs from `learned_layer_mix.yaml` in
  exactly that one key (asserted by
  `tests/test_config_loads.py::test_layer_mix_arms_differ_only_in_target_and_exposure`).
- `codec/scripts/run_learned_layer_mix_warmstart.sh` — a third `learn7` arm, its own
  `warmstart_learn7` output dir and `learn7-*` tags.
- `codec/scripts/report_layer_mix.py` — infers the exposure from the weight vector's length and
  prints the freedom/reach decomposition once `learn7` has scored checkpoints.

**Why selection rather than masking.** The first build did this the other way: expose all 24 and
hold 17 weights at zero via a mask, substituted out of the forward pass. That works, but the two
are *exactly* equivalent — a masked term contributes `0.0 * f`, and adding an exact zero to a float
is exact in IEEE-754, so both spellings give a bit-identical latent (asserted in
`tests/test_learned_layer_mix.py::test_restricting_by_exposure_equals_restricting_by_zero_weights`,
and confirmed on the real model: `torch.equal` on the latent, max abs diff 0.0).

Given equivalence, selection wins on everything that matters here. It needs no mask, no frozen
buffers, and no reasoning at all about gradients or AdamW's decoupled weight decay reaching a
weight that is meant to be held — a chain of reasoning subtle enough that the first write-up of it
in this repo was wrong. It is also cheaper: 17 fewer tensor multiply-adds and ~113MB fewer retained
features per forward. And it makes the arm legible, since "this arm reads 7 blocks" is the
experiment, stated in one config key.

What it gives up is the ability to pin a weight at a *nonzero* value — e.g. holding mira's deep
residual at 8/7 while everything else learns. No arm wants that; if one ever does, the mask comes
back in ~15 lines and it is in this repo's history.

**One trap worth knowing about.** `expose_layers` is a constructor argument, *not*
`config.encoder.aggregation_layers`, and that is load-bearing. Both finished arms' saved
`codec_config.yaml` files record `aggregation_layers: [11, 13, ..., 23]` while their checkpoints
hold **24** weights, because the old class overrode that field internally. Deriving the exposure
from the config would build a 7-weight encoder for those configs and fail the strict
`load_state_dict` — breaking re-scoring of all 50 existing checkpoints, and only at scoring time.
Defaulting the argument to all 24 keeps them loading; verified by loading both arms' step-200,000
checkpoints after the change.

## Still to do

Only the compute:

```bash
bash codec/scripts/run_learned_layer_mix_warmstart.sh 1 learn7    # 8k steps, ~1h + ~6min scoring
```

Repeat to ~200,000 to match the other arms (~25 hourly chunks). Do not stop early: the gap between
the existing two arms was still widening at 200,000, so a short `learn7` would understate whichever
component it measures.

## What each outcome means

- **`learn7` recovers most of the gain**: the effect is about weighting, not depth. mira's layer set
  is fine and its uniform weighting is what was leaving value on the table. This would be the
  strongest possible version of the Experiment 1 result, because it is compatible with the semantic
  rationale rather than in tension with it.
- **`learn7` stays near `control`**: the entire gain is shallow-layer access. The world-model
  caveat stops being a caveat and becomes the finding: a reconstruction objective, given the
  choice, abandons the semantic layers, which is presumably why the formula is shaped the way it
  is. Report it that way.
- **Intermediate**: report the split as a number. It is informative either way.

## Watch-outs

- `learn7` pays the same warm-restart penalty as the other arms, so its comparison point is
  `control` at a matched step. Expect it to fall below 24.885 first and climb back; that is the
  restart, not the arm.
- Both existing arms dip through 72k-96k and recover at 104,000. That is the shared seed schedule,
  established by running the control to full length, so expect `learn7` to show it too and do not
  read it as a result.
- This project has hit false plateaus more than once, including a multi-hour flat stretch followed
  by a jump over 1 dB. Do not call the outcome on one flat reading.
- Keep the per-chunk seed schedule identical so the arms stay paired on data.
- The latent-consistency loss stays pinned to the stock 7 (`KMIRA_PIN_CONSISTENCY_LAYERS`), as in
  every other arm. Nothing about the objective changes here.
