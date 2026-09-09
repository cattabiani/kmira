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

## Implementation (built 2026-09-09)

A `trainable_layers` option on `VideoCodecLearn7LayerMix`, a subclass of
`VideoCodecLearnedLayerMix`, rather than a second variant file:

- `src/kmira/codec/variants/learned_layer_mix.py` — `VideoCodecLearn7LayerMix`, plus
  `LearnedLayerMixEncoder.effective_layer_weights()`, which is an identity for the other two arms.
- `codec/configs/model/learned_layer_mix_learn7.yaml` — differs from `learned_layer_mix.yaml` in
  exactly one line, the `_target_` (asserted by
  `tests/test_config_loads.py::test_layer_mix_arms_differ_only_in_architecture_target`).
- `codec/scripts/run_learned_layer_mix_warmstart.sh` — a third `learn7` arm, with its own
  `warmstart_learn7` output dir and `learn7-*` benchmark tags.
- `codec/scripts/report_layer_mix.py` — now reports all three arms and prints the
  freedom/reach decomposition directly once `learn7` has scored checkpoints.

**How the exclusion is enforced, and why not the obvious way.** The excluded weights are
*substituted out of the forward pass* — the aggregation reads a constant for them — rather than
having their gradients masked. Masking the gradient is the obvious implementation and it silently
fails: AdamW's weight decay is *decoupled*, so it moves a parameter from its own value even when
the gradient is exactly zero. Masked weights would creep off their init and back into the latent,
handing this arm precisely the reach it exists to withhold, and invisibly — the run would still
train and still produce a number.

That failure is not hypothetical; the test written to catch it caught it, in a first draft that
relied on the excluded entries starting at zero. Note the guarantee's exact shape: the *effective*
weight is a constant, while the raw parameter entries still drift in storage under decay. They are
inert, and for this arm's real configuration they do not drift at all (init exactly 0.0, and
`p -= lr*wd*p` leaves 0.0 at 0.0). `tests/test_learned_layer_mix.py` pins both the bit-exact
invariant for this configuration and the general substitution guarantee where decay does bite.

The mask and frozen values are non-persistent buffers, so `state_dict` still holds exactly
`encoder.layer_weights` and this arm warm-starts from `checkpoint-304000` through the same
`KMIRA_FINETUNE_NEW_KEYS=encoder.layer_weights` path as the other two.

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
