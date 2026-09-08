# Experiment 2: decompose the Experiment 1 gain

**Status: planned, not run.** Written before the run, so the hypothesis and the falsification
conditions are on record in advance rather than fitted to whatever comes out.

## The confound

Experiment 1 changed two things at once relative to stock mira:

1. **Freedom**: the per-layer weights became learnable instead of fixed.
2. **Reach**: 17 additional DINOv3 layers became available, almost all of them shallower than
   anything the stock formula reads.

The no-op initialisation makes the *starting point* identical to the baseline, which is what makes
the +3.16 dB attributable to the intervention. It does not separate the two components of that
intervention. Since the learned weights ended with ~92% of their mass on layer 0, reach is the
obvious suspect, but "obvious" is not measured.

This matters beyond tidiness. mira keeps a residual on the deepest block to preserve semantics for
the world model. If the gain needs the shallow layers, it is in direct tension with that rationale.
If it does not, there may be a variant that is both better and compatible with it, which is a far
more useful result.

## Hypothesis

The gain is mostly reach, not freedom. Learning the weights *within* mira's own 7-layer set will
recover only a small part of the +3.16 dB.

## Design

One new arm, `learn7`: the Experiment 1 machinery with the 17 non-stock layer weights held at zero
and the 7 stock weights free. Warm-started from `checkpoint-304000`, identical protocol, seed
schedule and chunking to the existing arms, so the three are directly comparable:

| arm | reach | freedom | status |
|---|---|---|---|
| `control` | 24 layers exposed, weights frozen at stock-equivalent | none | measured, flat at 24.5-24.7 |
| `learn7` | stock 7 only | free | **this experiment** |
| `learned_mix` | all 24 | free | measured, 27.905 at 200k |

`learn7` minus `control` is the value of freedom alone. `learned_mix` minus `learn7` is the value
of reach.

Implementation: a trainable-index option on `VideoCodecLearnedLayerMix` rather than a second
variant file. The 17 excluded weights must be genuinely non-trainable, not merely initialised to
zero, or weight decay and gradient noise will let them drift and quietly reintroduce reach.

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
  `control` and the 24.747 plateau, never the annealed 24.885.
- This project has hit false plateaus more than once, including a multi-hour flat stretch followed
  by a jump over 1 dB. Do not call the outcome on one flat reading.
- Keep the per-chunk seed schedule identical so the arms stay paired on data.
- The latent-consistency loss stays pinned to the stock 7 (`KMIRA_PIN_CONSISTENCY_LAYERS`), as in
  every other arm. Nothing about the objective changes here.
