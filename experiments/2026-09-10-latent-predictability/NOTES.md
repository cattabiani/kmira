# Experiment 4: is the shallow latent harder to predict?

**Status: pre-registered, not run.** Written before any of it exists, so the hypothesis and the
falsification conditions are on record rather than fitted to whatever comes out.

**This does not retrain anything.** The three Experiment 1/2 arms stay frozen exactly as they are.
The new component — a small next-step predictor — is *evaluation apparatus*, fitted on latents the
existing checkpoints produce. Nothing is added to the codec's training objective. The version that
*does* change codec training is a different and much larger experiment; see "What comes after".

## Question

Experiment 1's reframing (`experiments/2026-08-13-learned-layer-mix/NOTES.md`) says mira's fixed
layer set is a **regularizer**: mira selects codecs on world-model metrics but trains them on
reconstruction, and the deep-ish layer set is what bridges the two. On that reading, learning the
weights removes a constraint rather than tuning one, and the +2.914 dB is a surrogate artifact that
should cost something downstream.

That is currently an argument, not a measurement. This experiment measures it — the part of it that
can be measured without a world model.

The property a world model actually needs is not reconstruction fidelity in any feature space. It
is **predictability**: given past latents (and actions), how much of the next latent is
determined? A latent carrying lots of unpredictable high-frequency detail is a liability for a
*generative* model, because that content cannot be predicted, only hallucinated — which is what
degraded gFID/gFVD/gFDD look like. So "what fraction of this latent's variance is predictable" is
directly the quantity of interest, and a small auxiliary predictor measures it.

## Hypothesis

`learned_mix`'s layer-0-dominant latent is **disproportionately** harder to predict than
`control`'s — more so than its extra information content alone would explain. `learn7` sits between
the two and closer to `control`, since depth is the axis that matters and layer 11 is only slightly
shallow.

## Design

Three frozen codecs at `checkpoint-200000`: `control`, `learn7`, `learned_mix`. Identical warm
start, seed schedule and step count, so they are already paired on everything but the aggregation.

1. Encode frame pairs `(t, t+h)` from the **test** split, using the same fixed clip set and
   `eval_seed 37` the benchmark uses, so all three arms see identical frames.
2. Fit a predictor `g: z_t -> z_{t+h}`. **Same architecture, same capacity, same training budget for
   every arm** — a capacity difference between arms would confound the whole thing.
3. Report **fraction of variance unexplained**, `FVU = MSE(g(z_t), z_{t+h}) / Var(z_{t+h})`, on
   held-out clips, with latents standardized per channel first.

`FVU` and not raw MSE, because **the arms' latents are not comparable in scale.** The aggregation
has an exact scaling symmetry with the bottleneck projection (scale the weights down, the
projection up, the latent is unchanged), and weight decay resolves it arbitrarily — Experiment 1's
weight vector collapsed from L2 1.1952 to 0.0127 while its PSNR climbed 3 dB. A raw-MSE comparison
across arms would be meaningless.

Two reference predictors to calibrate the scale, fitted the same way:

- **mean**, `g(z) = E[z]`: `FVU = 1` by construction. The "no information" floor.
- **copy**, `g(z) = z`: measures how much of the answer is just temporal smoothness.

Horizon sweep `h in {1, 2, 4, 8}` frames. At 20 fps consecutive frames are nearly identical, so at
`h=1` **copy** will be strong for every arm and differences will be compressed; the longer horizons
are where the arms should separate.

Cheap: encoding costs about what one benchmark row costs (~6 min/arm for 2048 clips), and fitting a
small predictor is minutes. Call it an afternoon for all three arms including writing the script,
against ~25 h of GPU for one training arm.

## The trap this design has to survive, stated up front

**Predictability alone is maximised by a constant latent.** `FVU` must never be read on its own —
a codec that encodes nothing scores perfectly. It is only meaningful *jointly* with reconstruction
quality.

And the arms genuinely do not carry the same information: `learned_mix` encodes more
high-frequency detail — that is *why* its PSNR is 2.9 dB higher — and high-frequency detail is
inherently less predictable. So some increase in `FVU` is expected and is **not** by itself a
defect.

The result is therefore a point per arm in `(PSNR, FVU)` space, and the real question is whether
`learned_mix` moved **along** a frontier or **off** it. Anyone reading a single `FVU` number here
without its PSNR alongside is reading it wrong.

## Pre-registered outcomes

- **`learned_mix`'s FVU rises disproportionately** (much worse predictability for its fidelity
  gain): the regularizer reading is confirmed by measurement. mira's constraint is doing real work,
  and Experiment 1's headline is a surrogate artifact. This is the hypothesis.
- **FVU rises roughly in proportion to the fidelity gain**: no free lunch, and the arms sit on a
  frontier. mira's point is then a defensible tradeoff rather than a mistake, and choosing between
  them depends on how much the world model values fidelity against predictability — which this rig
  cannot answer. Report it as a curve, not a winner.
- **FVU essentially unchanged**: mira left real value on the table, and the reconstruction gain may
  be close to free downstream. This is the surprising outcome and would substantially raise the
  value of the whole line — it would also be the one most in need of a replicate before being
  believed.
- **`learn7` between the two, nearer `control`**: consistent with depth being the operative axis.
  If instead `learn7` sits near `learned_mix` on FVU despite its much smaller PSNR gain, then
  *freedom* rather than reach is what costs predictability, which would be a genuinely unexpected
  and interesting result.

## Watch-outs

- **Scale.** Standardize per channel before fitting, or the comparison measures the arbitrary
  resolution of a scaling symmetry rather than predictability.
- **Capacity.** One predictor architecture and budget for all arms. A larger probe fits more of
  everything and would flatten real differences.
- **Copy dominance.** At `h=1` temporal smoothness supplies most of the answer; do not call the
  outcome on the shortest horizon.
- **This is a lower bound on predictability.** The probe conditions on a *single* past latent and
  no actions. A real world model conditions on a window of latents plus every player's actions, and
  actions in particular carry information this probe simply does not have. So the absolute numbers
  will overstate unpredictability for all arms; only the *comparison* between arms is meaningful.
- **Within-setup only.** This codec is image-only (`timesteps: 1`), so latents are per-frame at
  20 fps, where mira's real latent is 10 Hz with a temporal stride of 2. These FVU numbers are not
  comparable to anything in the paper, the same way the PSNRs are not.
- Hold out clips for evaluation; never score on what the probe was fitted on.
- This measures the latent, not the pipeline. A latent could be predictable and still decode badly.
  That is why PSNR travels with every reading.

## What this does not test, and what comes after

It does not test the world model. It tests one property a world model needs, on a single-frame
conditioning window, with no actions. A latent that scores well here could still generate badly.

The obvious follow-on — and the thing that would make this a *method* rather than a diagnosis — is
to move the predictor from evaluation into training: add a small auxiliary next-step predictor to
the codec's loss so the codec is pushed toward a predictable latent, in tension with
reconstruction (a predictive information bottleneck; predictability alone collapses the latent,
reconstruction prevents it). That keeps credit assignment short and cheap — a small local
predictor, not backprop through a 12 s rollout — and it is the principled version of what mira's
layer prior approximates crudely.

**That is deliberately not pre-registered here.** It only makes sense if this probe shows the arms
differ, and its design depends on *how* they differ. Writing it down now would be pre-registering
a design chosen before its motivating measurement exists.
