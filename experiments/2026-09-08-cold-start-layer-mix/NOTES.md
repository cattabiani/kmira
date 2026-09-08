# Experiment 3: does a cold start also go shallow?

**Status: planned, not run. Queued behind Experiment 2**, which is cheaper and resolves a confound
that limits how Experiment 1 can be reported.

## Question

In Experiment 1 the aggregation went shallow while sitting on a decoder that had already been
trained for 304k steps to decode a *deep-feature* latent. Is shallow dominance a property of the
objective, or an artifact of that starting point?

## Design

Cold-start `learned_mix` from scratch under the baseline's own protocol (constant-LR plateau search
then cosine anneal, `run_plateau.sh` / `run_anneal.sh`), compared against the existing locked
baseline, which is itself a cold-start run of the stock codec under exactly that protocol. So the
comparison arm already exists and only one new run is needed. Roughly 35 hours of GPU for the
plateau alone, on the same 0.45 s/step grid.

Read the learned weights, not only PSNR: the outcome of interest is *where the mass ends up*, and
that is visible long before any final number.

## Readings of the outcome

**If it also goes shallow**: strong evidence this is what the reconstruction objective wants,
independent of path. That sharpens the world-model concern rather than easing it, and makes
Experiment 1's finding more general.

**If it does not**, three explanations, and they are not equally interesting:

1. **Multimodal landscape**: two stable aggregation regimes, and initialisation selects between
   them. Interesting if true, but it is the hardest of the three to establish, since it requires
   ruling out the other two.
2. **Insufficient budget**: a cold start has to co-adapt decoder and aggregation from nothing and
   may simply not have arrived yet. This is the boring explanation and the most likely confound,
   so a negative result here is weak unless the weights are visibly static rather than still
   moving.
3. **Decoder-pretraining asymmetry**: a decoder already competent on a deep-feature latent may find
   shallow features an easy way to buy extra fidelity, where a jointly-trained decoder would not.
   This is the most interesting reading, because it would mean Experiment 1's gain is partly a
   property of fine-tuning a converged codec rather than of the aggregation itself.

Distinguishing 2 from 3 needs the weight trajectory, not the endpoint: if the cold-start weights
move toward shallow and stall, that is budget; if they stabilise elsewhere while still training
actively, that points at 3.
