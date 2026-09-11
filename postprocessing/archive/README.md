# Archived: the studies on the retired baseline

Everything here belongs to [`RESULTS-legacy-baseline.md`](RESULTS-legacy-baseline.md) — its page,
its figures, its scripts and its generated numbers, kept together so none of them can be mistaken
for current results.

```bash
pixi run python postprocessing/make_all.py --archive   # rebuilds figures/ and stats.md in here
```

Without `--archive`, `make_all.py` touches nothing in this directory, so a retired figure cannot
quietly reappear on the live page.

## Why it is archived rather than deleted

The baseline these studies were measured against trained, for its first 56,000 steps, on roughly
half the available data because of a data-loader seeding fault. Every **absolute** value here is
therefore wrong by an unknown amount.

What survives is everything **relative**. The arms were paired — same warm-start origin, same
per-chunk seed schedule, read at matched steps — so the arm-to-arm directions hold even though the
level they sit on does not. Those directions are summarised in section 5 of the live
[`../RESULTS.md`](../RESULTS.md).

## Why it is archived rather than obsolete

Most of these scripts come back. `plot_trajectories`, `plot_decomposition`, `plot_metric_panel`,
`plot_metric_signature` and `plot_layer_weights` are the analysis for Experiments 1 and 2, which
are queued to be redone on the new baseline; they will move back up a directory when there is data
for them. `plot_baseline_elbow` and `plot_seed_effects` are general and will be pointed at the new
runs. `plot_calibration` belongs to a three-run study that will not be repeated in that form —
one run per arm cannot estimate a spread — but its shape is the template for the paired
replacement.
