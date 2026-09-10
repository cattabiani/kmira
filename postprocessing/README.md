# postprocessing

Figures and readable write-ups of the codec results. Nothing here trains or scores anything — it
only reads what training and scoring already wrote.

**Start with [RESULTS.md](RESULTS.md).**

## Why this folder exists

`codec/results/benchmark.jsonl` is the record: one row per scored checkpoint, appended by
`kmira.benchmark.eval_codec`. It is authoritative and unreadable. Past a few dozen rows across
three arms and five metrics, reading it by eye stopped working — and two claims in this repo's own
write-ups turned out to be wrong in ways a plot would have caught immediately (a metric that never
actually separated the arms, and a weight share attributed to the wrong quantity). Hence: plots,
generated from the record, regenerable in one command.

## Layout

| file | what it is |
|---|---|
| `RESULTS.md` | the write-up, split into **established** and **provisional** |
| `stats.md` | **generated** — every number the write-up quotes. The numeric source of truth |
| `make_all.py` | regenerates all figures and `stats.md` |
| `lib.py` | data loading, the validated palette, matplotlib style |
| `plot_*.py` | one figure each; standalone-runnable |
| `extract_layer_weights.py` | the only script needing local checkpoints (see below) |
| `data/*.json` | small committed inputs that are not in `benchmark.jsonl` |
| `figures/*.png` | committed, so `RESULTS.md` renders without running anything |

## Regenerating

```bash
pixi run python postprocessing/make_all.py            # all figures + stats.md
pixi run python postprocessing/plot_trajectories.py   # or one at a time
```

This works on a fresh clone: it reads only `codec/results/benchmark.jsonl` and `data/*.json`, both
committed.

**The one exception** is `extract_layer_weights.py`, which reads `encoder.layer_weights` out of the
arms' `.pth` checkpoints. `checkpoints/` is gitignored and each file is ~4.4GB, so that script is a
separate step that writes a few hundred bytes of `data/layer_weights.json` — which *is* committed.
Re-run it after training more steps, then `make_all.py`:

```bash
pixi run python postprocessing/extract_layer_weights.py
```

It records the checkpoint path and step for each arm in its output, so a figure drawn from stale
weights is detectable rather than silent.

## Rules this folder keeps

Worth stating, because the point of the folder is that its numbers can be trusted.

- **No number is typed in from memory.** Figures read `benchmark.jsonl` or `data/*.json`. The prose
  in `RESULTS.md` quotes `stats.md`, which is generated. If the two disagree, `stats.md` wins.
- **External numbers carry provenance.** The only hand-transcribed values are published numbers
  from mira's technical report, in `data/mira_layer_ablation.json`, which records the source file,
  the table label, that it was transcribed by hand, and why absolute values are not comparable.
- **Established and provisional stay separate.** An arm that is still running, or an interpretation
  that hasn't been measured, goes in the provisional section and says why.
- **Arm-to-arm numbers only at matched steps.** All three arms were still improving when last
  scored, so an arm measured further along than another is not a comparison. `lib.matched_steps()`
  is what enforces this.
- **Shares, not raw magnitudes, for the layer weights.** The aggregation and the bottleneck
  projection have an exact scaling symmetry that weight decay resolves arbitrarily, so raw weight
  values carry no meaning across arms or across steps.
- **Colour follows the arm, never its rank**, and is fixed in `lib.COLOR` so an arm keeps its hue
  in every figure. The three-colour palette was validated (worst all-pairs CVD ΔE 9.2,
  normal-vision 24.0); aqua sits below 3:1 against the surface, which is why every series carries a
  visible direct label and `stats.md` exists as a table view.
- **Light surface, explicitly painted.** These render on pages that may be dark, so the figures
  never use a transparent background.
