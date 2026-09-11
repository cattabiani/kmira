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
| `RESULTS.md` | the write-up — **WIP**, rebuilding on `baseline_v2` |
| `archive/RESULTS-legacy-baseline.md` | the previous write-up, retired with its baseline |
| `stats.md` | **generated** — every number the write-up quotes. The numeric source of truth |
| `make_all.py` | regenerates all figures and `stats.md` |
| `lib.py` | data loading, the validated palette, matplotlib style |
| `plot_*.py` | one figure each; standalone-runnable |
| `extract_layer_weights.py` | extraction: layer weights out of the arms' checkpoints |
| `extract_setup_facts.py` | extraction: the codec's shape and parameter counts |
| `extract_run_metadata.py` | extraction: every run's val curves, seeds and LR schedule from its log |
| `data/*.json` | small committed inputs that are not in `benchmark.jsonl` |
| `figures/*.png` | committed, so `RESULTS.md` renders without running anything |

## Regenerating

```bash
pixi run python postprocessing/make_all.py            # all figures + stats.md
pixi run python postprocessing/plot_trajectories.py   # or one at a time
```

This works on a fresh clone: it reads only `codec/results/benchmark.jsonl` and `data/*.json`, both
committed.

**The two exceptions** are the extraction scripts, which need things the repo does not ship:

```bash
pixi run python postprocessing/extract_layer_weights.py   # needs local checkpoints
pixi run python postprocessing/extract_run_metadata.py   # needs local training logs
pixi run python postprocessing/extract_setup_facts.py    # needs the DINOv3 weights
```

`extract_layer_weights.py` reads `encoder.layer_weights` out of the arms' `.pth` files —
`checkpoints/` is gitignored and each file is ~4.4GB, so it writes a few hundred bytes of
`data/layer_weights.json` instead, which *is* committed. It records each arm's checkpoint path and
step, so a figure drawn from stale weights is detectable rather than silent. **Re-run it after
training more steps**, then `make_all.py`.

`extract_run_metadata.py` parses every `checkpoints/calibration/*.log` into
`data/run_metadata.json`: each run's validation readings, and the seed and LR schedule each hourly
chunk actually resolved to. `benchmark.jsonl` records scored PSNR rows and nothing else, so the logs
are the only record of both. It also *checks* the paired design — arms compared against each other
must have drawn the same data at the same steps — rather than leaving it asserted in prose. The
launch scripts call it themselves at session end, so this rarely needs running by hand. Note its
readings are validation *loss*, not PSNR, on a different sample set; the two never share an axis.

`extract_setup_facts.py` constructs the codec to count parameters (including the XL decoder, to
size the configuration that OOMs a 12GB card) and derives the compression ratio from the config
arithmetic. It writes `data/setup.json`. Only re-run it if the model configs change.

## Rules this folder keeps

Worth stating, because the point of the folder is that its numbers can be trusted.

- **No number is typed in from memory.** Figures read `benchmark.jsonl` or `data/*.json`. The prose
  in `RESULTS.md` quotes `stats.md`, which is generated. If the two disagree, `stats.md` wins.
- **External numbers carry provenance.** The only hand-transcribed values are published numbers
  from mira's technical report, in `data/mira_layer_ablation.json` and
  `data/mira_bottleneck_ablation.json`. Each records the source file, the table label, that it was
  transcribed by hand, and why absolute values are not comparable — only gaps between two of
  mira's own rows are ever used.
- **Claims with no data say so.** `RESULTS.md` carries a "Not here yet" table naming each queued
  run and what it would give, so the boundary between measured and pending is visible rather than
  implied by absence.
- **A retired foundation retires its page.** When the baseline was found to have trained on 53.4%
  of the data, the results page was archived whole rather than patched number by number — a page
  whose every absolute value is wrong is not fixable by editing the wrong ones.
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
- **Prose never hardcodes a running arm's frontier.** A number like "at 64k, freedom is 20% of the
  gap" is wrong by the next session. Claims about an in-progress arm are qualitative here and
  quantitative in `stats.md`, whose header carries its generation date.
