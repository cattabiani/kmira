# kmira

Experiments and possible improvements built on top of [mira](https://github.com/mira-wm/mira)
(MIRA: Multiplayer Interactive World Models with Representation Autoencoders). Currently focused on
the codec; other parts of mira (e.g. the world model) may get their own subfolder here later,
following the same pattern.

`mira` lives untouched as a sibling folder (`../mira`) and is installed editable as a dependency —
it is the ground-truth baseline and is never modified here. When an idea requires changing a
specific piece of the codec (e.g. the bottleneck), we fork just that file into `src/kmira/codec/`
rather than vendoring the whole codec, so diffs against mira stay small and explicit.

## Where this is

The codec benchmark is built and characterised. A reference baseline is training now. The world
model is untouched.

**Current state, honestly.** The baseline this project measured against for its first month was
retired on 2026-09-11: a data-loader seeding fault meant it trained on roughly half the available
dataset for its first 56,000 steps, which makes its training speed, its elbow and its plateau
artifacts rather than properties of the rig. A clean replacement is training now and the studies
are being redone on it. **No absolute dB figure in this repository should be quoted yet.**

**What is established.** The benchmark, the metrics, and the training protocol — in particular that
comparisons here *must* be paired. Per-chunk data slices on this rig are strikingly uneven (the
best 8,000-step slice observed is worth +0.795 dB against −0.177, −0.302 and −0.396 for
the worst three), so an
unpaired comparison measures the data draw as much as the intervention.

**Experiment 1, learned per-layer DINO aggregation.** mira aggregates a fixed set of 7 DINOv3
layers with uniform weights, RAEv2's k=7 default, adopted unchanged. Replacing that with 24 learned
per-layer weights, initialised to reproduce the stock formula exactly, substantially improved
reconstruction against its paired control, and a follow-up attributed most of that to gaining
access to *shallower* DINOv3 blocks rather than to the weights being free. Those arms were paired
against each other, so the **directions hold**; their **magnitudes rest on the retired baseline**
and are being remeasured.

**What that does not show.** This benchmark scores reconstruction only. The learned weights did not
reweight mira's layers — they largely *abandoned* them, moving mass onto blocks the stock formula
never reads and away from the deeper ones mira keeps in order to preserve semantics for the world
model that predicts in this latent. P-DINO, the benchmark's one paired DINO-feature perceptual
distance, never separated the arms. Treat it as a reconstruction result with an open question
attached, not as an improvement to MIRA.

## Where to look

- **[postprocessing/RESULTS.md](postprocessing/RESULTS.md): the results, in figures.** Start here
  if you want to know what came out rather than how to run it. **WIP** while the studies are redone
  on the new baseline; the superseded write-up is kept whole, with its figures and its own
  generated numbers, at
  [postprocessing/archive/](postprocessing/archive/RESULTS-legacy-baseline.md). Every number on
  both pages is generated from `codec/results/benchmark.jsonl` and the training logs rather than
  written by hand, and a test enforces it — see [postprocessing/](postprocessing/).
- [codec/README.md](codec/README.md): current state of the codec work, how to run training and
  evaluation, the paper's reference numbers and the calibration targets.
- [CHANGELOG.md](CHANGELOG.md): the narrative, what happened in what order and why.
- [AGENTS.md](AGENTS.md): orientation for a coding agent picking this up cold, plus the gotchas
  about mira and this environment that cost real time to find once already.
- [src/kmira/codec/variants/learned_layer_mix.py](src/kmira/codec/variants/learned_layer_mix.py):
  Experiment 1's full rationale, including why the latent-consistency loss had to be pinned, in the
  module docstring.
- `codec/results/benchmark.jsonl`: every scored checkpoint (tags `learned_mix-*`, `learn7-*`,
  `control-*`, `plateau-*`, `anneal-*`). The current numbers live here, not in prose — and
  `postprocessing/` turns them into figures rather than restating them.
- `git log`: the reasoning trail. Commit messages carry the detail deliberately.

## Layout

- `codec/` — everything for the codec benchmark: Hydra configs, training/launch scripts, output
  images. See [codec/README.md](codec/README.md) for current state and how to run things.
- `src/kmira/codec/` — thin assembly of a `VideoCodec` from mira's classes plus our own variants.
  - `variants/` — forked/modified pieces (e.g. an alternative bottleneck), one file per idea.
- `src/kmira/benchmark/` — codec benchmark scripts: `eval_codec.py` (reconstruction metrics — PSNR,
  SSIM, LPIPS, P-DINO, rFDD — mirroring the protocol mira's paper uses for codec ablations, Section
  6.3) and `visualize_reconstruction.py` (a before/after PNG for one image).
- `experiments/<date>-<idea-name>/` — one folder per experiment, with a `NOTES.md` describing the
  hypothesis, what changed, and results.
- `postprocessing/` — figures and write-ups generated from the results record. Reads only; never
  trains or scores. `pixi run python postprocessing/make_all.py` regenerates everything.
- `checkpoints/`, `data/` — gitignored; local only; shared across every part of the project (raw
  datasets, weight caches, trained checkpoints), not nested under `codec/`.

## Setup

```bash
pixi run setup      # GPU
pixi run setup-cpu  # CPU only
pixi run test
```

Two gated downloads are needed, from **different places** — a detail that is easy to conflate:

- **DINOv3 weights**, from [Meta's DINOv3 page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
  (not the HuggingFace `facebook/dinov3-*` repos, which ship `safetensors` rather than the `.pth`
  files mira's `torch.hub` loader wants). Both files go in `~/projects/shared/dino_weights/`,
  a shared store other projects point at too rather than each keeping a copy:
  `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (the codec's frozen encoder) and
  `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (used by the benchmark metrics).
- **The dataset**, [`kyutai/rocket-science`](https://huggingface.co/datasets/kyutai/rocket-science)
  on HuggingFace — gated separately from the weights. `hf auth login`, then pull a shard subset via
  `RocketScienceDataset.from_hub(shards=N)` and write **restricted** local `index.json` files
  (`data/rocket_science/{train,test}/`) listing only the downloaded shards — the raw snapshot's own
  `index.json` references the entire 2,821-shard/7.6TB split, so pointing a trainer at it directly
  fails on shards you never downloaded.

`.envrc` (direnv) activates the pixi environment and points `RS_DINO_WEIGHTS_DIR` at
`~/projects/shared/dino_weights` and `MIRA_TRAIN` at `../mira/scripts/train_codec.py`
automatically on `cd`; set them by hand if you don't use direnv. A value you have already
exported wins over the default, so a single run can point at different weights or a different
mira checkout without editing anything.
