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

The codec benchmark is built, calibrated and running. The world model is untouched.

**Locked baseline.** mira's stock codec, Base decoder, image-only, trained to `checkpoint-304000`:
PSNR 24.885 after annealing, 24.747 at the constant-LR plateau. Every experiment here compares
against that, not against the paper's published numbers. This setup is image-only and
reduced-scale, so absolute values are not comparable with mira's own (the reference table in
[codec/README.md](codec/README.md) says so explicitly, and the plateau is the number a warm-started
run should be read against).

**Experiment 1, learned per-layer DINO aggregation.** mira aggregates a fixed set of 7 DINOv3
layers with uniform weights, RAEv2's k=7 default, adopted unchanged. Replacing that with 24 learned
per-layer weights, initialised to reproduce the stock formula exactly, reached **27.905 dB against
its paired control's 24.992 at a matched 200k steps — +2.914 dB**, with SSIM, LPIPS and rFDD
improving alongside. Both arms are now complete at 200k, so that gap rests on a control observed
over the variant's full length rather than a quarter of it, and it is the control that makes the
gain attributable to the aggregation rather than to the warm restart. Extending the control also
showed that the variant's apparent takeoff at 104k is a feature of the shared seed schedule — both
arms dip through 72k-96k and recover at 104k — while the gap itself widens steadily throughout.

**What that does not show.** This benchmark scores reconstruction only. The learned weights put
92% of their normalized mass on the 17 layers mira's formula never reads — nearly half of it on
DINOv3's shallowest block alone — and away from the deeper blocks mira keeps in order to preserve
semantics for the world model that predicts in this latent.
Whether the gain survives downstream is untested here, and there are reasons to expect it might
not. Treat it as a reconstruction result with an open question attached, not as an improvement to
MIRA.

## Where to look

- [codec/README.md](codec/README.md): current state of the codec work, how to run training and
  evaluation, the paper's reference numbers and the calibration targets.
- [CHANGELOG.md](CHANGELOG.md): the narrative, what happened in what order and why.
- [AGENTS.md](AGENTS.md): orientation for a coding agent picking this up cold, plus the gotchas
  about mira and this environment that cost real time to find once already.
- [src/kmira/codec/variants/learned_layer_mix.py](src/kmira/codec/variants/learned_layer_mix.py):
  Experiment 1's full rationale, including why the latent-consistency loss had to be pinned, in the
  module docstring.
- `codec/results/benchmark.jsonl`: every scored checkpoint (tags `learned_mix-*`, `control-*`,
  `plateau-*`, `anneal-*`). The current numbers live here, not in prose.
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
