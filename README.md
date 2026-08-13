# kmira

Experiments and possible improvements built on top of [mira](https://github.com/mira-wm/mira)
(MIRA: Multiplayer Interactive World Models with Representation Autoencoders). Currently focused on
the codec; other parts of mira (e.g. the world model) may get their own subfolder here later,
following the same pattern.

`mira` lives untouched as a sibling folder (`../mira`) and is installed editable as a dependency —
it is the ground-truth baseline and is never modified here. When an idea requires changing a
specific piece of the codec (e.g. the bottleneck), we fork just that file into `src/kmira/codec/`
rather than vendoring the whole codec, so diffs against mira stay small and explicit.

## Layout

- `codec/` — everything for the codec benchmark: Hydra configs, training/launch scripts, output
  images. See [codec/README.md](codec/README.md) for the steps taken so far and how to reproduce
  them.
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
  files mira's `torch.hub` loader wants). Both files go in `data/dino_weights/`:
  `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (the codec's frozen encoder) and
  `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (used by the benchmark metrics).
- **The dataset**, [`kyutai/rocket-science`](https://huggingface.co/datasets/kyutai/rocket-science)
  on HuggingFace — gated separately from the weights. `hf auth login`, then see
  [codec/README.md](codec/README.md) step 5 for pulling a shard subset.

`.envrc` (direnv) activates the pixi environment and sets `RS_DINO_WEIGHTS_DIR` automatically on
`cd`; set it by hand if you don't use direnv.
