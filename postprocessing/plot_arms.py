"""Figure: every arm on one set of axes -- the scored metric and mira's four validation terms.

ONE FIGURE FOR THE WHOLE STUDY. This replaces a per-run progress figure plus a separate comparison
figure. The panels are the same quantities either one showed; the only change is that each panel now
carries every arm, which is the point -- a single arm's curve is not a result, and two figures
invited reading one without the other.

WHAT IS AND IS NOT AN ARM. `baseline_v2` and its cosine anneal are one run continued into its own
output directory, so the anneal is a phase, not an arm. It still gets its own colour, because the
alternative -- one hue distinguished by line style -- made a 32,000-step tail invisible against
264,000 steps of solid line. The frozen-bottleneck ablation is a different model and a real arm.

NO PER-CHUNK INCREMENT PANEL. It existed to answer "has this plateaued", which the trend readout on
the PSNR panel answers directly, and it could not carry several arms without becoming a mess of
overlapping bars.

THE LAST PANEL IS A ZOOM, NOT A NEW MEASUREMENT. A running arm has covered a fraction of the
baseline's steps, so on a full-range axis the comparison is a stub in the corner. The final panel
re-plots PSNR over only the range every arm has reached. Arms share a seed base, so at any step they
trained on the same data and the gap there is attributable to the intervention. Nothing reads an arm
against the baseline's ENDPOINT, which would confound the intervention with the extra steps.
"""

from __future__ import annotations

import json
import re

import lib
import matplotlib.pyplot as plt
from lib import GRID, INK, INK_SOFT, SURFACE

# Colour follows the entity and never its rank, so these hold across figures and reruns.
C_BASE = "#2a78d6"
C_ANNEAL = "#1baf7a"
C_FROZEN = "#eb6834"
C_SEED2 = "#7b5cd6"

BASELINE = "ablation_baseline"
FROZEN = "ablation_frozen_bneck"
SEED2 = "baseline_v2_seed2"

# An arm with no scored rows is simply absent from the figure rather than special-cased, so the
# second seed appears of its own accord the moment it has data.
ARMS = [
    (BASELINE, "baseline", C_BASE),
    (FROZEN, "frozen bottleneck", C_FROZEN),
    (SEED2, "baseline, seed 2", C_SEED2),
]

LOSS_TERMS = [
    ("loss_total", "the sum, dominated by lpips"),
    ("loss_mae", "pixel L1"),
    ("loss_lpips_perceptual", "perceptual"),
    ("loss_dino_latent_consistency", "latent consistency"),
]


def scored(run_dir: str) -> list[tuple[int, float]]:
    """(step, psnr) for one run, keyed off the checkpoint path rather than the tag.

    Tags are per-arm and one deliberately no longer matches its own directory, so a tag prefix can
    pair one run's curve with another's numbers. The path cannot drift from the run that wrote it.
    """
    out = []
    for row in lib.load_rows():
        m = re.search(rf"/{re.escape(run_dir)}/checkpoint-(\d+)/", row.get("checkpoint", ""))
        if m:
            out.append((int(m.group(1)), row["psnr"]))
    return sorted(out)


def readings(run_dir: str, term: str) -> list[tuple[int, float]]:
    """(step, value) for one validation term, from the metadata extracted out of the run logs."""
    blob = json.loads((lib.DATA / "run_metadata.json").read_text())["runs"]
    if run_dir not in blob:
        return []
    return sorted((r["step"], r[term]) for r in blob[run_dir]["readings"] if term in r)


def anneal_base() -> int | None:
    """The step the anneal began at, read from the window file the launcher wrote."""
    path = lib.REPO / "checkpoints" / "calibration" / BASELINE / "anneal_window.env"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("BASE_STEP="):
            return int(line.split("=", 1)[1])
    return None


def trend(points, n=10):
    """Slope per 10,000 steps over the last n points, its residual spread, and their ratio.

    A single chunk's gain carries the data it drew and bounces for reasons unrelated to the run, so
    the trend against that bounce is the readable quantity, not the last reading.
    """
    pts = points[-n:]
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys, strict=True)]
    sd = (sum(r * r for r in resid) / len(resid)) ** 0.5
    return b * 10_000, sd, (abs(b * (xs[-1] - xs[0])) / sd if sd else float("inf"))


def label_at(ax, point, text, color, dx=8, dy=0, ha="left"):
    ax.annotate(
        text,
        xy=point,
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        fontsize=8.5,
        fontweight="bold",
        color=color,
    )


def build():
    cut = anneal_base()
    psnr = {run: scored(run) for run, _, _ in ARMS}
    present = [(run, name, c) for run, name, c in ARMS if psnr[run]]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), facecolor=SURFACE)

    # ---- panel 1: the scored metric, every arm -------------------------------------------------
    ax = axes[0][0]
    for run, name, c in present:
        pts = psnr[run]
        const = [(s, p) for s, p in pts if run != BASELINE or cut is None or s <= cut]
        ax.plot([s for s, _ in const], [p for _, p in const], color=c, linewidth=2, zorder=3)
        label_at(ax, const[len(const) // 2], name, c, dy=-14)
        if run == BASELINE and cut is not None:
            ann = [(s, p) for s, p in pts if s >= cut]
            if len(ann) > 1:
                ax.plot(
                    [s for s, _ in ann],
                    [p for _, p in ann],
                    color=C_ANNEAL,
                    linewidth=2.4,
                    marker="o",
                    markersize=3.5,
                    zorder=4,
                )
                ax.axvline(cut, color=INK_SOFT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
                label_at(
                    ax,
                    ann[-1],
                    f"anneal {ann[-1][1]:.3f} dB",
                    C_ANNEAL,
                    dx=-4,
                    dy=10,
                    ha="right",
                )
    const_base = [(s, p) for s, p in psnr[BASELINE] if cut is None or s <= cut]
    rate, _sd, ratio = trend(const_base)
    ax.set_title("PSNR, scored every 8k  ↑ better", loc="left", fontweight="bold")
    ax.set_ylabel("PSNR (dB), 2048 held-out frames")
    ax.annotate(
        f"baseline before the anneal:\n{rate:+.3f} dB/10k, {ratio:.0f}x the bounce\n"
        f"{'still climbing when stopped' if ratio >= 3 else 'flat within noise'}",
        xy=(0.97, 0.06),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_SOFT,
    )

    # ---- panels 2-5: mira's four validation terms, every arm -----------------------------------
    for idx, (term, sub) in enumerate(LOSS_TERMS):
        a = axes[(idx + 1) // 3][(idx + 1) % 3]
        floored = False
        for run, name, c in present:
            pts = readings(run, term)
            if not pts:
                continue
            a.plot([s for s, _ in pts], [v for _, v in pts], color=c, linewidth=1.5)
            if len({v for _, v in pts[-8:]}) == 1:
                floored = True
        a.set_title(f"{term}  ↓ better", loc="left", fontweight="bold")
        a.set_xlabel(sub)
        if floored:
            a.annotate(
                "AT THE LOG'S 4-dp FLOOR —\nflatness here is rounding",
                xy=(0.97, 0.93),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8,
                fontweight="bold",
                color=INK_SOFT,
            )

    # ---- panel 6: the matched range, where a comparison is legible ------------------------------
    bx = axes[1][2]
    others = [(run, name, c) for run, name, c in present if run != BASELINE]
    hi = min((psnr[run][-1][0] for run, _, _ in others), default=None)
    if hi is not None:
        for run, name, c in [(BASELINE, "baseline", C_BASE), *others]:
            pts = [(s, p) for s, p in psnr[run] if s <= hi]
            bx.plot(
                [s for s, _ in pts],
                [p for _, p in pts],
                color=c,
                linewidth=2,
                marker="o",
                markersize=4,
            )
            if run == BASELINE:
                label_at(bx, pts[-1], name, c, dx=-6, dy=8, ha="right")
            else:
                label_at(bx, pts[0], name, c, dx=8, dy=-4, ha="left")
        bd = dict(psnr[BASELINE])
        gaps = []
        for run, _, _ in others:
            for s, p in psnr[run]:
                if s in bd and s <= hi:
                    bx.plot([s, s], [p, bd[s]], color=INK_SOFT, linewidth=0.8, zorder=1)
                    gaps.append(bd[s] - p)
        if gaps:
            bx.annotate(
                f"gap {min(gaps):.2f}–{max(gaps):.2f} dB",
                xy=(0.5, 0.42),
                xycoords="axes fraction",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
        bx.set_title(f"Matched steps, to {hi:,}  ↑ better", loc="left", fontweight="bold")
        bx.set_xlabel("panel 1, over the range every arm has reached")
        bx.set_ylabel("PSNR (dB)")
    else:
        bx.axis("off")

    for a in axes.flat:
        if a.has_data():
            a.set_facecolor(SURFACE)
            a.grid(True, color=GRID, linewidth=0.7)
            a.set_axisbelow(True)
            lib.spines(a)
            lib.thousands(a)

    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    fig.suptitle(
        "Every arm under the locked recipe — scored PSNR and mira's four validation terms",
        x=0.006,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.008,
        0.012,
        "All arms run constant LR 1e-4 in 8,000-step chunks from the same seed base, so at any step "
        "they have trained on the same data; the anneal is the baseline's own final phase.\n"
        "Validation terms are mira's val loop (512 samples, every 1,000 steps, replays excluded); "
        "PSNR is eval_codec on 2,048 held-out frames. Arms still running are read at matched steps "
        "only.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    return fig, stats_for(psnr, cut)


def stats_for(psnr, cut) -> str:
    base = psnr[BASELINE]
    lines = ["### The arms\n"]

    if cut is not None:
        pre = dict(base).get(cut)
        post = base[-1]
        rate, sd, ratio = trend([(s, p) for s, p in base if s <= cut])
        lines.append(
            f"**Baseline.** Constant LR to {cut:,} reached **{pre:.4f} dB**, still improving at "
            f"**{rate:+.3f} dB per 10,000 steps** against a between-reading spread of "
            f"**{sd:.3f} dB** — a ratio of **{ratio:.0f}x**, so it was stopped on budget "
            f"rather than at a ceiling. A {post[0] - cut:,}-step cosine decay to min_lr then took "
            f"it to **{post[1]:.4f} dB** at step **{post[0]:,}**, a gain of "
            f"**{post[1] - pre:+.4f} dB**. That endpoint is the fixed reference every arm is read "
            f"against.\n"
        )

    bd = dict(base)
    for run, name, _ in ARMS:
        if run == BASELINE or not psnr[run]:
            continue
        pts = psnr[run]
        gaps = [(s, p, bd[s] - p) for s, p in pts if s in bd]
        if not gaps:
            continue
        rows = "\n".join(f"| {s:,} | {p:.4f} | {bd[s]:.4f} | **{g:+.4f}** |" for s, p, g in gaps)
        only = [g for _, _, g in gaps]
        lines.append(
            f"**{name[0].upper()}{name[1:]}, at matched steps.** Same recipe and seed base as the "
            f"baseline, so the same data chunk for chunk:\n\n"
            "| step | this arm | baseline | gap |\n|---|---|---|---|\n"
            f"{rows}\n\n"
            f"The gap runs **{min(only):.4f}** to **{max(only):.4f} dB** over {len(only)} matched "
            f"readings, through step **{pts[-1][0]:,}**.\n"
        )

    missing = [name for run, name, _ in ARMS if not psnr[run]]
    if missing:
        lines.append(f"**Not yet run:** {', '.join(missing)}.\n")
    return "\n".join(lines)


def main() -> None:
    fig, _ = build()
    lib.save(fig, "arms")


if __name__ == "__main__":
    main()
