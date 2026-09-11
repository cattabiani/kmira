"""Every number in RESULTS.md must exist in the generated stats.md.

The rule the postprocessing folder is built on is that no figure and no quoted value is typed from
memory -- they come from `codec/results/benchmark.jsonl` and the training logs, via `make_all.py`.
That rule was stated in prose and enforced by discipline, which is exactly the kind of thing that
decays. This is the mechanical version: extract the numeric literals from the prose and assert each
one appears in the generated file.

It cannot catch a number that is generated but WRONG -- that is what the extraction scripts' own
provenance is for. What it does catch is the failure that actually happens: a value recalled from
an earlier session, or left behind when a run advanced and the stats were regenerated but the prose
was not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent / "postprocessing"

# Numbers that are structure or provenance rather than measurements, so they have no reason to
# appear in stats.md. Kept deliberately short: every addition is a hole in the check.
ALLOWED = {
    # section numbers and list counts
    "1", "2", "3", "4", "5", "6",
    # mira's own architecture, quoted as configuration rather than measured here
    "1152", "28", "40", "192",
    "7",   # mira's fixed aggregation reads 7 DINOv3 blocks
    "24",  # DINOv3-L/16 has 24 blocks, so the learned variant has 24 weights
    "2026",  # the retirement date
}

NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")


def _numbers(text: str) -> set[str]:
    """Numeric literals in prose, with markdown/code/link noise stripped first."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # fenced code
    text = re.sub(r"`[^`]*`", "", text)  # inline code: config keys, tags, paths
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)  # links, including their targets
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    return {m.group(1).replace(",", "") for m in NUMBER.finditer(text)}


# README.md is the front door, so a stale number there is the most visible kind. It quotes far
# fewer measurements than RESULTS.md but the same rule applies to the ones it does.
PAGES = {
    "postprocessing/RESULTS.md": ROOT / "RESULTS.md",
    "README.md": ROOT.parent / "README.md",
}
README_ALLOWED = {
    "7",  # mira's fixed 7-layer aggregation
    "24",  # DINOv3-L/16's 24 blocks
    "56000",  # the retired baseline's fixed-seed stretch, cited from its notes
    "2026",  # the retirement date
    "6.3",  # a section number in mira's paper
    "2821",  # shards in the upstream dataset split, a property of the download
}


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_number_quoted_appears_in_stats(page: str) -> None:
    results = PAGES[page].read_text()
    stats = (ROOT / "stats.md").read_text().replace(",", "")
    allowed = ALLOWED | (README_ALLOWED if page == "README.md" else set())

    # Token-bounded, not substring: "1.9" must not be satisfied by "21.9561" sitting in a table.
    # That false negative let a hand-rounded value through on the first run of this check.
    def present(n: str) -> bool:
        for form in {n, n.rstrip("0").rstrip(".") if "." in n else n}:
            if re.search(rf"(?<![\w.]){re.escape(form)}(?![\d])", stats):
                return True
        return False

    missing = sorted(n for n in _numbers(results) - allowed if not present(n))
    assert not missing, (
        f"{page} quotes {missing} which do not appear in the generated stats.md.\n"
        "Either regenerate (pixi run python postprocessing/make_all.py), make the value generated, "
        "or -- if it is genuinely structural -- add it to ALLOWED with a reason."
    )
