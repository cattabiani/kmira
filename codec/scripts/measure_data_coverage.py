"""How much of the training set does a run actually see? Replays the loader's own shard selection.

WHY THIS EXISTS. mira's train loader reseeds from `cfg.run.seed` on every process start and is not
checkpointed, so a chunked launcher that holds the seed fixed replays one slice of the dataset on
every restart. That is recorded as a gotcha, but "replays the same data" understates it: because a
chunk consumes only part of the stream before the process ends, a fixed seed means the rest of the
dataset is never reached at all. This measures that, rather than leaving it as an adjective.

It is what retired the first baseline. Its first seven chunks all ran seed 28 and it therefore
trained on 53.4% of the data for 56,000 steps, never touching the other 46.6%; a clean run with
per-chunk seeds reached 100% over the same span and was ~1.9 dB ahead by the end of it.

HOW IT WORKS. Mirrors training_loader.py exactly:
    _my_shards -> all_shards[rank::world_size][worker_id::num_workers]   (rank 0, world 1 here)
    rng        -> random.Random(seed + rank*1024 + worker_id)
    shard_order-> rng.shuffle(my_shards), walked in that order
A chunk ends when the process ends, so it only covers the prefix of that order it got through.
Shard assignment per worker is deterministic and seed-independent; only the ORDER varies, which is
why a fixed seed pins the same prefix every time.

    pixi run python codec/scripts/measure_data_coverage.py 28 28 28 28 28 28 28
    pixi run python codec/scripts/measure_data_coverage.py 1029 1030 1031 1032 1033 1034 1035
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

NUM_WORKERS = 6  # codec/configs/kmira_train_codec.yaml: dataloader.num_workers
CHUNK_STEPS = 8000  # one hourly chunk
BATCH = 4  # run.batch_size
INDEX = pathlib.Path(__file__).resolve().parents[2] / "data" / "rocket_science" / "train" / "index.json"


def load_shards() -> dict[str, int]:
    """Samples per shard. With clip_len=1 and n_players=1, every (chunk index, perspective) pair
    is one sample, so the count is the product summed over that shard's entries."""
    entries = json.loads(INDEX.read_text())["entries"]
    per_shard: collections.Counter[str] = collections.Counter()
    for e in entries:
        per_shard[e["shard"]] += len(e["chunk_indices"]) * e["n_players"]
    return dict(per_shard)


def chunk_shards(seed: int, all_shards: list[str], per_shard: dict[str, int]) -> set[str]:
    """Which shards one chunk actually reaches under this seed."""
    seen: set[str] = set()
    for worker in range(NUM_WORKERS):
        order = all_shards[worker::NUM_WORKERS]
        random.Random(seed + worker).shuffle(order)
        quota, got = CHUNK_STEPS * BATCH / NUM_WORKERS, 0.0
        for shard in order:
            if got >= quota:
                break
            seen.add(shard)
            got += per_shard[shard]
    return seen


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]]
    if not seeds:
        print(__doc__)
        raise SystemExit("give one seed per chunk, in order")

    per_shard = load_shards()
    all_shards = sorted(per_shard)
    pool = sum(per_shard.values())

    def cov(shards) -> float:
        return sum(per_shard[x] for x in shards) / pool

    print(f"pool {pool:,} samples over {len(all_shards)} shards; one chunk draws {CHUNK_STEPS*BATCH:,}\n")
    union: set[str] = set()
    for i, seed in enumerate(seeds, 1):
        got = chunk_shards(seed, all_shards, per_shard)
        union |= got
        print(f"  chunk {i:2d}  seed {seed:<6} {cov(got):6.1%} of the data   (running union {cov(union):6.1%})")
    print(f"\nUNION over {len(seeds)} chunks: {cov(union):.1%}  ({len(union)}/{len(all_shards)} shards)")
    print(f"NEVER SEEN:                {1 - cov(union):.1%}")


if __name__ == "__main__":
    main()
