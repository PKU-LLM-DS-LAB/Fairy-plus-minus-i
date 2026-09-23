"""Step 3: sample ~100B tokens from the tokenized RedPajama and pack them into 2048-token blocks.

1. Every subset gets a token budget proportional to its share of the full corpus
   (by default measured on the tokenized data itself, or the official dataset-card counts).
2. Inside a subset, documents are sampled uniformly at random without replacement:
   each document gets a random key and documents are taken in key order until the budget
   is covered. This is exactly a prefix of a random permutation of all documents, but only
   the documents with small keys are ever materialised in memory.
3. The sampled documents of a shard are streamed in random order into a buffer, the buffer
   is concatenated and cut into 2048-token blocks; the remainder (< 2048 tokens) stays in the
   buffer and is prepended to the following documents. The final remainder is dropped.
   Short documents are therefore concatenated with following ones, and long documents are
   split with their overflow feeding the next blocks.
4. All blocks are saved as a HF dataset with a single `input_ids` column of length 2048,
   loadable by `datasets.load_from_disk` in train/train.py.
"""

from common import (
    BLOCK_SIZE,
    OFFICIAL_TOKEN_COUNTS,
    SUBSETS,
    list_token_shards,
    parse_subsets,
)

import argparse
import json
import math
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from datasets.arrow_writer import ArrowWriter

# Matches MAX_TOKEN in train/train.py (100_000 × 512 × 2048), so training never runs a second epoch.
DEFAULT_TARGET_TOKENS = 104_857_600_000
FEATURES = Features({"input_ids": Sequence(Value("uint16"), length=BLOCK_SIZE)})


def doc_lengths(idx_path):
    return np.diff(np.load(idx_path, mmap_mode="r"))


def plan_subset(subset, shards, target_tokens, seed):
    """Pick random documents of a subset; returns a list of doc-id arrays, one per shard."""
    subset_id = SUBSETS.index(subset)
    total_tokens = sum(int(doc_lengths(idx).sum()) for _, idx in shards)
    # Packing drops < BLOCK_SIZE leftover tokens per shard, so over-sample by that much.
    needed = target_tokens + len(shards) * BLOCK_SIZE
    if needed > total_tokens:
        raise ValueError(
            f"{subset}: need {needed} tokens but only {total_tokens} are tokenized; "
            "tokenize more files or lower --target_tokens"
        )

    # Keep only documents whose key is below a threshold slightly above the needed
    # fraction, then sort those by key. Raise the threshold if it was not enough.
    threshold = min(1.0, needed / total_tokens * 1.05 + 1e-4)
    while True:
        keys, shard_ids, doc_ids, lengths = [], [], [], []
        for si, (_, idx_path) in enumerate(shards):
            lens = doc_lengths(idx_path)
            rng = np.random.default_rng([seed, subset_id, si])
            shard_keys = rng.random(len(lens))
            sel = np.nonzero(shard_keys < threshold)[0]
            keys.append(shard_keys[sel])
            shard_ids.append(np.full(len(sel), si, dtype=np.int32))
            doc_ids.append(sel)
            lengths.append(lens[sel])
        keys = np.concatenate(keys)
        shard_ids = np.concatenate(shard_ids)
        doc_ids = np.concatenate(doc_ids)
        lengths = np.concatenate(lengths)
        if lengths.sum() >= needed or threshold >= 1.0:
            break
        threshold = min(1.0, threshold * 1.5)

    order = np.argsort(keys, kind="stable")
    cum = np.cumsum(lengths[order])
    num_docs = int(np.searchsorted(cum, needed)) + 1
    chosen = order[:num_docs]
    chosen_shards, chosen_docs = shard_ids[chosen], doc_ids[chosen]

    # Grouping keeps the key order, so docs inside each shard stay in random order.
    per_shard = [chosen_docs[chosen_shards == si] for si in range(len(shards))]
    return per_shard, total_tokens, int(cum[num_docs - 1]), num_docs


def pack_shard(task):
    bin_path, idx_path, doc_ids, out_path, flush_tokens = task
    if out_path.exists():
        return out_path, len(Dataset.from_file(str(out_path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    offsets = np.load(idx_path, mmap_mode="r")

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    writer = ArrowWriter(features=FEATURES, path=str(tmp_path))
    num_blocks = 0

    def cut_blocks(buffer):
        nonlocal num_blocks
        stream = np.concatenate(buffer)
        n = len(stream) // BLOCK_SIZE
        if n:
            flat = pa.array(stream[: n * BLOCK_SIZE], type=pa.uint16())
            blocks = pa.FixedSizeListArray.from_arrays(flat, BLOCK_SIZE)
            writer.write_table(pa.Table.from_arrays([blocks], names=["input_ids"]))
            num_blocks += n
        return stream[n * BLOCK_SIZE:]

    buffer, buffered = [], 0
    for d in doc_ids:
        piece = tokens[offsets[d]:offsets[d + 1]]
        buffer.append(piece)
        buffered += len(piece)
        if buffered >= flush_tokens:
            rest = cut_blocks(buffer)
            buffer, buffered = [rest], len(rest)
    if buffer:
        cut_blocks(buffer)  # the last < BLOCK_SIZE tokens are dropped

    writer.finalize()
    os.replace(tmp_path, out_path)
    return out_path, num_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_dir", type=str, required=True,
                        help="Directory produced by tokenize_redpajama.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory; the final dataset is <output_dir>/dataset")
    parser.add_argument("--subsets", type=str, default=",".join(SUBSETS))
    parser.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--ratio_source", choices=["tokenized", "official"], default="tokenized",
                        help="Subset proportions from the tokenized data or the dataset card")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--flush_tokens", type=int, default=BLOCK_SIZE * 4096,
                        help="Buffer size (tokens) before cutting it into blocks")
    parser.add_argument("--keep_parts", action="store_true",
                        help="Keep intermediate per-shard arrow files")
    args = parser.parse_args()

    subsets = parse_subsets(args.subsets)
    shards = {s: list_token_shards(args.token_dir, s) for s in subsets}
    for s in subsets:
        if not shards[s]:
            raise ValueError(f"No tokenized shards for subset '{s}' in {args.token_dir}")

    if args.ratio_source == "official":
        counts = {s: OFFICIAL_TOKEN_COUNTS[s] for s in subsets}
    else:
        counts = {s: sum(int(doc_lengths(i).sum()) for _, i in shards[s]) for s in subsets}
    total = sum(counts.values())

    out_dir = Path(args.output_dir)
    parts_dir = out_dir / "parts"
    tasks, stats = [], {}
    for s in subsets:
        weight = counts[s] / total
        target = math.ceil(args.target_tokens * weight / BLOCK_SIZE) * BLOCK_SIZE
        per_shard, available, picked, num_docs = plan_subset(s, shards[s], target, args.seed)
        stats[s] = {
            "weight": weight,
            "target_tokens": target,
            "available_tokens": available,
            "picked_docs": num_docs,
            "picked_tokens": picked,
        }
        print(f"{s}: weight={weight:.4f} target={target} picked_docs={num_docs} "
              f"picked_tokens={picked} / available={available}")
        for si, ((bin_path, idx_path), doc_ids) in enumerate(zip(shards[s], per_shard)):
            if len(doc_ids):
                out_path = parts_dir / s / f"{si:05d}.arrow"
                tasks.append((bin_path, idx_path, doc_ids, out_path, args.flush_tokens))

    tasks.sort(key=lambda t: len(t[2]), reverse=True)
    parts = {s: [] for s in subsets}
    with Pool(args.num_workers) as pool:
        for i, (out_path, n) in enumerate(pool.imap_unordered(pack_shard, tasks), 1):
            parts[out_path.parent.name].append((out_path, n))
            print(f"[{i}/{len(tasks)}] {out_path}: {n} blocks", flush=True)

    for s in subsets:
        blocks = sum(n for _, n in parts[s])
        stats[s]["blocks"] = blocks
        stats[s]["tokens"] = blocks * BLOCK_SIZE
        print(f"{s}: {blocks} blocks, {blocks * BLOCK_SIZE} tokens "
              f"(target {stats[s]['target_tokens']})")

    part_files = sorted(p for s in subsets for p, n in parts[s] if n > 0)
    dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in part_files])
    dataset.save_to_disk(str(out_dir / "dataset"), num_proc=min(args.num_workers, 64))

    summary = {
        "block_size": BLOCK_SIZE,
        "target_tokens": args.target_tokens,
        "total_blocks": len(dataset),
        "total_tokens": len(dataset) * BLOCK_SIZE,
        "ratio_source": args.ratio_source,
        "seed": args.seed,
        "subsets": stats,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    del dataset
    if not args.keep_parts:
        shutil.rmtree(parts_dir)
    print(f"Saved dataset to {out_dir / 'dataset'}")


if __name__ == "__main__":
    main()
