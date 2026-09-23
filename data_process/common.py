import os
import gzip
import io
import json
from pathlib import Path

REPO_ID = "togethercomputer/RedPajama-Data-1T"
URL_BASE = "https://data.together.xyz/redpajama-data-1T/v1.0.0"
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"
BLOCK_SIZE = 2048

# "book" is defunct on the official source (copyright takedown), so it is excluded.
SUBSETS = ["arxiv", "c4", "common_crawl", "github", "stackexchange", "wikipedia"]

# Official LLaMA-tokenizer token counts from the RedPajama-Data-1T dataset card.
OFFICIAL_TOKEN_COUNTS = {
    "common_crawl": 878_000_000_000,
    "c4": 175_000_000_000,
    "github": 59_000_000_000,
    "book": 26_000_000_000,
    "arxiv": 28_000_000_000,
    "wikipedia": 24_000_000_000,
    "stackexchange": 20_000_000_000,
}

RAW_SUFFIXES = (".jsonl", ".jsonl.zst", ".jsonl.gz", ".json.gz")
TOKEN_BIN_SUFFIX = ".bin"
TOKEN_IDX_SUFFIX = ".idx.npy"


def parse_subsets(value):
    subsets = [s.strip() for s in value.split(",") if s.strip()]
    for s in subsets:
        if s not in SUBSETS:
            raise ValueError(f"Unknown subset '{s}', choose from {SUBSETS}")
    return subsets


def list_raw_files(raw_dir, subset):
    root = Path(raw_dir) / subset
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.name.endswith(RAW_SUFFIXES)
    )


def list_token_shards(token_dir, subset):
    """Return sorted (bin_path, idx_path) pairs of completely tokenized shards."""
    root = Path(token_dir) / subset
    if not root.exists():
        return []
    shards = []
    for idx_path in sorted(root.rglob("*" + TOKEN_IDX_SUFFIX)):
        bin_path = Path(str(idx_path)[: -len(TOKEN_IDX_SUFFIX)] + TOKEN_BIN_SUFFIX)
        if bin_path.exists():
            shards.append((bin_path, idx_path))
    return shards


def open_raw_text(path):
    path = str(path)
    if path.endswith(".zst"):
        import zstandard as zstd

        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_texts(path):
    with open_raw_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text")
            if text:
                yield text
