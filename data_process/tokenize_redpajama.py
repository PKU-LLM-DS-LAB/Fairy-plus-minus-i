"""Step 2: tokenize raw RedPajama files with the Llama-2 tokenizer.

Every raw file <raw_dir>/<subset>/<rel> becomes two files under <output_dir>/<subset>/:
  <rel>.bin      all documents' token ids concatenated, uint16
  <rel>.idx.npy  int64 offsets of length num_docs + 1; doc i = bin[idx[i]:idx[i+1]]
Each document is encoded as [BOS] + tokens + [EOS]. The .idx.npy file is written last,
so its existence marks a finished shard and reruns skip it.
"""

from common import (
    SUBSETS,
    TOKEN_BIN_SUFFIX,
    TOKEN_IDX_SUFFIX,
    TOKENIZER_NAME,
    iter_texts,
    list_raw_files,
    parse_subsets,
)

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_tokenizer = None


def _init_worker(tokenizer_name):
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_file(task):
    raw_path, bin_path, idx_path, batch_size = task
    if idx_path.exists():
        return str(raw_path), None, None, "skipped"
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bos, eos = _tokenizer.bos_token_id, _tokenizer.eos_token_id

    tmp_bin = bin_path.with_name(bin_path.name + ".tmp")
    offsets = [0]
    with open(tmp_bin, "wb") as fout:

        def flush(texts):
            ids_list = _tokenizer(texts, add_special_tokens=False)["input_ids"]
            docs = [np.asarray([bos] + ids + [eos], dtype=np.uint16) for ids in ids_list]
            for doc in docs:
                offsets.append(offsets[-1] + len(doc))
            fout.write(np.concatenate(docs).tobytes())

        batch = []
        for text in iter_texts(raw_path):
            batch.append(text)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    os.replace(tmp_bin, bin_path)
    tmp_idx = idx_path.with_name(idx_path.name + ".tmp.npy")
    np.save(tmp_idx, np.asarray(offsets, dtype=np.int64))
    os.replace(tmp_idx, idx_path)
    return str(raw_path), len(offsets) - 1, offsets[-1], "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory produced by download_redpajama.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store tokenized shards")
    parser.add_argument("--subsets", type=str, default=",".join(SUBSETS))
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_NAME)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Documents per tokenizer call")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    vocab_size = len(AutoTokenizer.from_pretrained(args.tokenizer))
    assert vocab_size <= np.iinfo(np.uint16).max + 1, "vocab too large for uint16"

    tasks = []
    for subset in parse_subsets(args.subsets):
        raw_root = Path(args.raw_dir) / subset
        files = list_raw_files(args.raw_dir, subset)
        print(f"{subset}: {len(files)} raw files")
        for raw_path in files:
            out_base = Path(args.output_dir) / subset / raw_path.relative_to(raw_root)
            bin_path = Path(str(out_base) + TOKEN_BIN_SUFFIX)
            idx_path = Path(str(out_base) + TOKEN_IDX_SUFFIX)
            tasks.append((raw_path, bin_path, idx_path, args.batch_size))

    # Large files first so the long tail of the pool is short.
    tasks.sort(key=lambda t: t[0].stat().st_size, reverse=True)
    with Pool(args.num_workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for i, (raw, n_docs, n_tokens, status) in enumerate(
            pool.imap_unordered(tokenize_file, tasks), 1
        ):
            info = f" docs={n_docs} tokens={n_tokens}" if status == "done" else ""
            print(f"[{i}/{len(tasks)}] {status}: {raw}{info}", flush=True)
    print("Tokenization finished.")


if __name__ == "__main__":
    main()
