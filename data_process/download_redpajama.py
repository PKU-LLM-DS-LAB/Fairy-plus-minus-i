"""Step 1: download the raw RedPajama-Data-1T files.

Files are stored as <output_dir>/<subset>/..., mirroring the official URL layout
(same layout as the RED_PAJAMA_DATA_DIR expected by the official HF loading script).
Downloads are resumable: partial files are kept as *.part and continued via HTTP Range.
"""

from common import REPO_ID, SUBSETS, URL_BASE, parse_subsets

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


def get_urls(subset):
    url_list = hf_hub_download(
        repo_id=REPO_ID, filename=f"urls/{subset}.txt", repo_type="dataset"
    )
    with open(url_list, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def local_path_for(url, output_dir):
    rel = url.split("/v1.0.0/", 1)[1] if "/v1.0.0/" in url else url.rsplit("/", 1)[1]
    return Path(output_dir) / rel


def download_file(url, dest, max_retries=10, chunk_size=1 << 20):
    if dest.exists():
        return dest, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(max_retries):
        try:
            done = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    tmp.rename(dest)
                    return dest, "downloaded"
                r.raise_for_status()
                mode = "ab" if done and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
            tmp.rename(dest)
            return dest, "downloaded"
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"[retry {attempt + 1}/{max_retries}] {url}: {e}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store raw RedPajama files")
    parser.add_argument("--subsets", type=str, default=",".join(SUBSETS),
                        help="Comma separated subsets to download")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--max_files_per_subset", type=int, default=None,
                        help="Only download the first N files of each subset (for testing)")
    parser.add_argument("--url_base", type=str, default=URL_BASE,
                        help="Replace the official URL prefix, e.g. with a mirror")
    args = parser.parse_args()

    jobs = []
    for subset in parse_subsets(args.subsets):
        urls = get_urls(subset)
        if args.max_files_per_subset is not None:
            urls = urls[: args.max_files_per_subset]
        print(f"{subset}: {len(urls)} files")
        for url in urls:
            dest = local_path_for(url, args.output_dir)
            if args.url_base != URL_BASE:
                url = url.replace(URL_BASE, args.url_base.rstrip("/"), 1)
            jobs.append((url, dest))

    failed = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        futures = {pool.submit(download_file, url, dest): url for url, dest in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                dest, status = fut.result()
                print(f"[{i}/{len(jobs)}] {status}: {dest}")
            except Exception as e:
                failed.append(futures[fut])
                print(f"[{i}/{len(jobs)}] FAILED: {e}")

    if failed:
        print(f"{len(failed)} files failed, rerun the script to resume:")
        for url in failed:
            print("  " + url)
        raise SystemExit(1)
    print("All files downloaded.")


if __name__ == "__main__":
    main()
