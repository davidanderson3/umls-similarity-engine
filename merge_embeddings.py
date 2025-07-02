#!/usr/bin/env python3
"""
merge_embeddings.py

Efficiently merges per-chunk .npy embeddings into one memmap and optional .npy.

Usage:
  python3 merge_embeddings.py \
    --parts output/part1 output/part2 output/part3 output/part4 output/part5 output/part6 \
    --out_dir final \
    --save_npy
"""

import argparse
import os
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Merge chunked embeddings into one dataset")
    p.add_argument(
        "--parts",
        nargs="+",
        required=True,
        help="List of chunk directories containing umls_embeddings.npy"
    )
    p.add_argument(
        "--out_dir",
        default="final",
        help="Output directory for merged embeddings"
    )
    p.add_argument(
        "--save_npy",
        action="store_true",
        help="Also save a .npy copy after merging"
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Determine total size and dimensionality
    total = 0
    dims = None
    for d in args.parts:
        path = os.path.join(d, "umls_embeddings.npy")
        arr = np.load(path, mmap_mode="r")
        if dims is None:
            _, dims = arr.shape
        total += arr.shape[0]
        print(f"Chunk {d}: {arr.shape[0]} vectors")
    print(f"Total vectors: {total}, dimension: {dims}")

    # 2) Create memmap
    merged_path = os.path.join(args.out_dir, "umls_embeddings.memmap")
    merged = np.memmap(merged_path, dtype="float32", mode="w+", shape=(total, dims))

    # 3) Copy chunks into memmap
    offset = 0
    for d in args.parts:
        path = os.path.join(d, "umls_embeddings.npy")
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]
        merged[offset: offset + n] = arr
        offset += n
        print(f"Copied {n} vectors from {d}")

    merged.flush()
    print(f"Saved merged memmap to {merged_path}")

    # 4) Optionally save .npy
    if args.save_npy:
        npy_out = os.path.join(args.out_dir, "umls_embeddings.npy")
        # Load into RAM in manageable chunks if needed
        full = np.array(merged)
        np.save(npy_out, full)
        print(f"Also saved .npy to {npy_out}")

if __name__ == "__main__":
    main()
