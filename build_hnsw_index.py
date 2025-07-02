#!/usr/bin/env python3
"""
build_hnsw_index.py

1) Memory-map final/umls_embeddings.npy
2) Batch-normalize & batch-add to an HNSW index
3) Save the index
"""

import argparse
import numpy as np
import faiss
import os

def parse_args():
    p = argparse.ArgumentParser(description="Streaming HNSW index builder")
    p.add_argument("--embeddings",  default="final/umls_embeddings.npy",
                   help="Path to merged embeddings (.npy)")
    p.add_argument("--index_out",   default="final/umls_index_hnsw.faiss",
                   help="Where to write the FAISS index")
    p.add_argument("--M",    type=int, default=32,
                   help="HNSW graph connectivity (M)")
    p.add_argument("--ef_construction", type=int, default=200,
                   help="efConstruction parameter")
    p.add_argument("--batch_size",      type=int, default=100_000,
                   help="How many vectors to add per batch")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Memory‐map the embeddings
    print(f"Loading embeddings (memmap) from {args.embeddings}")
    embs = np.load(args.embeddings, mmap_mode="r")
    N, d = embs.shape
    print(f"Embedding shape: {N}×{d}")

    # 2) Create HNSW index
    print(f"Initializing HNSW index: d={d}, M={args.M}")
    index = faiss.IndexHNSWFlat(d, args.M)
    index.hnsw.efConstruction = args.ef_construction

    # 3) Stream in batches
    bs = args.batch_size
    for start in range(0, N, bs):
        end = min(N, start + bs)
        batch = embs[start:end].astype('float32')
        # L2-normalize in place for IP-as-cosine
        faiss.normalize_L2(batch)
        print(f" Adding vectors {start}:{end} …", end='', flush=True)
        index.add(batch)
        print(" done")

    # 4) Save index
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    print(f"Writing index to {args.index_out}")
    faiss.write_index(index, args.index_out)
    print("Index build complete.")

if __name__ == "__main__":
    main()
