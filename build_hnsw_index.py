#!/usr/bin/env python3
"""
build_hnsw_index.py

Load embeddings .npy → build + save FAISS HNSW index.
"""

import argparse, numpy as np, faiss, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings",     required=True,
                   help="Path to umls_embeddings.npy")
    p.add_argument("--index_out",      required=True,
                   help="Output FAISS file (umls_index_hnsw.faiss)")
    p.add_argument("--M",    type=int, default=32)
    p.add_argument("--ef_construction", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    embs = np.load(args.embeddings, mmap_mode="r")
    N, d = embs.shape
    print(f"Loaded {N}×{d}")

    faiss.normalize_L2(embs)
    idx = faiss.IndexHNSWFlat(d, args.M)
    idx.hnsw.efConstruction = args.ef_construction
    idx.add(embs)
    print(f"Built HNSW index, ntotal={idx.ntotal}")

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    faiss.write_index(idx, args.index_out)
    print(f"Saved index → {args.index_out}")

if __name__ == "__main__":
    main()
