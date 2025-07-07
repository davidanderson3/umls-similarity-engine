#!/usr/bin/env python3
"""
identify_missed_synonyms.py

Uses the existing UMLS embeddings and FAISS index to find pairs of concepts
from different CUIs whose embeddings are very similar. These may represent
potential synonymy that is missing from the UMLS.

Outputs a CSV with the following columns:
    CUI1,STR1,CUI2,STR2,SIMILARITY

Similarity is the cosine similarity computed from the L2-normalized vectors.
"""

import argparse
import csv
import numpy as np
import pandas as pd
import faiss


def parse_args():
    p = argparse.ArgumentParser(
        description="Identify possible missed synonymy using embeddings"
    )
    p.add_argument(
        "--metadata",
        required=True,
        help="Path to umls_metadata.csv used to build the embeddings",
    )
    p.add_argument(
        "--index",
        required=True,
        help="Path to FAISS HNSW index built from the embeddings",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of nearest neighbours to examine per term",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold to flag potential synonyms",
    )
    p.add_argument(
        "--out",
        default="missed_synonyms.csv",
        help="Where to write the output CSV",
    )
    return p.parse_args()


def load_metadata(path):
    df = pd.read_csv(path, usecols=["CUI", "STR"]).dropna()
    cuis = df["CUI"].astype(str).tolist()
    strs = df["STR"].astype(str).tolist()
    return cuis, strs


def main():
    args = parse_args()

    cuis, strs = load_metadata(args.metadata)
    n = len(cuis)

    index = faiss.read_index(args.index)

    seen = set()
    results = []

    for i in range(n):
        vec = index.reconstruct(i)
        D, I = index.search(np.expand_dims(vec, 0), args.top_k + 1)
        for dist, j in zip(D[0], I[0]):
            if j == i:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            sim = 1.0 - float(dist) / 2.0
            if sim >= args.threshold and cuis[i] != cuis[j]:
                results.append((cuis[i], strs[i], cuis[j], strs[j], sim))

    results.sort(key=lambda x: -x[4])

    with open(args.out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["CUI1", "STR1", "CUI2", "STR2", "SIMILARITY"])
        for row in results:
            w.writerow(row)

    print(f"Wrote {len(results)} potential synonym pairs to {args.out}")


if __name__ == "__main__":
    main()
