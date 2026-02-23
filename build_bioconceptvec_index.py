#!/usr/bin/env python3
"""
Build a FAISS index from BioConceptVec concept-only embeddings.

Input:
  - concept_cbow.json (or other concept-only JSON from BioConceptVec)
    Format is expected to be a JSON object: { "concept_id": [float, ...], ... }

Output:
  - FAISS index (HNSW, cosine via L2 normalization + inner product)
  - ids.txt: one BioConceptVec concept ID per line, aligned to index rows
"""

import argparse
import json
import os
from typing import Dict, List

import faiss
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS index for BioConceptVec")
    p.add_argument("--concept_json", required=True,
                   help="Path to BioConceptVec concept-only JSON (e.g., concept_cbow.json)")
    p.add_argument("--index_out", default="final/bioconceptvec_cbow.faiss",
                   help="Output FAISS index path")
    p.add_argument("--id_out", default="final/bioconceptvec_ids.txt",
                   help="Output BioConceptVec concept IDs (one per line, aligned to index rows)")
    p.add_argument("--include_sources", nargs="+", default=["MESH", "OMIM"],
                   help="Keep only concept IDs with these source tokens (e.g., MESH OMIM)")
    p.add_argument("--M", type=int, default=32, help="HNSW M")
    p.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction")
    return p.parse_args()


def load_concept_json(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object mapping concept_id -> vector list")
    return data


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    include_sources = {s.strip().upper() for s in args.include_sources if s.strip()}

    print(f"Loading concept vectors from {args.concept_json} ...")
    concept = load_concept_json(args.concept_json)
    print(f"Total concepts in BioConceptVec file: {len(concept)}")

    ids: List[str] = []
    vecs: List[List[float]] = []
    skipped = 0
    for concept_id, vec in concept.items():
        parts = concept_id.split("_", 2)
        if len(parts) < 3:
            skipped += 1
            continue
        source = parts[1].upper()
        if source not in include_sources:
            skipped += 1
            continue
        ids.append(concept_id)
        vecs.append(vec)

    if not ids:
        raise RuntimeError("No concept IDs matched --include_sources")
    print(f"Selected IDs: {len(ids)} (skipped {skipped})")

    X = np.array(vecs, dtype="float32")
    faiss.normalize_L2(X)
    d = X.shape[1]

    print(f"Building HNSW index: n={X.shape[0]}, d={d}, M={args.M}")
    index = faiss.IndexHNSWFlat(d, args.M)
    index.hnsw.efConstruction = args.ef_construction
    index.add(X)

    print(f"Writing index to {args.index_out}")
    faiss.write_index(index, args.index_out)

    print(f"Writing concept IDs to {args.id_out}")
    with open(args.id_out, "w", encoding="utf8") as f:
        for concept_id in ids:
            f.write(concept_id + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
