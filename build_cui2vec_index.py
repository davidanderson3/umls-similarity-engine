#!/usr/bin/env python3
"""
Build a FAISS index from a cui2vec embedding file.

Supported input formats:
- word2vec-style text:
  C0000001 0.123 -0.456 ...
  (optional first line: <num_vectors> <dim>)
- JSON map:
  {"C0000001": [0.123, -0.456, ...], ...}
"""

import argparse
import csv
import json
import os

import faiss
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS index for cui2vec")
    p.add_argument("--input", required=True, help="Path to cui2vec embeddings file")
    p.add_argument("--format", choices=["auto", "word2vec", "json", "csv"], default="auto",
                   help="Input format (default: auto)")
    p.add_argument("--index_out", default="final/cui2vec.faiss", help="Output FAISS index path")
    p.add_argument("--cui_out", default="final/cui2vec_cuis.txt",
                   help="Output CUI list aligned to index rows")
    p.add_argument("--M", type=int, default=32, help="HNSW M")
    p.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction")
    return p.parse_args()


def load_from_json(path):
    with open(path, "r", encoding="utf8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("JSON input must be an object mapping CUI -> vector")
    cuis = []
    vecs = []
    for k, v in obj.items():
        if not isinstance(k, str) or not k.startswith("C"):
            continue
        if not isinstance(v, list) or not v:
            continue
        try:
            vec = [float(x) for x in v]
        except Exception:
            continue
        cuis.append(k)
        vecs.append(vec)
    return cuis, vecs


def load_from_word2vec_text(path):
    cuis = []
    vecs = []
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # optional header: "<num_vectors> <dim>"
            if i == 0 and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue
            if len(parts) < 3:
                continue
            cui = parts[0]
            if not cui.startswith("C"):
                continue
            try:
                vec = [float(x) for x in parts[1:]]
            except Exception:
                continue
            cuis.append(cui)
            vecs.append(vec)
    return cuis, vecs


def load_from_csv(path):
    cuis = []
    vecs = []
    with open(path, "r", encoding="utf8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row:
                continue
            if not header_seen:
                header_seen = True
                # Skip header if first cell is empty or not a CUI.
                first = row[0].strip().strip('"')
                if not first.startswith("C"):
                    continue
            cui = row[0].strip().strip('"')
            if not cui.startswith("C"):
                continue
            try:
                vec = [float(x) for x in row[1:] if x != ""]
            except Exception:
                continue
            if not vec:
                continue
            cuis.append(cui)
            vecs.append(vec)
    return cuis, vecs


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    fmt = args.format
    if fmt == "auto":
        low = args.input.lower()
        if low.endswith(".json"):
            fmt = "json"
        elif low.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "word2vec"

    if fmt == "json":
        cuis, vecs = load_from_json(args.input)
    elif fmt == "csv":
        cuis, vecs = load_from_csv(args.input)
    else:
        cuis, vecs = load_from_word2vec_text(args.input)

    if not vecs:
        raise RuntimeError("No CUI vectors parsed from input file.")

    X = np.array(vecs, dtype="float32")
    faiss.normalize_L2(X)
    d = X.shape[1]

    print(f"Parsed vectors: {len(cuis)} (dim={d})")
    index = faiss.IndexHNSWFlat(d, args.M)
    index.hnsw.efConstruction = args.ef_construction
    index.add(X)

    print(f"Writing index: {args.index_out}")
    faiss.write_index(index, args.index_out)

    print(f"Writing CUI list: {args.cui_out}")
    with open(args.cui_out, "w", encoding="utf8") as f:
        for cui in cuis:
            f.write(cui + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
