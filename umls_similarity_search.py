#!/usr/bin/env python3
"""
umls_similarity_search.py

1) Load UMLS metadata (CUI, STR, STY)
2) Load FAISS HNSW index
3) Encode a query string with SapBERT via its [CLS] token
4) Search the index and print top-k nearest concepts with CUI, STR, STY
"""

import argparse
import ast
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",  required=True,
                   help="Path to your merged umls_metadata.csv")
    p.add_argument("--index",     required=True,
                   help="Path to your final umls_index_hnsw.faiss")
    p.add_argument("--model",     default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                   help="SapBERT model ID or local dir")
    p.add_argument("--top_k",     type=int, default=10,
                   help="Number of results to return")
    p.add_argument("--ef_search", type=int, default=64,
                   help="HNSW efSearch (higher=more accurate/slower)")
    p.add_argument(
        "--exclude_sty",
        nargs="*",
        default=[],
        help="Semantic type names to exclude from results",
    )
    return p.parse_args()

def load_metadata(path):
    df = pd.read_csv(path, usecols=["CUI", "STR", "STY"]).dropna()
    cuis = df["CUI"].astype(str).tolist()
    terms = df["STR"].astype(str).tolist()
    stys_raw = df["STY"].astype(str).tolist()
    stys = []
    for s in stys_raw:
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                stys.append(parsed)
            else:
                stys.append([str(parsed)])
        except Exception:
            stys.append([s])
    return (cuis, terms, stys)

def encode_query(text, tokenizer, model, device):
    # tokenize
    enc = tokenizer([text],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        # out.last_hidden_state is (1, L, D); we take the [CLS] token at position 0
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")

    # normalize so search distances are comparable to the index vectors
    faiss.normalize_L2(cls_emb)
    return cls_emb  # shape (1, D)

def main():
    args = parse_args()

    # 1) Load metadata
    cuis, terms, stys = load_metadata(args.metadata)
    exclude_set = {s.lower() for s in args.exclude_sty}

    # 2) Load FAISS index
    index = faiss.read_index(args.index)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = args.ef_search

    # 3) Load SapBERT
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model     = AutoModel.from_pretrained(args.model).to(device).eval()

    # 4) Interactive loop
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break

        q_vec = encode_query(q, tokenizer, model, device)
        D, I  = index.search(q_vec, args.top_k * 5)

        print(f"\nTop {args.top_k} results for “{q}”:\n")
        shown = 0
        for idx, score in zip(I[0], D[0]):
            sty_list = stys[idx]
            if exclude_set and exclude_set.intersection(s.lower() for s in sty_list):
                continue
            sty_str = ", ".join(sty_list)
            shown += 1
            print(f"{shown:2d}. {cuis[idx]}  {terms[idx]:<50}  {sty_str:<20}  {score:.4f}")
            if shown >= args.top_k:
                break

if __name__ == "__main__":
    main()
