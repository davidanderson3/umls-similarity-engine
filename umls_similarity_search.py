#!/usr/bin/env python3
"""
umls_similarity_search.py

1) Load UMLS metadata (CUI, STR, STY)
2) Load FAISS HNSW index
3) Encode a query string with SapBERT via its [CLS] token
4) Search the index and print top-k nearest concepts with CUI, STR, STY
"""

import argparse
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
    return p.parse_args()

def load_metadata(path):
    df = pd.read_csv(path, usecols=["CUI","STR","STY"]).dropna()
    return (
        df["CUI"].astype(str).tolist(),
        df["STR"].astype(str).tolist(),
        df["STY"].astype(str).tolist(),
    )

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
    return cls_emb  # shape (1, D)

def main():
    args = parse_args()

    # 1) Load metadata
    cuis, terms, stys = load_metadata(args.metadata)

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
        D, I  = index.search(q_vec, args.top_k)

        print(f"\nTop {args.top_k} results for “{q}”:\n")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            print(f"{rank:2d}. {cuis[idx]}  {terms[idx]:<50}  {stys[idx]:<20}  {score:.4f}")

if __name__ == "__main__":
    main()
