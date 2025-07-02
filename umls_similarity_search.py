#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",  required=True)
    p.add_argument("--index",     required=True)
    p.add_argument("--model",     default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--top_k",     type=int, default=10)
    p.add_argument("--ef_search", type=int, default=64)
    return p.parse_args()

def load_terms(path):
    df = pd.read_csv(path, usecols=["STR"])
    return df["STR"].astype(str).tolist()

def encode_query(q, tok, model, dev):
    enc = tok([q], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1)
        vec = (out*mask).sum(1) / mask.sum(1)
    return vec.cpu().numpy().astype("float32")

def main():
    args = parse_args()
    terms = load_terms(args.metadata)

    idx = faiss.read_index(args.index)
    if hasattr(idx, "hnsw"):
        idx.hnsw.efSearch = args.ef_search

    dev = torch.device("cpu")
    tok   = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model).to(dev).eval()

    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break
        qv = encode_query(q, tok, model, dev)
        D, I = idx.search(qv, args.top_k)
        print(f"\nTop {args.top_k} results for “{q}”:\n")
        for rank, (i, score) in enumerate(zip(I[0], D[0]), start=1):
            print(f"{rank:2d}. {terms[i]:<50} {score:.4f}")

if __name__ == "__main__":
    main()
