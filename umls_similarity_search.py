#!/usr/bin/env python3
"""
umls_similarity_search.py

1) Loads precomputed UMLS metadata (STR column) and your SapBERT+FAISS embeddings
   from the specified outdir (default: ./processed).
2) Loads SapBERT via HuggingFace Transformers (mean-pooling).
3) Encodes a query string and retrieves the top-k most similar terms by NumPy.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ─── Supporting Helpers ────────────────────────────────────────────────────────

def load_metadata(metadata_csv: str):
    """
    Load the STR column from your metadata CSV.
    """
    df = pd.read_csv(metadata_csv, usecols=["STR"])
    df = df[df["STR"].notna()].reset_index(drop=True)
    return df["STR"].astype(str).tolist()

def encode_term(term: str,
                tokenizer: AutoTokenizer,
                model: AutoModel,
                device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    Encode a single term into a float32 vector using mean-pooling.
    Always runs on CPU to avoid CUDA teardown segfaults.
    """
    model.to(device)
    enc = tokenizer(
        [term],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**enc).last_hidden_state       # (1, L, D)
        mask = enc.attention_mask.unsqueeze(-1)     # (1, L, 1)
        summed = (out * mask).sum(1)                # (1, D)
        lengths = mask.sum(1)                       # (1, 1)
        emb = (summed / lengths).cpu().numpy().astype('float32')  # (1, D)

    return emb

def search_np_fallback(query_vec: np.ndarray,
                       embeddings_npy: str,
                       terms: list[str],
                       k: int):
    """
    Brute-force inner-product search using the .npy embeddings.
    """
    # 1) Memory-map the embeddings (N, D)
    embs = np.load(embeddings_npy, mmap_mode="r")
    # 2) Compute all inner products (N,)
    scores = embs.dot(query_vec.flatten())
    # 3) Top-k indices
    topk = np.argpartition(-scores, k)[:k]
    # 4) Sort top-k by descending score
    topk = topk[np.argsort(-scores[topk])]
    return [{"term": terms[i], "score": float(scores[i])} for i in topk]

# ─── Full Function Implementation ─────────────────────────────────────────────

def search_umls(query: str,
                k: int,
                outdir: str,
                metadata_csv: str,
                model_name: str):
    """
    Given a query string, return top-k nearest UMLS terms and similarity scores.
    """
    # Load metadata strings
    terms = load_metadata(metadata_csv)

    # Load model & tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Encode the query
    q_vec = encode_term(query, tokenizer, model)

    # Fallback pure-NumPy search
    emb_path = os.path.join(outdir, "umls_embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found at {emb_path}")
    return search_np_fallback(q_vec, emb_path, terms, k)

# ─── Example CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Search UMLS via SapBERT+NumPy")
    p.add_argument(
        "--outdir",
        default="processed",
        help="Directory containing umls_embeddings.npy (default: ./processed)"
    )
    p.add_argument(
        "--metadata",
        required=True,
        help="Path to umls_metadata.csv (must include STR column)"
    )
    p.add_argument(
        "--query",
        required=True,
        help="Search query text"
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    p.add_argument(
        "--model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="HuggingFace SapBERT model ID or local path"
    )
    args = p.parse_args()

    results = search_umls(
        query=args.query,
        k=args.top_k,
        outdir=args.outdir,
        metadata_csv=args.metadata,
        model_name=args.model
    )

    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['term']:<50} {r['score']:.4f}")
