#!/usr/bin/env python3
# test_umls.py

from umls_similarity_search import load_resources, encode_term
import numpy as np
import os

def main():
    # 1️⃣ Load resources (index, tokenizer, model, terms)
    print("1️⃣ load_resources…")
    idx, tok, model, terms = load_resources(
        outdir="processed",
        metadata_csv="processed/umls_metadata.csv",
        model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    print(f"   ✔️  Loaded {len(terms)} terms; index = {idx}")

    # 2️⃣ Encode term (now CPU-only)
    print("2️⃣ encode_term…")
    vec = encode_term("diabetes mellitus", tok, model)
    print("   ✔️  Encoded vector shape:", vec.shape)

    # 3️⃣ FAISS search
    print("3️⃣ FAISS search…")
    D, I = idx.search(vec, 10)
    print("   ✔️  Top scores:", D[0][:5])

if __name__ == "__main__":
    main()
