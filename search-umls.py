#!/usr/bin/env python3
"""
search_umls.py

A simple CLI to query the FAISS index of UMLS embeddings and return the top-K similar concepts.

Usage:
  python search_umls.py --metadata ./processed/umls_metadata.csv \
                        --index    ./processed/umls_index.faiss \
                        --embeddings ./processed/umls_embeddings.npy \
                        --query "Type 2 diabetes" \
                        --topk 10

Requirements:
  pip install transformers pandas numpy faiss-cpu
"""

import argparse
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Force single-thread
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load SapBERT model globally
def load_sapbert(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

# Embed a query string
def embed_query(text, tokenizer, model, device):
    enc = tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (1, L, D)
        mask = enc.attention_mask.unsqueeze(-1)
        summed = (out * mask).sum(1)
        lengths = mask.sum(1)
        emb = (summed / lengths).cpu().numpy()
    return emb

# Parse CLI args
def parse_args():
    p = argparse.ArgumentParser(description='Search UMLS embeddings')
    p.add_argument('--metadata',    required=True, help='Path to umls_metadata.csv')
    p.add_argument('--index',       required=True, help='Path to umls_index.faiss')
    p.add_argument('--embeddings',  required=True, help='Path to umls_embeddings.npy')
    p.add_argument('--model',       default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                   help='HuggingFace SapBERT model ID')
    p.add_argument('--query',       required=True, help='Query string')
    p.add_argument('--topk',        type=int, default=5, help='Number of similar concepts to return')
    return p.parse_args()

# Main CLI function
def main():
    args = parse_args()

    # Load resources
    meta = pd.read_csv(args.metadata)
    embs = np.load(args.embeddings)
    index = faiss.read_index(args.index)

    # Load model
    tokenizer, model, device = load_sapbert(args.model)

    # Embed query
    q_emb = embed_query(args.query, tokenizer, model, device)
    faiss.normalize_L2(q_emb)

    # Search
    D, I = index.search(q_emb, args.topk)

    # Display results
    print(f"Top {args.topk} results for '{args.query}':\n")
    for score, idx in zip(D[0], I[0]):
        cui = meta.loc[idx, 'CUI']
        term = meta.loc[idx, 'STR']
        print(f"{score:.4f}\t{cui}\t{term}")

if __name__ == '__main__':
    main()
