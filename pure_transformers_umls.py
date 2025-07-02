#!/usr/bin/env python3
"""
pure_transformers_umls.py

1) Load STRs from umls_metadata.csv
2) Encode with SapBERT + checkpointing (chunkable)
3) Save embeddings to .memmap and .npy (no FAISS)
"""

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
torch.set_num_threads(1)

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import torch.multiprocessing as _mp
_mp.set_sharing_strategy('file_system')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",   required=True, help="Path to umls_metadata.csv")
    p.add_argument("--outdir",     default="output", help="Where to save outputs")
    p.add_argument(
        "--model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="HuggingFace SapBERT model ID or local dir"
    )
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--start_idx",  type=int, default=None,
                   help="(Optional) Global index to start at")
    p.add_argument("--end_idx",    type=int, default=None,
                   help="(Optional) Global index to end before (exclusive)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load and sanitize the STR column
    df = pd.read_csv(args.metadata, usecols=["STR"]).dropna()
    terms = df["STR"].astype(str).tolist()
    total = len(terms)

    # 1a) Chunk selection
    s = args.start_idx if args.start_idx is not None else 0
    e = args.end_idx   if args.end_idx   is not None else total
    if not (0 <= s < e <= total):
        raise ValueError(f"Invalid slice [{s}:{e}] of {total}")
    terms = terms[s:e]
    N = len(terms)

    # Sanity check: ensure no off-by-one in slice
    expected = e - s
    if N != expected:
        raise RuntimeError(
            f"Chunk size mismatch: expected {expected} terms, got {N}. "
            "Please delete this chunk’s output and re-run it from scratch."
        )

    print(f"Processing {N} terms [{s}:{e}]")

    # paths for memmap and progress
    memmap_path   = os.path.join(args.outdir, "umls_embeddings.memmap")
    progress_path = os.path.join(args.outdir, "progress.json")

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model     = AutoModel.from_pretrained(args.model).eval()
    hidden_sz = model.config.hidden_size

    # prepare memmap
    mode = "r+" if os.path.exists(memmap_path) else "w+"
    embs = np.memmap(memmap_path, dtype="float32", mode=mode, shape=(N, hidden_sz))

    # determine resume index
    start = 0
    if os.path.exists(progress_path):
        try:
            st = json.load(open(progress_path))
            start = min(int(st.get("next_index", 0)), N)
            print(f"Resuming at {start}")
        except:
            nz = np.any(embs != 0, axis=1)
            if nz.any():
                start = int(np.where(nz)[0].max()) + 1
                print(f"Inferred resume index {start}")

    # encode loop
    if start < N:
        device = torch.device("cpu")
        model.to(device)
        batches = (N - start + args.batch_size - 1) // args.batch_size
        print(f"Encoding {N} terms from {start} in {batches} batches…")

        with torch.no_grad():
            for i in tqdm(range(start, N, args.batch_size),
                          initial=start // args.batch_size,
                          total=batches):
                batch = terms[i : i + args.batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)
                out  = model(**enc).last_hidden_state
                mask = enc.attention_mask.unsqueeze(-1)
                vecs = ((out * mask).sum(1) / mask.sum(1)).cpu().numpy()

                embs[i : i + vecs.shape[0]] = vecs
                if i % 100 == 0 or i + vecs.shape[0] >= N:
                    embs.flush()
                    json.dump(
                        {"next_index": i + vecs.shape[0]},
                        open(progress_path, "w")
                    )

    # finalize and save numpy array
    npy_path = os.path.join(args.outdir, "umls_embeddings.npy")
    np.save(npy_path, np.array(embs))
    print(f"Saved embeddings → {npy_path}")


if __name__ == "__main__":
    main()
