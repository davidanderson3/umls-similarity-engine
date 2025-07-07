# UMLS Similarity Engine

This repository contains utilities for building and searching an approximate nearest
neighbour (ANN) index over UMLS concept embeddings. The scripts use the
[SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
model to encode preferred English terms and FAISS to perform fast similarity
searches.

## Overview

The general workflow is:

1. **Prepare metadata**
   - `precompute.py` reads `MRCONSO.RRF` and `MRSTY.RRF` from the UMLS
     distribution. It filters for preferred English terms and writes
     `umls_metadata.csv` containing the concept unique identifier (CUI), term
     string and semantic type information.
2. **Encode with SapBERT**
   - `pure_transformers_umls.py` encodes the terms from
     `umls_metadata.csv` using SapBERT. It supports processing a slice of the
     data so that large corpora can be encoded in chunks. Results are stored in a
     memory‑mapped array (`umls_embeddings.memmap`) and an optional `.npy`
     file.
3. **Merge chunks (optional)**
   - If embeddings are produced in multiple output directories,
     `merge_embeddings.py` combines them into a single memmap (and optional
     `.npy`) in the `final` directory.
4. **Build a search index**
   - `build_hnsw_index.py` reads the merged embeddings and constructs a FAISS
     HNSW index (`umls_index_hnsw.faiss`) for efficient similarity search.
5. **Search**
   - `umls_similarity_search.py` provides a simple command‑line interface to
     query the index. A query string is encoded with SapBERT and the nearest
     concepts are printed along with their CUIs, terms and semantic types.
   - `umls_similarity_web.py` serves a small Flask web application offering the
     same functionality via a browser.
6. **Identify missed synonyms**
   - `identify_missed_synonyms.py` scans the index for pairs of different CUIs
     whose embeddings are very similar. This can help find potential synonym
     relations not present in the original UMLS data.

## Requirements

The scripts require Python 3 and the following packages:

- `pandas`
- `numpy`
- `torch`
- `transformers`
- `faiss-cpu`
- `flask` (for the web interface)
- `tqdm`

Install them with `pip` or your preferred package manager.

## Example usage

```bash
# 1. Build metadata CSV
python precompute.py --mrconso /path/to/MRCONSO.RRF \
                     --mrsty /path/to/MRSTY.RRF \
                     --out final/umls_metadata.csv

# 2. Encode terms (may be run in chunks)
python pure_transformers_umls.py --metadata final/umls_metadata.csv \
                                 --outdir output

# 3. Merge chunks (if needed)
python merge_embeddings.py --parts output --out_dir final --save_npy

# 4. Build HNSW index
python build_hnsw_index.py --embeddings final/umls_embeddings.npy \
                           --index_out final/umls_index_hnsw.faiss

# 5. Search interactively
python umls_similarity_search.py --metadata final/umls_metadata.csv \
                                 --index final/umls_index_hnsw.faiss
```

`umls_similarity_web.py` can be run similarly to launch the browser-based
search tool, and `identify_missed_synonyms.py` can be used to produce a CSV of
potential missing synonym pairs.


