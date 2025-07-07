# UMLS Similarity Engine

This repository provides scripts for building and querying a similarity search index over UMLS concepts. It encodes preferred English terms using the [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) model and performs nearest neighbour lookups with FAISS.

## Workflow

The typical pipeline is:

1. **Extract metadata**
   - `precompute.py` reads `MRCONSO.RRF` and `MRSTY.RRF` from a UMLS distribution.
     Preferred English terms are filtered and written to `umls_metadata.csv`.
2. **Build embeddings**
   - `build_umls_embeddings.py` encodes the terms from `umls_metadata.csv` with SapBERT.
     Large collections can be processed in slices using `--start_idx` and `--end_idx`.
     The script saves a memmap (`umls_embeddings.memmap`) and a NumPy file (`umls_embeddings.npy`).
3. **Merge chunks (optional)**
   - If embeddings were produced in multiple output directories, `merge_embeddings.py`
     combines them into a single dataset in the `final` directory.
4. **Create a search index**
   - `build_hnsw_index.py` normalizes the merged embeddings and builds a FAISS HNSW
     index (`umls_index_hnsw.faiss`) for fast approximate searches.
5. **Query the index**
   - `umls_similarity_search.py` offers a command line interface that encodes a query
     string with SapBERT and prints the nearest concepts with their CUIs, terms and
     semantic types.
   - `umls_similarity_web.py` runs a small Flask application providing the same
     functionality in a browser.
6. **Find missed synonyms (optional)**
   - `identify_missed_synonyms.py` uses the embeddings and index to locate pairs of
     concepts from different CUIs whose vectors are extremely similar.

## Requirements

Python 3 is required. Install the dependencies with:

```bash
pip install pandas numpy torch transformers faiss-cpu flask tqdm
```

## Getting started

The following commands illustrate a minimal setup:

```bash
# 1. Extract UMLS metadata
python precompute.py --mrconso /path/to/MRCONSO.RRF \
                     --mrsty /path/to/MRSTY.RRF \
                     --out final/umls_metadata.csv

# 2. Encode the terms (use --start_idx/--end_idx for chunking)
python build_umls_embeddings.py --metadata final/umls_metadata.csv \
                               --outdir output

# 3. Merge chunks if you encoded in pieces
python merge_embeddings.py --parts output --out_dir final --save_npy

# 4. Build the FAISS HNSW index
python build_hnsw_index.py --embeddings final/umls_embeddings.npy \
                           --index_out final/umls_index_hnsw.faiss

# 5. Run an interactive search session
python umls_similarity_search.py --metadata final/umls_metadata.csv \
                                 --index final/umls_index_hnsw.faiss
```

You can also start the web interface with:

```bash
python umls_similarity_web.py --metadata final/umls_metadata.csv \
                              --index final/umls_index_hnsw.faiss
```

The commands above assume all outputs are written to a directory named `final`.
Adjust the paths as needed for your environment.

