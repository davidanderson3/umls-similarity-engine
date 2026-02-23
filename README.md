# UMLS Similarity Engine

Find semantically related UMLS concepts using FAISS indexes and multiple embedding sources.

This repository supports:
- SapBERT concept retrieval
- BioConceptVec concept retrieval
- cui2vec concept retrieval
- Ensemble fusion across all three (optional MRREL-based relation boosting)

## Modes

`umls_similarity_web.py` supports four modes:
- `sapbert`: term-embedding similarity from SapBERT
- `bioconceptvec`: concept co-occurrence embeddings from BioConceptVec
- `cui2vec`: CUI co-occurrence embeddings
- `ensemble`: weighted fusion of SapBERT + BioConceptVec + cui2vec

Additional runtime capabilities:
- Long-query segmentation for `bioconceptvec`/`cui2vec` (`--chunk_long_query_min_chars`, `--chunk_max_segments`)
- Resolver strategies for seed concept resolution:
  - `--resolver exact` (direct string-to-term match)
  - `--resolver umls_api` (external/local resolver API)
- Optional MRREL graph boost in ensemble mode (`--mrrel*` flags)
- Optional cross-type filtering in BioConceptVec mode (`--cross_type_only`)

## Prerequisites

1. **UMLS License**  
   Register and accept UTS license:  
   https://uts.nlm.nih.gov/uts/signup-login

2. **UMLS files (recommended: 2025AA Full Metathesaurus Subset)**  
   https://download.nlm.nih.gov/umls/kss/2025AA/umls-2025AA-metathesaurus-full.zip

   Core files used here:
   - `MRCONSO.RRF`
   - `MRSTY.RRF`
   - `MRREL.RRF` (optional, ensemble relation boost)

3. **Create metadata CSV**
   ```bash
   python precompute.py \
     --mrconso /path/to/MRCONSO.RRF \
     --mrsty /path/to/MRSTY.RRF \
     --out final/umls_metadata.csv
   ```

## Environment Setup

Conda example:

```bash
conda create -n umls-similarity python=3.9
conda activate umls-similarity
conda install -c conda-forge pandas numpy flask tqdm transformers pytorch cpuonly faiss-cpu
```

## Build Artifacts

### 1) SapBERT index (UMLS terms)

Build embeddings in chunks:

```bash
python build_umls_embeddings.py \
  --metadata final/umls_metadata.csv \
  --outdir output/part1 \
  --start_idx 0 --end_idx 640000
```

Repeat as needed for additional ranges.

Merge chunks:

```bash
python merge_embeddings.py \
  --parts output/part1 output/part2 \
  --out_dir final \
  --save_npy
```

Build FAISS index:

```bash
python build_hnsw_index.py \
  --embeddings final/umls_embeddings.npy \
  --index_out final/umls_index_hnsw.faiss
```

### 2) BioConceptVec index (optional)

```bash
python build_bioconceptvec_index.py \
  --concept_json data/bioconceptvec/concept_cbow.json \
  --index_out final/bioconceptvec_cbow.faiss \
  --id_out final/bioconceptvec_ids.txt \
  --include_sources MESH OMIM
```

### 3) cui2vec index (optional)

```bash
python build_cui2vec_index.py \
  --input data/cui2vec/cui2vec_pretrained.csv \
  --format csv \
  --index_out final/cui2vec.faiss \
  --cui_out final/cui2vec_cuis.txt
```

## Querying

### CLI (SapBERT)

```bash
python umls_similarity_search.py \
  --metadata final/umls_metadata.csv \
  --index final/umls_index_hnsw.faiss \
  --model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --top_k 10 \
  --ef_search 128
```

### Web App (all modes)

Basic run:

```bash
python umls_similarity_web.py \
  --mode sapbert \
  --metadata final/umls_metadata.csv \
  --index final/umls_index_hnsw.faiss \
  --bioconceptvec_index final/bioconceptvec_cbow.faiss \
  --bioconceptvec_ids final/bioconceptvec_ids.txt \
  --cui2vec_index final/cui2vec.faiss \
  --cui2vec_cuis final/cui2vec_cuis.txt \
  --host 0.0.0.0 \
  --port 5000
```

Open `http://127.0.0.1:5000`.

## Resolver API Integration (optional)

Use `--resolver umls_api` to resolve long or free-text queries through an external/local resolver.

This project is configured to work with:
- https://github.com/davidanderson3/umls-search

When that service is running locally, you can point this app at its search endpoint (example below).

Example:

```bash
python umls_similarity_web.py \
  --mode ensemble \
  --resolver umls_api \
  --umls_api_base_url http://localhost:3000 \
  --umls_api_search_path /api/search \
  --umls_api_query_param q \
  --umls_api_page_param page \
  --umls_api_size_param size \
  --umls_api_limit_param size \
  --umls_api_limit 100 \
  --umls_api_results_key results \
  --metadata final/umls_metadata.csv \
  --index final/umls_index_hnsw.faiss \
  --bioconceptvec_index final/bioconceptvec_cbow.faiss \
  --bioconceptvec_ids final/bioconceptvec_ids.txt \
  --cui2vec_index final/cui2vec.faiss \
  --cui2vec_cuis final/cui2vec_cuis.txt
```

## Ensemble + MRREL Boost (optional)

If you provide `MRREL.RRF`, ensemble mode can boost candidates linked to seed CUIs by UMLS relations:

```bash
python umls_similarity_web.py \
  --mode ensemble \
  --mrrel /path/to/MRREL.RRF \
  --mrrel_boost 0.005 \
  --mrrel_count_boost 0.001 \
  --metadata final/umls_metadata.csv \
  --index final/umls_index_hnsw.faiss \
  --bioconceptvec_index final/bioconceptvec_cbow.faiss \
  --bioconceptvec_ids final/bioconceptvec_ids.txt \
  --cui2vec_index final/cui2vec.faiss \
  --cui2vec_cuis final/cui2vec_cuis.txt
```

## Latest Updates in This Repo

- Added multi-mode web search (`sapbert`, `bioconceptvec`, `cui2vec`, `ensemble`)
- Added long-query segmentation controls for concept-vector modes
- Added resolver abstraction (`exact` and `umls_api`)
- Added ensemble score breakdown columns in the UI
- Added optional MRREL relation boosting for ensemble ranking
- Added scripts to build BioConceptVec and cui2vec FAISS indexes
