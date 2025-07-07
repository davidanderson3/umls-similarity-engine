# UMLS Similarity Engine

Find similar UMLS concepts. Includes a pre-built FAISS index if you'd rather not take the time to make your own embeddings. 

Project encodes preferred English terms using the SapBERT model and performs nearest-neighbor lookups with FAISS.

---

## Prerequisites

1. **UMLS License**  
   You must have an active UMLS Terminology Services (UTS) license.  
   Register and accept the license at:  
   https://uts.nlm.nih.gov/uts/signup-login

2. **Download the 2025AA Full Metathesaurus Subset**  
   If you want to use the attached FAISS index, you must pre-process the “Full Metathesaurus Subset” (2025AA release):
   https://download.nlm.nih.gov/umls/kss/2025AA/umls-2025AA-metathesaurus-full.zip
   Unzip to obtain:
   - `MRCONSO.RRF`  
   - `MRSTY.RRF`  

3. **Extract metadata**  
   Filter for English preferred, un-suppressed terms (TS = P, STT = PF, ISPREF = Y, SUPPRESS = N, LAT = ENG) and write:
   ```bash
   python precompute.py \
     --mrconso /path/to/MRCONSO.RRF \
     --mrsty  /path/to/MRSTY.RRF \
     --out    final/umls_metadata.csv
   ```

## Requirements

Conda is recommended. Install conda then:

```bash
conda create -n umls-similarity python=3.9
conda activate umls-similarity
conda install -c conda-forge pandas numpy flask tqdm transformers pytorch cpuonly faiss-cpu
```

## Using Prebuilt Artifacts

To skip embedding computation, which can take a while, pull prebuilt files from Hugging Face Hub repo:

```bash
pip install huggingface_hub
mkdir -p final
python - <<'EOF'
from huggingface_hub import hf_hub_download
repo_id = "dvdndrsn/umls-sapbert-faiss"
fname   = "final/umls_index_hnsw.faiss"
# make sure final/ exists:
path = hf_hub_download(repo_id=repo_id, filename=fname, local_dir="final")
print("Downloaded index to:", path)
EOF
```

This will add the FAISS index:
- `final/umls_index_hnsw.faiss`

## Or - Build the artifacts yourself

1. **Build embeddings**  
   Encode terms in chunks (checkpoint on failure):
   ```bash
   python build_umls_embeddings.py \
     --metadata final/umls_metadata.csv \
     --outdir  output/part1 \
     --start_idx 0 --end_idx 640000

   # repeat for subsequent chunks…
   ```

2. **Merge chunks**  
   ```bash
   python merge_embeddings.py \
     --parts output/part1 output/part2 … \
     --out_dir final \
     --save_npy
   ```

3. **Build FAISS HNSW index**  
   ```bash
   python build_hnsw_index.py \
     --embeddings final/umls_embeddings.npy \
     --index_out final/umls_index_hnsw.faiss
   ```

## Querying

   **CLI**:
   ```bash
   python umls_similarity_search.py \
     --metadata final/umls_metadata.csv \
     --index    final/umls_index_hnsw.faiss \
     --top_k    10 \
     --ef_search 128
   ```

   **Web**:
   ```bash
   python umls_similarity_web.py \
     --metadata final/umls_metadata.csv \
     --index    final/umls_index_hnsw.faiss \
     --host 0.0.0.0 --port 5000
   ```
   Navigate to http://127.0.0.1:5000 in your web browser. 
---


