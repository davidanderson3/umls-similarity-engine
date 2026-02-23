# HTML interface for UMLS similarity search

import argparse
import ast
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, render_template_string
from transformers import AutoModel, AutoTokenizer

DEFAULT_TOP_K = 10
DEFAULT_METADATA = "final/umls_metadata.csv"
DEFAULT_INDEX = "final/umls_index_hnsw.faiss"
EXCLUDED_SEMANTIC_TYPES = {"Clinical Drug", "Intellectual Product"}


def parse_args():
    p = argparse.ArgumentParser(description="UMLS similarity search web app")
    p.add_argument("--mode", choices=["sapbert", "bioconceptvec", "cui2vec", "ensemble"], default="sapbert",
                   help="Default search mode shown in UI")
    p.add_argument("--metadata", default=DEFAULT_METADATA,
                   help=f"Path to umls_metadata.csv (default: {DEFAULT_METADATA})")
    p.add_argument("--semgroups", default="SemGroups.txt",
                   help="Path to SemGroups.txt for mapping semantic types to semantic groups")
    p.add_argument("--index", default=DEFAULT_INDEX,
                   help=f"Path to umls_index_hnsw.faiss (default: {DEFAULT_INDEX})")
    p.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                   help="SapBERT model ID or local dir")

    p.add_argument("--bioconceptvec_index", default="final/bioconceptvec_cbow.faiss",
                   help="FAISS index for BioConceptVec")
    p.add_argument("--bioconceptvec_ids", default="final/bioconceptvec_ids.txt",
                   help="BioConceptVec concept IDs aligned to index rows")

    p.add_argument("--cui2vec_index", default="final/cui2vec.faiss",
                   help="FAISS index for cui2vec")
    p.add_argument("--cui2vec_cuis", default="final/cui2vec_cuis.txt",
                   help="CUI list aligned to cui2vec index rows")

    p.add_argument("--mrconso", default="",
                   help="Path to MRCONSO.RRF for MSH/OMIM <-> CUI mapping")
    p.add_argument("--mrrel", default="",
                   help="Path to MRREL.RRF for optional relation boost")
    p.add_argument("--mrrel_boost", type=float, default=0.0,
                   help="Additive boost in ensemble mode if candidate is MRREL-neighbor of any seed CUI")
    p.add_argument("--mrrel_count_boost", type=float, default=0.0,
                   help="Additional additive boost per unique MRREL (REL, RELA, SAB) combo between candidate and seed CUIs")
    p.add_argument("--mrrel_rela", default="",
                   help="Comma-separated RELA filter for MRREL boost (optional)")

    p.add_argument("--resolver", choices=["exact", "umls_api"], default="exact",
                   help="Resolver used for bioconceptvec/cui2vec/ensemble modes")
    p.add_argument("--umls_api_base_url", default="", help="Base URL for local UMLS resolver API")
    p.add_argument("--umls_api_search_path", default="/search", help="Resolver API path")
    p.add_argument("--umls_api_query_param", default="q", help="Resolver API query parameter for input text")
    p.add_argument("--umls_api_page_param", default="page", help="Resolver API query parameter for page")
    p.add_argument("--umls_api_page", default="1", help="Resolver API page value")
    p.add_argument("--umls_api_limit_param", default="limit", help="Resolver API query parameter for max results")
    p.add_argument("--umls_api_size_param", default="size", help="Resolver API query parameter for page size")
    p.add_argument("--umls_api_limit", type=int, default=10, help="Resolver API max results")
    p.add_argument("--umls_api_fuzzy_param", default="fuzzy", help="Resolver API query parameter for fuzzy")
    p.add_argument("--umls_api_fuzzy", default="true", help="Resolver API fuzzy value")
    p.add_argument("--umls_api_results_key", default="results",
                   help="JSON key for result list; supports dot path, empty means root list")
    p.add_argument("--umls_api_sab_field", default="sab", help="Field name for source vocabulary")
    p.add_argument("--umls_api_code_field", default="code", help="Field name for source code")
    p.add_argument("--umls_api_timeout", type=float, default=10.0, help="Resolver API timeout in seconds")
    p.add_argument("--chunk_long_query_min_chars", type=int, default=400,
                   help="For bioconceptvec/cui2vec, split long queries when length exceeds this")
    p.add_argument("--chunk_max_segments", type=int, default=8,
                   help="Maximum number of query segments to use after splitting")

    p.add_argument("--ef_search", type=int, default=64, help="HNSW efSearch")
    p.add_argument("--cross_type_only", action="store_true",
                   help="In bioconceptvec mode, enforce disease<->drug cross-type neighbors")
    p.add_argument("--ensemble_sapbert_weight", type=float, default=0.2)
    p.add_argument("--ensemble_bioconceptvec_weight", type=float, default=0.4)
    p.add_argument("--ensemble_cui2vec_weight", type=float, default=0.4)

    p.add_argument("--host", default="0.0.0.0", help="Host for the web server")
    p.add_argument("--port", type=int, default=5000, help="Port for the web server")
    return p.parse_args()


def load_metadata(path):
    df = pd.read_csv(path, usecols=["CUI", "STR", "STY"]).dropna()
    cuis = df["CUI"].astype(str).tolist()
    terms = df["STR"].astype(str).tolist()
    stys_raw = df["STY"].astype(str).tolist()
    stys = []
    for s in stys_raw:
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                stys.append(parsed)
            else:
                stys.append([str(parsed)])
        except Exception:
            stys.append([s])
    return (cuis, terms, stys)


def load_semgroup_map(path):
    sty_to_group = {}
    if not path:
        return sty_to_group
    try:
        with open(path, encoding="utf8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 4:
                    continue
                group_name = parts[1].strip()
                sty_name = parts[3].strip()
                if group_name and sty_name:
                    sty_to_group[sty_name] = group_name
    except FileNotFoundError:
        return sty_to_group
    return sty_to_group


def encode_query(text, tokenizer, model, device):
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1)
        vec = ((out * mask).sum(1) / mask.sum(1)).cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)
    return vec


_space_re = re.compile(r"\s+")


def norm_term(s: str) -> str:
    return _space_re.sub(" ", s.strip().lower())


def split_query_segments(query: str, min_chars: int, max_segments: int):
    q = (query or "").strip()
    if not q:
        return []
    if len(q) < min_chars:
        return [q]
    # Sentence-like split for long paragraphs.
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", q) if p.strip()]
    if len(parts) <= 1:
        # Fallback split on commas/newlines if sentence splitting fails.
        parts = [p.strip() for p in re.split(r"[,\\n]+", q) if p.strip()]
    if not parts:
        return [q]
    return parts[:max(1, max_segments)]


def load_line_list(path: str):
    with open(path, "r", encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


def is_disease_sty(sty: str) -> bool:
    s = (sty or "").lower()
    return ("disease" in s) or ("syndrome" in s) or ("pathologic function" in s) or ("neoplastic process" in s)


def is_drug_sty(sty: str) -> bool:
    s = (sty or "").lower()
    return (
        ("pharmacologic substance" in s)
        or ("clinical drug" in s)
        or ("antibiotic" in s)
        or ("hormone" in s)
        or ("enzyme inhibitor" in s)
        or ("organic chemical" in s)
    )


def dominant_group(sty_list):
    if any(is_disease_sty(x) for x in sty_list):
        return "disease"
    if any(is_drug_sty(x) for x in sty_list):
        return "drug"
    return "other"


def canonical_sab(sab: str) -> str:
    s = (sab or "").strip().upper()
    if s in ("MSH", "MESH"):
        return "MSH"
    if s == "OMIM":
        return "OMIM"
    return s


def concept_ids_from_source(sab: str, code: str):
    sab = canonical_sab(sab)
    code = (code or "").strip()
    if not sab or not code:
        return []
    if sab == "MSH":
        return [f"Disease_MESH_{code}", f"Chemical_MESH_{code}"]
    if sab == "OMIM":
        return [f"Disease_OMIM_{code}"]
    return []


def build_source_mappings(mrconso_path: str):
    source_to_cuis = {}
    cui_to_sources = {}
    if not mrconso_path:
        return source_to_cuis, cui_to_sources
    with open(mrconso_path, encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 14:
                continue
            cui = parts[0]
            sab = canonical_sab(parts[11])
            code = parts[13]
            if sab not in ("MSH", "OMIM") or not code:
                continue
            key = (sab, code)
            source_to_cuis.setdefault(key, set()).add(cui)
            cui_to_sources.setdefault(cui, set()).add(key)
    return source_to_cuis, cui_to_sources


def build_rxnorm_flags(mrconso_path: str):
    rxnorm_any = set()
    rxnorm_allowed = set()
    if not mrconso_path:
        return rxnorm_any, rxnorm_allowed
    with open(mrconso_path, encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 13:
                continue
            cui = parts[0]
            sab = parts[11]
            tty = parts[12]
            if sab == "RXNORM":
                rxnorm_any.add(cui)
                if tty in ("IN", "PIN"):
                    rxnorm_allowed.add(cui)
    return rxnorm_any, rxnorm_allowed


def build_mrrel_neighbors(mrrel_path: str, rela_filter_csv: str):
    neighbors = defaultdict(set)
    edge_combos = defaultdict(lambda: defaultdict(set))
    if not mrrel_path:
        return neighbors, edge_combos
    rela_filter = {x.strip() for x in rela_filter_csv.split(",") if x.strip()} if rela_filter_csv else None
    with open(mrrel_path, encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 11:
                continue
            c1 = parts[0]
            c2 = parts[4]
            rel = parts[3]
            rela = parts[7]
            sab = parts[10]
            if rela_filter is not None and rela not in rela_filter:
                continue
            if c1.startswith("C") and c2.startswith("C") and c1 != c2:
                neighbors[c1].add(c2)
                neighbors[c2].add(c1)
                combo = (rel, rela, sab)
                edge_combos[c1][c2].add(combo)
                edge_combos[c2][c1].add(combo)
    return neighbors, edge_combos


def _extract_rows_from_payload(payload, results_key):
    if results_key == "":
        return payload
    rows = payload
    for part in results_key.split("."):
        if isinstance(rows, dict):
            rows = rows.get(part, [])
        else:
            return []
    return rows


def _call_umls_api(query: str, args):
    base = args.umls_api_base_url.rstrip("/")
    if not base:
        return []
    params = {
        args.umls_api_query_param: query,
        args.umls_api_limit_param: str(args.umls_api_limit),
        args.umls_api_size_param: str(args.umls_api_limit),
        args.umls_api_page_param: str(args.umls_api_page),
        args.umls_api_fuzzy_param: str(args.umls_api_fuzzy),
    }
    url = base + args.umls_api_search_path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=args.umls_api_timeout) as resp:
        payload = json.loads(resp.read().decode("utf8"))
    rows = _extract_rows_from_payload(payload, args.umls_api_results_key)
    return rows if isinstance(rows, list) else []


def resolve_with_umls_api(query: str, args):
    rows = _call_umls_api(query, args)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        codes = r.get("codes", [])
        if isinstance(codes, list):
            for c in codes:
                if not isinstance(c, dict):
                    continue
                sab = canonical_sab(str(c.get(args.umls_api_sab_field, "") or c.get("SAB", "") or c.get("rootSource", "") or c.get("source", "") or c.get("sab", "")))
                code = str(c.get(args.umls_api_code_field, "") or c.get("CODE", "") or c.get("code", "") or c.get("ui", "") or c.get("sourceUi", "")).strip()
                if sab in ("MSH", "OMIM") and code:
                    out.append((sab, code))
        sab = canonical_sab(str(r.get(args.umls_api_sab_field, "") or r.get("SAB", "") or r.get("rootSource", "") or r.get("source", "") or r.get("sab", "")))
        code = str(r.get(args.umls_api_code_field, "") or r.get("CODE", "") or r.get("code", "") or r.get("ui", "") or r.get("sourceUi", "")).strip()
        if sab in ("MSH", "OMIM") and code:
            out.append((sab, code))
    return out


def resolve_cuis_with_umls_api(query: str, args):
    rows = _call_umls_api(query, args)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cui = str(r.get("CUI", "") or r.get("cui", "")).strip()
        if cui.startswith("C"):
            out.append(cui)
    return out


def normalize_distance_map(dist_map):
    if not dist_map:
        return {}
    vals = list(dist_map.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {k: 1.0 for k in dist_map}
    return {k: (hi - v) / (hi - lo) for k, v in dist_map.items()}


def create_app(args):
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    cuis, terms, stys = load_metadata(args.metadata)
    sty_to_group = load_semgroup_map(args.semgroups)
    term_to_cui = {}
    cui_to_info = {}
    for cui, term, sty in zip(cuis, terms, stys):
        key = norm_term(term)
        if key not in term_to_cui:
            term_to_cui[key] = cui
        if cui not in cui_to_info:
            cui_to_info[cui] = (term, sty)

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    sap_index = faiss.read_index(args.index)
    if hasattr(sap_index, "hnsw"):
        sap_index.hnsw.efSearch = args.ef_search

    source_to_cuis, cui_to_sources = build_source_mappings(args.mrconso)
    rxnorm_any, rxnorm_allowed = build_rxnorm_flags(args.mrconso)
    mrrel_neighbors, mrrel_edge_combos = build_mrrel_neighbors(args.mrrel, args.mrrel_rela)

    bc_index = None
    bcv_ids = []
    bcv_to_row = {}
    try:
        bc_index = faiss.read_index(args.bioconceptvec_index)
        if hasattr(bc_index, "hnsw"):
            bc_index.hnsw.efSearch = args.ef_search
        bcv_ids = load_line_list(args.bioconceptvec_ids)
        bcv_to_row = {x: i for i, x in enumerate(bcv_ids)}
    except Exception:
        bc_index = None

    cui2_index = None
    cui2_cuis = []
    cui2_to_row = {}
    try:
        cui2_index = faiss.read_index(args.cui2vec_index)
        if hasattr(cui2_index, "hnsw"):
            cui2_index.hnsw.efSearch = args.ef_search
        cui2_cuis = load_line_list(args.cui2vec_cuis)
        cui2_to_row = {c: i for i, c in enumerate(cui2_cuis)}
    except Exception:
        cui2_index = None

    def run_sapbert(query, top_k):
        q_vec = encode_query(query, tokenizer, model, device)
        D, I = sap_index.search(q_vec, max(top_k * 5, top_k))
        dist = {}
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            cui = cuis[idx]
            if cui not in dist or score < dist[cui]:
                dist[cui] = float(score)
        seed_cuis = []
        q_cui = term_to_cui.get(norm_term(query))
        if q_cui:
            seed_cuis = [q_cui]
        return dist, seed_cuis, "Mode: SapBERT."

    def run_bioconceptvec(query, top_k):
        if bc_index is None:
            return {}, [], "BioConceptVec index not loaded."
        seed_ids = []
        seed_cuis = []
        segments = split_query_segments(query, args.chunk_long_query_min_chars, args.chunk_max_segments)
        if not segments:
            return {}, [], "Mode: BioConceptVec. Empty query."
        query_group = "other"
        if args.resolver == "exact":
            for seg in segments:
                q_cui = term_to_cui.get(norm_term(seg))
                if q_cui:
                    seed_cuis.append(q_cui)
                    for sab, code in sorted(cui_to_sources.get(q_cui, set())):
                        seed_ids.extend(concept_ids_from_source(sab, code))
                    _, q_stys = cui_to_info.get(q_cui, ("", []))
                    query_group = dominant_group(q_stys)
            status_msg = f"Mode: BioConceptVec (exact resolver). Segments used: {len(segments)}."
        else:
            try:
                resolved_total = []
                for seg in segments:
                    resolved = resolve_with_umls_api(seg, args)
                    resolved_total.extend(resolved)
                    for sab, code in resolved:
                        seed_ids.extend(concept_ids_from_source(sab, code))
                        seed_cuis.extend(list(source_to_cuis.get((sab, code), [])))
                if resolved_total:
                    top_sab, top_code = resolved_total[0]
                    top_cuis = list(source_to_cuis.get((top_sab, top_code), []))
                    if top_cuis:
                        _, q_stys = cui_to_info.get(top_cuis[0], ("", []))
                        query_group = dominant_group(q_stys)
                status_msg = (
                    f"Mode: BioConceptVec (API resolver). Segments used: {len(segments)}. "
                    f"Resolver returned {len(resolved_total)} source codes; {len(seed_ids)} seed IDs after expansion."
                )
            except Exception as e:
                return {}, [], f"Resolver API error: {e}"

        seed_rows = [bcv_to_row[x] for x in sorted(set(seed_ids)) if x in bcv_to_row]
        if not seed_rows:
            return {}, seed_cuis, status_msg + f" Matched seed IDs in index: {len(seed_rows)}."

        vecs = [bc_index.reconstruct(i) for i in seed_rows]
        q_vec = np.mean(np.vstack(vecs), axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(q_vec)
        D, I = bc_index.search(q_vec, max(top_k * 20, top_k))

        dist = {}
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            concept_id = bcv_ids[idx]
            parts = concept_id.split("_", 2)
            if len(parts) < 3:
                continue
            source = parts[1]
            code = parts[2]
            if source not in ("MESH", "OMIM"):
                continue
            sab = "MSH" if source == "MESH" else source
            for cui in source_to_cuis.get((sab, code), []):
                term, sty_list = cui_to_info.get(cui, (cui, []))
                if args.cross_type_only:
                    ng = dominant_group(sty_list)
                    if query_group == "disease" and ng != "drug":
                        continue
                    if query_group == "drug" and ng != "disease":
                        continue
                if cui not in dist or score < dist[cui]:
                    dist[cui] = float(score)
        return dist, seed_cuis, status_msg

    def run_cui2vec(query, top_k):
        if cui2_index is None:
            return {}, [], "cui2vec index not loaded."
        seed_cuis = []
        segments = split_query_segments(query, args.chunk_long_query_min_chars, args.chunk_max_segments)
        if not segments:
            return {}, [], "Mode: cui2vec. Empty query."
        if args.resolver == "exact":
            for seg in segments:
                q_cui = term_to_cui.get(norm_term(seg))
                if q_cui:
                    seed_cuis.append(q_cui)
            status_msg = f"Mode: cui2vec (exact resolver). Segments used: {len(segments)}."
        else:
            try:
                resolved_total = []
                for seg in segments:
                    resolved_cuis = resolve_cuis_with_umls_api(seg, args)
                    resolved_total.extend(resolved_cuis)
                seed_cuis.extend(resolved_total)
                status_msg = (
                    f"Mode: cui2vec (API resolver). Segments used: {len(segments)}. "
                    f"Resolver returned {len(resolved_total)} CUIs."
                )
            except Exception as e:
                return {}, [], f"Resolver API error: {e}"

        seed_rows = [cui2_to_row[c] for c in sorted(set(seed_cuis)) if c in cui2_to_row]
        if not seed_rows:
            return {}, seed_cuis, status_msg + f" Matched seed CUIs in index: {len(seed_rows)}."

        vecs = [cui2_index.reconstruct(i) for i in seed_rows]
        q_vec = np.mean(np.vstack(vecs), axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(q_vec)
        D, I = cui2_index.search(q_vec, max(top_k * 20, top_k))

        dist = {}
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            cui = cui2_cuis[idx]
            if cui not in dist or score < dist[cui]:
                dist[cui] = float(score)
        return dist, seed_cuis, status_msg

    app = Flask(__name__)

    TEMPLATE = """
    <!doctype html>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>UMLS Similarity Search</title>
    <style>
      :root {
        --bg: #f3f7f5;
        --panel: #ffffff;
        --text: #132119;
        --muted: #4f6557;
        --line: #d3e0d6;
        --brand: #0f6c4d;
        --brand-soft: #e5f4ee;
      }
      body {
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        color: var(--text);
        background:
          radial-gradient(1000px 600px at 100% -10%, #e3f2eb 0%, transparent 60%),
          radial-gradient(900px 500px at -10% 110%, #e9eefb 0%, transparent 50%),
          var(--bg);
      }
      .wrap { max-width: 1150px; margin: 28px auto; padding: 0 16px; }
      .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 10px 26px rgba(17, 36, 26, 0.08);
        padding: 18px;
      }
      h1 { margin: 0 0 16px 0; font-size: 1.6rem; letter-spacing: 0.01em; }
      .form-grid {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr;
        gap: 12px;
      }
      label { font-size: 0.95rem; color: var(--muted); display: block; }
      input, select {
        width: 100%;
        margin-top: 6px;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 11px;
        font-size: 0.95rem;
        background: #fff;
      }
      input:focus, select:focus {
        outline: none;
        border-color: var(--brand);
        box-shadow: 0 0 0 3px rgba(15, 108, 77, 0.12);
      }
      .mode-help {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px;
        background: var(--brand-soft);
        margin-top: 10px;
      }
      .mode-help p { margin: 4px 0; }
      .actions { margin-top: 12px; display: flex; gap: 12px; align-items: center; }
      .btn {
        border: 0;
        border-radius: 10px;
        background: var(--brand);
        color: #fff;
        padding: 10px 14px;
        font-weight: 600;
        cursor: pointer;
      }
      .status {
        margin-top: 12px;
        border-left: 4px solid var(--brand);
        background: #eef7f3;
        padding: 10px 12px;
        border-radius: 8px;
      }
      table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 10px; }
      th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }
      th {
        background: #f7fbf9;
        position: sticky;
        top: 0;
        z-index: 1;
      }
      tr:hover td { background: #f8fcfa; }
      .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 12px; }
      @media (max-width: 900px) {
        .form-grid { grid-template-columns: 1fr; }
      }
    </style>
    <div class="wrap">
    <div class="card">
    <h1>UMLS Similarity Search</h1>
    <form method="post" action="/search">
      <div class="form-grid">
      <label>Query:
        <input type="text" name="query" required value="{{query}}">
      </label>
      <label>Mode:
        <select name="mode">
          <option value="sapbert" {% if selected_mode == "sapbert" %}selected{% endif %}>SapBERT</option>
          <option value="bioconceptvec" {% if selected_mode == "bioconceptvec" %}selected{% endif %}>BioConceptVec</option>
          <option value="cui2vec" {% if selected_mode == "cui2vec" %}selected{% endif %}>cui2vec</option>
          <option value="ensemble" {% if selected_mode == "ensemble" %}selected{% endif %}>Ensemble</option>
        </select>
      </label>
      <label>Number of Results:
        <input type="number" name="top_k" value="{{default_top_k}}" min="1">
      </label>
      </div>
      <div class="mode-help">
        <p><strong>Embedding Sources</strong></p>
        <p><strong>SapBERT:</strong> best when you want terms with similar wording or synonyms.</p>
        <p><strong>BioConceptVec:</strong> best when you want concepts that co-occur in biomedical literature.</p>
        <p><strong>cui2vec:</strong> best when you want concepts that co-occur in clinical and literature data.</p>
        <p><strong>Ensemble:</strong> combines all three and can apply an MRREL relation boost.</p>
      </div>
      <label style="margin-top: 12px;">Score Threshold (optional):
        <input type="number" name="max_score" step="any" placeholder="minimum relevance, e.g. 0.4" value="{{default_max_score}}">
        <small>All modes: keep results with score >= threshold (higher is better).</small>
      </label>
      <div class="actions">
        <input class="btn" type="submit" value="Search">
      </div>
    </form>
    {% if status_msg %}
    <p class="status"><strong>{{status_msg}}</strong></p>
    {% endif %}
    {% if results %}
    <h2>Results for "{{query}}"</h2>
    <div id="sty-filter">
      <label for="sty-select"><strong>Semantic Group:</strong></label>
      <select id="sty-select">
        <option value="">All</option>
        {% for sty in sty_set %}
          <option value="{{sty}}">{{sty}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="table-wrap">
    <table>
      <tr>
        <th>Rank</th><th>CUI</th><th>Term</th><th>Semantic Type</th><th>Score</th>
        {% if selected_mode == "ensemble" %}
        <th>SapBERT</th><th>BioConceptVec</th><th>cui2vec</th><th>MRREL</th>
        {% endif %}
      </tr>
      {% for r in results %}
      <tr class="result-row" data-groups="{{r[4]}}">
        <td>{{loop.index}}</td><td>{{r[0]}}</td><td>{{r[1]}}</td><td>{{r[2]}}</td><td>{{"%.4f"|format(r[3])}}</td>
        {% if selected_mode == "ensemble" %}
        <td>{{"%.4f"|format(r[5])}}</td><td>{{"%.4f"|format(r[6])}}</td><td>{{"%.4f"|format(r[7])}}</td><td>{{"%.4f"|format(r[8])}}</td>
        {% endif %}
      </tr>
      {% endfor %}
    </table>
    </div>
    <script>
    function updateFilter() {
      const selected = document.getElementById('sty-select').value;
      document.querySelectorAll('.result-row').forEach(row => {
        const groups = (row.getAttribute('data-groups') || '').split('||');
        row.style.display = !selected || groups.includes(selected) ? '' : 'none';
      });
    }
    document.getElementById('sty-select').addEventListener('change', updateFilter);
    </script>
    {% endif %}
    </div>
    </div>
    """

    @app.route("/", methods=["GET"])
    def index_page():
        return render_template_string(
            TEMPLATE,
            results=None,
            query="",
            default_top_k=DEFAULT_TOP_K,
            default_max_score="",
            sty_set=[],
            status_msg="",
            selected_mode=args.mode,
        )

    @app.route("/search", methods=["POST"])
    def search():
        query = request.form.get("query", "").strip()
        selected_mode = request.form.get("mode", args.mode).strip().lower()
        if selected_mode not in ("sapbert", "bioconceptvec", "cui2vec", "ensemble"):
            selected_mode = args.mode

        try:
            top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
        except ValueError:
            top_k = DEFAULT_TOP_K
        max_score_in = request.form.get("max_score", "").strip()
        try:
            max_score = float(max_score_in) if max_score_in else None
        except ValueError:
            max_score = None

        if selected_mode == "sapbert":
            dist, _, status_msg = run_sapbert(query, top_k)
            rel = normalize_distance_map(dist)
            ranking = sorted(rel.items(), key=lambda x: x[1], reverse=True)
            status_msg += " Scores shown as normalized relevance."

        elif selected_mode == "bioconceptvec":
            dist, _, status_msg = run_bioconceptvec(query, top_k)
            rel = normalize_distance_map(dist)
            ranking = sorted(rel.items(), key=lambda x: x[1], reverse=True)
            status_msg += " Scores shown as normalized relevance."

        elif selected_mode == "cui2vec":
            dist, _, status_msg = run_cui2vec(query, top_k)
            rel = normalize_distance_map(dist)
            ranking = sorted(rel.items(), key=lambda x: x[1], reverse=True)
            status_msg += " Scores shown as normalized relevance."

        else:
            d1, s1, st1 = run_sapbert(query, top_k * 4)
            d2, s2, st2 = run_bioconceptvec(query, top_k * 4)
            d3, s3, st3 = run_cui2vec(query, top_k * 4)
            n1 = normalize_distance_map(d1)
            n2 = normalize_distance_map(d2)
            n3 = normalize_distance_map(d3)
            w1 = args.ensemble_sapbert_weight
            w2 = args.ensemble_bioconceptvec_weight
            w3 = args.ensemble_cui2vec_weight
            fused = {}
            contrib = {}
            all_cuis = set(n1) | set(n2) | set(n3)
            seed_cuis = set(s1) | set(s2) | set(s3)
            for cui in all_cuis:
                c1 = w1 * n1.get(cui, 0.0)
                c2 = w2 * n2.get(cui, 0.0)
                c3 = w3 * n3.get(cui, 0.0)
                boost = 0.0
                score = c1 + c2 + c3
                if (args.mrrel_boost > 0 or args.mrrel_count_boost > 0) and mrrel_neighbors and seed_cuis:
                    neigh = mrrel_neighbors.get(cui, set())
                    if any(s in neigh for s in seed_cuis):
                        relation_count = sum(len(mrrel_edge_combos.get(cui, {}).get(s, set())) for s in seed_cuis)
                        boost = args.mrrel_boost + (args.mrrel_count_boost * relation_count)
                        score += boost
                fused[cui] = score
                contrib[cui] = (c1, c2, c3, boost)
            ranking = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            status_msg = (
                "Mode: Ensemble. "
                f"Candidates sapbert={len(d1)}, bioconceptvec={len(d2)}, cui2vec={len(d3)}. "
                f"Weights=({w1:.2f},{w2:.2f},{w3:.2f}), "
                f"MRREL base boost={args.mrrel_boost:.2f}, MRREL combo boost={args.mrrel_count_boost:.2f}."
            )

        results = []
        sty_set = set()
        for cui, score in ranking:
            if len(results) >= top_k:
                break
            # Runtime filter: if a concept has any RXNORM atoms, keep it only
            # when at least one RXNORM atom has TTY=IN or TTY=PIN.
            if cui in rxnorm_any and cui not in rxnorm_allowed:
                continue
            term, sty_list = cui_to_info.get(cui, (cui, []))
            if not sty_list:
                sty_list = ["Unknown"]
            if sty_list == ["Unknown"]:
                continue
            if any(sty in EXCLUDED_SEMANTIC_TYPES for sty in sty_list):
                continue
            if max_score is not None:
                if score < max_score:
                    continue
            sty_str = ", ".join(sty_list)
            groups = sorted({sty_to_group.get(sty, "Unknown") for sty in sty_list})
            if not groups:
                groups = ["Unknown"]
            if selected_mode == "ensemble":
                c1, c2, c3, boost = contrib.get(cui, (0.0, 0.0, 0.0, 0.0))
            else:
                c1, c2, c3, boost = (0.0, 0.0, 0.0, 0.0)
            results.append((cui, term, sty_str, float(score), "||".join(groups), c1, c2, c3, boost))
            sty_set.update(groups)

        return render_template_string(
            TEMPLATE,
            results=results,
            query=query,
            default_top_k=top_k,
            default_max_score=max_score_in,
            sty_set=sorted(sty_set),
            status_msg=status_msg + f" Returned {len(results)} results.",
            selected_mode=selected_mode,
        )

    return app


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
