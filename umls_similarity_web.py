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
RESOLVER_EXCLUDED_SEMANTIC_TYPES = {
    "Biomedical Occupation or Discipline",
    "Clinical Attribute",
    "Finding",
    "Health Care Activity",
    "Intellectual Product",
    "Medical Device",
    "Professional or Occupational Group",
}
RESOLVER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "those", "to",
    "was", "were", "which", "while", "with", "within",
}
RESOLVER_GENERIC_TOKENS = {
    "cancer", "cell", "cells", "change", "changes", "clinical", "communication", "complexity",
    "course", "disease", "diseases", "disorder", "disorders", "dynamic", "effective",
    "editorial", "environmental", "evolving", "function", "genetic", "management",
    "involve", "involved", "involves", "mechanism", "mechanisms", "molecular", "nature", "network", "networks", "obstacle",
    "obstacles", "pathway", "pathways", "patient", "patients", "plastic", "presence",
    "progression", "research", "resistance", "review", "reviews", "series", "signal",
    "signaling", "spread", "strategy", "strategies", "study", "studies", "substantial",
    "therapeutic", "therapy", "therapies", "transition", "treatment", "treatments", "tumor", "tumors",
    "tumour", "tumours", "understanding", "variability",
}
TOKEN_NORMALIZATION_ALIASES = {
    "antibiotics": "antibiotic",
    "drug": "medication",
    "drugs": "medication",
    "medication": "medication",
    "medications": "medication",
    "medicine": "medication",
    "medicines": "medication",
}
LOW_VALUE_MODIFIER_TOKENS = {
    "analysis", "assessment", "attack", "bilateral", "check", "clinical", "complicated", "contralateral",
    "determination", "evaluation", "exam", "examination", "formation", "history", "management",
    "measure", "measurement", "medical", "method", "monitor", "monitoring", "panel", "prior", "probability",
    "reading", "recurrent", "reported", "secondary", "severe", "sign", "signs", "specimen",
    "symptom", "symptoms", "tak", "taking", "therapeutic", "therapy", "unspecified",
}
LOW_VALUE_QUERY_TOKENS = {
    "analysis", "assessment", "check", "diagnosis", "diagnostic", "exam", "examination",
    "history", "management", "measure", "measurement", "method", "monitor", "monitoring",
    "procedure", "reading", "screen", "screening", "sign", "signs", "symptom", "symptoms",
    "test", "testing",
}
RESOLVER_EDGE_TOKENS = RESOLVER_STOPWORDS | {
    "ability", "arises", "arise", "based", "core", "course", "create", "creates", "creating",
    "critical", "dedicated", "despite", "diverse", "drive", "driven", "drives", "effective",
    "emerging", "evolving", "highlighted", "impacting", "influences", "intrinsic", "making",
    "involve", "involved", "involves", "nature", "necessity", "new", "notably", "obstacles",
    "part", "presents", "presence",
    "precise", "proposing", "role", "roles", "series", "substantial", "such", "understanding",
}

TREATMENT_INTENT_TOKENS = {
    "treat", "treatment", "therapy", "therapeutic", "manage", "management",
    "medication", "medicine", "drug", "drugs", "agent", "agents",
}
MEDICATION_INTENT_TOKENS = {
    "medication", "medicine", "drug", "drugs", "antibiotic", "antibiotics",
    "pill", "pills", "tablet", "tablets", "statin", "statins",
}
DEVICE_INTENT_TOKENS = {
    "inhaler", "inhalers", "device", "devices", "pump", "pumps", "injector", "injectors",
}
DIAGNOSTIC_INTENT_TOKENS = {
    "diagnosis", "diagnostic", "screen", "screening", "test", "testing", "detect",
}

LOW_SIGNAL_GROUP_MULTIPLIER = {
    "Concepts & Ideas": 0.65,
    "Geographic Areas": 0.45,
    "Objects": 0.60,
    "Occupations": 0.45,
    "Organizations": 0.45,
}

QUERY_REWRITE_RULES = (
    (re.compile(r"\bmeds\b", re.I), "medication", 0.02),
    (re.compile(r"\bmed\b", re.I), "medication", 0.02),
    (re.compile(r"\bpills?\b", re.I), "medication", 0.03),
    (re.compile(r"\buti\b", re.I), "urinary tract infection", 0.02),
    (re.compile(r"\bcopd\b", re.I), "chronic obstructive pulmonary disease", 0.02),
)


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

    p.add_argument("--resolver", choices=["exact", "sapbert", "umls_api"], default="sapbert",
                   help="Resolver used for bioconceptvec/cui2vec/ensemble modes (sapbert works best for free-text queries; umls_api kept as alias)")
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
    p.add_argument("--resolver_chunk_max_words", type=int, default=6,
                   help="When the resolver misses on a segment, retry with smaller word chunks up to this size")
    p.add_argument("--resolver_chunk_min_words", type=int, default=2,
                   help="Smallest word chunk size to try for resolver fallback")
    p.add_argument("--resolver_chunk_max_queries", type=int, default=12,
                   help="Maximum resolver API queries to spend per segment, including the original segment")
    p.add_argument("--resolver_max_hits_per_segment", type=int, default=3,
                   help="Maximum validated resolver hits to keep per segment for BioConceptVec/cui2vec seeding")

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
_token_re = re.compile(r"[a-z0-9]+")


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


def rewrite_query_variants(query: str):
    q = _space_re.sub(" ", (query or "").strip())
    if not q:
        return []

    variants = [(q, 0.0)]
    seen = {norm_term(q)}

    def add_variant(text: str, penalty: float):
        cleaned = _space_re.sub(" ", (text or "").strip())
        if not cleaned:
            return
        key = norm_term(cleaned)
        if key in seen:
            return
        seen.add(key)
        variants.append((cleaned, penalty))

    for pattern, replacement, penalty in QUERY_REWRITE_RULES:
        if pattern.search(q):
            add_variant(pattern.sub(replacement, q), penalty)

    lowered = q.lower()
    if "blood pressure" in lowered and any(token in lowered for token in ("med", "meds", "medicine", "medication", "drug", "drugs", "pill", "pills")):
        add_variant(re.sub(r"(?i)blood pressure", "hypertension", q), 0.05)
        add_variant(re.sub(r"(?i)blood pressure", "high blood pressure", q), 0.04)
    if ("cholesterol" in lowered) and any(token in lowered for token in ("med", "meds", "medicine", "medication", "drug", "drugs", "pill", "pills")):
        add_variant(re.sub(r"(?i)cholesterol", "statin", q), 0.07)
        add_variant(re.sub(r"(?i)cholesterol", "cholesterol lowering", q), 0.05)
    if ("blood sugar" in lowered or "glucose" in lowered) and any(token in lowered for token in ("med", "meds", "medicine", "medication", "drug", "drugs", "pill", "pills")):
        add_variant(re.sub(r"(?i)blood sugar", "diabetes", q), 0.06)
        add_variant(re.sub(r"(?i)blood sugar", "blood glucose", q), 0.03)
    if "bladder infection" in lowered:
        add_variant(re.sub(r"(?i)bladder infection", "urinary tract infection", q), 0.04)
        add_variant(re.sub(r"(?i)bladder infection", "cystitis", q), 0.05)
    if "acid reflux" in lowered:
        add_variant(re.sub(r"(?i)acid reflux", "gastroesophageal reflux", q), 0.04)
        add_variant(re.sub(r"(?i)acid reflux", "heartburn", q), 0.05)
    if "kidney stone" in lowered:
        add_variant(re.sub(r"(?i)kidney stone", "nephrolithiasis", q), 0.05)
    if "heart failure" in lowered and any(token in lowered for token in ("med", "meds", "medicine", "medication", "drug", "drugs", "pill", "pills")):
        add_variant(re.sub(r"(?i)heart failure", "heart failure treatment", q), 0.05)
    if "asthma inhaler" in lowered:
        add_variant(re.sub(r"(?i)asthma inhaler", "bronchodilator inhaler", q), 0.04)
    return variants


def tokenize_text(text: str):
    return _token_re.findall((text or "").lower())


def normalize_token(token: str):
    t = (token or "").lower()
    alias = TOKEN_NORMALIZATION_ALIASES.get(t)
    if alias:
        return alias
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 6 and t.endswith("ers") and t[-4] == t[-5]:
        t = t[:-3]
    elif len(t) > 5 and t.endswith("er") and t[-3] == t[-4]:
        t = t[:-2]
    elif len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 6 and t.endswith("ality"):
        t = t[:-5]
    elif len(t) > 5 and t.endswith("ity"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("al"):
        t = t[:-2]
    elif len(t) > 5 and t.endswith("tion"):
        t = t[:-4]
    elif len(t) > 4 and t.endswith("s"):
        t = t[:-1]
    return TOKEN_NORMALIZATION_ALIASES.get(t, t)


NORMALIZED_LOW_VALUE_MODIFIER_TOKENS = {normalize_token(token) for token in LOW_VALUE_MODIFIER_TOKENS}
NORMALIZED_LOW_VALUE_QUERY_TOKENS = {normalize_token(token) for token in LOW_VALUE_QUERY_TOKENS}


def is_generic_resolver_token(token: str):
    t = (token or "").lower()
    return t in RESOLVER_GENERIC_TOKENS or normalize_token(t) in RESOLVER_GENERIC_TOKENS


def analyze_resolver_text(text: str):
    tokens = tokenize_text(text)
    content_tokens = [t for t in tokens if len(t) >= 3 and t not in RESOLVER_STOPWORDS]
    content_sigs = {normalize_token(t) for t in content_tokens if normalize_token(t)}
    distinctive_sigs = {
        normalize_token(t)
        for t in content_tokens
        if normalize_token(t) and not is_generic_resolver_token(t)
    }
    generic_count = sum(1 for t in content_tokens if is_generic_resolver_token(t))
    return {
        "tokens": tokens,
        "content_count": len(content_tokens),
        "content_sigs": content_sigs,
        "distinctive_sigs": distinctive_sigs,
        "generic_count": generic_count,
    }


def text_content_signatures(text: str):
    sigs = set()
    for token in tokenize_text(text):
        if token in RESOLVER_STOPWORDS:
            continue
        sig = normalize_token(token)
        if sig:
            sigs.add(sig)
    return sigs


def _query_text_options(query_text):
    if isinstance(query_text, (list, tuple)):
        return [str(x) for x in query_text if str(x).strip()]
    text = str(query_text or "").strip()
    return [text] if text else []


def lexical_overlap_features(query_text: str, candidate_text: str):
    candidate_sigs = text_content_signatures(candidate_text)
    candidate_total = max(1, len(candidate_sigs))
    best = None
    for query_option in _query_text_options(query_text):
        query_info = analyze_resolver_text(query_option)
        shared_content = query_info["content_sigs"] & candidate_sigs
        shared_distinctive = query_info["distinctive_sigs"] & candidate_sigs
        extra_candidate = candidate_sigs - query_info["content_sigs"]
        extra_low_value = {sig for sig in extra_candidate if sig in NORMALIZED_LOW_VALUE_MODIFIER_TOKENS}
        query_low_value = query_info["content_sigs"] & NORMALIZED_LOW_VALUE_QUERY_TOKENS
        norm_query = norm_term(query_option)
        norm_candidate = norm_term(candidate_text)
        exact_match = bool(norm_query and norm_candidate and norm_query == norm_candidate)
        substring_match = bool(
            norm_query
            and norm_candidate
            and not exact_match
            and (norm_query in norm_candidate or norm_candidate in norm_query)
        )
        query_total = max(1, len(query_info["content_sigs"]))
        current = {
            "exact_match": exact_match,
            "substring_match": substring_match,
            "shared_content": len(shared_content),
            "shared_distinctive": len(shared_distinctive),
            "query_coverage": len(shared_content) / query_total,
            "candidate_coverage": len(shared_content) / candidate_total,
            "query_content_total": len(query_info["content_sigs"]),
            "extra_candidate": len(extra_candidate),
            "extra_low_value": len(extra_low_value),
            "query_has_low_value_focus": bool(query_low_value),
        }
        rank = (
            int(current["exact_match"]),
            int(current["substring_match"]),
            current["shared_distinctive"],
            current["shared_content"],
            current["query_coverage"],
            -current["extra_low_value"],
            current["candidate_coverage"],
        )
        if best is None or rank > best[0]:
            best = (rank, current)
    if best is None:
        return {
            "exact_match": False,
            "substring_match": False,
            "shared_content": 0,
            "shared_distinctive": 0,
            "query_coverage": 0.0,
            "candidate_coverage": 0.0,
            "query_content_total": 0,
            "extra_candidate": 0,
            "extra_low_value": 0,
            "query_has_low_value_focus": False,
        }
    return best[1]


def lexical_relevance_multiplier(query_text: str, candidate_text: str):
    feats = lexical_overlap_features(query_text, candidate_text)
    if feats["query_content_total"] == 0:
        return 1.0
    if feats["exact_match"]:
        return 1.30

    score = 0.78 + (0.22 * feats["query_coverage"]) + (0.05 * feats["candidate_coverage"])
    if feats["shared_distinctive"] > 0:
        score += 0.10
    elif feats["shared_content"] > 0:
        score += 0.05
    elif feats["query_content_total"] >= 2:
        score -= 0.12
    if feats["substring_match"]:
        score += 0.08
    if feats["query_content_total"] <= 3 and feats["extra_candidate"] > 0:
        score -= min(0.10, 0.03 * feats["extra_candidate"])
    if feats["extra_low_value"] > 0 and not feats["query_has_low_value_focus"]:
        score -= min(0.24, 0.12 * feats["extra_low_value"])
    return max(0.55, min(1.20, score))


def ensemble_source_multiplier(query_text: str, candidate_text: str, c1: float, c2: float, c3: float):
    feats = lexical_overlap_features(query_text, candidate_text)
    total = float(c1) + float(c2) + float(c3)
    if total <= 0.0:
        return 1.0

    sap_share = float(c1) / total
    query_total = feats["query_content_total"]
    coverage = feats["query_coverage"]

    if query_total <= 1:
        return 1.0

    if feats["exact_match"]:
        return 1.12 if sap_share > 0.0 else 1.04

    mult = 1.0
    if coverage < 0.75:
        if sap_share < 0.20:
            mult *= 0.58
        elif sap_share < 0.35:
            mult *= 0.74
        elif sap_share < 0.50:
            mult *= 0.88

    if coverage <= 0.50 and not feats["substring_match"]:
        mult *= 0.90

    if feats["shared_distinctive"] == 0 and sap_share < 0.35:
        mult *= 0.88

    return max(0.45, min(1.12, mult))


def sapbert_anchor_strength(query_text: str, normalized_scores, cui_to_info):
    if not normalized_scores:
        return 0.0
    best_cui, best_score = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)[0]
    term, _ = cui_to_info.get(best_cui, (best_cui, []))
    feats = lexical_overlap_features(query_text, term)
    if feats["query_content_total"] <= 1:
        return 0.0
    if feats["exact_match"]:
        return 0.55
    if feats["substring_match"]:
        return 0.55 if feats["query_coverage"] >= 1.0 else 0.45
    if feats["shared_distinctive"] > 0 and feats["query_coverage"] >= 0.5 and best_score >= 0.9:
        return 0.40
    if feats["shared_content"] > 0 and best_score >= 0.9:
        return 0.25
    return 0.0


def sapbert_seed_semantic_priority(sty_list):
    lower = [(sty or "").lower() for sty in (sty_list or [])]
    if any("hazardous or poisonous substance" in sty for sty in lower):
        return 0
    if any(is_drug_sty(sty) for sty in lower):
        return 4
    if any(is_disease_sty(sty) for sty in lower):
        return 3
    if any("therapeutic or preventive procedure" in sty or "diagnostic procedure" in sty for sty in lower):
        return 3
    if any("finding" in sty or "sign or symptom" in sty or "pathologic function" in sty for sty in lower):
        return 2
    return 1


def sapbert_seed_rank(query_text: str, candidate_text: str, sty_list, distance: float):
    feats = lexical_overlap_features(query_text, candidate_text)
    has_overlap = int(
        feats["exact_match"]
        or feats["substring_match"]
        or feats["shared_distinctive"] > 0
        or feats["shared_content"] > 0
    )
    return (
        has_overlap,
        sapbert_seed_semantic_priority(sty_list),
        int(feats["exact_match"]),
        int(feats["substring_match"]),
        feats["shared_distinctive"],
        feats["shared_content"],
        feats["query_coverage"],
        feats["candidate_coverage"],
        -float(distance),
    )


def select_sapbert_seed_cuis_for_text(query_text: str, raw_scores, cui_to_info, limit: int):
    ranked = []
    seen = set()
    for cui, distance in sorted(raw_scores.items(), key=lambda x: x[1]):
        if cui in seen:
            continue
        seen.add(cui)
        term, sty_list = cui_to_info.get(cui, (cui, []))
        if any(sty in RESOLVER_EXCLUDED_SEMANTIC_TYPES for sty in sty_list):
            continue
        rank = sapbert_seed_rank(query_text, term, sty_list, float(distance))
        ranked.append((rank, cui))
    if not ranked:
        return []
    ranked.sort(key=lambda x: x[0], reverse=True)
    overlap_ranked = [item for item in ranked if item[0][0] > 0]
    chosen = overlap_ranked if overlap_ranked else ranked
    return [cui for _, cui in chosen[:max(1, limit)]]


def clean_resolver_chunk(text: str):
    words = [w.strip(" \t\r\n,;:.!?()[]{}\"'") for w in (text or "").split()]
    words = [w for w in words if w]
    while words and (normalize_token(words[0]) in RESOLVER_EDGE_TOKENS or len(words[0]) <= 2):
        words.pop(0)
    while words and (normalize_token(words[-1]) in RESOLVER_EDGE_TOKENS or len(words[-1]) <= 2):
        words.pop()
    return _space_re.sub(" ", " ".join(words)).strip()


def is_informative_resolver_chunk(text: str):
    info = analyze_resolver_text(text)
    if info["content_count"] < 2:
        return False
    if not info["distinctive_sigs"]:
        return False
    if len(info["distinctive_sigs"]) == 1 and info["content_count"] > 4:
        return False
    if info["generic_count"] >= info["content_count"]:
        return False
    return True


def build_resolver_query_levels(query: str, max_words: int, min_words: int, max_queries: int):
    q = _space_re.sub(" ", (query or "").strip())
    if not q:
        return []

    max_queries = max(1, max_queries)
    levels = [[q]]
    if max_queries == 1:
        return levels

    seen = {q}
    min_words = max(1, min_words)
    max_words = max(min_words, max_words)
    candidate_count = 1
    direct_level = []
    level_map = {}

    phrases = [clean_resolver_chunk(p) for p in re.split(r"(?i)[,;]|\b(?:and|or|which|that|including|notably|while)\b", q)]
    for phrase in phrases:
        if not phrase or phrase in seen:
            continue
        words = phrase.split()
        if len(words) <= max_words:
            if is_informative_resolver_chunk(phrase):
                direct_level.append(phrase)
                seen.add(phrase)
                candidate_count += 1
                if candidate_count >= max_queries:
                    break
            continue

        upper = min(max_words, len(words))
        for size in range(upper, min_words - 1, -1):
            if candidate_count >= max_queries:
                break
            for start in range(0, len(words) - size + 1):
                chunk = clean_resolver_chunk(" ".join(words[start:start + size]))
                if not chunk or chunk in seen or not is_informative_resolver_chunk(chunk):
                    continue
                level_map.setdefault(size, []).append(chunk)
                seen.add(chunk)
                candidate_count += 1
                if candidate_count >= max_queries:
                    break
        if candidate_count >= max_queries:
            break

    if direct_level:
        direct_level.sort(
            key=lambda chunk: (
                len(analyze_resolver_text(chunk)["distinctive_sigs"]),
                -analyze_resolver_text(chunk)["generic_count"],
                len(chunk.split()),
            ),
            reverse=True,
        )
        levels.append(direct_level)
    for size in sorted(level_map.keys(), reverse=True):
        levels.append(level_map[size])
    return levels


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


def is_therapeutic_agent_sty(sty: str) -> bool:
    s = (sty or "").lower()
    return (
        ("pharmacologic substance" in s)
        or ("clinical drug" in s)
        or ("antibiotic" in s)
        or ("hormone" in s)
        or ("enzyme inhibitor" in s)
    )


def is_device_sty(sty: str) -> bool:
    s = (sty or "").lower()
    return ("medical device" in s) or ("drug delivery device" in s)


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


def _call_umls_api(query: str, args, fuzzy_override=None):
    base = args.umls_api_base_url.rstrip("/")
    if not base:
        return []
    params = {
        args.umls_api_query_param: query,
        args.umls_api_limit_param: str(args.umls_api_limit),
        args.umls_api_size_param: str(args.umls_api_limit),
        args.umls_api_page_param: str(args.umls_api_page),
    }
    if args.umls_api_fuzzy_param:
        fuzzy_value = args.umls_api_fuzzy if fuzzy_override is None else fuzzy_override
        params[args.umls_api_fuzzy_param] = str(fuzzy_value)
    url = base + args.umls_api_search_path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=args.umls_api_timeout) as resp:
        payload = json.loads(resp.read().decode("utf8"))
    rows = _extract_rows_from_payload(payload, args.umls_api_results_key)
    return rows if isinstance(rows, list) else []


def _extract_semantic_types(row):
    sty = row.get("STY", row.get("sty", [])) if isinstance(row, dict) else []
    if isinstance(sty, list):
        return [str(x) for x in sty if str(x)]
    if isinstance(sty, str) and sty.strip():
        return [sty.strip()]
    return []


def _row_text_signatures(row):
    texts = []
    if isinstance(row, dict):
        pref = str(row.get("preferred_name", "") or row.get("name", "") or "").strip()
        if pref:
            texts.append(pref)
        codes = row.get("codes", [])
        if isinstance(codes, list):
            for c in codes:
                if not isinstance(c, dict):
                    continue
                c_pref = str(c.get("preferred_name", "") or "").strip()
                if c_pref:
                    texts.append(c_pref)
                strings = c.get("strings", [])
                if isinstance(strings, list):
                    texts.extend(str(x) for x in strings if str(x))
    sigs = set()
    for text in texts:
        for token in tokenize_text(text):
            if token in RESOLVER_STOPWORDS:
                continue
            sig = normalize_token(token)
            if sig:
                sigs.add(sig)
    return sigs


def _resolver_score_value(row):
    try:
        return float(row.get("_customScore") or 0.0)
    except Exception:
        return 0.0


def _is_truthy_string(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def _should_try_exact_first(query_info, is_root_query):
    if is_root_query:
        return False
    return len(query_info["tokens"]) <= 4 and bool(query_info["distinctive_sigs"])


def _should_allow_fuzzy(query_info, is_root_query, args):
    if is_root_query:
        return _is_truthy_string(args.umls_api_fuzzy)
    if not _is_truthy_string(args.umls_api_fuzzy):
        return False
    if not query_info["distinctive_sigs"]:
        return False
    return query_info["generic_count"] < query_info["content_count"]


def _score_resolver_row(query_text: str, row):
    query_info = analyze_resolver_text(query_text)
    sty_list = _extract_semantic_types(row)
    if any(sty in RESOLVER_EXCLUDED_SEMANTIC_TYPES for sty in sty_list):
        return None

    row_sigs = _row_text_signatures(row)
    if not row_sigs:
        return None

    distinct_shared = query_info["distinctive_sigs"] & row_sigs
    if not distinct_shared:
        return None

    content_shared = query_info["content_sigs"] & row_sigs
    if len(content_shared) == 1 and query_info["content_count"] > 4:
        return None

    norm_query = norm_term(query_text)
    exact_name_match = 0
    preferred = str(row.get("preferred_name", "") or row.get("name", "") or "").strip()
    if preferred and norm_term(preferred) == norm_query:
        exact_name_match = 1
    codes = row.get("codes", [])
    if isinstance(codes, list):
        for c in codes:
            if not isinstance(c, dict):
                continue
            strings = c.get("strings", [])
            if isinstance(strings, list) and any(norm_term(str(s)) == norm_query for s in strings):
                exact_name_match = 1
                break

    match_type = str(row.get("matchType", "") or "").strip().lower()
    exact_bonus = 1 if match_type == "exact" else 0
    return (
        exact_name_match,
        len(distinct_shared),
        len(content_shared),
        exact_bonus,
        _resolver_score_value(row),
    )


def _row_cui(row):
    if not isinstance(row, dict):
        return ""
    return str(row.get("CUI", "") or row.get("cui", "")).strip()


def _extract_source_codes_from_row(row, args):
    out = []
    if not isinstance(row, dict):
        return out
    codes = row.get("codes", [])
    if isinstance(codes, list):
        for c in codes:
            if not isinstance(c, dict):
                continue
            sab = canonical_sab(str(c.get(args.umls_api_sab_field, "") or c.get("SAB", "") or c.get("rootSource", "") or c.get("source", "") or c.get("sab", "")))
            code = str(c.get(args.umls_api_code_field, "") or c.get("CODE", "") or c.get("code", "") or c.get("ui", "") or c.get("sourceUi", "")).strip()
            if sab in ("MSH", "OMIM") and code:
                out.append((sab, code))
    sab = canonical_sab(str(row.get(args.umls_api_sab_field, "") or row.get("SAB", "") or row.get("rootSource", "") or row.get("source", "") or row.get("sab", "")))
    code = str(row.get(args.umls_api_code_field, "") or row.get("CODE", "") or row.get("code", "") or row.get("ui", "") or row.get("sourceUi", "")).strip()
    if sab in ("MSH", "OMIM") and code:
        out.append((sab, code))
    return out


def _extract_cui_from_row(row):
    cui = _row_cui(row)
    return cui if cui.startswith("C") else ""


def _resolve_with_umls_api_fallback(query: str, args):
    kept_rows = []
    api_queries_tried = 0
    hit_queries = []
    levels = build_resolver_query_levels(
        query,
        args.resolver_chunk_max_words,
        args.resolver_chunk_min_words,
        args.resolver_chunk_max_queries,
    )
    budget = max(1, args.resolver_chunk_max_queries)
    for level_idx, level in enumerate(levels):
        if api_queries_tried >= budget:
            break
        level_candidates = []
        level_hits = set()
        for subquery in level:
            if api_queries_tried >= budget:
                break
            query_info = analyze_resolver_text(subquery)
            modes = []
            if _should_try_exact_first(query_info, is_root_query=(level_idx == 0)):
                modes.append("false")
            if _should_allow_fuzzy(query_info, is_root_query=(level_idx == 0), args=args):
                fuzzy_mode = None if level_idx == 0 else str(args.umls_api_fuzzy)
                modes.append(fuzzy_mode)
            elif not modes:
                modes.append("false")

            seen_modes = set()
            ordered_modes = []
            for mode in modes:
                key = "__default__" if mode is None else str(mode).lower()
                if key in seen_modes:
                    continue
                seen_modes.add(key)
                ordered_modes.append(mode)

            matched_rows = []
            for mode in ordered_modes:
                if api_queries_tried >= budget:
                    break
                rows = _call_umls_api(subquery, args, fuzzy_override=mode)
                api_queries_tried += 1
                scored = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    score = _score_resolver_row(subquery, row)
                    if score is None:
                        continue
                    scored.append((score, row))
                if scored:
                    matched_rows = scored
                    break

            if matched_rows:
                level_hits.add(subquery)
                for score, row in matched_rows:
                    level_candidates.append((score, subquery, row))

        if level_candidates:
            seen_cuis = set()
            level_candidates.sort(key=lambda x: x[0], reverse=True)
            for score, subquery, row in level_candidates:
                cui = _row_cui(row)
                cui_key = cui or f"ROW::{subquery}::{row.get('preferred_name', '')}"
                if cui_key in seen_cuis:
                    continue
                seen_cuis.add(cui_key)
                kept_rows.append(row)
                if len(kept_rows) >= max(1, args.resolver_max_hits_per_segment):
                    break
            hit_queries.extend(sorted(level_hits))
            break
    return kept_rows, api_queries_tried, hit_queries


def resolve_with_umls_api(query: str, args):
    return _resolve_with_umls_api_fallback(query, args)


def resolve_cuis_with_umls_api(query: str, args):
    return _resolve_with_umls_api_fallback(query, args)


def normalize_distance_map(dist_map):
    if not dist_map:
        return {}
    vals = list(dist_map.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {k: 1.0 for k in dist_map}
    return {k: (hi - v) / (hi - lo) for k, v in dist_map.items()}


def normalize_weight_map(weight_map):
    positive = {k: float(v) for k, v in weight_map.items() if float(v) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {k: 0.0 for k in weight_map}
    return {k: positive.get(k, 0.0) / total for k in weight_map}


def summarize_status_detail(status_msg: str):
    text = (status_msg or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[.])\s+", text) if part.strip()]
    for part in parts:
        lowered = part.lower()
        if (
            "requires --mrconso" in lowered
            or "not loaded" in lowered
            or "no exact term match found" in lowered
            or "matched seed ids in index: 0" in lowered
            or "matched seed cuis in index: 0" in lowered
        ):
            return part
    return ""


def search_sapbert_with_variants(query, search_k, tokenizer, model, device, sap_index, cuis):
    merged = {}
    variants = rewrite_query_variants(query) or [(query, 0.0)]
    for variant_text, variant_penalty in variants:
        q_vec = encode_query(variant_text, tokenizer, model, device)
        D, I = sap_index.search(q_vec, search_k)
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            cui = cuis[idx]
            adjusted = float(score) + float(variant_penalty)
            if cui not in merged or adjusted < merged[cui]:
                merged[cui] = adjusted
    return merged, variants


def infer_query_semantic_profile(query, seed_cuis, term_to_cui, cui_to_info, sty_to_group):
    tokens = set(tokenize_text(query))
    treatment_intent = bool(tokens & TREATMENT_INTENT_TOKENS)
    medication_intent = bool(tokens & MEDICATION_INTENT_TOKENS)
    device_intent = bool(tokens & DEVICE_INTENT_TOKENS)
    diagnostic_intent = bool(tokens & DIAGNOSTIC_INTENT_TOKENS)

    group_counts = defaultdict(int)
    for cui in sorted(set(seed_cuis or [])):
        _, sty_list = cui_to_info.get(cui, ("", []))
        for sty in sty_list:
            group = sty_to_group.get(sty)
            if group:
                group_counts[group] += 1

    if not group_counts:
        q_cui = term_to_cui.get(norm_term(query or ""))
        if q_cui:
            _, sty_list = cui_to_info.get(q_cui, ("", []))
            for sty in sty_list:
                group = sty_to_group.get(sty)
                if group:
                    group_counts[group] += 1

    dominant_group = None
    if group_counts:
        dominant_group = sorted(group_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    if medication_intent and not device_intent and dominant_group in (
        None,
        "Activities & Behaviors",
        "Concepts & Ideas",
        "Physiology",
        "Procedures",
    ):
        dominant_group = "Chemicals & Drugs"
    elif diagnostic_intent and dominant_group in (None, "Chemicals & Drugs", "Physiology"):
        dominant_group = "Procedures"
    elif device_intent and not medication_intent and dominant_group is None:
        dominant_group = "Objects"

    return {
        "dominant_group": dominant_group,
        "treatment_intent": treatment_intent,
        "medication_intent": medication_intent,
        "device_intent": device_intent,
        "diagnostic_intent": diagnostic_intent,
    }


def intent_match_multiplier(sty_list, query_profile):
    styles = sty_list or []
    medication_intent = query_profile.get("medication_intent", False)
    device_intent = query_profile.get("device_intent", False)
    diagnostic_intent = query_profile.get("diagnostic_intent", False)

    if not (medication_intent or device_intent or diagnostic_intent):
        return 1.0

    has_therapeutic_agent = any(is_therapeutic_agent_sty(sty) for sty in styles)
    has_device = any(is_device_sty(sty) for sty in styles)
    has_disease = any(is_disease_sty(sty) for sty in styles)
    has_diagnostic = any(
        ("diagnostic procedure" in (sty or "").lower())
        or ("laboratory procedure" in (sty or "").lower())
        for sty in styles
    )
    has_procedure = has_diagnostic or any(
        ("therapeutic or preventive procedure" in (sty or "").lower())
        or ("health care activity" in (sty or "").lower())
        for sty in styles
    )
    has_finding = any(
        ("finding" in (sty or "").lower())
        or ("sign or symptom" in (sty or "").lower())
        or ("organism function" in (sty or "").lower())
        or ("health care activity" in (sty or "").lower())
        or ("clinical attribute" in (sty or "").lower())
        for sty in styles
    )

    mult = 1.0
    if medication_intent:
        if has_therapeutic_agent:
            mult *= 1.34
        elif any(is_drug_sty(sty) for sty in styles):
            mult *= 1.12
        elif has_device and device_intent:
            mult *= 1.10
        elif has_diagnostic:
            mult *= 0.66
        elif has_procedure:
            mult *= 0.76
        elif has_disease:
            mult *= 0.80
        elif has_finding:
            mult *= 0.62

    if device_intent:
        if has_device:
            mult *= 1.22
        elif has_therapeutic_agent and medication_intent:
            mult *= 1.02
        elif has_diagnostic:
            mult *= 0.76
        elif has_disease:
            mult *= 0.84

    if diagnostic_intent:
        if has_diagnostic:
            mult *= 1.18
        elif has_therapeutic_agent:
            mult *= 0.82

    return max(0.45, min(1.35, mult))


def concept_form_multiplier(query_text: str, candidate_text: str, sty_list, query_profile):
    feats = lexical_overlap_features(query_text, candidate_text)
    if feats["query_content_total"] == 0:
        return 1.0

    styles = [(sty or "").lower() for sty in (sty_list or [])]
    short_query = feats["query_content_total"] <= 3
    medication_intent = query_profile.get("medication_intent", False)
    device_intent = query_profile.get("device_intent", False)
    diagnostic_intent = query_profile.get("diagnostic_intent", False)

    has_finding = any(
        ("finding" in sty)
        or ("sign or symptom" in sty)
        or ("clinical attribute" in sty)
        or ("organism function" in sty)
        for sty in styles
    )
    has_symptom = any("sign or symptom" in sty for sty in styles)
    has_measurement = any(
        ("diagnostic procedure" in sty)
        or ("laboratory procedure" in sty)
        or ("health care activity" in sty)
        for sty in styles
    )
    has_body_substance = any("body substance" in sty for sty in styles)
    dominant_group = query_profile.get("dominant_group")

    mult = 1.0
    if short_query and feats["extra_candidate"] > 0:
        mult *= max(0.82, 1.0 - (0.05 * feats["extra_candidate"]))
    if feats["extra_low_value"] > 0 and not feats["query_has_low_value_focus"]:
        mult *= max(0.60, 1.0 - (0.18 * feats["extra_low_value"]))
        if short_query:
            mult *= 0.88
        if short_query and (has_finding or has_measurement or has_body_substance):
            mult *= 0.84
        if short_query and dominant_group == "Disorders":
            mult *= 0.78
        if medication_intent and has_finding:
            mult *= 0.72
    if short_query and not (medication_intent or device_intent or diagnostic_intent):
        if has_measurement:
            mult *= 0.78
        elif has_body_substance:
            mult *= 0.82
        elif has_finding and feats["extra_low_value"] > 0:
            mult *= 0.86
        if dominant_group == "Disorders" and has_symptom:
            mult *= 0.78
    return max(0.45, min(1.05, mult))


def semantic_compatibility_multiplier(candidate_groups, query_profile, support_count):
    groups = {g for g in candidate_groups if g and g != "Unknown"}
    if not groups:
        return 0.75

    low_signal = 1.0
    for group in groups:
        low_signal = min(low_signal, LOW_SIGNAL_GROUP_MULTIPLIER.get(group, 1.0))

    dominant = query_profile.get("dominant_group")
    if not dominant:
        return low_signal
    if dominant in groups:
        return low_signal

    treatment_intent = query_profile.get("treatment_intent", False)
    medication_intent = query_profile.get("medication_intent", False)
    diagnostic_intent = query_profile.get("diagnostic_intent", False)

    cross = 0.72
    if dominant == "Disorders":
        if "Chemicals & Drugs" in groups or "Procedures" in groups:
            cross = 0.96 if treatment_intent else 0.78
        elif "Genes & Molecular Sequences" in groups or "Physiology" in groups:
            cross = 0.82
        elif "Anatomy" in groups:
            cross = 0.70
        else:
            cross = 0.68
    elif dominant == "Chemicals & Drugs":
        if "Disorders" in groups:
            if medication_intent:
                cross = 0.78
            else:
                cross = 0.92 if (treatment_intent or diagnostic_intent) else 0.85
        elif "Procedures" in groups:
            cross = 0.72 if medication_intent else 0.80
        elif "Genes & Molecular Sequences" in groups or "Physiology" in groups:
            cross = 0.82
        else:
            cross = 0.70
    elif dominant == "Procedures":
        if "Disorders" in groups:
            cross = 0.90
        elif "Chemicals & Drugs" in groups:
            cross = 0.86
        else:
            cross = 0.76
    else:
        if groups & {"Disorders", "Chemicals & Drugs", "Procedures", "Physiology", "Genes & Molecular Sequences"}:
            cross = 0.83
        else:
            cross = 0.65

    mult = cross * low_signal
    if support_count >= 2 and mult < 0.95:
        mult = min(0.95, mult + 0.07)
    return max(0.25, mult)


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
    has_source_mappings = bool(source_to_cuis) and bool(cui_to_sources)

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
        dist, variants = search_sapbert_with_variants(
            query,
            max(top_k * 8, 200),
            tokenizer,
            model,
            device,
            sap_index,
            cuis,
        )
        seed_cuis = []
        q_cui = term_to_cui.get(norm_term(query))
        if q_cui:
            seed_cuis = [q_cui]
        else:
            seed_cuis = select_sapbert_seed_cuis_for_text(
                query,
                dist,
                cui_to_info,
                args.resolver_max_hits_per_segment,
            )
        status = "Mode: SapBERT."
        if len(variants) > 1:
            status += f" Query rewrites used: {len(variants)}."
        return dist, seed_cuis, status

    def _seed_cuis_from_sapbert_segment(seg: str):
        if not seg:
            return []
        # Pull a wider SapBERT candidate pool, then keep only high-quality unique CUIs.
        search_k = max(50, max(1, args.resolver_max_hits_per_segment) * 20)
        dist_map, _ = search_sapbert_with_variants(
            seg,
            search_k,
            tokenizer,
            model,
            device,
            sap_index,
            cuis,
        )
        best_by_cui = {}
        for cui, score in sorted(dist_map.items(), key=lambda x: x[1]):
            term, sty_list = cui_to_info.get(cui, (cui, []))
            if any(sty in RESOLVER_EXCLUDED_SEMANTIC_TYPES for sty in sty_list):
                continue
            rank = sapbert_seed_rank(seg, term, sty_list, float(score))
            prev = best_by_cui.get(cui)
            if prev is None or rank > prev[0]:
                best_by_cui[cui] = (rank, float(score))

        if not best_by_cui:
            return []

        ranked = sorted(best_by_cui.items(), key=lambda x: x[1][0], reverse=True)
        overlap_ranked = [item for item in ranked if item[1][0][0] > 0]
        chosen = overlap_ranked if overlap_ranked else ranked
        return [cui for cui, _ in chosen[:max(1, args.resolver_max_hits_per_segment)]]

    def run_bioconceptvec(query, top_k):
        if bc_index is None:
            return {}, [], "BioConceptVec index not loaded."
        if not has_source_mappings:
            return {}, [], "BioConceptVec requires --mrconso for MSH/OMIM <-> CUI mapping."
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
            segment_hits = 0
            for seg in segments:
                seg_seed_cuis = _seed_cuis_from_sapbert_segment(seg)
                if not seg_seed_cuis:
                    continue
                segment_hits += 1
                for row_cui in seg_seed_cuis:
                    seed_cuis.append(row_cui)
                    for sab, code in sorted(cui_to_sources.get(row_cui, set())):
                        seed_ids.extend(concept_ids_from_source(sab, code))
                        seed_cuis.extend(list(source_to_cuis.get((sab, code), [])))
            if seed_cuis:
                _, q_stys = cui_to_info.get(seed_cuis[0], ("", []))
                query_group = dominant_group(q_stys)
            resolver_label = "SapBERT resolver"
            if args.resolver == "umls_api":
                resolver_label += " (umls_api alias)"
            status_msg = (
                f"Mode: BioConceptVec ({resolver_label}). Segments used: {len(segments)}. "
                f"Segments with hits: {segment_hits}. "
                f"Seed CUIs: {len(seed_cuis)}; {len(seed_ids)} seed IDs after expansion."
            )

        seed_rows = [bcv_to_row[x] for x in sorted(set(seed_ids)) if x in bcv_to_row]
        if not seed_rows:
            extra = f" Matched seed IDs in index: {len(seed_rows)}."
            if args.resolver == "exact" and not seed_cuis:
                extra += " No exact term match found; try --resolver sapbert for free-text queries."
            return {}, seed_cuis, status_msg + extra

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
            segment_hits = 0
            for seg in segments:
                seg_seed_cuis = _seed_cuis_from_sapbert_segment(seg)
                if not seg_seed_cuis:
                    continue
                segment_hits += 1
                for cui in seg_seed_cuis:
                    seed_cuis.append(cui)
                    for sab, code in sorted(cui_to_sources.get(cui, set())):
                        seed_cuis.extend(list(source_to_cuis.get((sab, code), [])))
            resolver_label = "SapBERT resolver"
            if args.resolver == "umls_api":
                resolver_label += " (umls_api alias)"
            status_msg = (
                f"Mode: cui2vec ({resolver_label}). Segments used: {len(segments)}. "
                f"Segments with hits: {segment_hits}. "
                f"Seed CUIs: {len(seed_cuis)}."
            )

        seed_rows = [cui2_to_row[c] for c in sorted(set(seed_cuis)) if c in cui2_to_row]
        if not seed_rows:
            extra = f" Matched seed CUIs in index: {len(seed_rows)}."
            if args.resolver == "exact" and not seed_cuis:
                extra += " No exact term match found; try --resolver sapbert for free-text queries."
            return {}, seed_cuis, status_msg + extra

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
        <p><strong>BioConceptVec:</strong> best when you want concepts that co-occur in biomedical literature. Requires <code>--mrconso</code> source mappings.</p>
        <p><strong>cui2vec:</strong> best when you want concepts that co-occur in clinical and literature data.</p>
        <p><strong>Ensemble:</strong> combines all three, renormalizes over active sources, and can apply an MRREL relation boost.</p>
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
        query_variants = [text for text, _ in rewrite_query_variants(query)] or [query]
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
        contrib = {}
        mode_seed_cuis = []

        if selected_mode == "sapbert":
            dist, mode_seed_cuis, status_msg = run_sapbert(query, top_k)
            rel = normalize_distance_map(dist)
            ranking = sorted(rel.items(), key=lambda x: x[1], reverse=True)
            status_msg += " Scores shown as normalized relevance."

        elif selected_mode == "bioconceptvec":
            dist, mode_seed_cuis, status_msg = run_bioconceptvec(query, top_k)
            rel = normalize_distance_map(dist)
            ranking = sorted(rel.items(), key=lambda x: x[1], reverse=True)
            status_msg += " Scores shown as normalized relevance."

        elif selected_mode == "cui2vec":
            dist, mode_seed_cuis, status_msg = run_cui2vec(query, top_k)
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
            anchor_strength = sapbert_anchor_strength(query_variants, n1, cui_to_info)
            configured_weights = {
                "sapbert": args.ensemble_sapbert_weight,
                "bioconceptvec": args.ensemble_bioconceptvec_weight,
                "cui2vec": args.ensemble_cui2vec_weight,
            }
            active_weights = normalize_weight_map({
                "sapbert": configured_weights["sapbert"] if d1 else 0.0,
                "bioconceptvec": configured_weights["bioconceptvec"] if d2 else 0.0,
                "cui2vec": configured_weights["cui2vec"] if d3 else 0.0,
            })
            fused = {}
            contrib = {}
            all_cuis = set(n1) | set(n2) | set(n3)
            seed_cuis = set(s1) | set(s2) | set(s3)
            for cui in all_cuis:
                c1 = active_weights["sapbert"] * n1.get(cui, 0.0)
                c2 = active_weights["bioconceptvec"] * n2.get(cui, 0.0)
                c3 = active_weights["cui2vec"] * n3.get(cui, 0.0)
                boost = 0.0
                score = c1 + c2 + c3
                if anchor_strength > 0.0:
                    score = ((1.0 - anchor_strength) * score) + (anchor_strength * n1.get(cui, 0.0))
                if (args.mrrel_boost > 0 or args.mrrel_count_boost > 0) and mrrel_neighbors and seed_cuis:
                    neigh = mrrel_neighbors.get(cui, set())
                    if any(s in neigh for s in seed_cuis):
                        relation_count = sum(len(mrrel_edge_combos.get(cui, {}).get(s, set())) for s in seed_cuis)
                        boost = args.mrrel_boost + (args.mrrel_count_boost * relation_count)
                        score += boost
                fused[cui] = score
                contrib[cui] = (c1, c2, c3, boost)
            ranking = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            mode_seed_cuis = sorted(seed_cuis)
            notes = []
            for label, status in (("BioConceptVec", st2), ("cui2vec", st3)):
                detail = summarize_status_detail(status)
                if detail:
                    notes.append(f"{label}: {detail}")
            status_msg = (
                "Mode: Ensemble. "
                f"Candidates sapbert={len(d1)}, bioconceptvec={len(d2)}, cui2vec={len(d3)}. "
                f"Configured weights=({configured_weights['sapbert']},{configured_weights['bioconceptvec']},{configured_weights['cui2vec']}), "
                f"effective weights=({active_weights['sapbert']:.3f},{active_weights['bioconceptvec']:.3f},{active_weights['cui2vec']:.3f}), "
                f"sapbert anchor={anchor_strength:.3f}, "
                f"MRREL base boost={args.mrrel_boost}, MRREL combo boost={args.mrrel_count_boost}."
            )
            if notes:
                status_msg += " " + " ".join(notes)

        query_profile = infer_query_semantic_profile(query, mode_seed_cuis, term_to_cui, cui_to_info, sty_to_group)
        reranked = []
        for cui, base_score in ranking:
            term, sty_list = cui_to_info.get(cui, (cui, []))
            groups = {sty_to_group.get(sty, "Unknown") for sty in sty_list} if sty_list else {"Unknown"}
            if selected_mode == "ensemble":
                c1, c2, c3, _ = contrib.get(cui, (0.0, 0.0, 0.0, 0.0))
                support_count = int(c1 > 0) + int(c2 > 0) + int(c3 > 0)
            else:
                c1, c2, c3 = (0.0, 0.0, 0.0)
                support_count = 1
            semantic_mult = semantic_compatibility_multiplier(groups, query_profile, support_count)
            lexical_mult = lexical_relevance_multiplier(query_variants, term)
            intent_mult = intent_match_multiplier(sty_list, query_profile)
            form_mult = concept_form_multiplier(query_variants, term, sty_list, query_profile)
            source_mult = ensemble_source_multiplier(query_variants, term, c1, c2, c3) if selected_mode == "ensemble" else 1.0
            reranked.append((cui, float(base_score) * semantic_mult * lexical_mult * intent_mult * form_mult * source_mult))
        ranking = sorted(reranked, key=lambda x: x[1], reverse=True)
        dominant_group = query_profile.get("dominant_group") or "unknown"
        status_msg += f" Semantic, lexical, intent, and form rerank dominant group: {dominant_group}."

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
