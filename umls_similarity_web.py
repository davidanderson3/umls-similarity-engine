# HTML interface for UMLS similarity search

import argparse
import faiss
import pandas as pd
import numpy as np
import torch
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModel

DEFAULT_TOP_K = 10
DEFAULT_METADATA = "final/umls_metadata.csv"
DEFAULT_INDEX = "final/umls_index_hnsw.faiss"

def parse_args():
    p = argparse.ArgumentParser(description="UMLS similarity search web app")
    p.add_argument("--metadata", default=DEFAULT_METADATA,
                   help=f"Path to umls_metadata.csv (default: {DEFAULT_METADATA})")
    p.add_argument("--index", default=DEFAULT_INDEX,
                   help=f"Path to umls_index_hnsw.faiss (default: {DEFAULT_INDEX})")
    p.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                   help="SapBERT model ID or local dir")
    p.add_argument("--ef_search", type=int, default=64,
                   help="HNSW efSearch (higher=more accurate/slower)")
    p.add_argument("--host", default="0.0.0.0", help="Host for the web server")
    p.add_argument("--port", type=int, default=5000, help="Port for the web server")
    return p.parse_args()


def load_metadata(path):
    df = pd.read_csv(path, usecols=["CUI", "STR", "STY"]).dropna()
    return (
        df["CUI"].astype(str).tolist(),
        df["STR"].astype(str).tolist(),
        df["STY"].astype(str).tolist(),
    )


def encode_query(text, tokenizer, model, device):
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
    return cls_emb


def create_app(args):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    cuis, terms, stys = load_metadata(args.metadata)

    index = faiss.read_index(args.index)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = args.ef_search

    app = Flask(__name__)

    TEMPLATE = """
    <!doctype html>
    <title>UMLS Similarity Search</title>
    <h1>UMLS Similarity Search</h1>
    <form method="post" action="/search">
      <label>Query:<br><input type="text" name="query" size="60" required></label><br><br>
      <label>Number of results:<br><input type="number" name="top_k" value="{{default_top_k}}" min="1"></label><br><br>
      <label>Semantic Types (comma separated, optional):<br>
        <input type="text" name="semantic_types" size="60" value="{{semantic_types_text}}">
      </label><br><br>
      <input type="submit" value="Search">
    </form>
    {% if results %}
    <h2>Results for "{{query}}"</h2>
    <div id="sty-filters">
      <strong>Filter by Semantic Type:</strong>
      {% for sty in sty_set %}
        <label><input type="checkbox" class="sty-filter" value="{{sty}}" checked> {{sty}}</label>
      {% endfor %}
    </div>
    <br>
    <table border="1" cellpadding="5" cellspacing="0">
      <tr><th>Rank</th><th>CUI</th><th>Term</th><th>Semantic Type</th><th>Score</th></tr>
      {% for r in results %}
      <tr class="result-row" data-sty="{{r[2]}}">
        <td>{{loop.index}}</td><td>{{r[0]}}</td><td>{{r[1]}}</td><td>{{r[2]}}</td><td>{{"%.4f"|format(r[3])}}</td>
      </tr>
      {% endfor %}
    </table>
    <script>
    function updateFilters() {
      const checked = Array.from(document.querySelectorAll('.sty-filter:checked')).map(cb => cb.value);
      document.querySelectorAll('.result-row').forEach(row => {
        const sty = row.getAttribute('data-sty');
        row.style.display = checked.includes(sty) ? '' : 'none';
      });
    }
    document.querySelectorAll('.sty-filter').forEach(cb => cb.addEventListener('change', updateFilters));
    </script>
    {% endif %}
    """

    @app.route("/", methods=["GET"])
    def index_page():
        return render_template_string(
            TEMPLATE,
            results=None,
            default_top_k=DEFAULT_TOP_K,
            semantic_types_text="",
            sty_set=[],
        )

    @app.route("/search", methods=["POST"])
    def search():
        query = request.form.get("query", "").strip()
        try:
            top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
        except ValueError:
            top_k = DEFAULT_TOP_K
        sty_input = request.form.get("semantic_types", "")
        semantic_types = [s.strip() for s in sty_input.split(",") if s.strip()]

        q_vec = encode_query(query, tokenizer, model, device)
        D, I = index.search(q_vec, max(top_k * 5, top_k))

        results = []
        for idx, score in zip(I[0], D[0]):
            sty = stys[idx]
            if not semantic_types or sty in semantic_types:
                results.append((cuis[idx], terms[idx], sty, float(score)))
                if len(results) >= top_k:
                    break

        sty_set = sorted({r[2] for r in results})

        return render_template_string(
            TEMPLATE,
            results=results,
            query=query,
            default_top_k=top_k,
            semantic_types_text=sty_input,
            sty_set=sty_set,
        )

    return app


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
