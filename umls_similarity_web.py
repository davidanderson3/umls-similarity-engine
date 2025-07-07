# HTML interface for UMLS similarity search

import argparse
import faiss
import pandas as pd
import numpy as np
import torch
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModel


def parse_args():
    p = argparse.ArgumentParser(description="UMLS similarity search web app")
    p.add_argument("--metadata", required=True, help="Path to umls_metadata.csv")
    p.add_argument("--index", required=True, help="Path to umls_index_hnsw.faiss")
    p.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                   help="SapBERT model ID or local dir")
    p.add_argument("--top_k", type=int, default=10, help="Default number of results")
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
        <input type="text" name="semantic_types" size="60">
      </label><br><br>
      <input type="submit" value="Search">
    </form>
    {% if results %}
    <h2>Results for "{{query}}"</h2>
    <table border="1" cellpadding="5" cellspacing="0">
      <tr><th>Rank</th><th>CUI</th><th>Term</th><th>Semantic Type</th><th>Score</th></tr>
      {% for r in results %}
      <tr>
        <td>{{loop.index}}</td><td>{{r[0]}}</td><td>{{r[1]}}</td><td>{{r[2]}}</td><td>{{"%.4f"|format(r[3])}}</td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}
    """

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(TEMPLATE, results=None, default_top_k=args.top_k)

    @app.route("/search", methods=["POST"])
    def search():
        query = request.form.get("query", "").strip()
        try:
            top_k = int(request.form.get("top_k", args.top_k))
        except ValueError:
            top_k = args.top_k
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

        return render_template_string(
            TEMPLATE,
            results=results,
            query=query,
            default_top_k=top_k,
        )

    return app


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
