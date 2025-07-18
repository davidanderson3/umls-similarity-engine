# HTML interface for UMLS similarity search

import argparse
import ast
import faiss
import pandas as pd
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


def encode_query(text, tokenizer, model, device):
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")

    # normalize query so distance scores align with the index vectors
    faiss.normalize_L2(cls_emb)
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
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      table { border-collapse: collapse; }
      th, td { padding: 4px 8px; border: 1px solid #ccc; }
      th { background-color: #f2f2f2; }
    </style>
    <h1>UMLS Similarity Search</h1>
    <form method="post" action="/search">
      <label>Query:<br><input type="text" name="query" size="60" required value="{{query}}"></label><br><br>
      <label>Number of results:<br><input type="number" name="top_k" value="{{default_top_k}}" min="1"></label><br><br>
      <label>Maximum score (optional):<br>
        <input type="number" name="max_score" step="any" placeholder="e.g. 0.5" value="{{default_max_score}}">
        <small>Increase to allow more distant matches</small>
      </label><br><br>
      <input type="submit" value="Search">
    </form>
    {% if results %}
    <h2>Results for "{{query}}"</h2>
    <div id="sty-filter">
      <label for="sty-select"><strong>Semantic Type:</strong></label>
      <select id="sty-select">
        <option value="">All</option>
        {% for sty in sty_set %}
          <option value="{{sty}}">{{sty}}</option>
        {% endfor %}
      </select>
    </div>
    <br>
    <table border="1" cellpadding="5" cellspacing="0">
      <tr><th>Rank</th><th>CUI</th><th>Term</th><th>Semantic Type</th><th>Score</th></tr>
      {% for r in results %}
      <tr class="result-row" data-sty="{{r[4]}}">
        <td>{{loop.index}}</td><td>{{r[0]}}</td><td>{{r[1]}}</td><td>{{r[2]}}</td><td>{{"%.4f"|format(r[3])}}</td>
      </tr>
      {% endfor %}
    </table>
    <script>
    function updateFilter() {
      const selected = document.getElementById('sty-select').value;
      document.querySelectorAll('.result-row').forEach(row => {
        const sty = row.getAttribute('data-sty');
        row.style.display = !selected || sty === selected ? '' : 'none';
      });
    }
    document.getElementById('sty-select').addEventListener('change', updateFilter);
    </script>
    {% endif %}
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
        )

    @app.route("/search", methods=["POST"])
    def search():
        query = request.form.get("query", "").strip()
        try:
            top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
        except ValueError:
            top_k = DEFAULT_TOP_K
        max_score_in = request.form.get("max_score", "").strip()
        try:
            max_score = float(max_score_in) if max_score_in else None
        except ValueError:
            max_score = None

        q_vec = encode_query(query, tokenizer, model, device)
        D, I = index.search(q_vec, max(top_k * 5, top_k))

        results = []
        sty_set = set()
        for idx, score in zip(I[0], D[0]):
            if len(results) >= top_k:
                break
            if max_score is not None and score > max_score:
                continue
            sty_list = stys[idx]
            sty_set.update(sty_list)
            sty_str = ", ".join(sty_list)
            results.append((cuis[idx], terms[idx], sty_str, float(score), sty_list[0]))

        sty_set = sorted(sty_set)

        return render_template_string(
            TEMPLATE,
            results=results,
            query=query,
            default_top_k=top_k,
            default_max_score=max_score_in,
            sty_set=sty_set,
        )

    return app


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
