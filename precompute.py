#!/usr/bin/env python3
"""
precompute.py

  - Must use 2025AA Full Metathesaurus Subset from https://download.nlm.nih.gov/umls/kss/2025AA/umls-2025AA-metathesaurus-full.zip

Reads:
  - MRCONSO.RRF (atom strings)
  - MRSTY.RRF  (semantic types)

Filters MRCONSO to:
  LAT='ENG', TS='P', STT='PF', ISPREF='Y', SUPPRESS='N'

Joins on CUI and aggregates all TUIs and STYs per CUI.

Outputs a CSV with columns: CUI,STR,TUI,STY
"""

import argparse
from collections import defaultdict
import pandas as pd

def load_conso(path):
    """
    Manually parse MRCONSO.RRF, splitting on '|' so we never trip over embedded quotes.
    Returns a list of (CUI, STR) filtered by LAT, TS, STT, ISPREF, SUPPRESS.
    """
    out = []
    with open(path, encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            # ensure we have at least 17 cols
            if len(parts) < 17:
                continue
            cui      = parts[0]
            lat      = parts[1]
            ts       = parts[2]
            stt      = parts[4]
            ispref   = parts[6]
            strval   = parts[14]
            suppress = parts[16]
            # apply filters
            if (
                lat      == "ENG" and
                ts       == "P"   and
                stt      == "PF"  and
                ispref   == "Y"   and
                suppress == "N"
            ):
                out.append((cui, strval))
    # drop duplicate (CUI,STR) pairs
    return list(dict.fromkeys(out))

def load_sty(path):
    """
    Manually parse MRSTY.RRF, returning a mapping CUI -> list of (TUI,STY).
    """
    mapping = defaultdict(list)
    with open(path, encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[0]
            tui = parts[1]
            sty = parts[3]
            mapping[cui].append((tui, sty))
    # dedupe each list
    return {cui: list(dict.fromkeys(pairs)) for cui, pairs in mapping.items()}

def main():
    p = argparse.ArgumentParser(description="Build filtered UMLS metadata CSV")
    p.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF")
    p.add_argument("--mrsty",   required=True, help="Path to MRSTY.RRF")
    p.add_argument("--out",     required=True, help="Output CSV path")
    args = p.parse_args()

    # 1) Load & filter MRCONSO
    conso = load_conso(args.mrconso)
    # conso is [(CUI,STR),...]
    # 2) Load MRSTY and aggregate per CUI
    sty_map = load_sty(args.mrsty)
    records = []
    for cui, strval in conso:
        pairs = sty_map.get(cui, [])
        tuis = [t for t, _ in pairs]
        stys = [s for _, s in pairs]
        records.append({
            "CUI": cui,
            "STR": strval,
            "TUI": repr(tuis),
            "STY": repr(stys)
        })

    # 3) Write out
    df_out = pd.DataFrame(records, columns=["CUI","STR","TUI","STY"])
    df_out.to_csv(args.out, index=False)
    print(f"Wrote {len(df_out)} rows to {args.out}")

if __name__ == "__main__":
    main()
