"""
screen_social_science.py
========================
Screens 10 Social Science keyword candidates from Dimensions.ai (1975-2024).
For each candidate it fetches the corpus, computes:
  - n_papers          : total articles
  - papers_with_refs  : articles with at least one reference
  - mean_ref_len      : average reference-list length
  - n_internal_edges  : citation edges where both papers are in the corpus
  - internal_ratio    : n_internal_edges / total_refs  (target: 10-30%)
  - gcc_fraction      : giant weak component / n_papers (target: >= 50%)
  - n_components      : number of connected components

Results are saved incrementally to screen_social_science_report.csv.

Run from the directory containing Dim_key.txt:

    python screen_social_science.py

Requirements: dimcli pandas pyarrow networkx
"""

import dimcli
import time
import datetime
import pandas as pd
import networkx as nx
from pathlib import Path

HERE = Path(__file__).parent
with open(HERE / "Dim_key.txt") as f:
    api_key = f.read().strip()

dimcli.login(key=api_key, endpoint="https://app.dimensions.ai/api/dsl.json")
dsl = dimcli.Dsl()

YEAR_MIN = 1975
YEAR_MAX = 2024
FIELDS   = "id+year+title+abstract+reference_ids"

CANDIDATES = [
    ("income_inequality",       ["income inequality"]),
    ("social_capital",          ["social capital"]),
    ("immigration_integration", ["immigration integration"]),
    ("public_opinion",          ["public opinion"]),
    ("organizational_behavior", ["organizational behavior"]),
    ("social_movements",        ["social movements"]),
    ("gender_labor_market",     ["gender labor market"]),
    ("urban_poverty",           ["urban poverty"]),
    ("ethnic_conflict",         ["ethnic conflict"]),
    ("educational_attainment",  ["educational attainment"]),
]

REPORT_CSV = Path("screen_social_science_report.csv")


def extract_ref_ids(raw_refs):
    if not isinstance(raw_refs, list):
        return []
    return [str(item) for item in raw_refs
            if isinstance(item, str) and str(item).startswith("pub.")]


def fetch_corpus(name, keywords):
    all_records = {}
    for kw in keywords:
        print(f"  Keyword: \"{kw}\"", flush=True)
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            print(f"  [{datetime.datetime.now():%H:%M:%S}] Year {year} ...",
                  end=" ", flush=True)
            try:
                result = dsl.query_iterative(f"""
                    search publications
                    in title_abstract_only
                    for "{kw}"
                    where year = {year}
                    and type = "article"
                    return publications[{FIELDS}]
                """)
                df_year = result.as_dataframe()
                added = 0
                for rec in df_year.to_dict(orient="records"):
                    pid = rec.get("id")
                    if pid and pid not in all_records:
                        rec["abstract"] = str(rec.get("abstract") or "")
                        all_records[pid] = rec
                        added += 1
                print(f"{len(df_year):,} fetched, {added:,} new "
                      f"(total: {len(all_records):,})", flush=True)
            except Exception as e:
                print(f"ERROR: {e} — skipping", flush=True)
            time.sleep(1)

    if not all_records:
        return pd.DataFrame()

    raw_list = list(all_records.values())
    df = pd.DataFrame({
        "id":            [r.get("id", "") for r in raw_list],
        "year":          [int(r.get("year", 0)) for r in raw_list],
        "reference_ids": [extract_ref_ids(r.get("reference_ids")) for r in raw_list],
    })
    df = df[df["year"] > 0].reset_index(drop=True)
    return df


def compute_stats(name, df):
    n_papers = len(df)
    if n_papers == 0:
        return None

    id_set = set(df["id"].astype(str))
    ref_lens = df["reference_ids"].apply(len)
    total_refs = int(ref_lens.sum())
    mean_ref_len = round(float(ref_lens.mean()), 1)
    papers_with_refs = int((ref_lens > 0).sum())

    edges = []
    for _, row in df.iterrows():
        citing = str(row["id"])
        for r in row["reference_ids"]:
            r = str(r)
            if r in id_set and r != citing:
                edges.append((citing, r))

    n_internal = len(edges)
    internal_ratio = round(n_internal / total_refs, 4) if total_refs > 0 else 0.0

    G = nx.Graph()
    G.add_nodes_from(id_set)
    G.add_edges_from(edges)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc_size = len(components[0]) if components else 0
    gcc_fraction = round(gcc_size / n_papers, 4)
    n_components = nx.number_connected_components(G)
    density = (2 * n_internal) / (n_papers * (n_papers - 1)) if n_papers > 1 else 0.0

    ok_ir  = 0.10 <= internal_ratio <= 0.30
    ok_gcc = gcc_fraction >= 0.50
    verdict = "PASS" if (ok_ir and ok_gcc) else "FAIL"

    return {
        "dataset":          name,
        "n_papers":         n_papers,
        "papers_with_refs": papers_with_refs,
        "mean_ref_len":     mean_ref_len,
        "total_refs":       total_refs,
        "n_internal_edges": n_internal,
        "internal_ratio":   internal_ratio,
        "density":          f"{density:.2e}",
        "gcc_size":         gcc_size,
        "gcc_fraction":     gcc_fraction,
        "n_components":     n_components,
        "verdict":          verdict,
    }


if REPORT_CSV.exists():
    done_df = pd.read_csv(REPORT_CSV)
    done_names = set(done_df["dataset"].tolist())
    rows = done_df.to_dict(orient="records")
    print(f"Resuming — {len(done_names)} already done: {done_names}", flush=True)
else:
    done_names = set()
    rows = []

for name, keywords in CANDIDATES:
    if name in done_names:
        print(f"\n[{name}] Already done — skipping.", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"CANDIDATE: {name}  |  keywords: {keywords}", flush=True)
    print(f"{'='*60}", flush=True)

    df = fetch_corpus(name, keywords)
    stats = compute_stats(name, df)

    if stats is None:
        print(f"  [SKIP] No records returned.", flush=True)
        continue

    rows.append(stats)
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)

    print(f"\n  RESULT: n={stats['n_papers']:,}  "
          f"internal_ratio={stats['internal_ratio']:.1%}  "
          f"gcc_fraction={stats['gcc_fraction']:.1%}  "
          f"[{stats['verdict']}]", flush=True)

report = pd.DataFrame(rows).sort_values("gcc_fraction", ascending=False)
print("\n\n" + "="*70)
print("SOCIAL SCIENCE CANDIDATES — SUMMARY (sorted by gcc_fraction)")
print("="*70)
cols = ["dataset","n_papers","mean_ref_len","n_internal_edges",
        "internal_ratio","gcc_fraction","n_components","verdict"]
cols = [c for c in cols if c in report.columns]
print(report[cols].to_string(index=False))
print(f"\nFull report saved to: {REPORT_CSV.resolve()}")
