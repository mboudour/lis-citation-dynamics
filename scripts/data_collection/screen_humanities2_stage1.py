"""
screen_humanities2_stage1.py
============================
STAGE 1 — Fast pre-screening of Humanities candidates.
Fetches only 3 sample years (2010, 2015, 2020) per candidate.
Computes internal_ratio and gcc_fraction on the sample.
Candidates scoring internal_ratio >= 0.06 AND gcc_fraction >= 0.30
are flagged as PROMISING and worth a full Stage 2 run.

Results saved to screen_humanities2_stage1_report.csv.

Run from the directory containing Dim_key.txt:
    python screen_humanities2_stage1.py

Requirements: dimcli pandas networkx
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

SAMPLE_YEARS = [2005, 2010, 2015, 2020]
FIELDS = "id+year+title+abstract+reference_ids"

# Stage 1 thresholds (relaxed — full corpus will be higher)
MIN_IR  = 0.06   # internal ratio >= 6%
MIN_GCC = 0.30   # gcc fraction >= 30%

CANDIDATES = [
    ("history_of_science",      "history of science"),
    ("historiography",          "historiography"),
    ("intellectual_history",    "intellectual history"),
    ("science_technology_studies", "science technology studies"),
    ("analytic_philosophy",     "analytic philosophy"),
    ("phenomenology",           "phenomenology"),
    ("postmodernism",           "postmodernism"),
    ("continental_philosophy",  "continental philosophy"),
    ("philosophy_of_language",  "philosophy of language"),
    ("cultural_memory",         "cultural memory"),
    ("gender_studies",          "gender studies"),
    ("postcolonial_studies",    "postcolonial studies"),
]

REPORT_CSV = Path("screen_humanities2_stage1_report.csv")


def extract_ref_ids(raw_refs):
    if not isinstance(raw_refs, list):
        return []
    return [str(item) for item in raw_refs
            if isinstance(item, str) and str(item).startswith("pub.")]


def fetch_sample(name, keyword):
    all_records = {}
    for year in SAMPLE_YEARS:
        print(f"  [{datetime.datetime.now():%H:%M:%S}] Year {year} ...",
              end=" ", flush=True)
        try:
            result = dsl.query_iterative(f"""
                search publications
                in title_abstract_only
                for "{keyword}"
                where year = {year}
                and type = "article"
                return publications[{FIELDS}]
            """)
            df_year = result.as_dataframe()
            added = 0
            for rec in df_year.to_dict(orient="records"):
                pid = rec.get("id")
                if pid and pid not in all_records:
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
    return df[df["year"] > 0].reset_index(drop=True)


def compute_stats(name, df):
    n_papers = len(df)
    if n_papers == 0:
        return None

    id_set = set(df["id"].astype(str))
    ref_lens = df["reference_ids"].apply(len)
    total_refs = int(ref_lens.sum())
    mean_ref_len = round(float(ref_lens.mean()), 1)

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

    promising = (internal_ratio >= MIN_IR) and (gcc_fraction >= MIN_GCC)
    verdict = "PROMISING" if promising else "skip"

    return {
        "dataset":          name,
        "sample_years":     str(SAMPLE_YEARS),
        "n_papers":         n_papers,
        "mean_ref_len":     mean_ref_len,
        "total_refs":       total_refs,
        "n_internal_edges": n_internal,
        "internal_ratio":   internal_ratio,
        "gcc_size":         gcc_size,
        "gcc_fraction":     gcc_fraction,
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

for name, keyword in CANDIDATES:
    if name in done_names:
        print(f"\n[{name}] Already done — skipping.", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"CANDIDATE: {name}  |  keyword: \"{keyword}\"", flush=True)
    print(f"{'='*60}", flush=True)

    df = fetch_sample(name, keyword)
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
print("HUMANITIES STAGE 1 — SUMMARY (sorted by gcc_fraction)")
print(f"Thresholds: internal_ratio >= {MIN_IR:.0%}, gcc_fraction >= {MIN_GCC:.0%}")
print("="*70)
cols = ["dataset","n_papers","mean_ref_len","n_internal_edges",
        "internal_ratio","gcc_fraction","verdict"]
print(report[cols].to_string(index=False))

promising = report[report["verdict"] == "PROMISING"]
print(f"\nPROMISING candidates for Stage 2 full fetch ({len(promising)}):")
for _, row in promising.iterrows():
    print(f"  {row['dataset']}  IR={row['internal_ratio']:.1%}  GCC={row['gcc_fraction']:.1%}")

print(f"\nFull report saved to: {REPORT_CSV.resolve()}")
