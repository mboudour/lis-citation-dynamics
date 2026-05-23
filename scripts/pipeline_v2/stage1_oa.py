"""
Stage 1 — Feature Engineering (OpenAlex v2)
=============================================
Fixes vs v1:
  1. Year-capped prestige: in-degree at year Y-1 from internal citation network
  2. Year-capped activity: prestige of citing paper at year Y-1 (proxy)
  3. Candidate-set constraints: cited paper must precede citing paper by >= 1 year
  4. Hard negatives: year-proximity ±3 years, not actual citation
     (efficient: binary search on sorted year arrays, rejection sampling)
  5. Deterministic seed: SEED=42

Features:
  prestige_cited, activity_citing, temporal_gap, common_refs, jaccard_refs, common_citers

Outputs (saved to results/oa/):
  oa_pairs_v2.parquet
  oa_stats_v2.json
"""

import ast, bisect, json, os, time
import numpy as np
import pandas as pd
from collections import defaultdict

SEED      = 42
DATA_PATH = "/home/ubuntu/data/OpenAlex_LIS_1975_2024.parquet"
OUT_DIR   = "/home/ubuntu/pipeline_v2/results/oa"
LOG_FILE  = f"{OUT_DIR}/stage1_oa.log"
MAX_TRIES = 20
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def parse_refs(x):
    if pd.isna(x) or str(x).strip() in ("", "nan", "None", "[]"):
        return []
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, list) else []
    except Exception:
        try:
            return json.loads(str(x))
        except Exception:
            return []

# Load data
log("Loading OpenAlex parquet...")
df = pd.read_parquet(DATA_PATH)
log(f"Loaded {len(df):,} articles, columns: {df.columns.tolist()}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)
df = df.reset_index(drop=True)

df["refs"] = df["reference_ids"].apply(parse_refs)

id_to_int = {pid: i for i, pid in enumerate(df["id"])}
n         = len(df)
years     = df["year"].values
ids       = df["id"].values

log(f"Year range: {years.min()}-{years.max()}")

# Year-capped prestige
log("Building citation network for year-capped prestige...")
cited_by_year = defaultdict(list)
for row in df.itertuples():
    ci = id_to_int[row.id]
    for ref in row.refs:
        cd = id_to_int.get(ref)
        if cd is not None:
            cited_by_year[cd].append(row.year)

for pid in cited_by_year:
    cited_by_year[pid].sort()

def get_prestige_at(cd_int, max_year):
    yrs = cited_by_year.get(cd_int, [])
    return bisect.bisect_right(yrs, max_year)

log("Year-capped prestige index built.")

# Build positive pairs
log("Building positive pairs with candidate-set constraints...")
pos_pairs = []
for row in df.itertuples():
    ci = id_to_int[row.id]
    for ref in row.refs:
        cd = id_to_int.get(ref)
        if cd is None:
            continue
        cited_year = years[cd]
        if cited_year >= row.year:
            continue
        pos_pairs.append((ci, cd, row.year, int(cited_year)))

log(f"Positive pairs after temporal constraint: {len(pos_pairs):,}")

# Hard negatives - binary search on sorted year arrays
log("Building hard negatives (year-proximity +-3, not actual citation)...")

actual_citations_int = set()
for row in df.itertuples():
    ci = id_to_int[row.id]
    for ref in row.refs:
        cd = id_to_int.get(ref)
        if cd is not None:
            actual_citations_int.add(ci * n + cd)

log(f"Internal citations index: {len(actual_citations_int):,} pairs")

sorted_order = np.argsort(years, kind="stable")
sorted_years = years[sorted_order].astype(np.int32)
sorted_ids   = sorted_order.astype(np.int32)

rng = np.random.default_rng(SEED)
neg_pairs = []
skipped   = 0

for ci, cd, citing_year, cited_year in pos_pairs:
    lo = bisect.bisect_left(sorted_years, cited_year - 3)
    hi = bisect.bisect_right(sorted_years, cited_year + 3)

    if hi <= lo:
        skipped += 1
        continue

    cand_ints = sorted_ids[lo:hi]
    cand_ints = cand_ints[(cand_ints != ci) & (cand_ints != cd)]

    if len(cand_ints) == 0:
        skipped += 1
        continue

    found = False
    for _ in range(MAX_TRIES):
        idx  = rng.integers(len(cand_ints))
        cand = int(cand_ints[idx])
        if (ci * n + cand) not in actual_citations_int:
            neg_pairs.append((ci, cand, citing_year, int(years[cand])))
            found = True
            break
    if not found:
        skipped += 1

log(f"Negative pairs collected: {len(neg_pairs):,} (skipped {skipped:,})")

n_pairs  = min(len(pos_pairs), len(neg_pairs))
pos_list = list(pos_pairs)
neg_list = list(neg_pairs)
rng.shuffle(pos_list)
rng.shuffle(neg_list)
pos_pairs = pos_list[:n_pairs]
neg_pairs = neg_list[:n_pairs]
log(f"Balanced dataset: {n_pairs:,} positive + {n_pairs:,} negative = {2*n_pairs:,} total pairs")

# Compute structural features
log("Computing structural features for all pairs...")

cited_by = defaultdict(set)
ref_sets = {}
for row in df.itertuples():
    ci = id_to_int[row.id]
    ref_sets[ci] = {id_to_int[r] for r in row.refs if r in id_to_int}
    for ref in row.refs:
        cd = id_to_int.get(ref)
        if cd is not None:
            cited_by[cd].add(ci)

def compute_features(ci, cd, citing_year, cited_year):
    prestige_cited  = get_prestige_at(cd, citing_year - 1)
    activity_citing = get_prestige_at(ci, citing_year - 1)
    temporal_gap    = citing_year - cited_year
    refs_a          = ref_sets.get(ci, set())
    refs_b          = ref_sets.get(cd, set())
    common_refs     = len(refs_a & refs_b)
    union_refs      = len(refs_a | refs_b)
    jaccard         = common_refs / union_refs if union_refs > 0 else 0.0
    common_citers   = len(cited_by.get(ci, set()) & cited_by.get(cd, set()))
    return [prestige_cited, activity_citing, temporal_gap, common_refs, jaccard, common_citers]

log("Computing positive pair features...")
pos_rows = []
for ci, cd, citing_year, cited_year in pos_pairs:
    feats = compute_features(ci, cd, citing_year, cited_year)
    pos_rows.append([ids[ci], ids[cd], citing_year, cited_year] + feats + [1])

log(f"Positive features done: {len(pos_rows):,}")

log("Computing negative pair features...")
neg_rows = []
for ci, cd, citing_year, cited_year in neg_pairs:
    feats = compute_features(ci, cd, citing_year, cited_year)
    neg_rows.append([ids[ci], ids[cd], citing_year, cited_year] + feats + [0])

log(f"Negative features done: {len(neg_rows):,}")

# Save
cols = ["citing_id","cited_id","citing_year","cited_year",
        "prestige_cited","activity_citing","temporal_gap",
        "common_refs","jaccard_refs","common_citers","label"]

all_rows = pos_rows + neg_rows
rng.shuffle(all_rows)
pairs_df = pd.DataFrame(all_rows, columns=cols)
out_path = f"{OUT_DIR}/oa_pairs_v2.parquet"
pairs_df.to_parquet(out_path, index=False)
log(f"Saved {len(pairs_df):,} pairs to {out_path}")

stats = {
    "dataset": "OpenAlex",
    "n_articles": int(n),
    "year_min": int(years.min()),
    "year_max": int(years.max()),
    "n_positive_pairs": int(len(pos_rows)),
    "n_negative_pairs": int(len(neg_rows)),
    "n_total_pairs": int(len(pairs_df)),
    "features": ["prestige_cited","activity_citing","temporal_gap",
                 "common_refs","jaccard_refs","common_citers"],
    "hard_negatives": True,
    "year_capped_prestige": True,
    "seed": SEED
}
with open(f"{OUT_DIR}/oa_stats_v2.json", "w") as f:
    json.dump(stats, f, indent=2)

log("Stage 1 OpenAlex complete.")
