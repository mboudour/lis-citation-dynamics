"""
oa_stage1_features.py  (v2 — fast vectorized negative sampling)
---------------------
Stage 1: Feature engineering for OpenAlex LIS dataset.
Builds citation pairs (up to 500K balanced) with 6 structural features.
Semantic similarity (feature 7) is added in Stage 2.
"""

import pandas as pd
import numpy as np
import networkx as nx
import random
import pickle
import json
from collections import defaultdict

RANDOM_SEED  = 42
N_PAIRS_EACH = 250_000   # positive + negative
MAX_COAUTH   = 20
OUTPUT_DIR   = "/home/ubuntu/lis/results_oa"

rng = np.random.default_rng(RANDOM_SEED)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading OpenAlex LIS dataset...")
df = pd.read_pickle("/home/ubuntu/lis/OpenAlex_LIS_1975_2024.pkl")
df = df.reset_index(drop=True)
print(f"  {len(df):,} papers, {df['year'].min()}–{df['year'].max()}")

ids   = df['id'].tolist()
years = df['year'].tolist()
id_to_idx = {pid: i for i, pid in enumerate(ids)}

# ── Positive pairs ─────────────────────────────────────────────────────────
print("Building positive pairs...")
pos_pairs = []
for i, row in df.iterrows():
    cy = row['year']
    refs = row['reference_ids'] if isinstance(row['reference_ids'], list) else []
    for ref in refs:
        j = id_to_idx.get(ref)
        if j is not None and years[j] <= cy:
            pos_pairs.append((i, j))

print(f"  {len(pos_pairs):,} valid positive pairs")
if len(pos_pairs) > N_PAIRS_EACH:
    idx_sel = rng.choice(len(pos_pairs), N_PAIRS_EACH, replace=False)
    pos_pairs = [pos_pairs[k] for k in idx_sel]
print(f"  Using {len(pos_pairs):,} positive pairs")

# ── Fast negative sampling ─────────────────────────────────────────────────
# Pre-index paper indices by year
print("Building negative pairs (fast vectorized)...")
year_to_indices = defaultdict(list)
for i, y in enumerate(years):
    year_to_indices[y].append(i)

# For each year t, pool = all paper indices with year <= t
cumulative_pool = {}
sorted_years = sorted(year_to_indices.keys())
pool = []
for y in sorted_years:
    pool = pool + year_to_indices[y]
    cumulative_pool[y] = np.array(pool, dtype=np.int32)

# cited set per citing paper (for exclusion)
citing_cited = defaultdict(set)
for ci, cd in pos_pairs:
    citing_cited[ci].add(cd)

neg_pairs = []
for ci, _ in pos_pairs:
    cy = years[ci]
    pool_arr = cumulative_pool[cy]
    # Sample candidates, exclude self and already-cited
    excluded = citing_cited[ci] | {ci}
    # Try up to 10 random draws
    for _ in range(10):
        j = int(rng.choice(pool_arr))
        if j not in excluded:
            neg_pairs.append((ci, j))
            break

print(f"  {len(neg_pairs):,} negative pairs")

# ── Citation graph for PageRank ────────────────────────────────────────────
print("Building citation graph for PageRank...")
G_cite = nx.DiGraph()
G_cite.add_nodes_from(range(len(df)))
for i, row in df.iterrows():
    refs = row['reference_ids'] if isinstance(row['reference_ids'], list) else []
    for ref in refs:
        j = id_to_idx.get(ref)
        if j is not None:
            G_cite.add_edge(i, j)

print(f"  Graph: {G_cite.number_of_nodes():,} nodes, {G_cite.number_of_edges():,} edges")
print("  Computing PageRank...")
pr = nx.pagerank(G_cite, alpha=0.85, max_iter=100, tol=1e-6)
print("  PageRank done.")

# ── Co-authorship graph ────────────────────────────────────────────────────
print("Building co-authorship graph...")
G_coauth = nx.Graph()
for _, row in df.iterrows():
    authors = row['author_ids'] if isinstance(row['author_ids'], list) else []
    for a in range(len(authors)):
        for b in range(a+1, len(authors)):
            G_coauth.add_edge(authors[a], authors[b])
print(f"  Co-authorship: {G_coauth.number_of_nodes():,} nodes, {G_coauth.number_of_edges():,} edges")

# ── Feature extraction ─────────────────────────────────────────────────────
print("Extracting features...")

journals    = df['journal'].tolist()
is_oa_list  = df['is_oa'].tolist()
author_ids  = df['author_ids'].tolist()

def coauth_dist(i, j):
    ai = author_ids[i] if isinstance(author_ids[i], list) else []
    aj = author_ids[j] if isinstance(author_ids[j], list) else []
    if not ai or not aj:
        return MAX_COAUTH
    best = MAX_COAUTH
    for a in ai:
        for b in aj:
            if a == b:
                return 0
            if G_coauth.has_node(a) and G_coauth.has_node(b):
                try:
                    d = nx.shortest_path_length(G_coauth, a, b)
                    if d < best:
                        best = d
                except nx.NetworkXNoPath:
                    pass
    return best

all_pairs = [(ci, cd, 1) for ci, cd in pos_pairs] + [(ci, cd, 0) for ci, cd in neg_pairs]
total = len(all_pairs)
records = []

for k, (ci, cd, label) in enumerate(all_pairs):
    if k % 50000 == 0:
        print(f"  {k:,}/{total:,} ({100*k/total:.1f}%)")
    records.append({
        'citing_id':         ids[ci],
        'cited_id':          ids[cd],
        'prestige_cited':    pr.get(cd, 0.0),
        'prestige_citing':   pr.get(ci, 0.0),
        'temporal_distance': int(years[ci]) - int(years[cd]),
        'coauth_distance':   coauth_dist(ci, cd),
        'same_journal':      int(journals[ci] == journals[cd] and journals[ci] != ''),
        'is_oa':             int(bool(is_oa_list[cd])),
        'label':             label,
    })

pairs_df = pd.DataFrame(records)
print(f"\nPairs: {len(pairs_df):,} | pos: {pairs_df['label'].sum():,} | neg: {(pairs_df['label']==0).sum():,}")

out = f"{OUTPUT_DIR}/oa_stage1_pairs.pkl"
pairs_df.to_pickle(out)
print(f"Saved: {out}")

# Dataset stats
stats = {
    "source": "OpenAlex",
    "n_papers": len(df),
    "year_min": int(df['year'].min()),
    "year_max": int(df['year'].max()),
    "abstract_coverage": round((df['abstract'].str.len() > 10).mean(), 4),
    "reference_coverage": round((df['reference_ids'].apply(len) > 0).mean(), 4),
    "mean_citations": round(float(df['times_cited'].mean()), 2),
    "median_citations": float(df['times_cited'].median()),
    "n_positive_pairs": int(pairs_df['label'].sum()),
    "n_negative_pairs": int((pairs_df['label'] == 0).sum()),
    "n_total_pairs": len(pairs_df),
}
with open(f"{OUTPUT_DIR}/oa_dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Stage 1 complete.")
