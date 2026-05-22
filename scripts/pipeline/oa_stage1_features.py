"""
OpenAlex Stage 1: Feature Engineering
Reads from oa_data.parquet (numpy-version-independent).
Builds citation pairs (positive + negative) with 6 structural features.
Strict temporal causality: cited paper must predate citing paper.
"""
import json, random
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

WORK_DIR = Path("/home/ubuntu/oa_work")
OUT_DIR  = WORK_DIR / "results"
SEED     = 42
N_PAIRS  = 250_000

random.seed(SEED)
np.random.seed(SEED)

print("Loading oa_data.parquet...")
df = pd.read_parquet(WORK_DIR / "oa_data.parquet")
df["reference_ids"] = df["reference_ids_json"].apply(json.loads)
print(f"  {len(df):,} articles, years {df['year'].min()}–{df['year'].max()}")

# ID → row index
id_to_idx = {row.id: i for i, row in df.iterrows()}

# ── Build citation graph ─────────────────────────────────────────────────────
print("Building citation graph...")
G = nx.DiGraph()
G.add_nodes_from(range(len(df)))
for i, row in df.iterrows():
    for ref_id in row["reference_ids"]:
        j = id_to_idx.get(ref_id)
        if j is not None:
            G.add_edge(i, j)
print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
successors   = {n: set(G.successors(n))   for n in G.nodes()}
predecessors = {n: set(G.predecessors(n)) for n in G.nodes()}

# ── Collect positive pairs ───────────────────────────────────────────────────
print("Collecting positive pairs...")
pos_pairs = []
for i, row in df.iterrows():
    for ref_id in row["reference_ids"]:
        j = id_to_idx.get(ref_id)
        if j is not None and df.at[j, "year"] < row["year"]:
            pos_pairs.append((i, j))
print(f"  Total positive: {len(pos_pairs):,}")
if len(pos_pairs) > N_PAIRS:
    pos_pairs = random.sample(pos_pairs, N_PAIRS)
print(f"  Sampled: {len(pos_pairs):,}")

# ── Negative sampling ────────────────────────────────────────────────────────
print("Sampling negative pairs...")
pos_set  = set(pos_pairs)
all_ids  = list(range(len(df)))
neg_pairs = []
attempts  = 0
while len(neg_pairs) < N_PAIRS and attempts < N_PAIRS * 30:
    i = random.choice(all_ids)
    j = random.choice(all_ids)
    attempts += 1
    if i == j: continue
    if df.at[j, "year"] >= df.at[i, "year"]: continue
    if (i, j) in pos_set: continue
    if G.has_edge(i, j): continue
    neg_pairs.append((i, j))
print(f"  Sampled: {len(neg_pairs):,} (attempts: {attempts:,})")

# ── Feature computation ──────────────────────────────────────────────────────
def featurize(citing_idx, cited_idx):
    prestige     = in_deg.get(cited_idx, 0)
    activity     = out_deg.get(citing_idx, 0)
    temp_gap     = int(df.at[citing_idx, "year"]) - int(df.at[cited_idx, "year"])
    refs_a       = successors.get(citing_idx, set())
    refs_b       = successors.get(cited_idx,  set())
    common_refs  = len(refs_a & refs_b)
    citers_a     = predecessors.get(citing_idx, set())
    citers_b     = predecessors.get(cited_idx,  set())
    common_citers = len(citers_a & citers_b)
    union_refs   = len(refs_a | refs_b)
    jaccard      = common_refs / union_refs if union_refs > 0 else 0.0
    return [prestige, activity, temp_gap, common_refs, common_citers, jaccard]

print("Computing features...")
all_pairs = pos_pairs + neg_pairs
labels    = [1]*len(pos_pairs) + [0]*len(neg_pairs)
feat_names = ["prestige_cited","activity_citing","temporal_gap",
              "common_refs","common_citers","jaccard_refs"]
rows = []
for k, (i, j) in enumerate(all_pairs):
    if k % 100_000 == 0:
        print(f"  {k:,}/{len(all_pairs):,}")
    rows.append(featurize(i, j))

feat_df = pd.DataFrame(rows, columns=feat_names)
feat_df["label"]      = labels
feat_df["citing_idx"] = [p[0] for p in all_pairs]
feat_df["cited_idx"]  = [p[1] for p in all_pairs]

out_path = OUT_DIR / "oa_pairs_features.parquet"
feat_df.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
print(f"Shape: {feat_df.shape}")
print(feat_df[feat_names].describe().round(3).to_string())

# Dataset stats
stats = {
    "total_articles": int(len(df)),
    "year_min": int(df["year"].min()),
    "year_max": int(df["year"].max()),
    "with_abstract_pct": round(100*(df["abstract"].str.len()>10).mean(),1),
    "with_references_pct": round(100*(df["reference_ids"].apply(len)>0).mean(),1),
    "mean_times_cited": round(float(df["times_cited"].mean()),2),
    "total_positive_pairs": len(pos_pairs),
    "total_negative_pairs": len(neg_pairs),
    "graph_nodes": G.number_of_nodes(),
    "graph_edges": G.number_of_edges(),
}
with open(OUT_DIR / "oa_dataset_stats.json","w") as f:
    json.dump(stats, f, indent=2)
print("\nDataset stats:"); print(json.dumps(stats, indent=2))

# Save texts for Stage 2
texts_df = df[["id","title","abstract","year"]].copy()
texts_df["text"] = (texts_df["title"].fillna("") + " " +
                    texts_df["abstract"].fillna("")).str.strip()
texts_df = texts_df[texts_df["text"].str.len() > 10].reset_index(drop=True)
texts_df[["id","text","year"]].to_parquet(OUT_DIR / "oa_texts.parquet", index=False)
print(f"\nSaved oa_texts.parquet: {len(texts_df):,} texts")
