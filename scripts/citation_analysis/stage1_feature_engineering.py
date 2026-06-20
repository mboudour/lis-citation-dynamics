"""
Stage 1: Feature Engineering for 10 Dimensions Datasets
=========================================================
Generates temporally constrained positive and negative citation pairs
and computes structural features for all 10 datasets.

Methodology (from LIS manuscript):
1. Positive pairs: internal citations where cited_year < citing_year.
2. Negative pairs: hard negative sampling. We pick a non-cited paper 
   published within ±3 years of the true cited paper.
   (We avoid FoR code constraints to keep it general across all datasets).
3. Features computed (all temporally constrained to year <= citing_year - 1):
   - temporal_indegree  : in-degree of cited paper at citing_year - 1
   - temporal_pagerank  : PageRank of cited paper in subgraph at citing_year - 1
   - citation_time_gap  : citing_year - cited_year
   (directional_similarity is added in Stage 2b)

Outputs:
  computations/citation_analysis_scripts/features/{dataset}_pairs_stage1.parquet
"""

import ast, bisect, json, os, time, sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from pathlib import Path

SEED = 42

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data_collection" / "data"
OUT_DIR = SCRIPT_DIR / "features"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    "protein_folding", "CRISPR",
    "neuroblastoma", "osteosarcoma",
    "additive_manufacturing", "corrosion_protection",
    "income_inequality", "organizational_behavior",
    "film_studies", "memory_studies",
]

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def parse_refs(x):
    # Already a list (parquet stores lists natively)
    if isinstance(x, list):
        return x
    # Scalar NA check (safe for non-array types)
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if s in ("", "nan", "None", "[]"):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return []

def process_dataset(dataset_name):
    log(f"\n{'='*50}\nProcessing {dataset_name}\n{'='*50}")
    
    data_path = DATA_DIR / f"Dimensions_{dataset_name}_1975_2024.parquet"
    if not data_path.exists():
        log(f"ERROR: {data_path} not found. Skipping.")
        return
        
    log(f"Loading {data_path.name}...")
    df = pd.read_parquet(data_path, columns=["id", "reference_ids", "year"])
    log(f"Loaded {len(df):,} articles")
    
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.reset_index(drop=True)
    
    df["refs"] = df["reference_ids"].apply(parse_refs)
    
    id_to_int = {pid: i for i, pid in enumerate(df["id"])}
    n = len(df)
    years = df["year"].values
    ids = df["id"].values
    
    # ── Build full edge list for temporal subgraph construction ──
    log("Building edge list for temporal subgraph...")
    # edges: list of (citing_int, cited_int, citing_year)
    all_edges = []
    for row in df.itertuples():
        ci = id_to_int[row.id]
        for ref in row.refs:
            cd = id_to_int.get(ref)
            if cd is not None and years[cd] < row.year:
                all_edges.append((ci, cd, row.year))
    
    # Sort edges by year for efficient subgraph construction
    all_edges.sort(key=lambda e: e[2])
    edge_years = np.array([e[2] for e in all_edges])
    
    # ── Year-capped indegree index ──
    log("Building temporal indegree index...")
    cited_by_year = defaultdict(list)
    for ci, cd, ey in all_edges:
        cited_by_year[cd].append(ey)
    for pid in cited_by_year:
        cited_by_year[pid].sort()
        
    def get_indegree_at(cd_int, max_year):
        yrs = cited_by_year.get(cd_int, [])
        return bisect.bisect_right(yrs, max_year)
    
    # ── Temporal PageRank: compute once per unique citing year ──
    log("Computing temporal PageRank per unique citing year...")
    unique_years = sorted(set(years))
    pagerank_cache = {}  # max_year -> {node_int: pr_score}
    
    for max_year in unique_years:
        # Build subgraph of all edges with citing_year <= max_year
        hi = bisect.bisect_right(edge_years, max_year)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for ci, cd, _ in all_edges[:hi]:
            G.add_edge(ci, cd)
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        pagerank_cache[max_year] = pr
    
    log(f"PageRank computed for {len(unique_years)} time points.")
    
    # ── Positive pairs ──
    log("Building positive pairs...")
    pos_pairs = []
    for row in df.itertuples():
        ci = id_to_int[row.id]
        for ref in row.refs:
            cd = id_to_int.get(ref)
            if cd is None: continue
            cited_year = years[cd]
            if cited_year >= row.year: continue
            pos_pairs.append((ci, cd, row.year, int(cited_year)))
            
    log(f"Positive pairs: {len(pos_pairs):,}")
    
    # ── Hard negatives ──
    log("Building hard negatives (±3 years)...")
    actual_citations_int = set()
    for ci, cd, ey in all_edges:
        actual_citations_int.add(ci * n + cd)
                
    # Sort all ids by year for binary search
    sorted_indices = np.argsort(years)
    sorted_years = years[sorted_indices]
    sorted_ids = sorted_indices
    
    rng = np.random.default_rng(SEED)
    neg_pairs = []
    skipped = 0
    MAX_TRIES = 20
    
    for ci, cd, citing_year, cited_year in pos_pairs:
        lo = bisect.bisect_left(sorted_years, cited_year - 3)
        hi = bisect.bisect_right(sorted_years, cited_year + 3)
        
        if hi > lo:
            cands = sorted_ids[lo:hi]
            cands = cands[(cands != ci) & (cands != cd)]
            
            if len(cands) > 0:
                found = False
                for _ in range(MAX_TRIES):
                    cand = int(rng.choice(cands))
                    if (ci * n + cand) not in actual_citations_int:
                        neg_pairs.append((ci, cand, citing_year, int(years[cand])))
                        found = True
                        break
                if not found:
                    skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1
            
    log(f"Negative pairs: {len(neg_pairs):,} (skipped {skipped:,})")
    
    n_pairs = min(len(pos_pairs), len(neg_pairs))
    pos_list = list(pos_pairs)
    neg_list = list(neg_pairs)
    rng.shuffle(pos_list)
    rng.shuffle(neg_list)
    pos_pairs = pos_list[:n_pairs]
    neg_pairs = neg_list[:n_pairs]
    log(f"Balanced: {n_pairs:,} pos + {n_pairs:,} neg = {2*n_pairs:,} total")
    
    # ── Compute features ──
    log("Computing features...")
    
    def compute_features(ci, cd, citing_year, cited_year):
        max_year = citing_year - 1
        temporal_indegree = get_indegree_at(cd, max_year)
        citation_time_gap = citing_year - cited_year
        pr_map = pagerank_cache.get(max_year, pagerank_cache.get(max(k for k in pagerank_cache if k <= max_year), {}))
        temporal_pagerank = pr_map.get(cd, 0.0)
        return [temporal_indegree, citation_time_gap, temporal_pagerank]
        
    all_rows = []
    for ci, cd, citing_year, cited_year in pos_pairs:
        feats = compute_features(ci, cd, citing_year, cited_year)
        all_rows.append([ids[ci], ids[cd], citing_year, cited_year] + feats + [1])
        
    for ci, cd, citing_year, cited_year in neg_pairs:
        feats = compute_features(ci, cd, citing_year, cited_year)
        all_rows.append([ids[ci], ids[cd], citing_year, cited_year] + feats + [0])
        
    rng.shuffle(all_rows)
    cols = ["citing_id", "cited_id", "citing_year", "cited_year",
            "temporal_indegree", "citation_time_gap", "temporal_pagerank", "label"]
            
    pairs_df = pd.DataFrame(all_rows, columns=cols)
    out_path = OUT_DIR / f"{dataset_name}_pairs_stage1.parquet"
    pairs_df.to_parquet(out_path, index=False)
    log(f"Saved {len(pairs_df):,} pairs to {out_path.name}")
    
    stats = {
        "dataset": dataset_name,
        "n_articles": int(n),
        "n_positive_pairs": int(len(pos_pairs)),
        "n_negative_pairs": int(len(neg_pairs)),
        "n_total_pairs": int(len(pairs_df)),
        "features": ["temporal_indegree", "citation_time_gap", "temporal_pagerank"]
    }
    with open(OUT_DIR / f"{dataset_name}_stats_stage1.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific dataset
        process_dataset(sys.argv[1])
    else:
        # Run all
        for ds in DATASETS:
            process_dataset(ds)
    log("\nStage 1 complete.")
