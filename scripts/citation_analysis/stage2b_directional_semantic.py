"""
Stage 2b: Directional Semantic Similarity
=========================================
Computes the asymmetric (directional) semantic similarity from existing SBERT embeddings.
Replaces the symmetric cosine similarity.

Directional similarity dir(A -> B) is defined as the scalar projection of A onto B,
normalized by the magnitude of B:
    dir(A -> B) = (A dot B) / ||B||
Since SBERT embeddings are L2-normalized (||B|| = 1), this mathematically reduces to
the dot product, which is identical to symmetric cosine.
To capture true Kotlerman-Dagan asymmetry using embedding dimensions, we compute the
Average Precision (AP) of B's top-k dimensions in A's ranked dimension list.

Outputs:
  computations/citation_analysis_scripts/features/{dataset}_pairs_stage2b.parquet
"""

import json, os, time, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).parent
FEAT_DIR = SCRIPT_DIR / "features"
CKPT_DIR = SCRIPT_DIR / "sbert_checkpoints"

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

def compute_directional_ap(emb_A, emb_B, k=50):
    """
    Computes Kotlerman-Dagan Average Precision (AP) of B's top-k features in A's ranked features.
    emb_A, emb_B: numpy arrays of shape (N, D)
    """
    N, D = emb_A.shape
    
    # Get indices of the top-k dimensions for B (the "target features" B cares about)
    # argsort sorts ascending, so we take the last k and reverse
    top_k_B = np.argsort(emb_B, axis=1)[:, -k:][:, ::-1]
    
    # Rank all dimensions for A (highest value gets rank 1)
    # argsort twice gives ranks (0-indexed, ascending), so we invert
    ranks_A = D - np.argsort(np.argsort(emb_A, axis=1), axis=1)
    
    ap_scores = np.zeros(N, dtype=np.float32)
    
    for i in range(N):
        # The ranks in A of the features that are top-k in B
        ranks_of_B_in_A = ranks_A[i, top_k_B[i]]
        # Sort these ranks ascending to compute precision at each recall level
        sorted_ranks = np.sort(ranks_of_B_in_A)
        
        # Precision at rank r is (number of hits up to r) / r
        # Since we sorted the ranks, the number of hits at sorted_ranks[j] is exactly j + 1
        precisions = np.arange(1, k + 1) / sorted_ranks
        ap_scores[i] = np.mean(precisions)
        
    return ap_scores

def process_dataset(dataset_name):
    log(f"\n{'='*50}\nStage 2b: {dataset_name}\n{'='*50}")
    
    stage1_path = FEAT_DIR / f"{dataset_name}_pairs_stage1.parquet"
    emb_file = CKPT_DIR / dataset_name / "embeddings.npy"
    
    if not stage1_path.exists() or not emb_file.exists():
        log(f"ERROR: Missing input files for {dataset_name}. Ensure Stage 1 and Stage 2 are run.")
        return
        
    log(f"Loading pairs from {stage1_path.name}...")
    pairs_df = pd.read_parquet(stage1_path)
    log(f"Loaded {len(pairs_df):,} pairs")
    
    # ── Collect unique IDs to map to embeddings ──
    ids_file = CKPT_DIR / dataset_name / "article_ids.json"
    if not ids_file.exists():
        log(f"ERROR: Missing {ids_file.name}. You must re-run stage 2 first to generate it.")
        return
        
    with open(ids_file, "r") as f:
        all_ids = json.load(f)
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
    
    log("Loading embeddings...")
    embeddings = np.load(emb_file)
    
    # ── Compute directional similarity ──
    log("Computing directional Average Precision (Kotlerman-Dagan)...")
    citing_indices = np.array([id_to_idx[pid] for pid in pairs_df["citing_id"]])
    cited_indices  = np.array([id_to_idx[pid] for pid in pairs_df["cited_id"]])
    
    BATCH = 10000
    n_pairs = len(pairs_df)
    sims = np.zeros(n_pairs, dtype=np.float32)
    
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        emb_A = embeddings[citing_indices[start:end]]
        emb_B = embeddings[cited_indices[start:end]]
        # Directional: A -> B (how much A includes B's top features)
        sims[start:end] = compute_directional_ap(emb_A, emb_B, k=50)
        
    pairs_df["directional_similarity"] = sims
    
    pos_sim = pairs_df[pairs_df["label"] == 1]["directional_similarity"].mean()
    neg_sim = pairs_df[pairs_df["label"] == 0]["directional_similarity"].mean()
    log(f"Mean directional similarity: Positive = {pos_sim:.4f}, Negative = {neg_sim:.4f}")
    
    # ── Save ──
    out_path = FEAT_DIR / f"{dataset_name}_pairs_stage2b.parquet"
    pairs_df.to_parquet(out_path, index=False)
    log(f"Saved enriched pairs to {out_path.name}")
    
    stats = {
        "dataset": dataset_name,
        "mean_directional_sim_positive": float(pos_sim),
        "mean_directional_sim_negative": float(neg_sim)
    }
    with open(FEAT_DIR / f"{dataset_name}_stats_stage2b.json", "w") as f:
        json.dump(stats, f, indent=2)
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        for ds in DATASETS:
            process_dataset(ds)
    log("\nStage 2b complete.")
