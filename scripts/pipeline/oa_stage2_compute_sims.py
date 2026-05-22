"""
oa_stage2_compute_sims.py
--------------------------
Loads SBERT checkpoint files and computes cosine similarities
in a memory-efficient way (processes pairs in batches, never
loads all embeddings into RAM at once).
"""

import pandas as pd
import numpy as np
import pickle
import os

CKPT_DIR   = "/home/ubuntu/lis/results_oa/sbert_checkpoints"
PAIRS_FILE = "/home/ubuntu/lis/results_oa/oa_stage1_pairs.pkl"
OUTPUT     = "/home/ubuntu/lis/results_oa/oa_pairs_with_features.pkl"
BATCH_SIZE = 50000

print("Loading pairs...")
pairs_df = pd.read_pickle(PAIRS_FILE)
print(f"  {len(pairs_df):,} pairs")

# Load all checkpoint files into a single dict
print("Loading embeddings from checkpoints...")
embeddings_map = {}
ckpt_files = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("chunk_") and f.endswith(".pkl")])
for fname in ckpt_files:
    with open(os.path.join(CKPT_DIR, fname), 'rb') as f:
        chunk_data = pickle.load(f)
    embeddings_map.update(chunk_data)
    del chunk_data
print(f"  {len(embeddings_map):,} embeddings loaded")

# Compute similarities in batches to keep peak RAM low
print("Computing semantic similarities in batches...")
zero = np.zeros(384, dtype=np.float32)
n = len(pairs_df)
sims = np.zeros(n, dtype=np.float32)

citing_ids = pairs_df['citing_id'].tolist()
cited_ids  = pairs_df['cited_id'].tolist()

for start in range(0, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)
    c_embs = np.array([embeddings_map.get(pid, zero) for pid in citing_ids[start:end]])
    d_embs = np.array([embeddings_map.get(pid, zero) for pid in cited_ids[start:end]])
    sims[start:end] = (c_embs * d_embs).sum(axis=1)
    print(f"  {end:,}/{n:,} ({100*end/n:.1f}%)")

pairs_df['semantic_similarity'] = sims.astype(np.float32)

pos_mask = pairs_df['label'] == 1
print(f"  Mean sim (positive): {pairs_df.loc[pos_mask,'semantic_similarity'].mean():.4f}")
print(f"  Mean sim (negative): {pairs_df.loc[~pos_mask,'semantic_similarity'].mean():.4f}")

pairs_df.to_pickle(OUTPUT)
print(f"Saved: {OUTPUT}")
print("Stage 2 complete.")
