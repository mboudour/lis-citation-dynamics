"""
oa_stage2_sbert.py
------------------
Stage 2: SBERT encoding for OpenAlex LIS dataset.
Encodes all unique paper texts (title + abstract) with all-MiniLM-L6-v2,
computes cosine similarity for each pair, and adds it as semantic_similarity.

Checkpoint-based: saves progress every CHUNK_SIZE texts so it can resume
after sandbox hibernation.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

CHUNK_SIZE   = 5000
BATCH_SIZE   = 64
OUTPUT_DIR   = "/home/ubuntu/lis/results_oa"
CKPT_DIR     = f"{OUTPUT_DIR}/sbert_checkpoints"
PROGRESS_FILE = f"{CKPT_DIR}/progress.json"

os.makedirs(CKPT_DIR, exist_ok=True)

# ── Load pairs ─────────────────────────────────────────────────────────────
print("Loading OpenAlex stage1 pairs...")
pairs_df = pd.read_parquet("/home/ubuntu/lis/results_oa/oa_stage1_pairs.parquet")
print(f"  {len(pairs_df):,} pairs")

# ── Load dataset for texts ─────────────────────────────────────────────────
print("Loading OpenAlex dataset for texts...")
df = pd.read_parquet("/home/ubuntu/lis/OpenAlex_LIS_1975_2024.parquet")
id_to_text = {}
for _, row in df.iterrows():
    title    = str(row['title'] or '').strip()
    abstract = str(row['abstract'] or '').strip()
    text = (title + '. ' + abstract).strip() if abstract else title
    id_to_text[row['id']] = text if text else 'unknown'

# Collect all unique paper IDs needed
needed_ids = set(pairs_df['citing_id'].tolist()) | set(pairs_df['cited_id'].tolist())
all_ids    = [pid for pid in needed_ids if pid in id_to_text]
all_texts  = [id_to_text[pid] for pid in all_ids]
print(f"  {len(all_ids):,} unique papers to encode")

# ── Load checkpoint ────────────────────────────────────────────────────────
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    start_chunk = progress['chunks_done']
    print(f"Resuming from chunk {start_chunk + 1}")
else:
    start_chunk = 0
    progress = {'chunks_done': 0, 'total_chunks': 0}

# ── SBERT model ────────────────────────────────────────────────────────────
print("Loading SBERT model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("  Model loaded.")

# ── Encode in chunks ───────────────────────────────────────────────────────
n = len(all_ids)
chunks = list(range(0, n, CHUNK_SIZE))
total_chunks = len(chunks)
progress['total_chunks'] = total_chunks

# Load already-encoded embeddings
embeddings_map = {}
for c in range(start_chunk):
    ckpt_file = f"{CKPT_DIR}/chunk_{c:04d}.pkl"
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'rb') as f:
            chunk_data = pickle.load(f)
        embeddings_map.update(chunk_data)

print(f"Encoding {n:,} texts in {total_chunks} chunks of {CHUNK_SIZE}...")
for c_idx in range(start_chunk, total_chunks):
    start = chunks[c_idx]
    end   = min(start + CHUNK_SIZE, n)
    chunk_ids   = all_ids[start:end]
    chunk_texts = all_texts[start:end]

    embs = model.encode(chunk_texts, batch_size=BATCH_SIZE,
                        show_progress_bar=False, convert_to_numpy=True,
                        normalize_embeddings=True)

    chunk_data = {pid: embs[i] for i, pid in enumerate(chunk_ids)}
    ckpt_file = f"{CKPT_DIR}/chunk_{c_idx:04d}.pkl"
    with open(ckpt_file, 'wb') as f:
        pickle.dump(chunk_data, f)

    embeddings_map.update(chunk_data)
    progress['chunks_done'] = c_idx + 1
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

    pct = 100 * (c_idx + 1) / total_chunks
    print(f"  Chunk {c_idx+1}/{total_chunks} ({pct:.1f}%) — {end:,}/{n:,} texts encoded")

print(f"All {len(embeddings_map):,} embeddings ready.")

# ── Compute cosine similarities ────────────────────────────────────────────
print("Computing semantic similarities for all pairs...")
zero = np.zeros(384, dtype=np.float32)

citing_embs = np.array([embeddings_map.get(pid, zero) for pid in pairs_df['citing_id']])
cited_embs  = np.array([embeddings_map.get(pid, zero) for pid in pairs_df['cited_id']])

# Embeddings are already L2-normalized, so dot product = cosine similarity
sims = (citing_embs * cited_embs).sum(axis=1)
pairs_df['semantic_similarity'] = sims.astype(np.float32)

print(f"  Mean sim (positive): {pairs_df.loc[pairs_df['label']==1,'semantic_similarity'].mean():.4f}")
print(f"  Mean sim (negative): {pairs_df.loc[pairs_df['label']==0,'semantic_similarity'].mean():.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
out = f"{OUTPUT_DIR}/oa_pairs_with_features.pkl"
pairs_df.to_pickle(out)
print(f"Saved: {out}")
print("Stage 2 complete.")
