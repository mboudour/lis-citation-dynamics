"""
LIS Citation Dynamics — Stage 2 (Checkpoint-based)
SBERT Semantic Similarity Encoding with checkpointing to survive sandbox hibernation.
Encodes in chunks of 5000 texts, saving progress after each chunk.
"""
import pandas as pd
import numpy as np
import json
import os
import gc
import pickle
from sentence_transformers import SentenceTransformer

RESULTS = "/home/ubuntu/lis/results"
TEXTS_CSV = f"{RESULTS}/lis_texts.csv"
CHECKPOINT_DIR = f"{RESULTS}/sbert_checkpoints"
EMBEDDINGS_FILE = f"{CHECKPOINT_DIR}/embeddings_partial.npy"
PROGRESS_FILE = f"{CHECKPOINT_DIR}/progress.json"
IDS_FILE = f"{CHECKPOINT_DIR}/all_ids.pkl"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("STAGE 2: SBERT Encoding (Checkpoint-based)")
log("=" * 60)

# ── Load pairs ──
log("Loading Stage 1 pairs...")
df = pd.read_pickle(f"{RESULTS}/lis_stage1_pairs.pkl")
log(f"  Loaded {len(df):,} pairs")

# ── Load text lookup ──
log("Loading text lookup from CSV...")
texts_df = pd.read_csv(TEXTS_CSV)
text_map = texts_df.set_index('id')['text'].to_dict()
del texts_df
gc.collect()
log(f"  Text map built for {len(text_map):,} papers")

# ── Build or load ID list ──
if os.path.exists(IDS_FILE):
    log("  Loading existing ID list from checkpoint...")
    with open(IDS_FILE, 'rb') as f:
        all_ids = pickle.load(f)
    log(f"  Loaded {len(all_ids):,} IDs from checkpoint")
else:
    all_ids = list(set(df['citing_id'].tolist() + df['cited_id'].tolist()))
    with open(IDS_FILE, 'wb') as f:
        pickle.dump(all_ids, f)
    log(f"  Unique papers in pairs: {len(all_ids):,}")

texts = [str(text_map.get(pid, '')) for pid in all_ids]
id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
del text_map
gc.collect()

n_texts = len(texts)
CHUNK_SIZE = 5000  # Encode 5000 texts at a time, save checkpoint

# ── Load or initialize embeddings ──
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(PROGRESS_FILE):
    log("  Resuming from checkpoint...")
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
    start_chunk = progress['completed_chunks']
    log(f"  Resuming from chunk {start_chunk} ({start_chunk * CHUNK_SIZE:,} / {n_texts:,} texts done)")
else:
    log("  Starting fresh encoding...")
    embeddings = np.zeros((n_texts, 384), dtype=np.float32)  # all-MiniLM-L6-v2 dim = 384
    start_chunk = 0

# ── Load SBERT model ──
log("Loading SBERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# ── Encode in chunks ──
n_chunks = (n_texts + CHUNK_SIZE - 1) // CHUNK_SIZE
log(f"Encoding {n_texts:,} texts in {n_chunks} chunks of {CHUNK_SIZE}...")

for chunk_idx in range(start_chunk, n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_texts)
    chunk_texts = texts[start:end]
    
    log(f"  Chunk {chunk_idx + 1}/{n_chunks}: texts {start:,}-{end:,}...")
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings[start:end] = chunk_embeddings
    
    # Save checkpoint after each chunk
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'completed_chunks': chunk_idx + 1, 'total_chunks': n_chunks,
                   'texts_done': end, 'total_texts': n_texts}, f)
    log(f"    Checkpoint saved: {end:,}/{n_texts:,} texts encoded ({100*end/n_texts:.1f}%)")

log(f"  All {n_texts:,} texts encoded. Embeddings shape: {embeddings.shape}")

del model, texts
gc.collect()

# ── Compute cosine similarities ──
log("Computing cosine similarities...")
citing_indices = np.array([id_to_idx[pid] for pid in df['citing_id']])
cited_indices  = np.array([id_to_idx[pid] for pid in df['cited_id']])

BATCH = 50000
n = len(df)
similarities = np.zeros(n, dtype=np.float32)
for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    sims = (embeddings[citing_indices[start:end]] * embeddings[cited_indices[start:end]]).sum(axis=1)
    similarities[start:end] = sims
    log(f"  Processed {end:,} / {n:,} pairs")

df['semantic_similarity'] = similarities

log(f"\n  Similarity stats:")
log(f"    mean pos: {df[df['label']==1]['semantic_similarity'].mean():.4f}")
log(f"    mean neg: {df[df['label']==0]['semantic_similarity'].mean():.4f}")
log(f"    overall mean: {similarities.mean():.4f}")
log(f"    std: {similarities.std():.4f}")

# ── Save results ──
log("\nSaving feature matrix...")
df.to_pickle(f"{RESULTS}/lis_pairs_with_features.pkl")
log(f"  Saved {len(df):,} pairs with {len(df.columns)} features")
log(f"  Columns: {list(df.columns)}")

stats = {
    'total_pairs': len(df),
    'positive_pairs': int((df['label'] == 1).sum()),
    'negative_pairs': int((df['label'] == 0).sum()),
    'semantic_similarity_method': 'SBERT all-MiniLM-L6-v2 cosine similarity',
    'mean_semantic_sim_positive': round(float(df[df['label'] == 1]['semantic_similarity'].mean()), 4),
    'mean_semantic_sim_negative': round(float(df[df['label'] == 0]['semantic_similarity'].mean()), 4),
}
with open(f"{RESULTS}/lis_dataset_stats_final.json", 'w') as f:
    json.dump(stats, f, indent=2)
log("  Stats JSON saved.")

# ── Clean up checkpoints ──
import shutil
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
log("  Checkpoints cleaned up.")

log("\nStage 2 complete.")
