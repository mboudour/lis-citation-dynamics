"""
OpenAlex Stage 2: SBERT Encoding + Semantic Similarity
Encodes all texts with all-MiniLM-L6-v2, saves checkpoints every 5000 texts,
then computes cosine similarity for each pair in oa_pairs_features.parquet.
"""
import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

WORK_DIR   = Path("/home/ubuntu/oa_work")
OUT_DIR    = WORK_DIR / "results"
CKPT_DIR   = WORK_DIR / "checkpoints"
CHUNK_SIZE = 5000

# ── Load texts ───────────────────────────────────────────────────────────────
print("Loading texts...")
texts_df = pd.read_parquet(OUT_DIR / "oa_texts.parquet")
texts    = texts_df["text"].tolist()
ids      = texts_df["id"].tolist()
N        = len(texts)
print(f"  {N:,} texts to encode")

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading SBERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded.")

# ── Checkpoint resume ────────────────────────────────────────────────────────
prog_file = CKPT_DIR / "progress.json"
if prog_file.exists():
    with open(prog_file) as f:
        prog = json.load(f)
    chunks_done = prog["chunks_done"]
else:
    chunks_done = 0

n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"  Chunks: {n_chunks}  Already done: {chunks_done}")

# ── Encode ───────────────────────────────────────────────────────────────────
for chunk_idx in range(chunks_done, n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end   = min(start + CHUNK_SIZE, N)
    batch = texts[start:end]
    embs  = model.encode(batch, batch_size=256, show_progress_bar=False,
                         convert_to_numpy=True)
    ckpt_path = CKPT_DIR / f"chunk_{chunk_idx:04d}.npy"
    np.save(ckpt_path, embs)
    chunks_done = chunk_idx + 1
    with open(prog_file, "w") as f:
        json.dump({"chunks_done": chunks_done, "total_chunks": n_chunks}, f)
    pct = 100 * end / N
    print(f"  Chunk {chunk_idx+1}/{n_chunks} ({pct:.1f}%) — {end:,}/{N:,} encoded")

# ── Assemble full embedding matrix ───────────────────────────────────────────
print("\nAssembling embedding matrix...")
parts = []
for chunk_idx in range(n_chunks):
    parts.append(np.load(CKPT_DIR / f"chunk_{chunk_idx:04d}.npy"))
embeddings = np.vstack(parts)
print(f"  Shape: {embeddings.shape}")

# Map id → embedding index
id_to_emb = {pid: i for i, pid in enumerate(ids)}

# ── Load pairs ───────────────────────────────────────────────────────────────
print("Loading pairs...")
pairs_df = pd.read_parquet(OUT_DIR / "oa_pairs_features.parquet")
print(f"  {len(pairs_df):,} pairs")

# Map pair indices to paper IDs
data_df = pd.read_parquet(WORK_DIR / "oa_data.parquet")
idx_to_id = data_df["id"].to_dict()

# ── Compute cosine similarity in batches ─────────────────────────────────────
print("Computing semantic similarity...")
BATCH = 10000
sims = np.zeros(len(pairs_df), dtype=np.float32)

citing_ids = [idx_to_id.get(int(i), "") for i in pairs_df["citing_idx"]]
cited_ids  = [idx_to_id.get(int(i), "") for i in pairs_df["cited_idx"]]

for start in range(0, len(pairs_df), BATCH):
    end   = min(start + BATCH, len(pairs_df))
    c_ids = citing_ids[start:end]
    r_ids = cited_ids[start:end]
    c_embs = np.array([embeddings[id_to_emb[i]] if i in id_to_emb
                       else np.zeros(384) for i in c_ids])
    r_embs = np.array([embeddings[id_to_emb[i]] if i in id_to_emb
                       else np.zeros(384) for i in r_ids])
    # Row-wise cosine similarity
    norms_c = np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-10
    norms_r = np.linalg.norm(r_embs, axis=1, keepdims=True) + 1e-10
    sims[start:end] = (c_embs / norms_c * r_embs / norms_r).sum(axis=1)
    if start % 100000 == 0:
        print(f"  {start:,}/{len(pairs_df):,}")

pairs_df["semantic_similarity"] = sims

# ── Stats ────────────────────────────────────────────────────────────────────
pos_sim = sims[pairs_df["label"] == 1].mean()
neg_sim = sims[pairs_df["label"] == 0].mean()
print(f"\nMean cosine sim — positive: {pos_sim:.4f}  negative: {neg_sim:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "oa_pairs_with_sbert.parquet"
pairs_df.to_parquet(out_path, index=False)
print(f"Saved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

sbert_stats = {
    "n_texts_encoded": int(N),
    "embedding_dim": int(embeddings.shape[1]),
    "mean_sim_positive": round(float(pos_sim), 4),
    "mean_sim_negative": round(float(neg_sim), 4),
}
with open(OUT_DIR / "oa_sbert_stats.json", "w") as f:
    json.dump(sbert_stats, f, indent=2)
print("Saved: oa_sbert_stats.json")
print(json.dumps(sbert_stats, indent=2))
