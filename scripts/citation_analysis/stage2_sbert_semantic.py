"""
Stage 2: SBERT Semantic Similarity (Intel Mac Compatible)
=========================================================
Computes cosine similarity of titles+abstracts using all-MiniLM-L6-v2.
Designed for CPU (Intel Mac) with chunked checkpointing to survive interruptions.

Outputs:
  computations/citation_analysis_scripts/features/{dataset}_pairs_stage2.parquet
"""

import json, os, time, gc, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data_collection" / "data"
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

def process_dataset(dataset_name):
    log(f"\n{'='*50}\nStage 2: {dataset_name}\n{'='*50}")
    
    stage1_path = FEAT_DIR / f"{dataset_name}_pairs_stage1.parquet"
    data_path = DATA_DIR / f"Dimensions_{dataset_name}_1975_2024.parquet"
    
    if not stage1_path.exists() or not data_path.exists():
        log(f"ERROR: Missing input files for {dataset_name}. Run Stage 1 first.")
        return
        
    log(f"Loading pairs from {stage1_path.name}...")
    pairs_df = pd.read_parquet(stage1_path)
    log(f"Loaded {len(pairs_df):,} pairs")
    
    # ── Collect unique IDs to encode ──
    all_ids = list(set(pairs_df["citing_id"].tolist() + pairs_df["cited_id"].tolist()))
    log(f"Unique papers to encode: {len(all_ids):,}")
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
    
    # Save the id mapping so stage2b can use the exact same indexing
    ds_ckpt_dir = CKPT_DIR / dataset_name
    os.makedirs(ds_ckpt_dir, exist_ok=True)
    with open(ds_ckpt_dir / "article_ids.json", "w") as f:
        json.dump(all_ids, f)
    
    # ── Load texts ──
    log(f"Loading texts from {data_path.name}...")
    df_raw = pd.read_parquet(data_path, columns=["id", "title", "abstract"])
    
    def make_text(row):
        title = str(row.title or "")
        abstract = str(row.abstract or "")
        if abstract and abstract != "nan":
            return (title + " " + abstract).strip()
        return title.strip()
        
    df_raw["text"] = df_raw.apply(make_text, axis=1)
    text_map = df_raw.set_index("id")["text"].to_dict()
    del df_raw
    gc.collect()
    
    texts = [str(text_map.get(pid, "")) for pid in all_ids]
    del text_map
    gc.collect()
    
    # ── SBERT Encoding with Checkpoints ──
    ds_ckpt_dir = CKPT_DIR / dataset_name
    os.makedirs(ds_ckpt_dir, exist_ok=True)
    
    emb_file = ds_ckpt_dir / "embeddings.npy"
    prog_file = ds_ckpt_dir / "progress.json"
    
    n_texts = len(texts)
    CHUNK_SIZE = 5000
    n_chunks = (n_texts + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    if emb_file.exists() and prog_file.exists():
        log("Resuming from checkpoint...")
        embeddings = np.load(emb_file)
        with open(prog_file, "r") as f:
            prog = json.load(f)
        start_chunk = prog["completed_chunks"]
        log(f"Resuming from chunk {start_chunk} ({start_chunk*CHUNK_SIZE:,} done)")
    else:
        log("Starting fresh encoding...")
        embeddings = np.zeros((n_texts, 384), dtype=np.float32)
        start_chunk = 0
        
    if start_chunk < n_chunks:
        log("Loading SBERT model (all-MiniLM-L6-v2) on CPU...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        for chunk_idx in range(start_chunk, n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_texts)
            chunk_texts = texts[start:end]
            
            log(f"  Encoding chunk {chunk_idx+1}/{n_chunks} (texts {start:,}-{end:,})...")
            chunk_embs = model.encode(
                chunk_texts,
                batch_size=256,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings[start:end] = chunk_embs
            
            np.save(emb_file, embeddings)
            with open(prog_file, "w") as f:
                json.dump({"completed_chunks": chunk_idx + 1, "total_chunks": n_chunks}, f)
                
        del model
        gc.collect()
        log("Encoding complete.")
    else:
        log("Encoding already complete.")
        
    del texts
    gc.collect()
    
    # ── Compute pairwise cosine similarities ──
    log("Computing pairwise cosine similarities...")
    citing_indices = np.array([id_to_idx[pid] for pid in pairs_df["citing_id"]])
    cited_indices  = np.array([id_to_idx[pid] for pid in pairs_df["cited_id"]])
    
    BATCH = 50000
    n_pairs = len(pairs_df)
    sims = np.zeros(n_pairs, dtype=np.float32)
    
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        # Dot product is equivalent to cosine similarity since embeddings are normalized
        sims[start:end] = (embeddings[citing_indices[start:end]] * embeddings[cited_indices[start:end]]).sum(axis=1)
        
    pairs_df["semantic_similarity"] = sims
    
    pos_sim = pairs_df[pairs_df["label"] == 1]["semantic_similarity"].mean()
    neg_sim = pairs_df[pairs_df["label"] == 0]["semantic_similarity"].mean()
    log(f"Mean similarity: Positive = {pos_sim:.4f}, Negative = {neg_sim:.4f}")
    
    # ── Save ──
    out_path = FEAT_DIR / f"{dataset_name}_pairs_stage2.parquet"
    pairs_df.to_parquet(out_path, index=False)
    log(f"Saved enriched pairs to {out_path.name}")
    
    stats = {
        "dataset": dataset_name,
        "mean_semantic_sim_positive": float(pos_sim),
        "mean_semantic_sim_negative": float(neg_sim)
    }
    with open(FEAT_DIR / f"{dataset_name}_stats_stage2.json", "w") as f:
        json.dump(stats, f, indent=2)
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        for ds in DATASETS:
            process_dataset(ds)
    log("\nStage 2 complete.")
