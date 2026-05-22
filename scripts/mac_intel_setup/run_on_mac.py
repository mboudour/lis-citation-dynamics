"""
run_on_mac.py
=============
Self-contained script for Intel-based macOS.
Does two things:
  1. Converts a LIS PKL dataset to a clean Parquet file.
  2. (Optional) Runs SBERT encoding on the abstracts and saves embeddings.

REQUIREMENTS — install once in a fresh conda environment:
  conda create -n lis_mac python=3.10 -y
  conda activate lis_mac
  pip install numpy==1.24.4 pandas==2.0.3 pyarrow==14.0.1 \
              torch==2.1.0 sentence-transformers==2.2.2 scikit-learn tqdm

USAGE:
  python run_on_mac.py --pkl Dimensions_LIS_1975_2024.pkl --mode convert
  python run_on_mac.py --pkl OpenAlex_LIS_1975_2024.pkl  --mode convert
  python run_on_mac.py --pkl OpenAlex_LIS_1975_2024.pkl  --mode sbert
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Silence tokenizer parallelism warning on macOS ──────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── 1. PKL → Parquet conversion ──────────────────────────────────────────────

def convert_pkl_to_parquet(pkl_path: str) -> str:
    """Load a PKL file and save it as a Parquet file in the same directory."""
    pkl_path = Path(pkl_path)
    parquet_path = pkl_path.with_suffix(".parquet")

    print(f"Loading {pkl_path.name} ...")
    df = pd.read_pickle(pkl_path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()}")

    # Cast all object columns to str to avoid pyarrow mixed-type errors
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  Saved → {parquet_path.name}  ({size_mb:.1f} MB)")
    return str(parquet_path)


# ── 2. SBERT encoding ─────────────────────────────────────────────────────────

def run_sbert(pkl_path: str, batch_size: int = 256, chunk_size: int = 5000):
    """
    Encode all abstracts in the dataset using SBERT (all-MiniLM-L6-v2).
    Saves embeddings as a numpy .npy file and a progress JSON checkpoint.
    Resumes automatically if interrupted.
    """
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    pkl_path = Path(pkl_path)
    out_dir = pkl_path.parent / "sbert_output"
    out_dir.mkdir(exist_ok=True)
    emb_path = out_dir / (pkl_path.stem + "_embeddings.npy")
    progress_path = out_dir / "progress.json"

    print(f"Loading {pkl_path.name} ...")
    df = pd.read_pickle(pkl_path)
    print(f"  {df.shape[0]:,} articles")

    # Build text list: title + " " + abstract (fallback to title only)
    texts = []
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        abstract = str(row.get("abstract", "") or "")
        text = (title + " " + abstract).strip() if abstract not in ("", "nan", "None") else title
        texts.append(text)

    n = len(texts)
    n_chunks = (n + chunk_size - 1) // chunk_size

    # Resume from checkpoint
    start_chunk = 0
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        start_chunk = prog.get("last_chunk_done", -1) + 1
        print(f"  Resuming from chunk {start_chunk}/{n_chunks}")

    # Load or initialise embeddings array
    if emb_path.exists() and start_chunk > 0:
        embeddings = np.load(emb_path)
    else:
        embeddings = np.zeros((n, 384), dtype=np.float32)

    print("Loading SBERT model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for i in tqdm(range(start_chunk, n_chunks), desc="Encoding chunks"):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        chunk_texts = texts[start:end]
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings[start:end] = chunk_emb
        np.save(emb_path, embeddings)
        progress_path.write_text(json.dumps({"last_chunk_done": i, "n_chunks": n_chunks}))

    print(f"\nEmbeddings saved → {emb_path}  shape: {embeddings.shape}")
    print("SBERT encoding complete.")
    return str(emb_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LIS dataset utilities for Intel Mac")
    parser.add_argument("--pkl",  required=True, help="Path to the .pkl dataset file")
    parser.add_argument("--mode", required=True, choices=["convert", "sbert"],
                        help="'convert': PKL→Parquet only.  'sbert': SBERT encoding only.")
    parser.add_argument("--batch_size",  type=int, default=256,
                        help="SBERT batch size (default 256; reduce to 64 if OOM)")
    parser.add_argument("--chunk_size",  type=int, default=5000,
                        help="Checkpoint every N texts (default 5000)")
    args = parser.parse_args()

    if not Path(args.pkl).exists():
        print(f"ERROR: file not found: {args.pkl}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "convert":
        convert_pkl_to_parquet(args.pkl)
    elif args.mode == "sbert":
        run_sbert(args.pkl, batch_size=args.batch_size, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
