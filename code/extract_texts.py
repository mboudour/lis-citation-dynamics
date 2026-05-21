"""
Extract title+abstract text for each paper from the LIS dataset.
Saves a lightweight CSV (id, text) to avoid loading the full 1.2GB pickle in Stage 2.
"""
import pandas as pd
import gc

print("Loading LIS dataset...", flush=True)
df = pd.read_pickle("/home/ubuntu/upload/Dimensions_LIS_1975_2024.pkl")
print(f"  Loaded {len(df):,} records", flush=True)
print(f"  Columns: {df.columns.tolist()}", flush=True)

def make_text(row):
    title = str(row.get('title', '') or '')
    abstract = str(row.get('abstract', '') or '')
    if abstract and abstract != 'nan':
        return (title + ' ' + abstract).strip()
    return title.strip()

print("Building text column...", flush=True)
df['text'] = df.apply(make_text, axis=1)
texts_df = df[['id', 'text']].copy()
del df
gc.collect()

print(f"  Saving {len(texts_df):,} texts to CSV...", flush=True)
texts_df.to_csv("/home/ubuntu/lis/results/lis_texts.csv", index=False)
print(f"  Done. File size: ", flush=True)

import os
size = os.path.getsize("/home/ubuntu/lis/results/lis_texts.csv")
print(f"  {size / 1024**2:.1f} MB", flush=True)
print("Text extraction complete.", flush=True)
