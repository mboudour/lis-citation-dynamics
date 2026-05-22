"""
Ablation study using Random Forest (fast, n_jobs=-1).
Uses AUC from the full-feature RF (0.9972) as baseline.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

WORK_DIR  = Path("/home/ubuntu/oa_work")
OUT_DIR   = WORK_DIR / "results"
SEED      = 42
FEAT_COLS = ["prestige_cited","activity_citing","temporal_gap",
             "common_refs","common_citers","jaccard_refs","semantic_similarity"]

print("Loading pairs...", flush=True)
df = pd.read_parquet(OUT_DIR / "oa_pairs_with_sbert.parquet")
X  = df[FEAT_COLS].values.astype("float32")
y  = df["label"].values.astype(int)
print(f"  {len(df):,} pairs", flush=True)

cv = json.load(open(OUT_DIR / "oa_cv_results.json"))
base_auc = cv["Random Forest"]["roc_auc"]["mean"]
ablation = {"all_features": round(base_auc, 4)}
print(f"Baseline (Random Forest, all features): AUC={base_auc:.4f}", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for drop_feat in FEAT_COLS:
    keep = [f for f in FEAT_COLS if f != drop_feat]
    Xi   = df[keep].values.astype("float32")
    aucs = []
    for tr, te in skf.split(Xi, y):
        m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
        m.fit(Xi[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xi[te])[:,1]))
    v = float(np.mean(aucs))
    ablation[f"without_{drop_feat}"] = round(v, 4)
    print(f"  without {drop_feat}: AUC={v:.4f}  (drop={base_auc-v:+.4f})", flush=True)

with open(OUT_DIR / "oa_ablation_results.json","w") as f:
    json.dump({"model":"Random Forest","ablation":ablation}, f, indent=2)
print("\nSaved: oa_ablation_results.json", flush=True)
print(json.dumps(ablation, indent=2))
