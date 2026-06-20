"""
Stage 3b: S-T-N_prom Decomposition
====================================
Implements the full seven-model decomposition of citation formation into:
  S   = Semantic proximity        (directional_similarity)
  T   = Temporal recency          (citation_time_gap)
  N   = Citation-network prominence (temporal_indegree + temporal_pagerank)

Seven sub-models trained via 5-fold stratified Gradient Boosting CV:
  1. S only
  2. T only
  3. N only
  4. S + T
  5. S + N
  6. T + N
  7. S + T + N  (full model — same as stage3 baseline)

Marginal contribution deltas:
  Delta_S     = AUC(S+T) - AUC(T)          [semantic gain over temporal alone]
  Delta_T     = AUC(S+T) - AUC(S)          [temporal gain over semantic alone]
  Delta_N|ST  = AUC(S+T+N) - AUC(S+T)     [prominence gain over S+T]

T-N_prom collinearity diagnostics (Pearson & Spearman):
  corr(citation_time_gap, temporal_pagerank)
  corr(citation_time_gap, temporal_indegree)
  — computed separately for positive pairs and negative pairs

Outputs (per dataset, in results/):
  {dataset}_stn_decomposition.json

Summary table (in results/):
  stn_decomposition_summary.csv
  stn_decomposition_summary.md

Run:
  python stage3b_stn_decomposition.py              # all 10 datasets
  python stage3b_stn_decomposition.py CRISPR       # single dataset
"""

import json, os, time, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).parent
FEAT_DIR   = SCRIPT_DIR / "features"
OUT_DIR    = SCRIPT_DIR / "results"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    "protein_folding", "CRISPR",
    "neuroblastoma", "osteosarcoma",
    "additive_manufacturing", "corrosion_protection",
    "income_inequality", "organizational_behavior",
    "film_studies", "memory_studies",
]

# Feature blocks
FEAT_S = ["directional_similarity"]
FEAT_T = ["citation_time_gap"]
FEAT_N = ["temporal_indegree", "temporal_pagerank"]
FEAT_ALL = FEAT_S + FEAT_T + FEAT_N   # order must match parquet columns

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def cv_auc(X, y, skf, seed=SEED):
    """5-fold CV AUC with Gradient Boosting."""
    gb = GradientBoostingClassifier(n_estimators=200, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        gb.fit(sc.fit_transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], gb.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))

def process_dataset(dataset_name):
    log(f"\n{'='*55}\nStage 3b: {dataset_name}\n{'='*55}")

    data_path = FEAT_DIR / f"{dataset_name}_pairs_stage2b.parquet"
    if not data_path.exists():
        log(f"  ERROR: {data_path.name} not found. Skipping.")
        return None

    df = pd.read_parquet(data_path)
    log(f"  Loaded {len(df):,} pairs  |  positives: {df['label'].sum():,}")

    # Verify all required columns are present
    missing = [c for c in FEAT_ALL + ["label"] if c not in df.columns]
    if missing:
        log(f"  ERROR: Missing columns: {missing}. Skipping.")
        return None

    y = df["label"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── Seven sub-models ──────────────────────────────────────────────────────
    models = {
        "S":     FEAT_S,
        "T":     FEAT_T,
        "N":     FEAT_N,
        "S+T":   FEAT_S + FEAT_T,
        "S+N":   FEAT_S + FEAT_N,
        "T+N":   FEAT_T + FEAT_N,
        "S+T+N": FEAT_ALL,
    }

    auc_results = {}
    for label, feats in models.items():
        X = df[feats].values.astype(float)
        mean_auc, std_auc = cv_auc(X, y, skf)
        auc_results[label] = {"mean": mean_auc, "std": std_auc}
        log(f"  AUC({label:<6}) = {mean_auc:.4f} ± {std_auc:.4f}")

    # ── Marginal contribution deltas ─────────────────────────────────────────
    delta_S    = auc_results["S+T"]["mean"]   - auc_results["T"]["mean"]
    delta_T    = auc_results["S+T"]["mean"]   - auc_results["S"]["mean"]
    delta_N_ST = auc_results["S+T+N"]["mean"] - auc_results["S+T"]["mean"]

    log(f"  ΔS (semantic gain over T alone)   = {delta_S:+.4f}")
    log(f"  ΔT (temporal gain over S alone)   = {delta_T:+.4f}")
    log(f"  ΔN|ST (prominence gain over S+T)  = {delta_N_ST:+.4f}")

    # ── T-N_prom collinearity diagnostics ────────────────────────────────────
    collinearity = {}
    for split_label, mask in [("all_pairs",  np.ones(len(df), dtype=bool)),
                               ("positives",  y == 1),
                               ("negatives",  y == 0)]:
        sub = df[mask]
        for n_feat in FEAT_N:
            pearson  = float(sub["citation_time_gap"].corr(sub[n_feat]))
            spear, _ = spearmanr(sub["citation_time_gap"].values,
                                 sub[n_feat].values)
            collinearity[f"{split_label}__T_vs_{n_feat}"] = {
                "pearson":  round(pearson, 4),
                "spearman": round(float(spear), 4),
                "n":        int(mask.sum()),
            }
            log(f"  corr(T, {n_feat}) [{split_label}]: "
                f"Pearson={pearson:.3f}, Spearman={float(spear):.3f}")

    # ── Save per-dataset JSON ─────────────────────────────────────────────────
    out = {
        "dataset":      dataset_name,
        "n_pairs":      int(len(df)),
        "n_positives":  int(y.sum()),
        "auc":          auc_results,
        "deltas": {
            "Delta_S":    round(delta_S, 4),
            "Delta_T":    round(delta_T, 4),
            "Delta_N_ST": round(delta_N_ST, 4),
        },
        "collinearity": collinearity,
    }
    out_path = OUT_DIR / f"{dataset_name}_stn_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"  Saved → {out_path.name}")
    return out

def build_summary(all_results):
    """Build a wide-format summary CSV and Markdown table."""
    rows = []
    for r in all_results:
        if r is None:
            continue
        row = {"dataset": r["dataset"]}
        for model in ["S", "T", "N", "S+T", "S+N", "T+N", "S+T+N"]:
            row[f"AUC_{model}"]     = round(r["auc"][model]["mean"], 4)
            row[f"AUC_{model}_std"] = round(r["auc"][model]["std"],  4)
        row["Delta_S"]    = r["deltas"]["Delta_S"]
        row["Delta_T"]    = r["deltas"]["Delta_T"]
        row["Delta_N_ST"] = r["deltas"]["Delta_N_ST"]
        # Collinearity: T vs temporal_pagerank, all pairs
        key_pr = "all_pairs__T_vs_temporal_pagerank"
        key_id = "all_pairs__T_vs_temporal_indegree"
        row["corr_T_PageRank_pearson"]  = r["collinearity"].get(key_pr, {}).get("pearson",  None)
        row["corr_T_InDegree_pearson"]  = r["collinearity"].get(key_id, {}).get("pearson",  None)
        row["corr_T_PageRank_spearman"] = r["collinearity"].get(key_pr, {}).get("spearman", None)
        row["corr_T_InDegree_spearman"] = r["collinearity"].get(key_id, {}).get("spearman", None)
        rows.append(row)

    df_sum = pd.DataFrame(rows)

    csv_path = OUT_DIR / "stn_decomposition_summary.csv"
    df_sum.to_csv(csv_path, index=False)
    log(f"\nSummary CSV → {csv_path}")

    # Markdown table (main AUC columns + deltas only, no std)
    md_cols = ["dataset",
               "AUC_S", "AUC_T", "AUC_N",
               "AUC_S+T", "AUC_S+N", "AUC_T+N", "AUC_S+T+N",
               "Delta_S", "Delta_T", "Delta_N_ST",
               "corr_T_PageRank_pearson", "corr_T_InDegree_pearson"]
    md_df = df_sum[md_cols].copy()
    md_df.columns = ["Dataset",
                     "S", "T", "N",
                     "S+T", "S+N", "T+N", "S+T+N",
                     "ΔS", "ΔT", "ΔN|ST",
                     "r(T,PR)", "r(T,ID)"]
    md_path = OUT_DIR / "stn_decomposition_summary.md"
    with open(md_path, "w") as f:
        f.write("# S-T-N_prom Decomposition Summary\n\n")
        f.write("AUC values are 5-fold CV means (Gradient Boosting). "
                "ΔS = AUC(S+T)−AUC(T), ΔT = AUC(S+T)−AUC(S), "
                "ΔN|ST = AUC(S+T+N)−AUC(S+T). "
                "r(T,PR) and r(T,ID) are Pearson correlations between "
                "citation_time_gap and temporal_pagerank / temporal_indegree "
                "(all pairs).\n\n")
        f.write(md_df.to_markdown(index=False))
        f.write("\n")
    log(f"Summary Markdown → {md_path}")

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    all_results = []
    for ds in targets:
        result = process_dataset(ds)
        all_results.append(result)

    valid = [r for r in all_results if r is not None]
    if valid:
        build_summary(valid)

    log("\nStage 3b complete.")
