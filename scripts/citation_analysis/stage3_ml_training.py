"""
Stage 3: ML Training, SHAP, Ablation, Temporal Hold-out
=========================================================
Trains classifiers on the Stage 2 pairs for each of the 10 datasets.
Follows the methodology of the LIS manuscript:
1. 5-fold Stratified CV for LR, SVM, RF, GB.
2. Metrics: AUC, PR-AUC, F1, Accuracy, Precision, Recall, MCC, Log-Loss.
3. SHAP on best model (Gradient Boosting).
4. Leave-one-out ablation (Random Forest).
5. Temporal hold-out evaluation (Train <= 2015, Test 2016-2020).

Outputs:
  computations/citation_analysis_scripts/results/{dataset}_cv.json
  computations/citation_analysis_scripts/results/{dataset}_shap.json
  computations/citation_analysis_scripts/results/{dataset}_ablation.json
  computations/citation_analysis_scripts/results/{dataset}_temporal.json
"""

import json, os, time, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, precision_score,
                             recall_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import StandardScaler
import shap

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).parent
FEAT_DIR = SCRIPT_DIR / "features"
OUT_DIR = SCRIPT_DIR / "results"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    "protein_folding", "CRISPR",
    "neuroblastoma", "osteosarcoma",
    "additive_manufacturing", "corrosion_protection",
    "income_inequality", "organizational_behavior",
    "film_studies", "memory_studies",
]

FEATURES = ["temporal_indegree", "citation_time_gap", "temporal_pagerank", "directional_similarity"]

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def process_dataset(dataset_name):
    log(f"\n{'='*50}\nStage 3: {dataset_name}\n{'='*50}")
    
    data_path = FEAT_DIR / f"{dataset_name}_pairs_stage2b.parquet"
    if not data_path.exists():
        log(f"ERROR: {data_path.name} not found. Run Stage 2 first.")
        return
        
    log(f"Loading pairs from {data_path.name}...")
    df = pd.read_parquet(data_path)
    log(f"Loaded {len(df):,} pairs")
    
    X = df[FEATURES].values.astype(float)
    y = df["label"].values
    
    # ── 5-fold CV ──
    log("Running 5-fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    }
    
    cv_results = {}
    for name, model in models.items():
        log(f"  Training {name}...")
        fold_metrics = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
            
            model.fit(Xtr, ytr)
            yprob = model.predict_proba(Xte)[:, 1]
            ypred = model.predict(Xte)
            
            fold_metrics.append({
                "auc":       roc_auc_score(yte, yprob),
                "pr_auc":    average_precision_score(yte, yprob),
                "f1":        f1_score(yte, ypred),
                "accuracy":  accuracy_score(yte, ypred),
                "precision": precision_score(yte, ypred),
                "recall":    recall_score(yte, ypred),
                "mcc":       matthews_corrcoef(yte, ypred),
                "log_loss":  log_loss(yte, yprob)
            })
            
        means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        stds  = {k: float(np.std( [m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        cv_results[name] = {"mean": means, "std": stds, "folds": fold_metrics}
        log(f"    Mean AUC = {means['auc']:.4f} ± {stds['auc']:.4f}")
        
    with open(OUT_DIR / f"{dataset_name}_cv.json", "w") as f:
        json.dump(cv_results, f, indent=2)
        
    # ── SHAP ──
    log("Computing SHAP values (Gradient Boosting)...")
    scaler_full = StandardScaler()
    Xsc = scaler_full.fit_transform(X)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
    gb.fit(Xsc, y)
    
    explainer = shap.TreeExplainer(gb)
    # Use a sample of 5000 to keep it fast
    sample_size = min(5000, len(Xsc))
    shap_vals = explainer.shap_values(Xsc[:sample_size])
    mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
    
    shap_out = {"features": FEATURES, "mean_abs_shap": mean_abs}
    with open(OUT_DIR / f"{dataset_name}_shap.json", "w") as f:
        json.dump(shap_out, f, indent=2)
        
    # ── Ablation ──
    log("Running leave-one-out ablation (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    ablation = {}
    
    base_aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        rf.fit(sc.fit_transform(X[tr]), y[tr])
        base_aucs.append(roc_auc_score(y[te], rf.predict_proba(sc.transform(X[te]))[:,1]))
    ablation["all_features"] = float(np.mean(base_aucs))
    log(f"  Baseline AUC: {ablation['all_features']:.4f}")
    
    for i, feat in enumerate(FEATURES):
        mask = [j for j in range(len(FEATURES)) if j != i]
        Xm = X[:, mask]
        aucs = []
        for tr, te in skf.split(Xm, y):
            sc = StandardScaler()
            rf.fit(sc.fit_transform(Xm[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], rf.predict_proba(sc.transform(Xm[te]))[:,1]))
        ablation[f"without_{feat}"] = float(np.mean(aucs))
        drop = ablation["all_features"] - ablation[f"without_{feat}"]
        log(f"  without {feat}: AUC={ablation[f'without_{feat}']:.4f} (drop={drop:.4f})")
        
    with open(OUT_DIR / f"{dataset_name}_ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)
        
    # ── Temporal hold-out ──
    log("Running temporal hold-out (Train <=2015, Test 2016-2020)...")
    train_mask = df["citing_year"] <= 2015
    test_mask = (df["citing_year"] >= 2016) & (df["citing_year"] <= 2020)
    
    if test_mask.sum() > 0 and train_mask.sum() > 0:
        Xtr_t, ytr_t = X[train_mask], y[train_mask]
        Xte_t, yte_t = X[test_mask], y[test_mask]
        log(f"  Train pairs: {train_mask.sum():,} | Test pairs: {test_mask.sum():,}")
        
        temporal_results = {}
        for name, model in models.items():
            sc = StandardScaler()
            model.fit(sc.fit_transform(Xtr_t), ytr_t)
            yprob = model.predict_proba(sc.transform(Xte_t))[:, 1]
            ypred = model.predict(sc.transform(Xte_t))
            
            temporal_results[name] = {
                "auc":      float(roc_auc_score(yte_t, yprob)),
                "pr_auc":   float(average_precision_score(yte_t, yprob)),
                "f1":       float(f1_score(yte_t, ypred)),
                "mcc":      float(matthews_corrcoef(yte_t, ypred)),
                "log_loss": float(log_loss(yte_t, yprob))
            }
            log(f"    {name} temporal AUC: {temporal_results[name]['auc']:.4f}")
            
        with open(OUT_DIR / f"{dataset_name}_temporal.json", "w") as f:
            json.dump(temporal_results, f, indent=2)
    else:
        log("  Skipping temporal hold-out: Insufficient data in time ranges.")
        
    # ── S / N / SN Model Decomposition (Gradient Boosting) ──
    log("Running S / N / SN decomposition (Gradient Boosting 5-fold CV)...")
    idx_S = FEATURES.index("directional_similarity")
    idx_N = [i for i in range(len(FEATURES)) if i != idx_S]
    
    X_S = X[:, [idx_S]]
    X_N = X[:, idx_N]
    
    decomp_results = {"S": [], "N": [], "SN": []}
    gb_decomp = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
    
    for tr, te in skf.split(X, y):
        # Model S
        sc_S = StandardScaler()
        gb_decomp.fit(sc_S.fit_transform(X_S[tr]), y[tr])
        decomp_results["S"].append(roc_auc_score(y[te], gb_decomp.predict_proba(sc_S.transform(X_S[te]))[:,1]))
        
        # Model N
        sc_N = StandardScaler()
        gb_decomp.fit(sc_N.fit_transform(X_N[tr]), y[tr])
        decomp_results["N"].append(roc_auc_score(y[te], gb_decomp.predict_proba(sc_N.transform(X_N[te]))[:,1]))
        
        # Model SN (all features)
        sc_SN = StandardScaler()
        gb_decomp.fit(sc_SN.fit_transform(X[tr]), y[tr])
        decomp_results["SN"].append(roc_auc_score(y[te], gb_decomp.predict_proba(sc_SN.transform(X[te]))[:,1]))
        
    decomp_summary = {
        "AUC_S": float(np.mean(decomp_results["S"])),
        "AUC_N": float(np.mean(decomp_results["N"])),
        "AUC_SN": float(np.mean(decomp_results["SN"]))
    }
    log(f"  Decomposition AUC: S={decomp_summary['AUC_S']:.4f}, N={decomp_summary['AUC_N']:.4f}, SN={decomp_summary['AUC_SN']:.4f}")
    
    with open(OUT_DIR / f"{dataset_name}_decomposition.json", "w") as f:
        json.dump(decomp_summary, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        for ds in DATASETS:
            process_dataset(ds)
    log("\nStage 3 complete.")
