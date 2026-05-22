"""
OpenAlex Stage 3: ML Training, SHAP, Ablation
5-fold CV for 6 classifiers using all 7 features (6 structural + semantic_similarity).
Computes: AUC, F1, Accuracy, Precision, Recall, Log-loss, MCC.
Runs SHAP feature importance on best model.
Runs ablation: remove one feature at a time.
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score,
                             log_loss, matthews_corrcoef)
import shap
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = Path("/home/ubuntu/oa_work")
OUT_DIR  = WORK_DIR / "results"
SEED     = 42

print("Loading pairs with SBERT features...")
df = pd.read_parquet(OUT_DIR / "oa_pairs_with_sbert.parquet")
print(f"  Shape: {df.shape}")

FEAT_COLS = ["prestige_cited","activity_citing","temporal_gap",
             "common_refs","common_citers","jaccard_refs","semantic_similarity"]

X = df[FEAT_COLS].values.astype(np.float32)
y = df["label"].values.astype(int)
print(f"  Features: {FEAT_COLS}")
print(f"  Positive: {y.sum():,}  Negative: {(1-y).sum():,}")

# ── Classifiers ──────────────────────────────────────────────────────────────
def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=SEED),
    }

# ── 5-fold CV ────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

for name, model in make_models().items():
    print(f"\n{name}...", flush=True)
    fold_metrics = {m: [] for m in ["roc_auc","f1","accuracy","precision",
                                     "recall","log_loss","mcc"]}
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        model.fit(X[tr], y[tr])
        y_prob = model.predict_proba(X[te])[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        fold_metrics["roc_auc"].append(roc_auc_score(y[te], y_prob))
        fold_metrics["f1"].append(f1_score(y[te], y_pred))
        fold_metrics["accuracy"].append(accuracy_score(y[te], y_pred))
        fold_metrics["precision"].append(precision_score(y[te], y_pred))
        fold_metrics["recall"].append(recall_score(y[te], y_pred))
        fold_metrics["log_loss"].append(log_loss(y[te], y_prob))
        fold_metrics["mcc"].append(matthews_corrcoef(y[te], y_pred))
        print(f"  Fold {fold+1}: AUC={fold_metrics['roc_auc'][-1]:.4f}", flush=True)
    cv_results[name] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for m, v in fold_metrics.items()}
    print(f"  → AUC={cv_results[name]['roc_auc']['mean']:.4f} "
          f"± {cv_results[name]['roc_auc']['std']:.4f}", flush=True)

with open(OUT_DIR / "oa_cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)
print("\nSaved: oa_cv_results.json")

# ── SHAP on best model (Gradient Boosting or Neural Network) ─────────────────
best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc"]["mean"])
print(f"\nRunning SHAP on best model: {best_name}")

best_models = make_models()
best_model  = best_models[best_name]
best_model.fit(X, y)

# Use a subsample for SHAP
rng   = np.random.default_rng(SEED)
idx_s = rng.choice(len(X), size=min(5000, len(X)), replace=False)
X_s   = X[idx_s]

# Get the underlying estimator if Pipeline
if hasattr(best_model, "named_steps"):
    scaler = best_model.named_steps.get("scaler")
    clf    = best_model.named_steps["clf"]
    X_shap = scaler.transform(X_s) if scaler else X_s
else:
    clf    = best_model
    X_shap = X_s

try:
    if hasattr(clf, "predict_proba"):
        explainer = shap.TreeExplainer(clf) if hasattr(clf, "estimators_") \
                    else shap.Explainer(clf.predict_proba, X_shap)
    else:
        explainer = shap.Explainer(clf, X_shap)
    shap_vals = explainer(X_shap)
    if hasattr(shap_vals, "values"):
        sv = np.abs(shap_vals.values)
        if sv.ndim == 3:
            sv = sv[:,:,1]
    else:
        sv = np.abs(shap_vals)
    mean_shap = sv.mean(axis=0)
    shap_dict = {feat: float(mean_shap[i]) for i, feat in enumerate(FEAT_COLS)}
except Exception as e:
    print(f"  SHAP error: {e} — using permutation importance fallback")
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best_model, X_s, y[idx_s], n_repeats=5,
                                random_state=SEED, scoring="roc_auc")
    shap_dict = {feat: float(pi.importances_mean[i])
                 for i, feat in enumerate(FEAT_COLS)}

shap_dict = dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))
with open(OUT_DIR / "oa_shap_values.json", "w") as f:
    json.dump({"model": best_name, "shap_mean_abs": shap_dict}, f, indent=2)
print("Saved: oa_shap_values.json")
print(json.dumps(shap_dict, indent=2))

# ── Ablation study ───────────────────────────────────────────────────────────
print(f"\nAblation study (leave-one-out features) on {best_name}...")
ablation = {}
# Baseline: all features
base_auc = cv_results[best_name]["roc_auc"]["mean"]
ablation["all_features"] = round(base_auc, 4)

for drop_feat in FEAT_COLS:
    keep = [f for f in FEAT_COLS if f != drop_feat]
    Xi   = df[keep].values.astype(np.float32)
    fold_aucs = []
    abl_model = make_models()[best_name]
    for tr, te in skf.split(Xi, y):
        abl_model.fit(Xi[tr], y[tr])
        y_prob = abl_model.predict_proba(Xi[te])[:,1]
        fold_aucs.append(roc_auc_score(y[te], y_prob))
    abl_auc = float(np.mean(fold_aucs))
    ablation[f"without_{drop_feat}"] = round(abl_auc, 4)
    drop = round(base_auc - abl_auc, 4)
    print(f"  without {drop_feat}: AUC={abl_auc:.4f}  (drop={drop:+.4f})")

with open(OUT_DIR / "oa_ablation_results.json", "w") as f:
    json.dump({"model": best_name, "ablation": ablation}, f, indent=2)
print("Saved: oa_ablation_results.json")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== FINAL RESULTS ===")
print(f"{'Model':<22} {'AUC':>7} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'MCC':>7}")
for name, res in cv_results.items():
    print(f"{name:<22} "
          f"{res['roc_auc']['mean']:>7.4f} "
          f"{res['f1']['mean']:>7.4f} "
          f"{res['accuracy']['mean']:>7.4f} "
          f"{res['precision']['mean']:>7.4f} "
          f"{res['recall']['mean']:>7.4f} "
          f"{res['mcc']['mean']:>7.4f}")
