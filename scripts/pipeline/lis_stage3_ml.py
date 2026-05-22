"""
Stage 3: ML model training, evaluation, SHAP, and ablation — LIS dataset.
Adapted from neuroblastoma Stage 3 for the Library & Information Science corpus.
"""
import sys
import pandas as pd
import numpy as np
import json, os, warnings, gc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             accuracy_score, roc_auc_score, log_loss,
                             matthews_corrcoef, roc_curve)
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

os.makedirs("/home/ubuntu/lis/figures", exist_ok=True)
os.makedirs("/home/ubuntu/lis/results", exist_ok=True)

def log(msg):
    print(msg, flush=True)

SEED = 42
np.random.seed(SEED)

log("=" * 60)
log("STAGE 3: ML Training & Evaluation — LIS Dataset")
log("=" * 60)

# ── Load data ──
df = pd.read_pickle("/home/ubuntu/lis/results/lis_pairs_with_features.pkl")
log(f"Loaded: {df.shape}")
log(f"Columns: {df.columns.tolist()}")

FEATURE_COLS = ['prestige_cited', 'prestige_citing', 'temporal_distance',
                'coauth_distance', 'semantic_similarity', 'same_journal', 'cited_oa']
FEATURE_LABELS = [
    'Prestige (cited)', 'Prestige (citing)', 'Temporal Distance',
    'Co-auth Distance', 'Semantic Similarity', 'Same Journal', 'Open Access'
]

X = df[FEATURE_COLS].values.copy()
y = df['label'].values

# Cap disconnected co-authorship distance
coauth_idx = FEATURE_COLS.index('coauth_distance')
X[:, coauth_idx] = np.where(X[:, coauth_idx] >= 999, 20.0, X[:, coauth_idx])

log(f"Feature matrix: {X.shape}, Label distribution: {np.bincount(y.astype(int))}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Linear SVM':          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
    'k-NN (k=5)':          KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    'Neural Network':      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED),
}

results = {}
roc_data = {}

log("\nRunning 5-fold cross-validation...")
for name, model in MODELS.items():
    log(f"  {name}...")
    metrics = {k: [] for k in ['f1', 'recall', 'precision', 'accuracy', 'roc_auc', 'log_loss', 'mcc']}
    fold_roc = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        metrics['log_loss'].append(log_loss(y_test, y_proba))
        metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fold_roc.append((fpr.tolist(), tpr.tolist()))
        log(f"    Fold {fold+1}: AUC={metrics['roc_auc'][-1]:.4f}")
    results[name] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                     for k, v in metrics.items()}
    roc_data[name] = fold_roc
    log(f"  → AUC={results[name]['roc_auc']['mean']:.4f} ± {results[name]['roc_auc']['std']:.4f} | F1={results[name]['f1']['mean']:.4f}")

with open("/home/ubuntu/lis/results/lis_cv_results.json", 'w') as f:
    json.dump(results, f, indent=2)
log("CV results saved.")

# ── Best model ──
best_model_name = max(results, key=lambda k: results[k]['roc_auc']['mean'])
log(f"\nBest model: {best_model_name} (AUC={results[best_model_name]['roc_auc']['mean']:.4f})")

# Retrain on full data
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
best_model = MODELS[best_model_name]
best_model.fit(X_scaled, y)

# ── SHAP ──
log("Computing SHAP values (5000 samples)...")
sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
X_sample = X_scaled[sample_idx]

if 'Random Forest' in best_model_name or 'Gradient' in best_model_name:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
else:
    bg = shap.sample(X_scaled, 200)
    explainer = shap.KernelExplainer(best_model.predict_proba, bg)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

shap_importance = np.abs(shap_values).mean(axis=0)
shap_dict = dict(zip(FEATURE_COLS, shap_importance.tolist()))
with open("/home/ubuntu/lis/results/lis_shap_values.json", 'w') as f:
    json.dump(shap_dict, f, indent=2)
log(f"SHAP importance: {shap_dict}")

# ── Ablation ──
log("\nRunning ablation study...")
rf_abl = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
feature_subsets = {
    'All features':              FEATURE_COLS,
    'No semantic similarity':    [f for f in FEATURE_COLS if f != 'semantic_similarity'],
    'No co-authorship distance': [f for f in FEATURE_COLS if f != 'coauth_distance'],
    'No prestige':               [f for f in FEATURE_COLS if 'prestige' not in f],
    'Semantic similarity only':  ['semantic_similarity'],
    'Network features only':     ['coauth_distance', 'prestige_cited', 'prestige_citing'],
    'Temporal + semantic':       ['temporal_distance', 'semantic_similarity'],
}
ablation_results = {}
for subset_name, cols in feature_subsets.items():
    col_idx = [FEATURE_COLS.index(c) for c in cols]
    X_sub = X[:, col_idx]
    aucs = []
    for train_idx, test_idx in skf.split(X_sub, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_sub[train_idx])
        X_te = sc.transform(X_sub[test_idx])
        rf_abl.fit(X_tr, y[train_idx])
        proba = rf_abl.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    ablation_results[subset_name] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}
    log(f"  {subset_name}: AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

with open("/home/ubuntu/lis/results/lis_ablation.json", 'w') as f:
    json.dump(ablation_results, f, indent=2)
log("Ablation results saved.")

# ── Dataset stats ──
log("\nComputing dataset statistics...")
df_raw = pd.read_pickle("/home/ubuntu/upload/Dimensions_LIS_1975_2024.pkl")
log(f"Raw dataset: {df_raw.shape}")

pos = df[df['label'] == 1]
neg = df[df['label'] == 0]

# Co-authorship graph stats from Stage 1 (approximate from pairs data)
# These are computed from the pairs data since we don't have the graph object
stats = {
    'n_articles': int(len(df_raw)),
    'year_min': int(df_raw['year'].min()),
    'year_max': int(df_raw['year'].max()),
    'n_positive_pairs': int(len(pos)),
    'n_negative_pairs': int(len(neg)),
    'n_total_pairs': int(len(df)),
    'pct_abstract': round(float(df_raw['abstract'].notna().mean() * 100), 1),
    'pct_reference_ids': round(float(df_raw['reference_ids'].notna().mean() * 100), 1),
    'mean_times_cited': round(float(df_raw['times_cited'].mean()), 2),
    'median_times_cited': round(float(df_raw['times_cited'].median()), 1),
    'pct_disconnected_coauth': round(float((df['coauth_distance'] >= 20).mean() * 100), 1),
    'mean_temporal_distance_pos': round(float(pos['temporal_distance'].mean()), 2),
    'mean_temporal_distance_neg': round(float(neg['temporal_distance'].mean()), 2),
    'mean_semantic_sim_pos': round(float(pos['semantic_similarity'].mean()), 4),
    'mean_semantic_sim_neg': round(float(neg['semantic_similarity'].mean()), 4),
    'mean_prestige_cited_pos': round(float(pos['prestige_cited'].mean()), 4),
    'mean_prestige_cited_neg': round(float(neg['prestige_cited'].mean()), 4),
    'best_model': best_model_name,
    'best_auc': round(results[best_model_name]['roc_auc']['mean'], 4),
    'best_f1': round(results[best_model_name]['f1']['mean'], 4),
    'best_accuracy': round(results[best_model_name]['accuracy']['mean'], 4),
    'best_precision': round(results[best_model_name]['precision']['mean'], 4),
    'best_recall': round(results[best_model_name]['recall']['mean'], 4),
    'best_mcc': round(results[best_model_name]['mcc']['mean'], 4),
}

with open("/home/ubuntu/lis/results/lis_dataset_stats.json", 'w') as f:
    json.dump(stats, f, indent=2)
log("\nDataset stats:")
log(json.dumps(stats, indent=2))
log("\nStage 3 complete.")
