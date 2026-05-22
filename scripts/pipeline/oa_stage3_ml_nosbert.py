"""
oa_stage3_ml_nosbert.py
-----------------------
Stage 3 ML training WITHOUT semantic similarity feature.
Runs immediately after Stage 1 while SBERT encodes in background.
Results will be merged with SBERT results in the full Stage 3 run.
"""
import pandas as pd
import numpy as np
import json, os, warnings
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
warnings.filterwarnings('ignore')

os.makedirs("/home/ubuntu/lis/results_oa", exist_ok=True)

def log(msg):
    print(msg, flush=True)

SEED = 42
np.random.seed(SEED)

log("=" * 60)
log("STAGE 3 (no SBERT): ML Training — OpenAlex LIS Dataset")
log("=" * 60)

df = pd.read_pickle("/home/ubuntu/lis/results_oa/oa_stage1_pairs.pkl")
log(f"Loaded: {df.shape}")

# Features WITHOUT semantic similarity
FEATURE_COLS = ['prestige_cited', 'prestige_citing', 'temporal_distance',
                'coauth_distance', 'same_journal', 'is_oa']
FEATURE_LABELS = [
    'Prestige (cited)', 'Prestige (citing)', 'Temporal Distance',
    'Co-auth Distance', 'Same Journal', 'Open Access'
]

X = df[FEATURE_COLS].values.copy()
y = df['label'].values
coauth_idx = FEATURE_COLS.index('coauth_distance')
X[:, coauth_idx] = np.where(X[:, coauth_idx] >= 999, 20.0, X[:, coauth_idx])

log(f"Feature matrix: {X.shape} (6 features, no semantic similarity)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Linear SVM':          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
    'k-NN (k=5)':          KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    'Neural Network':      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED),
}

results_nosbert = {}

log("\nRunning 5-fold CV (no semantic similarity)...")
for name, model in MODELS.items():
    log(f"  {name}...")
    metrics = {k: [] for k in ['f1', 'recall', 'precision', 'accuracy', 'roc_auc', 'log_loss', 'mcc']}
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model.fit(X_tr, y[train_idx])
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        metrics['f1'].append(f1_score(y[test_idx], y_pred))
        metrics['recall'].append(recall_score(y[test_idx], y_pred))
        metrics['precision'].append(precision_score(y[test_idx], y_pred))
        metrics['accuracy'].append(accuracy_score(y[test_idx], y_pred))
        metrics['roc_auc'].append(roc_auc_score(y[test_idx], y_proba))
        metrics['log_loss'].append(log_loss(y[test_idx], y_proba))
        metrics['mcc'].append(matthews_corrcoef(y[test_idx], y_pred))
        log(f"    Fold {fold+1}: AUC={metrics['roc_auc'][-1]:.4f}")
    results_nosbert[name] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                              for k, v in metrics.items()}
    log(f"  → AUC={results_nosbert[name]['roc_auc']['mean']:.4f} ± {results_nosbert[name]['roc_auc']['std']:.4f}")

with open("/home/ubuntu/lis/results_oa/oa_cv_results_nosbert.json", 'w') as f:
    json.dump(results_nosbert, f, indent=2)
log("\nNo-SBERT CV results saved.")
log("Stage 3 (no SBERT) complete.")
