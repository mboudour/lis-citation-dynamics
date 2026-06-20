"""
Stage 5: Per-Dataset Multiplots (4×3 grids, 10 panels + 2 hidden)
==================================================================
Reads the Stage 2b pair files and Stage 3 JSON results and produces
four publication-quality multiplot figures, each a 4×3 grid of subplots
(one panel per dataset, last 2 cells hidden):

  fig5_roc_multiplot_.png      – CV ROC curves (all 4 classifiers)
  fig6_rolling_multiplot_.png  – Rolling temporal AUC (Gradient Boosting)
  fig7_ablation_multiplot_.png – Leave-one-out ablation AUC drop per feature
  fig8_shap_multiplot_.png     – Normalized SHAP importance per feature

All outputs go to:
  <script_dir>/figures/

Usage:
  python stage5_multiplots.py
"""

import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
FEAT_DIR    = SCRIPT_DIR / "features"
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR     = SCRIPT_DIR / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────
DATASETS = [
    "protein_folding", "CRISPR",
    "neuroblastoma", "osteosarcoma",
    "additive_manufacturing", "corrosion_protection",
    "income_inequality", "organizational_behavior",
    "film_studies", "memory_studies",
]

LABELS = {
    "protein_folding":          "Protein Folding",
    "CRISPR":                   "CRISPR",
    "neuroblastoma":            "Neuroblastoma",
    "osteosarcoma":             "Osteosarcoma",
    "additive_manufacturing":   "Additive Manuf.",
    "corrosion_protection":     "Corrosion Prot.",
    "income_inequality":        "Income Inequality",
    "organizational_behavior":  "Org. Behavior",
    "film_studies":             "Film Studies",
    "memory_studies":           "Memory Studies",
}

DISCIPLINE_COLORS = {
    "protein_folding":          "#1f77b4",   # Science – blue
    "CRISPR":                   "#1f77b4",
    "neuroblastoma":            "#2ca02c",   # BioMed – green
    "osteosarcoma":             "#2ca02c",
    "additive_manufacturing":   "#ff7f0e",   # Engineering – orange
    "corrosion_protection":     "#ff7f0e",
    "income_inequality":        "#d62728",   # Social Science – red
    "organizational_behavior":  "#d62728",
    "film_studies":             "#9467bd",   # Humanities – purple
    "memory_studies":           "#9467bd",
}

FEATURES = ["temporal_indegree", "citation_time_gap", "temporal_pagerank", "directional_similarity"]

FEATURE_LABELS = {
    "temporal_indegree":       "Temp. Indegree",
    "citation_time_gap":       "Citation Lag",
    "temporal_pagerank":       "Temp. PageRank",
    "directional_similarity":  "Dir. Sim.",
}

MODEL_COLORS = {
    "Logistic Regression": "#4878cf",
    "Linear SVM":          "#6acc65",
    "Random Forest":       "#d65f5f",
    "Gradient Boosting":   "#b47cc7",
}

# ── Rolling windows ────────────────────────────────────────────────────────
# Windows are generated dynamically per dataset based on its actual year range,
# so post-2012 datasets (CRISPR, additive_manufacturing) still get 4 points.
MIN_TRAIN = 2   # minimum training samples for a window to be used
MIN_TEST  = 2   # minimum test samples for a window to be used
N_WINDOWS = 4   # target number of rolling windows per dataset

def make_windows(years_array):
    """Generate N_WINDOWS rolling windows over the dataset's actual year range.

    Strategy: divide [y_min, y_max] into N_WINDOWS+1 equal segments.
    For window i, training uses all years up to boundary[i],
    and testing uses years in (boundary[i], boundary[i+1]].
    This guarantees each window has a non-empty training set (all prior data)
    and a non-empty test set (the next segment).
    """
    y_min = int(years_array.min())
    y_max = int(years_array.max())
    span  = y_max - y_min
    if span < N_WINDOWS + 1:
        # Too short: use 1-year test windows from y_min+1 onward
        windows = []
        for i in range(N_WINDOWS):
            train_end  = y_min + i
            test_start = train_end + 1
            test_end   = test_start
            if test_end > y_max:
                break
            windows.append((f"{test_start}-{test_end}", train_end, test_start, test_end))
        return windows
    # Place N_WINDOWS+1 evenly-spaced boundary points
    boundaries = [y_min + round(i * span / N_WINDOWS) for i in range(N_WINDOWS + 1)]
    windows = []
    for i in range(N_WINDOWS):
        train_end  = boundaries[i]
        test_start = boundaries[i] + 1
        test_end   = boundaries[i + 1]
        label = f"{test_start}-{test_end}"
        windows.append((label, train_end, test_start, test_end))
    return windows

# ══════════════════════════════════════════════════════════════════════════
# Helper: load pairs
# ══════════════════════════════════════════════════════════════════════════
def load_pairs(dataset):
    """Load the richest available pair file for a dataset.

    Priority: stage2b > merge(stage1+stage2) > stage2 > stage1.
    Ensures directional_similarity column always present (filled 0 if absent).
    """
    path2b = FEAT_DIR / f"{dataset}_pairs_stage2b.parquet"
    path2  = FEAT_DIR / f"{dataset}_pairs_stage2.parquet"
    path1  = FEAT_DIR / f"{dataset}_pairs_stage1.parquet"

    if path2b.exists():
        return pd.read_parquet(path2b)

    if path2.exists() and path1.exists():
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)
        if "directional_similarity" in df2.columns:
            df = df1.merge(
                df2[["citing_id", "cited_id", "directional_similarity"]],
                on=["citing_id", "cited_id"], how="left"
            )
            df["directional_similarity"] = df["directional_similarity"].fillna(0.0)
            return df
        return df1

    if path2.exists():
        df = pd.read_parquet(path2)
        if "directional_similarity" not in df.columns:
            df["directional_similarity"] = 0.0
        return df

    if path1.exists():
        df = pd.read_parquet(path1)
        if "directional_similarity" not in df.columns:
            df["directional_similarity"] = 0.0
        return df

    return None


def _hide_extra_axes(axes, n_used):
    for ax in axes[n_used:]:
        ax.set_visible(False)


# ══════════════════════════════════════════════════════════════════════════
# Fig 5 – CV ROC multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_roc_multiplot():
    print("Generating fig5_roc_multiplot_.png ...")
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        df = load_pairs(ds)
        if df is None:
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        X = df[FEATURES].values.astype(float)
        y = df["label"].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
            "Linear SVM":          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
            "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        }

        for mname, model in models.items():
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 200)
            for tr, te in skf.split(X, y):
                sc = StandardScaler()
                model.fit(sc.fit_transform(X[tr]), y[tr])
                yp = model.predict_proba(sc.transform(X[te]))[:, 1]
                fpr, tpr, _ = roc_curve(y[te], yp)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(roc_auc_score(y[te], yp))
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = np.mean(aucs)
            ax.plot(mean_fpr, mean_tpr,
                    color=MODEL_COLORS[mname], lw=1.5,
                    label=f"{mname} ({mean_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower left")

    _hide_extra_axes(axes, len(DATASETS))
    fig.suptitle("CV ROC Curves Across 10 Datasets", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig5_roc_multiplot_.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 6 – Rolling temporal AUC multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_rolling_multiplot():
    print("Generating fig6_rolling_multiplot_.png ...")
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        df = load_pairs(ds)
        if df is None:
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        if "citing_year" not in df.columns:
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No year col", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        X     = df[FEATURES].values.astype(float)
        y     = df["label"].values
        years = df["citing_year"].values
        gb    = GradientBoostingClassifier(n_estimators=100, random_state=SEED)

        windows = make_windows(years)
        window_labels, window_aucs = [], []
        for wlabel, train_end, test_start, test_end in windows:
            tr_mask = years <= train_end
            te_mask = (years >= test_start) & (years <= test_end)
            if tr_mask.sum() < MIN_TRAIN or te_mask.sum() < MIN_TEST:
                continue
            if len(np.unique(y[te_mask])) < 2:
                continue
            sc = StandardScaler()
            gb.fit(sc.fit_transform(X[tr_mask]), y[tr_mask])
            yp  = gb.predict_proba(sc.transform(X[te_mask]))[:, 1]
            auc = roc_auc_score(y[te_mask], yp)
            window_labels.append(wlabel)
            window_aucs.append(auc)

        if window_aucs:
            color = DISCIPLINE_COLORS[ds]
            ax.plot(range(len(window_labels)), window_aucs,
                    marker="o", color=color, lw=2, ms=6)
            ax.set_xticks(range(len(window_labels)))
            ax.set_xticklabels(window_labels, rotation=30, ha="right", fontsize=7)
            y_floor = min(0.35, min(window_aucs) - 0.05)
            ax.set_ylim([y_floor, 1.02])
            for i, v in enumerate(window_aucs):
                ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_ylabel("AUC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    _hide_extra_axes(axes, len(DATASETS))
    fig.suptitle("Rolling Temporal AUC (Gradient Boosting) Across 10 Datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig6_rolling_multiplot_.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 7 – Ablation multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_ablation_multiplot():
    print("Generating fig7_ablation_multiplot_.png ...")
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        path = RESULTS_DIR / f"{ds}_ablation.json"
        if not path.exists():
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        with open(path) as f:
            ablation = json.load(f)

        baseline = ablation.get("all_features", 0)
        drops, feat_names = [], []
        for feat in FEATURES:
            key = f"without_{feat}"
            if key in ablation:
                drop = max(0.0, baseline - ablation[key])
                drops.append(drop)
                feat_names.append(FEATURE_LABELS[feat])

        if not drops:
            ax.text(0.5, 0.5, "No ablation data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(LABELS[ds])
            continue

        color = DISCIPLINE_COLORS[ds]
        bars = ax.barh(feat_names, drops, color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, drops):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)

        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_xlabel("AUC Drop", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    _hide_extra_axes(axes, len(DATASETS))
    fig.suptitle("Feature Ablation: AUC Drop per Feature Across 10 Datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig7_ablation_multiplot_.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 8 – SHAP multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_shap_multiplot():
    print("Generating fig8_shap_multiplot_.png ...")
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        path = RESULTS_DIR / f"{ds}_shap.json"
        if not path.exists():
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        with open(path) as f:
            shap_data = json.load(f)

        feats = shap_data.get("features", FEATURES)
        vals  = np.array(shap_data.get("mean_abs_shap", []))
        if vals.sum() > 0:
            vals = vals / vals.sum()   # normalize to sum=1

        feat_names = [FEATURE_LABELS.get(f, f) for f in feats]

        color = DISCIPLINE_COLORS[ds]
        bars = ax.barh(feat_names, vals, color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7)

        ax.set_xlim([0, 1.05])
        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_xlabel("Norm. SHAP", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    _hide_extra_axes(axes, len(DATASETS))
    fig.suptitle("Normalized SHAP Feature Importance Across 10 Datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig8_shap_multiplot_.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_roc_multiplot()
    make_rolling_multiplot()
    make_ablation_multiplot()
    make_shap_multiplot()
    print("\nStage 5 complete. Four multiplots saved to figures/")
