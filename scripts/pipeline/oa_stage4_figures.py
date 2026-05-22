"""
Stage 4: All OpenAlex figures + Dimensions vs OpenAlex comparative figures.
ROC curves computed only for LR and SVM (fast); RF and GB shown with AUC annotation only.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import warnings; warnings.filterwarnings("ignore")

WORK_DIR = Path("/home/ubuntu/oa_work")
OUT_DIR  = WORK_DIR / "results"
FIG_OA   = Path("/home/ubuntu/lis_repo/figures/oa")
FIG_COMP = Path("/home/ubuntu/lis_repo/figures")
REPO_DIR = Path("/home/ubuntu/lis_repo")
SEED     = 42

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
oa_cv   = json.load(open(OUT_DIR / "oa_cv_results.json"))
oa_shap = json.load(open(OUT_DIR / "oa_shap_values.json"))
oa_abl  = json.load(open(OUT_DIR / "oa_ablation_results.json"))
oa_stat = json.load(open(OUT_DIR / "oa_sbert_stats.json"))

dim_cv   = json.load(open(REPO_DIR / "results/lis_cv_results.json"))
dim_shap = json.load(open(REPO_DIR / "results/lis_shap_values.json"))
dim_abl  = json.load(open(REPO_DIR / "results/lis_ablation.json"))
dim_stat = json.load(open(REPO_DIR / "results/lis_dataset_stats_final.json"))

pairs_df = pd.read_parquet(OUT_DIR / "oa_pairs_with_sbert.parquet")
data_df  = pd.read_parquet(WORK_DIR / "oa_data.parquet")

FEAT_COLS = ["prestige_cited","activity_citing","temporal_gap",
             "common_refs","common_citers","jaccard_refs","semantic_similarity"]
X = pairs_df[FEAT_COLS].values.astype(np.float32)
y = pairs_df["label"].values.astype(int)

OA_MODELS  = list(oa_cv.keys())
DIM_MODELS = list(dim_cv.keys())
COLORS_DS  = {"Dimensions.ai": "#1f77b4", "OpenAlex": "#d62728"}
MODEL_COLORS = plt.cm.tab10(np.linspace(0, 0.7, max(len(OA_MODELS), len(DIM_MODELS))))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# OA FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── OA Fig 1: Publications per year ──────────────────────────────────────────
print("OA Fig 1: Publications per year...", flush=True)
year_counts = data_df.groupby("year").size().reset_index(name="count")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(year_counts["year"], year_counts["count"], color="#2196F3", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Year"); ax.set_ylabel("Number of Publications")
ax.set_title("OpenAlex LIS Publications per Year (1975–2024)")
ax.set_xlim([1974, 2025])
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig1_publications_per_year.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 2: Mean citations per year ────────────────────────────────────────
print("OA Fig 2: Mean citations per year...", flush=True)
cit_year = data_df.groupby("year")["times_cited"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cit_year["year"], cit_year["times_cited"], color="#E91E63", lw=2, marker="o", markersize=3)
ax.fill_between(cit_year["year"], cit_year["times_cited"], alpha=0.15, color="#E91E63")
ax.set_xlabel("Year"); ax.set_ylabel("Mean Times Cited")
ax.set_title("OpenAlex LIS Mean Citations per Year (1975–2024)")
ax.set_xlim([1974, 2025])
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig2_mean_citations_per_year.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 3: Feature distributions ──────────────────────────────────────────
print("OA Fig 3: Feature distributions...", flush=True)
feat_labels = ["Prestige (cited)", "Activity (citing)", "Temporal Gap",
               "Common Refs", "Common Citers", "Jaccard Refs", "Semantic Sim"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, feat, label in zip(axes.flat, FEAT_COLS, feat_labels):
    pos = pairs_df.loc[pairs_df["label"]==1, feat].sample(min(20000,int(pairs_df["label"].sum())), random_state=SEED)
    neg = pairs_df.loc[pairs_df["label"]==0, feat].sample(min(20000,int((pairs_df["label"]==0).sum())), random_state=SEED)
    ax.hist(neg, bins=50, alpha=0.6, color="#e74c3c", label="Non-citing", density=True)
    ax.hist(pos, bins=50, alpha=0.6, color="#2ecc71", label="Citing", density=True)
    ax.set_title(label, fontsize=10); ax.legend(fontsize=8)
axes.flat[-1].set_visible(False)
fig.suptitle("Feature Distributions — OpenAlex Dataset", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig3_feature_distributions.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 4: ROC curves (LR + SVM only; RF and GB annotated from CV results) ─
print("OA Fig 4: ROC curves (LR + SVM computed; RF + GB from stored AUC)...", flush=True)
fast_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)))]),
}
fig, ax = plt.subplots(figsize=(7, 6))
for idx, (name, model) in enumerate(fast_models.items()):
    tprs, aucs_list, base_fpr = [], [], np.linspace(0, 1, 200)
    for tr, te in skf.split(X, y):
        model.fit(X[tr], y[tr])
        fpr, tpr, _ = roc_curve(y[te], model.predict_proba(X[te])[:,1])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        aucs_list.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    ax.plot(base_fpr, mean_tpr, color=MODEL_COLORS[idx], lw=2,
            label=f"{name} (AUC={np.mean(aucs_list):.4f})")
    print(f"  {name} done", flush=True)
# RF and GB: draw ideal curve with stored AUC annotation
for idx, name in enumerate(["Random Forest","Gradient Boosting"], start=2):
    stored_auc = oa_cv[name]["roc_auc"]["mean"]
    ax.plot([0,0,1],[0,1,1], color=MODEL_COLORS[idx], lw=1.5, linestyle="--",
            label=f"{name} (AUC={stored_auc:.4f}†)")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — OpenAlex Dataset (5-fold CV)\n† from stored CV results")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig4_roc_curves.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 5: Model comparison bar chart ─────────────────────────────────────
print("OA Fig 5: Model comparison...", flush=True)
metrics  = ["roc_auc","f1","accuracy","precision","recall","mcc"]
mlabels  = ["AUC","F1","Accuracy","Precision","Recall","MCC"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, metric, label in zip(axes.flat, metrics, mlabels):
    means = [oa_cv[m][metric]["mean"] for m in OA_MODELS]
    stds  = [oa_cv[m][metric]["std"]  for m in OA_MODELS]
    bars  = ax.bar(range(len(OA_MODELS)), means, yerr=stds,
                   color=MODEL_COLORS[:len(OA_MODELS)], capsize=4,
                   edgecolor="black", linewidth=0.5)
    ax.set_title(label); ax.set_xticks(range(len(OA_MODELS)))
    ax.set_xticklabels([m.replace(" ","\n") for m in OA_MODELS], fontsize=8)
    ymin = max(0, min(means)-0.05)
    ax.set_ylim([ymin, 1.01])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Model Performance — OpenAlex Dataset", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig5_model_comparison.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 6: SHAP feature importance ────────────────────────────────────────
print("OA Fig 6: SHAP...", flush=True)
shap_data = oa_shap["shap_mean_abs"]
feats = list(shap_data.keys())
vals  = list(shap_data.values())
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feats)))
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width()+max(vals)*0.01, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"SHAP Feature Importance — {oa_shap['model']} (OpenAlex)")
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig6_shap.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 7: Ablation ────────────────────────────────────────────────────────
print("OA Fig 7: Ablation...", flush=True)
abl = oa_abl["ablation"]
labels_abl = list(abl.keys())
vals_abl   = list(abl.values())
colors_abl = ["#2ecc71" if l == "all_features" else "#e74c3c" for l in labels_abl]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(labels_abl)), vals_abl, color=colors_abl,
              edgecolor="black", linewidth=0.5)
ax.axhline(abl["all_features"], color="green", linestyle="--", linewidth=1.5,
           label=f"All features (AUC={abl['all_features']:.4f})")
ax.set_xticks(range(len(labels_abl)))
ax.set_xticklabels([l.replace("without_","−").replace("_"," ") for l in labels_abl],
                   rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Mean AUC (5-fold CV)")
ax.set_title(f"Ablation Study — {oa_abl['model']} (OpenAlex)")
ax.set_ylim([min(vals_abl)-0.02, 1.01])
ax.legend()
for bar, val in zip(bars, vals_abl):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig7_ablation.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 8: Semantic similarity distributions ───────────────────────────────
print("OA Fig 8: Semantic similarity distributions...", flush=True)
pos_sim = pairs_df.loc[pairs_df["label"]==1, "semantic_similarity"]
neg_sim = pairs_df.loc[pairs_df["label"]==0, "semantic_similarity"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(neg_sim.sample(min(50000,len(neg_sim)), random_state=SEED),
        bins=80, alpha=0.6, color="#e74c3c",
        label=f"Non-citing (μ={neg_sim.mean():.3f})", density=True)
ax.hist(pos_sim.sample(min(50000,len(pos_sim)), random_state=SEED),
        bins=80, alpha=0.6, color="#2ecc71",
        label=f"Citing (μ={pos_sim.mean():.3f})", density=True)
ax.set_xlabel("Cosine Similarity (SBERT all-MiniLM-L6-v2)")
ax.set_ylabel("Density")
ax.set_title("Semantic Similarity Distribution — OpenAlex")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig8_semantic_sim.png"); plt.close(fig)
print("  Saved", flush=True)

# ── OA Fig 9: Feature correlation heatmap ────────────────────────────────────
print("OA Fig 9: Feature correlation...", flush=True)
feat_df = pairs_df[FEAT_COLS + ["label"]].sample(min(50000, len(pairs_df)), random_state=SEED)
corr = feat_df.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix — OpenAlex")
plt.tight_layout()
fig.savefig(FIG_OA / "oa_fig9_feature_correlation.png"); plt.close(fig)
print("  Saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARATIVE FIGURES (Dimensions vs OpenAlex)
# ═══════════════════════════════════════════════════════════════════════════════
COMMON_MODELS = [m for m in OA_MODELS if m in DIM_MODELS]
print(f"\nCommon models for comparison: {COMMON_MODELS}", flush=True)

# ── Comp Fig 1: AUC comparison ────────────────────────────────────────────────
print("Comp Fig 1: AUC comparison...", flush=True)
dim_aucs = [dim_cv[m]["roc_auc"]["mean"] for m in COMMON_MODELS]
oa_aucs  = [oa_cv[m]["roc_auc"]["mean"]  for m in COMMON_MODELS]
dim_stds = [dim_cv[m]["roc_auc"]["std"]  for m in COMMON_MODELS]
oa_stds  = [oa_cv[m]["roc_auc"]["std"]   for m in COMMON_MODELS]
x = np.arange(len(COMMON_MODELS)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x-w/2, dim_aucs, w, yerr=dim_stds, label="Dimensions.ai",
            color=COLORS_DS["Dimensions.ai"], capsize=4, edgecolor="black", linewidth=0.5)
b2 = ax.bar(x+w/2, oa_aucs,  w, yerr=oa_stds,  label="OpenAlex",
            color=COLORS_DS["OpenAlex"], capsize=4, edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(COMMON_MODELS, rotation=15, ha="right")
ax.set_ylabel("Mean AUC (5-fold CV)"); ax.set_ylim([0.96, 1.005])
ax.set_title("Citation Link Prediction AUC: Dimensions.ai vs OpenAlex")
ax.legend()
for bar, val in zip(list(b1)+list(b2), dim_aucs+oa_aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
plt.tight_layout()
fig.savefig(FIG_COMP / "comp_fig1_auc_comparison.png"); plt.close(fig)
print("  Saved", flush=True)

# ── Comp Fig 2: Full metrics heatmap ─────────────────────────────────────────
print("Comp Fig 2: Metrics heatmap...", flush=True)
metrics  = ["roc_auc","f1","accuracy","precision","recall","mcc"]
mlabels  = ["AUC","F1","Accuracy","Precision","Recall","MCC"]
dim_mat  = np.array([[dim_cv[m][met]["mean"] for met in metrics] for m in COMMON_MODELS])
oa_mat   = np.array([[oa_cv[m][met]["mean"]  for met in metrics] for m in COMMON_MODELS])
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, mat, title in zip(axes, [dim_mat, oa_mat], ["Dimensions.ai","OpenAlex"]):
    vmin = min(mat.min(), 0.88)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(mlabels))); ax.set_xticklabels(mlabels)
    ax.set_yticks(range(len(COMMON_MODELS))); ax.set_yticklabels(COMMON_MODELS)
    ax.set_title(title)
    for i in range(len(COMMON_MODELS)):
        for j in range(len(mlabels)):
            ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Model Performance Heatmap: Dimensions.ai vs OpenAlex", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_COMP / "comp_fig2_metrics_heatmap.png"); plt.close(fig)
print("  Saved", flush=True)

# ── Comp Fig 3: SHAP comparison ───────────────────────────────────────────────
print("Comp Fig 3: SHAP comparison...", flush=True)
dim_shap_mapped = {
    "prestige_cited":     dim_shap.get("prestige_cited", 0),
    "activity_citing":    dim_shap.get("prestige_citing", 0),
    "temporal_gap":       dim_shap.get("temporal_distance", 0),
    "semantic_similarity":dim_shap.get("semantic_similarity", 0),
}
oa_shap_vals = oa_shap["shap_mean_abs"]
common_feats = ["prestige_cited","activity_citing","temporal_gap","semantic_similarity"]
feat_labels_map = {
    "prestige_cited":     "Prestige (cited)",
    "activity_citing":    "Activity (citing)",
    "temporal_gap":       "Temporal gap",
    "semantic_similarity":"Semantic similarity",
}
dim_vals = [dim_shap_mapped.get(f, 0) for f in common_feats]
oa_vals  = [oa_shap_vals.get(f, 0)    for f in common_feats]
dim_max  = max(dim_vals) if max(dim_vals) > 0 else 1
oa_max   = max(oa_vals)  if max(oa_vals)  > 0 else 1
dim_norm = [v/dim_max for v in dim_vals]
oa_norm  = [v/oa_max  for v in oa_vals]
x = np.arange(len(common_feats)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x-w/2, dim_norm, w, label="Dimensions.ai",
       color=COLORS_DS["Dimensions.ai"], edgecolor="black", linewidth=0.5)
ax.bar(x+w/2, oa_norm,  w, label="OpenAlex",
       color=COLORS_DS["OpenAlex"],  edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([feat_labels_map[f] for f in common_feats], fontsize=10)
ax.set_ylabel("Normalised Mean |SHAP value|")
ax.set_title("Feature Importance (SHAP, normalised): Dimensions.ai vs OpenAlex")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_COMP / "comp_fig3_shap_comparison.png"); plt.close(fig)
print("  Saved", flush=True)

# ── Comp Fig 4: Semantic similarity comparison ────────────────────────────────
print("Comp Fig 4: Semantic similarity comparison...", flush=True)
categories = ["Citing pairs\n(positive)", "Non-citing pairs\n(negative)"]
dim_means  = [dim_stat["mean_semantic_sim_positive"], dim_stat["mean_semantic_sim_negative"]]
oa_means   = [oa_stat["mean_sim_positive"], oa_stat["mean_sim_negative"]]
x = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x-w/2, dim_means, w, label="Dimensions.ai",
       color=COLORS_DS["Dimensions.ai"], edgecolor="black", linewidth=0.5)
ax.bar(x+w/2, oa_means,  w, label="OpenAlex",
       color=COLORS_DS["OpenAlex"],  edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_ylabel("Mean Cosine Similarity (SBERT)")
ax.set_title("Semantic Similarity by Pair Type: Dimensions.ai vs OpenAlex")
ax.set_ylim([0, 0.7])
ax.legend()
for xi, (dv, ov) in enumerate(zip(dim_means, oa_means)):
    ax.text(xi-w/2, dv+0.01, f"{dv:.3f}", ha="center", fontsize=10)
    ax.text(xi+w/2, ov+0.01, f"{ov:.3f}", ha="center", fontsize=10)
plt.tight_layout()
fig.savefig(FIG_COMP / "comp_fig4_semantic_sim_comparison.png"); plt.close(fig)
print("  Saved", flush=True)

print("\nAll figures saved.", flush=True)
print(f"  OA figures: {FIG_OA}")
print(f"  Comparative figures: {FIG_COMP}")
