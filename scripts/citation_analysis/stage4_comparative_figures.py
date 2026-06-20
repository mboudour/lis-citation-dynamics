"""
Stage 4: Comparative Figures
=========================================================
Generates comparative figures across all 10 datasets based on the Stage 3 results.
Figures include:
1. AUC comparison across datasets (bar chart)
2. SHAP feature importance comparison (heatmap)
3. Ablation drops comparison (heatmap)
4. Temporal holdout stability (AUC drop from CV)

Outputs:
  computations/citation_analysis_scripts/figures/
"""

import json, os, time, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).parent
RES_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

DATASETS = [
    "protein_folding", "CRISPR",
    "neuroblastoma", "osteosarcoma",
    "additive_manufacturing", "corrosion_protection",
    "income_inequality", "organizational_behavior",
    "film_studies", "memory_studies",
]

DISCIPLINES = {
    "protein_folding":         "Science",
    "CRISPR":                  "Science",
    "neuroblastoma":           "BioMed",
    "osteosarcoma":            "BioMed",
    "additive_manufacturing":  "Engineering",
    "corrosion_protection":    "Engineering",
    "income_inequality":       "Social Science",
    "organizational_behavior": "Social Science",
    "film_studies":            "Humanities",
    "memory_studies":          "Humanities",
}

FEATURES = ["temporal_indegree", "citation_time_gap", "temporal_pagerank", "directional_similarity"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def generate_figures():
    log(f"\n{'='*50}\nStage 4: Generating Comparative Figures\n{'='*50}")
    
    # ── Load all results ──
    cv_data = {}
    shap_data = {}
    abl_data = {}
    temp_data = {}
    
    for ds in DATASETS:
        cv_path = RES_DIR / f"{ds}_cv.json"
        shap_path = RES_DIR / f"{ds}_shap.json"
        abl_path = RES_DIR / f"{ds}_ablation.json"
        temp_path = RES_DIR / f"{ds}_temporal.json"
        
        if cv_path.exists():
            with open(cv_path) as f: cv_data[ds] = json.load(f)
        if shap_path.exists():
            with open(shap_path) as f: shap_data[ds] = json.load(f)
        if abl_path.exists():
            with open(abl_path) as f: abl_data[ds] = json.load(f)
        if temp_path.exists():
            with open(temp_path) as f: temp_data[ds] = json.load(f)
            
    if not cv_data:
        log("No CV results found. Run Stage 3 first.")
        return
        
    # ── Fig 1: AUC Comparison ──
    log("Generating AUC comparison plot...")
    gb_aucs = []
    gb_stds = []
    labels = []
    colors = []
    
    palette = {"Science": "#1f77b4", "Engineering": "#ff7f0e", 
               "BioMed": "#2ca02c", "Social Science": "#d62728", "Humanities": "#9467bd"}
               
    for ds in DATASETS:
        if ds in cv_data:
            gb_aucs.append(cv_data[ds]["Gradient Boosting"]["mean"]["auc"])
            gb_stds.append(cv_data[ds]["Gradient Boosting"]["std"]["auc"])
            labels.append(ds)
            colors.append(palette[DISCIPLINES[ds]])
            
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, gb_aucs, yerr=gb_stds, color=colors, capsize=4, edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean AUC (Gradient Boosting)")
    ax.set_title("Citation Link Prediction AUC Across 10 Datasets")
    ax.set_ylim([min(gb_aucs) - 0.05, 1.0])
    
    # Add values on top
    for bar, val in zip(bars, gb_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
                
    # Custom legend for disciplines — placed outside the plot area
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[d], label=d) for d in palette]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_auc_comparison.png")
    plt.close(fig)
    
    # ── Fig 2: SHAP Heatmap ──
    if shap_data:
        log("Generating SHAP importance heatmap...")
        shap_matrix = np.zeros((len(DATASETS), len(FEATURES)))
        for i, ds in enumerate(DATASETS):
            if ds in shap_data:
                vals = shap_data[ds]["mean_abs_shap"]
                # Normalize so rows sum to 1 for comparability
                shap_matrix[i, :] = np.array(vals) / sum(vals)
                
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(shap_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=FEATURES, yticklabels=DATASETS, ax=ax)
        ax.set_title("Normalized SHAP Feature Importance Across Datasets")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig2_shap_heatmap.png")
        plt.close(fig)
        
    # ── Fig 3: Ablation Drops Heatmap ──
    if abl_data:
        log("Generating Ablation drops heatmap...")
        abl_matrix = np.zeros((len(DATASETS), len(FEATURES)))
        for i, ds in enumerate(DATASETS):
            if ds in abl_data:
                base = abl_data[ds]["all_features"]
                for j, feat in enumerate(FEATURES):
                    drop = base - abl_data[ds].get(f"without_{feat}", base)
                    abl_matrix[i, j] = max(0, drop) # Ignore negative drops
                    
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(abl_matrix, annot=True, fmt=".3f", cmap="Reds",
                    xticklabels=FEATURES, yticklabels=DATASETS, ax=ax)
        ax.set_title("AUC Drop When Feature Removed (Ablation)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig3_ablation_heatmap.png")
        plt.close(fig)
        
    # ── Fig 4: Temporal Holdout Stability ──
    if temp_data:
        log("Generating Temporal Holdout Stability plot...")
        cv_aucs = []
        temp_aucs = []
        valid_labels = []
        
        for ds in DATASETS:
            if ds in cv_data and ds in temp_data:
                cv_aucs.append(cv_data[ds]["Gradient Boosting"]["mean"]["auc"])
                temp_aucs.append(temp_data[ds]["Gradient Boosting"]["auc"])
                valid_labels.append(ds)
                
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(valid_labels))
        w = 0.35
        
        ax.bar(x - w/2, cv_aucs, w, label="CV (Full Time Range)", color="#1f77b4", edgecolor="black")
        ax.bar(x + w/2, temp_aucs, w, label="Temporal Holdout (Test 2016-2020)", color="#ff7f0e", edgecolor="black")
        
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, rotation=45, ha="right")
        ax.set_ylabel("AUC")
        ax.set_title("Temporal Generalization: CV vs Holdout")
        ax.set_ylim([min(min(cv_aucs), min(temp_aucs)) - 0.05, 1.0])
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig4_temporal_stability.png")
        plt.close(fig)

    log("Stage 4 complete.")

if __name__ == "__main__":
    generate_figures()
