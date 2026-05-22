# Figure Data Reference Guide

This document lists the exact data source for every figure in the repository. Use it to make precise micro-edits to any figure by modifying the underlying data or the plotting script.

---

## Dimensions.ai Figures (`figures/dimensions/`)

| Figure | File | Data Source | Script |
|---|---|---|---|
| Fig 1: Publications per year | `fig1_publications_per_year.png` | `data_df.groupby("year").size()` from the Dimensions PKL | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 2: Mean citations per year | `fig2_mean_citations_per_year.png` | `data_df.groupby("year")["times_cited"].mean()` | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 3: Feature distributions | `fig3_feature_distributions.png` | `pairs_df[FEAT_COLS]` split by `label` column | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 4: Model comparison | `fig4_model_comparison.png` | `results/lis_cv_results.json` — keys: `roc_auc`, `f1`, `accuracy`, `precision`, `recall`, `mcc` | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 5: ROC curves | `fig5_roc_curves.png` | Recomputed from `pairs_df` with LR and SVM; RF/GB/MLP annotated from `results/lis_cv_results.json` | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 6: SHAP importance | `fig6_shap_importance.png` | `results/lis_shap_values.json` — values: `prestige_cited=0.1369`, `semantic_similarity=0.3052`, `coauth_distance=0.0217`, `temporal_distance=0.0175`, `prestige_citing=0.0127`, `same_journal=0.0141`, `cited_oa=0.0` | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 7: Ablation | `fig7_ablation.png` | `results/lis_ablation.json` — All features: 0.9868, No semantic sim: 0.8709, No prestige: 0.9618, Semantic only: 0.9513, Network only: 0.8512, Temporal+semantic: 0.9576, No coauth: 0.9859 | `scripts/pipeline/lis_stage4_figures.py` |
| Fig 8: Feature correlation | `fig8_feature_correlation.png` | `pairs_df[FEAT_COLS + ["label"]].sample(50000).corr()` | `scripts/pipeline/lis_stage4_figures.py` |

---

## OpenAlex Figures (`figures/oa/`)

| Figure | File | Data Source | Script |
|---|---|---|---|
| OA Fig 1: Publications per year | `oa_fig1_publications_per_year.png` | `oa_data.parquet` → `groupby("year").size()` | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 2: Mean citations per year | `oa_fig2_mean_citations_per_year.png` | `oa_data.parquet` → `groupby("year")["times_cited"].mean()` | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 3: Feature distributions | `oa_fig3_feature_distributions.png` | `oa_pairs_with_sbert.parquet` — columns: `prestige_cited`, `activity_citing`, `temporal_gap`, `common_refs`, `common_citers`, `jaccard_refs`, `semantic_similarity`, `label` | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 4: ROC curves | `oa_fig4_roc_curves.png` | LR and SVM recomputed from `oa_pairs_with_sbert.parquet`; RF AUC=0.9972 and GB AUC=0.9979 annotated from `results/oa/oa_cv_results.json` | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 5: Model comparison | `oa_fig5_model_comparison.png` | `results/oa/oa_cv_results.json` — LR: 0.9872, SVM: 0.9863, RF: 0.9972, GB: **0.9979** | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 6: SHAP importance | `oa_fig6_shap.png` | `results/oa/oa_shap_values.json` — activity_citing=4.715, prestige_cited=4.498, semantic_similarity=1.956, temporal_gap=0.796, common_citers=0.643, jaccard_refs=0.620, common_refs=0.241 | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 7: Ablation | `oa_fig7_ablation.png` | `results/oa/oa_ablation_results.json` — all_features: 0.9972, w/o prestige_cited: 0.9847, w/o activity_citing: 0.9890, w/o temporal_gap: 0.9960, w/o semantic_similarity: 0.9914, w/o common_refs: 0.9972, w/o common_citers: 0.9970, w/o jaccard_refs: 0.9972 | `scripts/pipeline/oa_stage3b_ablation.py` |
| OA Fig 8: Semantic similarity | `oa_fig8_semantic_sim.png` | `oa_pairs_with_sbert.parquet["semantic_similarity"]` split by `label`; μ_positive=0.5183, μ_negative=0.2161 | `scripts/pipeline/oa_stage4_figures.py` |
| OA Fig 9: Feature correlation | `oa_fig9_feature_correlation.png` | `oa_pairs_with_sbert.parquet[FEAT_COLS + ["label"]].sample(50000).corr()` | `scripts/pipeline/oa_stage4_figures.py` |

---

## Comparative Figures (`figures/`)

| Figure | File | Data Source | Script |
|---|---|---|---|
| Comp Fig 1: AUC comparison | `comp_fig1_auc_comparison.png` | `results/lis_cv_results.json` + `results/oa/oa_cv_results.json` — common models: LR, SVM, RF, GB | `scripts/pipeline/oa_stage4_figures.py` |
| Comp Fig 2: Metrics heatmap | `comp_fig2_metrics_heatmap.png` | Both CV JSONs — all 6 metrics for 4 common models | `scripts/pipeline/oa_stage4_figures.py` |
| Comp Fig 3: SHAP comparison | `comp_fig3_shap_comparison.png` | `results/lis_shap_values.json` + `results/oa/oa_shap_values.json` — 4 common features, normalised to [0,1] | `scripts/pipeline/oa_stage4_figures.py` |
| Comp Fig 4: Semantic similarity | `comp_fig4_semantic_sim_comparison.png` | Dimensions: μ_pos=0.5573, μ_neg=0.1624 (`results/lis_dataset_stats_final.json`); OpenAlex: μ_pos=0.5183, μ_neg=0.2161 (`results/oa/oa_sbert_stats.json`) | `scripts/pipeline/oa_stage4_figures.py` |

---

## PKL to Parquet Conversion Commands

Use these commands to convert the raw pickle files to portable Parquet format. The Parquet format is fully numpy-version-independent and can be read in Python, R, and DuckDB.

### Requirements

```bash
pip install numpy==2.2.0 pandas pyarrow
```

### Dimensions.ai PKL → Parquet

```python
import pandas as pd

# Load the Dimensions pickle (requires numpy >= 2.x, as it was saved with numpy 2.x)
df = pd.read_pickle("Dimensions_LIS_1975_2024.pkl")

# Inspect
print(df.shape, df.columns.tolist())

# Save as Parquet
df.to_parquet("Dimensions_LIS_1975_2024.parquet", index=False, engine="pyarrow")
print("Saved:", df.shape)
```

### OpenAlex PKL → Parquet

The OpenAlex Parquet is already available in the repository as `OpenAlex_LIS_1975_2024.parquet`. If you need to regenerate it from the original PKL:

```python
import pandas as pd

df = pd.read_pickle("OpenAlex_LIS_1975_2024.pkl")
df.to_parquet("OpenAlex_LIS_1975_2024.parquet", index=False, engine="pyarrow")
print("Saved:", df.shape)
```

### Reading Parquet files

**Python:**
```python
import pandas as pd
df = pd.read_parquet("OpenAlex_LIS_1975_2024.parquet")
```

**R:**
```r
library(arrow)
df <- read_parquet("OpenAlex_LIS_1975_2024.parquet")
```

**DuckDB:**
```sql
SELECT * FROM 'OpenAlex_LIS_1975_2024.parquet' LIMIT 10;
```
