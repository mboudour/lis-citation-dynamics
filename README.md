# Citation Dynamics

**Temporally-Aware Citation Prediction Across Disciplines: A Machine Learning Framework**

By Moses Boudourides

---

## Overview

This repository contains all scripts and tools for a comparative study of citation dynamics across 10 topical bibliographic datasets spanning 5 broad scholarly domains (Science, Engineering, BioMed, Social Science, Humanities). We apply a temporally-aware machine learning framework to predict whether one scientific paper will cite another, using keyword-defined corpora of approximately 50,000–70,000 peer-reviewed articles each, retrieved from [Dimensions.ai](https://www.dimensions.ai) (1975–2024).

A central design principle is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the future-topology data leakage that affects many prior citation prediction studies.

The key research question is: **What survives after temporal leakage is removed?** We decompose citation prediction into Semantic (S) features (SBERT cosine similarity) and Network/Structural (N) features (temporal gap, indegree, PageRank). Our findings reveal that **semantic similarity overwhelmingly dominates the predictive signal across almost all scholarly domains**, making citation prediction fundamentally a semantic retrieval task augmented by temporal context, rather than a purely network-theoretic problem.

---

## Datasets

Ten thematic bibliographic datasets are retrieved from Dimensions.ai using keyword searches in title and abstract, restricted to journal articles published 1975–2024. 

*See `outputs/tables/table1_corpus_overview.csv` for detailed corpus construction queries and abstract availability.*

| Domain | Dataset |
|---|---|
| Science | Protein Folding |
| Science | CRISPR |
| Engineering | Additive Manufacturing |
| Engineering | Corrosion Protection |
| BioMed | Neuroblastoma |
| BioMed | Osteosarcoma |
| Social Science | Income Inequality |
| Social Science | Organizational Behavior |
| Humanities | Film Studies |
| Humanities | Memory Studies |

---

## Data Collection

All datasets are fetched using the scripts in `scripts/data_collection/` (e.g., `fetch_dimensions_crispr.py`, `fetch_6_new_datasets.py`), which iterate year by year (1975–2024) for each keyword and deduplicate across keywords within the same corpus.

**Requirements:** A valid [Dimensions.ai](https://www.dimensions.ai) API key.

---

## Pipeline and Scripts

The machine learning pipeline is executed sequentially through five stages located in `scripts/citation_analysis/`:

1. **`stage1_feature_engineering.py`**: Constructs temporally-capped directed acyclic citation graphs (DAGs) and samples positive and hard-negative pairs within strict $\pm 3$ year windows, computing structural features (N) strictly prior to the citing year.
2. **`stage2_sbert_semantic.py` & `stage2b_directional_semantic.py`**: Computes semantic text embeddings using Sentence-BERT (`all-MiniLM-L6-v2`) on titles and abstracts, yielding the Semantic (S) directional similarity feature.
3. **`stage3_ml_training.py`**: Trains classifiers (Logistic Regression, Linear SVM, Random Forest, Gradient Boosting) using 5-fold cross-validation, temporal holdout, and S-N-SN feature ablation.
4. **`stage4_comparative_figures.py`**: Generates comparative performance visualizations (AUC bar charts, temporal holdout stability, SHAP heatmaps).
5. **`stage5_multiplots.py`**: Generates multi-panel figures across all 10 datasets (rolling temporal AUC, ablation multiplots, SHAP multiplots).

### Outputs

All generated figures and tables are available in the `outputs/` directory:
- `outputs/figures/`: Multi-dataset visualizations (fig1–fig8)
- `outputs/tables/`: Dataset overviews and citation graph statistics
- `outputs/reports/`: S-N-SN Interpretation analysis

---

## Requirements

```
Python 3.11
dimcli>=1.0
pandas>=2.0
pyarrow>=14.0
networkx>=3.0
numpy>=1.26
scikit-learn>=1.3
sentence-transformers>=2.7
torch>=2.1
matplotlib>=3.7
seaborn>=0.12
tabulate>=0.9
```

---

## Copyright

© 2026 Moses Boudourides. All rights reserved.

This repository and its contents are made available for academic peer review purposes only. No part of this work may be reproduced, distributed, or used in any form without the express written permission of the authors.
