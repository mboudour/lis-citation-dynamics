# LIS Citation Dynamics

**Predicting Citation Links in Library and Information Science: Where Network Science Meets Machine Learning**

Moses A. Boudourides · Giannis Tsakonas

*Submitted to Scientometrics*

---

## Overview

This repository contains all scripts, results, and figures for a study on citation dynamics in the Library and Information Science (LIS) literature. We apply a machine learning framework to predict whether one scientific paper will cite another, using a corpus of 259,220 peer-reviewed articles (Dimensions.ai, 1975–2024) and a replication corpus from OpenAlex (168,901 articles). A central design principle is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the data leakage that affects many prior citation prediction studies.

---

## Repository Structure

```
lis-citation-dynamics/
├── scripts/
│   ├── data_collection/
│   │   ├── fetch_dimensions_lis.py   # Retrieve LIS data from Dimensions.ai (dimcli)
│   │   └── fetch_openalex_lis.py     # Retrieve LIS data from OpenAlex API
│   └── pipeline/
│       ├── extract_texts.py          # Stage 0: extract title+abstract texts
│       ├── lis_stage2_sbert_checkpoint.py  # Stage 2: SBERT encoding (Dimensions)
│       ├── lis_stage3_ml.py          # Stage 3: ML training, SHAP, ablation (Dimensions)
│       ├── lis_stage4_figures.py     # Stage 4: figures (Dimensions)
│       ├── oa_stage1_features.py     # Stage 1: feature engineering (OpenAlex)
│       ├── oa_stage2_sbert.py        # Stage 2: SBERT encoding (OpenAlex)
│       ├── oa_stage3_ml_nosbert.py   # Stage 3: ML without semantic similarity (OpenAlex)
│       └── oa_stage2_compute_sims.py # Similarity computation from checkpoints
├── figures/
│   └── dimensions/                   # 8 manuscript figures (PNG)
├── results/
│   ├── dimensions/                   # CV scores, SHAP values, ablation (JSON)
│   └── oa/                           # OpenAlex results (in progress)
└── paper/
    ├── lis_manuscript.pdf
    └── lis_manuscript.md
```

---

## Data

### Dimensions.ai (Primary Dataset)

Retrieved using `scripts/data_collection/fetch_dimensions_lis.py` (requires a valid Dimensions API key and the `dimcli` library). Due to Dimensions' Terms of Service, the raw dataset is not redistributed here; researchers with Dimensions access can reproduce it exactly using the provided script.

| Property | Value |
|---|---|
| Total articles | 259,220 |
| Year range | 1975–2024 |
| Abstract coverage | 96.4% |
| Reference list coverage | 65.2% |
| Mean citations per paper | 20.98 |

### OpenAlex (Replication Dataset)

Retrieved using `scripts/data_collection/fetch_openalex_lis.py`. Filters by five primary topic IDs covering core LIS and scientometrics (T10712, T14330, T13166, T13673, T10102). Fully open and reproducible without any subscription.

| Property | Value |
|---|---|
| Total articles | 168,901 |
| Year range | 1975–2024 |
| Abstract coverage | 64.3% |
| Reference list coverage | 29.8% |

---

## Pipeline

| Stage | Script | Description |
|---|---|---|
| 0 | `extract_texts.py` | Extract title + abstract texts |
| 1 | `oa_stage1_features.py` | Build citation pairs; compute 6 structural features |
| 2 | `*_stage2_sbert.py` | Encode texts with SBERT (`all-MiniLM-L6-v2`); compute cosine similarity |
| 3 | `*_stage3_ml.py` | 5-fold CV with 6 classifiers; SHAP analysis; ablation study |
| 4 | `*_stage4_figures.py` | Generate manuscript figures |

---

## Results (Dimensions.ai)

Best model: **Multi-Layer Perceptron (MLP) Neural Network**, AUC = **0.9906**

| Model | AUC | F1 | Accuracy | Precision | Recall | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9897 | 0.9522 | 0.9521 | 0.9505 | 0.9539 | 0.9042 |
| Linear SVM | 0.9897 | 0.9522 | 0.9521 | 0.9505 | 0.9540 | 0.9043 |
| k-NN (k=5) | 0.9773 | 0.9485 | 0.9483 | 0.9459 | 0.9511 | 0.8967 |
| Random Forest | 0.9873 | 0.9503 | 0.9502 | 0.9473 | 0.9533 | 0.9003 |
| Gradient Boosting | 0.9904 | 0.9542 | 0.9540 | 0.9511 | 0.9573 | 0.9081 |
| **MLP Neural Network** | **0.9906** | **0.9540** | **0.9539** | **0.9518** | **0.9562** | **0.9078** |

OpenAlex replication results will be added upon completion of the comparative pipeline.

---

## Requirements

```
pip install pandas numpy scikit-learn networkx sentence-transformers shap matplotlib seaborn
```

- Python 3.11
- SBERT model: `all-MiniLM-L6-v2` (downloaded automatically)
- Dimensions.ai access: subscription required
- OpenAlex access: fully open

---

## Citation

> Boudourides, M.A. & Tsakonas, G. (2026). Predicting Citation Links in Library and Information Science: Where Network Science Meets Machine Learning. *Scientometrics* (submitted).
