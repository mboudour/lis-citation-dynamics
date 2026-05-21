# Citation Dynamics in Library and Information Science (LIS)
### A Temporally-Aware Machine Learning Framework for Predicting Scientific Citations

---

## Overview

This repository contains the code outputs, figures, and results for a study on citation dynamics in the Library and Information Science (LIS) literature. We apply a machine learning framework to predict whether one scientific paper will cite another, using a corpus of 259,220 peer-reviewed articles published between 1975 and 2024. A central design principle of this work is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the data leakage that affects many prior citation prediction studies.

---

## Research Questions

This study addresses four core research questions:

**RQ1 — Predictability:** To what extent are citation events in LIS predictable from observable features of the papers and their authors?

**RQ2 — Temporal validity:** Does eliminating temporal data leakage — a methodological flaw common in the citation prediction literature — substantially reduce model performance, or do citation dynamics remain highly predictable under strict causal constraints?

**RQ3 — Feature importance:** Which mechanisms drive citation behavior? Specifically, what is the relative contribution of (a) semantic content similarity, (b) social proximity via co-authorship networks, and (c) prestige (the Matthew effect)?

**RQ4 — Feature interactions:** Is it the combination of semantic and structural network features that enables accurate prediction, or can a single feature class alone explain citation behavior?

---

## Dataset

The dataset was extracted from [Dimensions.ai](https://www.dimensions.ai/) — a comprehensive scholarly database — categorized under Library and Information Science (LIS). 

| Property | Value |
| :--- | :--- |
| **Source** | Dimensions.ai |
| **Domain** | Library and Information Science (LIS) |
| **Publication type** | Peer-reviewed journal articles |
| **Time span** | 1975–2024 (50 years) |
| **Total articles** | 259,220 |
| **Abstract coverage** | 96.4% |
| **Reference list coverage** | 65.2% |
| **Mean citations per paper** | 20.98 |
| **Median citations per paper** | 4.0 |

### Citation Pair Construction

The classification task is: given a pair of papers $(A, B)$ where $B$ was published no later than $A$, predict whether $A$ cites $B$.

| Split | Count |
| :--- | :--- |
| **Positive pairs** (observed citations within corpus) | 250,000 |
| **Negative pairs** (temporally valid non-citations) | 250,000 |
| **Total pairs** | 500,000 |

Negative pairs are sampled strictly: for each positive pair $(A, B)$, a negative pair $(A, B')$ is constructed by drawing $B'$ uniformly from all papers published on or before $A$'s publication year that $A$ does not actually cite. This prevents the model from exploiting impossible temporal orderings.

---

## Methodology

### Temporally-Aware Feature Engineering

Seven features are engineered across four conceptual categories. All features that depend on the state of the literature (prestige, co-authorship) are computed using a **rolling causal window**: for a citing paper published in year $t$, only data from years $\leq t-1$ are used.

| Feature | Category | Description |
| :--- | :--- | :--- |
| **Semantic Similarity** | Semantic | Cosine similarity of SBERT (all-MiniLM-L6-v2) embeddings of title + abstract |
| **Prestige of Cited Paper** | Prestige | PageRank centrality in the citation network up to year $t_A - 1$ |
| **Prestige of Citing Paper** | Prestige | PageRank centrality of the citing paper up to year $t_A - 1$ |
| **Temporal Distance** | Temporal | $t_A - t_B$ (difference in publication years) |
| **Co-authorship Distance** | Network | Dijkstra shortest path in the cumulative co-authorship graph up to year $t_A - 1$; capped at 20 for disconnected pairs |
| **Same Journal** | Metadata | Binary: 1 if $A$ and $B$ appear in the same journal |
| **Open Access** | Metadata | Binary: 1 if $B$ is Open Access |

### Machine Learning Models

Six models are evaluated using **5-fold cross-validation** on the balanced dataset:

- Logistic Regression
- Linear Support Vector Machine (LinearSVC)
- k-Nearest Neighbours (k=5)
- Random Forest
- Gradient Boosting
- Multi-Layer Perceptron (MLP) Neural Network

### Explainability

SHAP (SHapley Additive exPlanations) values are computed for the best-performing model (Neural Network) to quantify global feature importance and interpret the contribution of each feature to individual predictions.

---

## Results

### Model Performance

| Model | ROC-AUC | F1-Score | Accuracy | Precision | Recall | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP Neural Network** | **0.9906** | **0.9540** | **0.9539** | **0.9518** | 0.9562 | **0.9078** |
| Gradient Boosting | 0.9904 | 0.9542 | 0.9540 | 0.9511 | **0.9573** | 0.9081 |
| Logistic Regression | 0.9897 | 0.9522 | 0.9521 | 0.9505 | 0.9539 | 0.9042 |
| Linear SVM | 0.9897 | 0.9522 | 0.9521 | 0.9505 | 0.9540 | 0.9043 |
| Random Forest | 0.9873 | 0.9503 | 0.9502 | 0.9473 | 0.9533 | 0.9003 |
| k-NN (k=5) | 0.9773 | 0.9485 | 0.9483 | 0.9459 | 0.9511 | 0.8967 |

The MLP Neural Network achieves the highest ROC-AUC (0.9906), closely followed by Gradient Boosting. The extremely high performance across all models, including linear baselines, indicates that the engineered features (especially semantic similarity) contain an overwhelming signal for citation prediction in LIS.

### Feature Importance (SHAP)

| Feature | Mean \|SHAP Value\| | Rank |
| :--- | :--- | :--- |
| Semantic Similarity | 0.3052 | 1 |
| Prestige (cited) | 0.1369 | 2 |
| Co-authorship Distance | 0.0217 | 3 |
| Temporal Distance | 0.0175 | 4 |
| Same Journal | 0.0141 | 5 |
| Prestige (citing) | 0.0127 | 6 |
| Open Access | 0.0000 | 7 |

**Semantic similarity is the overwhelmingly dominant predictor** — papers that are conceptually close are far more likely to be cited in LIS. Prestige of the cited paper (the Matthew effect) is the second strongest driver. Social proximity (co-authorship distance) plays a significant but relatively minor role compared to content relevance.

### Ablation Study

| Feature Subset | ROC-AUC | Δ vs. All Features |
| :--- | :--- | :--- |
| **All features** | **0.9868** | — |
| No semantic similarity | 0.8709 | −11.59% |
| No prestige | 0.9618 | −2.50% |
| No co-authorship distance | 0.9859 | −0.09% |
| Semantic similarity only | 0.9513 | −3.55% |
| Network features only | 0.8512 | −13.56% |

The ablation results directly answer **RQ4**: removing semantic similarity causes a massive drop in performance (−11.59 pp), whereas removing social features like co-authorship distance has a negligible effect. In LIS, content matters far more than social connections.

---

## Repository Structure

```
├── paper/                   # Author-compiled manuscript (PDF and LaTeX)
├── code/                    # Python scripts for data extraction, ML, and figures
├── figures/                 # All 8 manuscript figures (PNG)
└── results/                 # Raw computed results (JSON)
```

---

## Dependencies

- Python 3.11
- pandas, numpy, scikit-learn, networkx
- sentence-transformers (SBERT: all-MiniLM-L6-v2)
- shap, matplotlib, seaborn
- Data source: [Dimensions.ai](https://www.dimensions.ai/) (API access required)
