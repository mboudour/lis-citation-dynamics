# Citation Dynamics in Library and Information Science (LIS)
### A Temporally-Aware Machine Learning Framework for Predicting Scientific Citations

---

## Overview

This repository contains the code, figures, and results for a study on citation dynamics in the Library and Information Science (LIS) literature. We apply a machine learning framework to predict whether one scientific paper will cite another, using a corpus of 259,220 peer-reviewed articles published between 1975 and 2024. A central design principle of this work is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the data leakage that affects many prior citation prediction studies.

The pipeline is designed to be run on **two independent data sources** — Dimensions.ai and OpenAlex — enabling a cross-database robustness check that strengthens the generalizability of our findings. See the [Comparative Pipeline](#comparative-pipeline-dimensions-vs-openalex) section below.

---

## Research Questions

This study addresses four core research questions:

**RQ1 — Predictability:** To what extent are citation events in LIS predictable from observable features of the papers and their authors?

**RQ2 — Temporal validity:** Does eliminating temporal data leakage — a methodological flaw common in the citation prediction literature — substantially reduce model performance, or do citation dynamics remain highly predictable under strict causal constraints?

**RQ3 — Feature importance:** Which mechanisms drive citation behavior? Specifically, what is the relative contribution of (a) semantic content similarity, (b) social proximity via co-authorship networks, and (c) prestige (the Matthew effect)?

**RQ4 — Feature interactions:** Is it the combination of semantic and structural network features that enables accurate prediction, or can a single feature class alone explain citation behavior?

---

## Dataset

### Dimensions.ai (Primary Dataset)

The primary dataset was retrieved from [Dimensions.ai](https://www.dimensions.ai/) using the Python script below (requires a valid Dimensions API key and the `dimcli` library). Due to Dimensions' proprietary Terms of Service, the raw dataset cannot be redistributed; researchers with Dimensions access can reproduce it exactly using this script.

```python
%%time
import dimcli
import pandas as pd
import time
import datetime

# --- Authentication ---
with open("../key.txt", "r") as f:
    api_key = f.read().strip()

dimcli.login(key=api_key, endpoint="https://app.dimensions.ai/api/dsl.json")
dsl = dimcli.Dsl()

# --- Configuration ---
YEARS   = list(range(1975, 2025))
FIELDS  = ("id+year+date+title+authors+journal+abstract+"
           "times_cited+reference_ids+"
           "category_for+concepts+open_access+doi")

KEYWORDS = [
    "knowledge organization",
    "digital libraries",
    "information literacy",
    "academic libraries"
]

# --- Helper ---
def safe_query(dsl, query_str, label=""):
    for attempt in range(3):
        try:
            result = dsl.query_iterative(query_str).as_dataframe()
            print(f"  {label} → {len(result)} records", flush=True)
            return result
        except Exception as e:
            print(f"  {label} error: {e}  (attempt {attempt+1}/3)", flush=True)
            time.sleep(10)
    return pd.DataFrame()

# --- Year-by-year loop ---
all_frames = []

for y in YEARS:
    print(f"\n[{datetime.datetime.now():%H:%M:%S}]  Year {y}", flush=True)
    year_parts = []

    # Query A — FoR 4610
    q = safe_query(dsl, f"""
        search publications
        where category_for.id = "4610"
        and year = {y}
        and type = "article"
        return publications[{FIELDS}]
    """, label="FoR 4610")
    year_parts.append(q)

    # Query B — one query per keyword
    for kw in KEYWORDS:
        q = safe_query(dsl, f"""
            search publications
            in title_abstract_only
            for "{kw}"
            where year = {y}
            and type = "article"
            return publications[{FIELDS}]
        """, label=f'"{kw}"')
        year_parts.append(q)
        time.sleep(1)

    # Combine and deduplicate for this year
    year_df = pd.concat(year_parts, ignore_index=True)
    year_df.drop_duplicates(subset="id", inplace=True)
    all_frames.append(year_df)
    print(f"  Year {y} deduplicated total: {len(year_df)}", flush=True)
    time.sleep(2)

# --- Final assembly ---
df = pd.concat(all_frames, ignore_index=True)
df.drop_duplicates(subset="id", inplace=True)
df.sort_values("year", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n{'='*50}")
print(f"TOTAL RECORDS : {len(df)}")
print(f"YEAR RANGE    : {int(df['year'].min())} – {int(df['year'].max())}")

df.to_pickle("Dimensions_LIS_1975_2024.pkl")
print("Saved → Dimensions_LIS_1975_2024.pkl")
```

The query combines **FoR 4610** (ANZSRC Field of Research code for Library and Information Studies) with keyword searches in title/abstract for four core LIS terms. Results are deduplicated by Dimensions publication ID.

| Property | Value |
| :--- | :--- |
| **Total articles** | 259,220 |
| **Abstract coverage** | 96.4% |
| **Reference list coverage** | 65.2% |
| **Mean citations per paper** | 20.98 |
| **Median citations per paper** | 4.0 |

### OpenAlex (Replication Dataset)

A replication on [OpenAlex](https://openalex.org/) data is in progress. The data collection script and query will be added here once the methodology is finalised by the authors. The resulting pickle file (`OpenAlex_LIS_1975_2024.pkl`) follows the same column schema as the Dimensions dataset, enabling the pipeline to be run without modification.

### Citation Pair Construction

The classification task is: given a pair of papers $(A, B)$ where $B$ was published no later than $A$, predict whether $A$ cites $B$.

| Split | Count (Dimensions) |
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

SHAP (SHapley Additive exPlanations) values are computed for the best-performing model (MLP Neural Network) to quantify global feature importance and interpret the contribution of each feature to individual predictions.

---

## Results (Dimensions.ai)

### Model Performance

| Model | ROC-AUC | F1-Score | Accuracy | Precision | Recall | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP Neural Network** | **0.9906** | **0.9540** | **0.9539** | **0.9518** | 0.9562 | **0.9078** |
| Gradient Boosting | 0.9904 | 0.9542 | 0.9540 | 0.9511 | **0.9573** | 0.9081 |
| Logistic Regression | 0.9897 | 0.9522 | 0.9521 | 0.9505 | 0.9539 | 0.9042 |
| Linear SVM | 0.9897 | 0.9522 | 0.9521 | 0.9505 | 0.9540 | 0.9043 |
| Random Forest | 0.9873 | 0.9503 | 0.9502 | 0.9473 | 0.9533 | 0.9003 |
| k-NN (k=5) | 0.9773 | 0.9485 | 0.9483 | 0.9459 | 0.9511 | 0.8967 |

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

### Ablation Study

| Feature Subset | ROC-AUC | Δ vs. All Features |
| :--- | :--- | :--- |
| **All features** | **0.9868** | — |
| No semantic similarity | 0.8709 | −11.59% |
| No prestige | 0.9618 | −2.50% |
| No co-authorship distance | 0.9859 | −0.09% |
| Semantic similarity only | 0.9513 | −3.55% |
| Network features only | 0.8512 | −13.56% |

---

## Comparative Pipeline: Dimensions vs. OpenAlex

To assess the robustness of our findings across independent bibliographic databases, we run the **identical pipeline** on both Dimensions.ai and OpenAlex data. The comparison tests whether the dominance of semantic similarity, the Matthew effect, and the relative unimportance of social proximity are properties of LIS citation behavior in general, rather than artifacts of a particular database's coverage or classification scheme.

### How to run the OpenAlex pipeline

**Step 1 — Fetch the data:**
```bash
pip install requests pandas
python data/fetch_openalex_lis.py --email your@email.com --output OpenAlex_LIS_1975_2024.pkl
```
The `--email` flag is strongly recommended: it places your requests in OpenAlex's "polite pool" with higher rate limits. The script takes approximately 10–20 minutes to fetch ~260,000 records.

**Step 2 — Run Stage 1 (feature engineering):**
```bash
python code/stage1_features.py --dataset OpenAlex_LIS_1975_2024.pkl --source openalex
```

**Step 3 — Run Stage 2 (SBERT encoding):**
```bash
python code/stage2_sbert.py --source openalex
```

**Step 4 — Run Stage 3 (ML training):**
```bash
python code/stage3_ml.py --source openalex
```

**Step 5 — Run Stage 4 (figures):**
```bash
python code/stage4_figures.py --source openalex --compare
```
The `--compare` flag generates side-by-side comparison figures for both datasets.

### Expected differences

| Property | Dimensions.ai | OpenAlex (expected) |
| :--- | :--- | :--- |
| Reference list coverage | 65.2% | ~85–90% |
| Abstract coverage | 96.4% | ~80–90% |
| Corpus size | 259,220 | ~200,000–280,000 |
| API access | Subscription required | Fully open |

The higher reference list coverage in OpenAlex is expected to yield a denser citation network and potentially stronger network features. The core findings (semantic similarity dominance, Matthew effect) are expected to be robust.

---

## Repository Structure

```
├── paper/                          # Author-compiled manuscript (PDF and LaTeX source)
│   ├── lis_manuscript.pdf
│   └── lis_manuscript.tex
├── code/                           # Python scripts for the full pipeline
│   ├── extract_texts.py            # Stage 0: extract title+abstract texts
│   ├── stage2_sbert.py             # Stage 2: SBERT encoding (checkpoint-based)
│   ├── stage3_ml.py                # Stage 3: ML training, SHAP, ablation
│   └── stage4_figures.py           # Stage 4: manuscript figures
├── data/                           # Data acquisition scripts
│   └── fetch_openalex_lis.py       # Fetch LIS data from OpenAlex API
├── figures/                        # All 8 manuscript figures (PNG and PDF)
│   ├── fig1_publications_per_year.*
│   ├── fig2_mean_citations_per_year.*
│   ├── fig3_feature_distributions.*
│   ├── fig4_model_comparison.*
│   ├── fig5_roc_curves.*
│   ├── fig6_shap_importance.*
│   ├── fig7_ablation.*
│   └── fig8_feature_correlation.*
└── results/                        # Raw computed results (JSON)
    ├── lis_cv_results.json          # 5-fold CV scores for all 6 models
    ├── lis_shap_values.json         # SHAP feature importances (MLP Neural Network)
    ├── lis_ablation.json            # Ablation study AUC scores
    └── lis_dataset_stats.json       # Dataset summary statistics
```

---

## Dependencies

```
pip install pandas numpy scikit-learn networkx sentence-transformers shap matplotlib seaborn requests
```

- Python 3.11
- SBERT model: `all-MiniLM-L6-v2` (downloaded automatically by sentence-transformers)
- Primary data source: [Dimensions.ai](https://www.dimensions.ai/) (subscription required)
- Replication data source: [OpenAlex](https://openalex.org/) (fully open)

---

## Authors

Moses Boudourides — School of Professional Studies, Northwestern University  
Giannis Tsakonas — Library Information Center, University of Patras

---

## Citation

If you use this work, please cite the manuscript (forthcoming in *Scientometrics*).
