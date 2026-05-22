# LIS Citation Dynamics

**Predicting Citation Links in Library and Information Science: Where Network Science Meets Machine Learning**

Moses A. Boudourides · Giannis Tsakonas

---

## Overview

This repository contains all scripts, results, and figures for a study on citation dynamics in the Library and Information Science (LIS) literature. We apply a machine learning framework to predict whether one scientific paper will cite another, using a corpus of 259,220 peer-reviewed articles (Dimensions.ai, 1975–2024) and a replication corpus from OpenAlex (168,901 articles). A central design principle is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the data leakage that affects many prior citation prediction studies.

---

## Data

### Dimensions.ai (Primary Dataset)

Retrieved using `scripts/data_collection/fetch_dimensions_lis.py` (requires a valid Dimensions API key and the `dimcli` library). Due to Dimensions' Terms of Service, the raw dataset is not redistributed here; researchers with Dimensions access can reproduce it exactly using the provided script.

The script issues two types of queries for each year from 1975 to 2024:

- **Query A** — Field of Research (FoR) code `4610` (Library and Information Studies), filtering for journal articles only.
- **Query B** — Title/abstract keyword search for each of the following terms, again restricted to journal articles: `"knowledge organization"`, `"digital libraries"`, `"information literacy"`, `"academic libraries"`.

Results from both query types are deduplicated on Dimensions article ID. The following fields are retrieved for each article: `id`, `year`, `date`, `title`, `authors`, `journal`, `abstract`, `times_cited`, `reference_ids`, `category_for`, `concepts`, `open_access`, `doi`.

| Property | Value |
|---|---|
| Total articles | 259,220 |
| Year range | 1975–2024 |
| Abstract coverage | 96.4% |
| Reference list coverage | 65.2% |
| Mean citations per paper | 20.98 |

### OpenAlex (Replication Dataset)

Retrieved using `scripts/data_collection/fetch_openalex_lis.py`. Fully open and reproducible without any subscription.

The script filters works by `primary_topic.id` using five specific OpenAlex topic IDs that correspond to core LIS and scientometrics research areas, combined with a publication year range of 1975–2024:

| Topic ID | Topic Label | Works at time of retrieval |
|---|---|---|
| T10712 | Library Science and Information Literacy | 114,018 |
| T14330 | Library Science and Information Systems | 98,052 |
| T13166 | Information Science and Libraries | 55,834 |
| T13673 | Library Science and Information | 25,971 |
| T10102 | Scientometrics and Bibliometrics Research | 99,916 |

The following fields are retrieved for each article: `id`, `doi`, `title`, `abstract_inverted_index`, `authorships`, `publication_year`, `primary_location`, `open_access`, `referenced_works`, `cited_by_count`.

| Property | Value |
|---|---|
| Total articles | 168,901 |
| Year range | 1975–2024 |
| Abstract coverage | 64.3% |
| Reference list coverage | 29.8% |

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

All results are 5-fold stratified cross-validation on 500,000 pairs (250,000 positive + 250,000 negative).

**Feature ablation (Logistic Regression, leave-one-out AUC):**

| Feature removed | AUC | Drop |
|---|---|---|
| None (all features) | 0.9868 | — |
| semantic\_similarity | 0.8709 | −0.0159 |
| prestige | 0.9618 | −0.0250 |
| temporal\_distance | 0.9576 | −0.0292 |
| coauth\_distance | 0.9859 | −0.0009 |
| Semantic similarity only | 0.9513 | — |
| Network features only | 0.8512 | — |

## Results (OpenAlex)

Best model: **Gradient Boosting**, AUC = **0.9979**

| Model | AUC | F1 | Accuracy | Precision | Recall | MCC |
|---|---|---|---|---|---|---------|
| Logistic Regression | 0.9872 | 0.9478 | 0.9495 | 0.9574 | 0.9385 | 0.8991 |
| Linear SVM | 0.9863 | 0.9451 | 0.9468 | 0.9549 | 0.9355 | 0.8937 |
| Random Forest | 0.9972 | 0.9793 | 0.9796 | 0.9729 | 0.9857 | 0.9592 |
| **Gradient Boosting** | **0.9979** | **0.9810** | **0.9813** | **0.9752** | **0.9868** | **0.9626** |

All results are 5-fold stratified cross-validation on 489,349 pairs (239,349 positive + 250,000 negative). k-NN and Neural Network were excluded from the OpenAlex run due to memory constraints on the 489k-pair dataset; the four models above are directly comparable with the Dimensions results.

**Feature ablation (Random Forest, leave-one-out AUC):**

| Feature removed | AUC | Drop |
|---|---|---|
| None (all features) | 0.9972 | — |
| prestige\_cited | 0.9847 | −0.0125 |
| activity\_citing | 0.9890 | −0.0082 |
| semantic\_similarity | 0.9914 | −0.0058 |
| temporal\_gap | 0.9960 | −0.0012 |
| common\_citers | 0.9970 | −0.0002 |
| jaccard\_refs | 0.9972 | 0.0000 |
| common\_refs | 0.9972 | 0.0000 |

---

## Requirements

```
Python 3.11
numpy==1.26.4
pandas>=2.0
scikit-learn>=1.3
networkx>=3.0
sentence-transformers==2.7.0
torch==2.1.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
pyarrow>=14.0
dimcli>=1.0
# Dimensions.ai only — requires a valid API key
```
