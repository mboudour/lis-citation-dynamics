# Citation Dynamics: S-N-SN Interpretation of Link Prediction Findings

**Author:** Manus AI  
**Date:** June 20, 2026

## 1. Overview of the S-N-SN Framework

The central analytical framework of this study decomposes citation link prediction into three models based on the underlying feature representations:
- **S (Semantic):** Features capturing the textual and conceptual similarity between papers, represented here exclusively by `directional_similarity` derived from SBERT embeddings.
- **N (Network/Structural):** Features capturing the graph topology and temporal dynamics of the citation network, comprising `temporal_indegree`, `citation_time_gap`, and `temporal_pagerank`.
- **SN (Semantic + Network):** The combined model utilizing all four features.

By evaluating the performance (AUC), feature importance (SHAP), and ablation impact across 10 datasets spanning 5 academic domains, we can identify which domains are driven by semantic content versus structural network effects.

## 2. Predictive Performance Across Domains (SN Model)

The baseline predictive performance of the SN model using Gradient Boosting (Figure 1 and Figure 5) reveals significant variation across academic domains. 

| Domain | Dataset | CV AUC (GB) |
| :--- | :--- | :--- |
| **Science** | Protein Folding | 0.751 |
| **Science** | CRISPR | 0.739 |
| **Humanities** | Film Studies | 0.702 |
| **BioMed** | Osteosarcoma | 0.694 |
| **BioMed** | Neuroblastoma | 0.643 |
| **Social Science** | Org. Behavior | 0.605 |
| **Engineering** | Additive Manuf. | 0.600 |
| **Humanities** | Memory Studies | 0.575 |
| **Engineering** | Corrosion Prot. | 0.539 |
| **Social Science** | Income Inequality | 0.526 |

The Science domain exhibits the highest predictive accuracy (mean AUC 0.719), followed by BioMed (0.649) and Humanities (0.620). Engineering (0.563) and Social Science (0.568) are notably harder to predict using these four features. Figure 5 further demonstrates that tree-based models (Gradient Boosting and Random Forest) consistently outperform linear models (Logistic Regression and Linear SVM), indicating non-linear relationships between the features and the probability of citation.

## 3. The Dominance of Network Structure (N)

The most striking finding across nearly all datasets is the overwhelming dominance of the Network (N) features, specifically the `citation_time_gap`. 

As shown in the SHAP analysis (Figure 2 and Figure 8), the N features account for the majority of the predictive weight in 9 out of 10 datasets. The `citation_time_gap` alone carries a normalized SHAP value ranging from 0.38 (Additive Manufacturing) to 0.65 (Protein Folding). 

This dominance is confirmed by the feature ablation results (Figure 3 and Figure 7). Removing the `citation_time_gap` feature causes catastrophic drops in AUC for the Science datasets: a 0.548 drop for Protein Folding and a 0.274 drop for CRISPR. This indicates that in the hard sciences, the temporal proximity between publication and citation is the single most critical determinant of link formation.

Interestingly, the other two N features—`temporal_indegree` and `temporal_pagerank`—contribute almost nothing to the model. Their SHAP values are consistently low (≤0.18), and their ablation drops are negligible (≤0.032). The "Network" signal is almost entirely a "Temporal Gap" signal.

## 4. The Limited Role of Semantics (S)

The Semantic (S) feature, `directional_similarity`, plays a surprisingly limited role in most domains. 

In the Engineering and Social Science domains, the S feature appears to be entirely inert. Despite having moderate SHAP values (0.39 to 0.45), removing the S feature results in zero or near-zero AUC drop in Corrosion Protection (0.000), Income Inequality (0.000), and Organizational Behavior (0.002). This suggests that while semantic similarity correlates with citations, it provides no unique predictive information beyond what is already captured by the temporal gap.

The BioMed and Science domains show slight S contribution. Removing the S feature drops AUC by 0.058 in Neuroblastoma and 0.042 in Osteosarcoma, indicating a small but measurable semantic signal.

### The Humanities Exception: Memory Studies

The sole exception to N-dominance is the Humanities dataset, Memory Studies. In this dataset, the S feature achieves near-parity in SHAP importance (0.48 vs 0.40 for citation gap). Crucially, the ablation analysis reveals that removing the S feature causes a larger AUC drop (0.126) than removing the N feature (0.085). 

This makes Memory Studies the only **S-dominated** dataset in the corpus. In this subfield, the conceptual alignment between papers is more predictive of a citation link than the temporal proximity of publication. Film Studies, the other Humanities dataset, also shows a meaningful S contribution (ablation drop 0.051), suggesting that Humanities disciplines may rely more heavily on semantic linkages than STEM fields.

## 5. Temporal Dynamics and Generalization

The temporal generalization analysis (Figure 4) tests whether models trained on older data can predict citations in a recent holdout period (2016–2020). 

| Domain | Dataset | CV AUC | Holdout AUC | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Science** | CRISPR | 0.734 | 0.798 | +0.064 |
| **Engineering** | Additive Manuf. | 0.595 | 0.660 | +0.065 |
| **Engineering** | Corrosion Prot. | 0.530 | 0.615 | +0.085 |
| **Humanities** | Film Studies | 0.685 | 0.583 | -0.102 |
| **BioMed** | Osteosarcoma | 0.672 | 0.605 | -0.067 |

For rapidly evolving fields like CRISPR and Additive Manufacturing, the model performs *better* on the recent holdout data than in cross-validation. This suggests that citation behavior in these fields has become more predictable and structurally rigid in recent years. Conversely, older or more stable fields like Film Studies and Osteosarcoma show significant performance drops in the holdout period, indicating a shift in citation dynamics.

The rolling temporal AUC analysis (Figure 6) confirms these non-stationary dynamics. Fields like Organizational Behavior and Neuroblastoma show dramatic improvements in predictability after the early 2000s, while Additive Manufacturing shows a steady decline in predictability over its brief, explosive history.

## 6. Conclusion: The S-N Balance

The S-N-SN framework reveals that citation link prediction is not a monolithic problem. The balance between Semantic (S) and Network (N) signals varies systematically by academic domain:

1. **Strong N-Dominance (Science):** Protein Folding and CRISPR are heavily reliant on the temporal gap between papers. Semantic similarity provides almost no unique predictive value.
2. **Moderate N-Dominance (BioMed):** Neuroblastoma and Osteosarcoma rely primarily on temporal structure but show small, distinct contributions from semantic similarity.
3. **N-Dominant but S-Inert (Engineering & Social Science):** In fields like Corrosion Protection and Income Inequality, semantic similarity is highly correlated with citations but provides zero independent predictive power when temporal gaps are known.
4. **S-Dominance (Humanities):** Memory Studies is the only dataset where semantic similarity outweighs temporal structure, highlighting the unique citation culture of the humanities where conceptual linkage transcends temporal recency.

These findings suggest that a universal "SN" model is sub-optimal. Citation prediction architectures must be domain-aware, prioritizing temporal network features for STEM fields while elevating semantic embeddings for the humanities.
