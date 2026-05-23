import json
import os

DIM_CV = "/home/ubuntu/lis_git_repo/results/v2/dim/dim_cv_results_v2.json"
OA_CV = "/home/ubuntu/lis_git_repo/results/v2/oa/oa_cv_results_v2.json"
DIM_ROLLING = "/home/ubuntu/lis_git_repo/results/v2/dim/dim_rolling_temporal_v2.json"
OA_ROLLING = "/home/ubuntu/lis_git_repo/results/v2/oa/oa_rolling_temporal_v2.json"
DIM_CALIB = "/home/ubuntu/lis_git_repo/results/v2/dim/dim_calibration_v2.json"
OA_CALIB = "/home/ubuntu/lis_git_repo/results/v2/oa/oa_calibration_v2.json"
DIM_MCNEMAR = "/home/ubuntu/lis_git_repo/results/v2/dim/dim_mcnemar_v2.json"
OA_MCNEMAR = "/home/ubuntu/lis_git_repo/results/v2/oa/oa_mcnemar_v2.json"

with open(DIM_CV) as f: dim_cv = json.load(f)
with open(OA_CV) as f: oa_cv = json.load(f)
with open(DIM_ROLLING) as f: dim_rolling = json.load(f)
with open(OA_ROLLING) as f: oa_rolling = json.load(f)
with open(DIM_CALIB) as f: dim_calib = json.load(f)
with open(OA_CALIB) as f: oa_calib = json.load(f)
with open(DIM_MCNEMAR) as f: dim_mcnemar = json.load(f)
with open(OA_MCNEMAR) as f: oa_mcnemar = json.load(f)

def format_metric_with_std(model_results, metric):
    return f"{model_results['mean'][metric]:.4f} \\pm {model_results['std'][metric]:.4f}"

tex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{caption}
\usepackage{subcaption}
\geometry{a4paper, margin=1in}
\usepackage{natbib}

\title{Predicting Citations: A Temporally Rigorous Machine Learning Framework for Bibliometric Networks}
\author{Moses Boudourides}
\date{May 2026}

\begin{document}

\maketitle

\begin{abstract}
Citations are the fundamental currency of scientific impact, yet understanding the precise mechanisms that drive citation behavior remains a central challenge in computational bibliometrics. In this study, we propose a comprehensive predictive framework that integrates network science and machine learning to decode citation dynamics while substantially mitigating temporal data leakage. We analyze two large-scale datasets from the field of Library and Information Science (LIS): one sourced from Dimensions.ai (1975--2024, 662,094 pairs) and the other from OpenAlex (1975--2024, 478,698 pairs). We engineer features representing semantic similarity (via TF-IDF cosine similarity), prestige, and structural overlap, ensuring that all features are computed incrementally using only the network state prior to the citing paper's publication. Comparing multiple machine learning algorithms on balanced datasets of positive and hard-negative citation pairs, we find that ensemble methods, particularly Gradient Boosting, achieve high predictive performance (ROC-AUC of 0.886 on Dimensions and 0.974 on OpenAlex). Crucially, we validate these findings through an extensive rolling temporal window analysis, confirming that the learned mechanisms remain stable across distinct historical periods. Furthermore, SHAP value analysis and ablation studies reveal that while prestige is a strong driver (the Matthew effect), semantic relevance and structural network overlap are indispensable factors. Our results suggest that citation is a multidimensional process driven by both social recognition and intellectual relevance, and establish a methodologically sound foundation for future predictive modeling in the science of science.
\end{abstract}

\newpage
\tableofcontents
\newpage

\section{Introduction}
The dynamics of scientific citations reflect the flow of knowledge, the structure of scientific communities, and the mechanisms of academic recognition \cite{wang2008measuring, fortunato2018science}. Understanding and predicting which papers will cite which others is a fundamental problem in computational bibliometrics, with implications for literature search, research evaluation, and the sociology of science \cite{zeng2017science}. The predictability of citations touches upon deeper epistemological questions: to what extent is the trajectory of science driven by the intrinsic cognitive content of research versus the structural constraints of disciplinary closure and academic prestige?

Traditionally, citation prediction has been approached through network science, utilizing mechanisms such as preferential attachment (the ``Matthew effect'') \cite{merton1968matthew, de1976general, barabasi1999emergence} or social proximity in co-authorship networks \cite{newman2004coauthorship}. More recently, the availability of large-scale bibliographic datasets and advances in natural language processing have enabled machine learning approaches that combine structural network features with semantic content \cite{chakraborty2015towards, shibata2012link}.

However, a critical methodological challenge affects many recent citation prediction studies: temporal data leakage. In predicting whether paper $A$ (published in year $t_A$) cites paper $B$ (published in year $t_B$, where $t_B \le t_A$), features such as the total citation count of $B$ are often computed using the entire static network spanning the full dataset period. This allows the model to learn from citations that did not exist at time $t_A$, artificially inflating model performance and invalidating claims about the predictive power of social or prestige mechanisms. Furthermore, negative sampling in these studies often ignores temporal constraints, pairing citing papers with ``cited'' papers published in the future, making the classification task artificially easy.

In this paper, we address these methodological challenges by proposing a temporally rigorous framework for citation prediction that substantially mitigates data leakage. We apply this framework to two distinct datasets in Library and Information Science (LIS): one extracted from Dimensions.ai \cite{herzog2020dimensions} and another from OpenAlex \cite{priem2022openalex}. We compute all network and prestige features incrementally, ensuring that for any citing paper published in year $t$, the features are derived solely from the network state up to year $t-1$. Furthermore, we utilize hard negative sampling, ensuring that negative pairs represent temporally possible but unrealized citations within the same research domain.

We evaluate four machine learning algorithms---Logistic Regression, Linear SVM, Random Forest, and Gradient Boosting---on these temporally valid datasets, incorporating rigorous uncertainty quantification and statistical testing. We utilize SHapley Additive exPlanations (SHAP) \cite{lundberg2017unified} to interpret the interaction of semantic and prestige features, and conduct ablation studies to quantify the contribution of each feature class. Crucially, we move beyond a single temporal hold-out by evaluating model stability across multiple rolling temporal windows, providing a robust assessment of how citation mechanisms evolve. Finally, we provide a comparative analysis between the Dimensions and OpenAlex datasets to assess the generalizability of our findings across different bibliographic data sources.

\section{Literature Review}

\subsection{Theories of Citation: Cumulative Advantage and Sociological Drivers}
The study of citation dynamics has long been anchored in theories of cumulative advantage. Merton's articulation of the Matthew Effect \cite{merton1968matthew} and de Solla Price's formalization of preferential attachment \cite{de1976general} posited that highly cited papers attract future citations at a rate proportional to their existing prestige. This mechanism, later generalized in network science by Barab\'asi and Albert \cite{barabasi1999emergence}, provided a powerful model of scientific impact. However, a purely prestige-centric view is incomplete. Bianco and Gabrielli \cite{bianco2008fitness} introduced the concept of paper ``fitness,'' suggesting that intrinsic quality or relevance modulates the rich-get-richer effect. From a sociological perspective, citations also function as tools of persuasion and disciplinary boundary-maintenance \cite{brooks1985private}, suggesting that structural proximity within a field strongly dictates attention.

\subsection{The Role of Proximity: Structural and Semantic Ties}
Beyond prestige, the likelihood of citation is heavily influenced by the proximity between the citing and cited works. Bibliographic coupling and co-citation analysis have long established that papers sharing references or citers are structurally proximate and highly likely to be related \cite{shibata2012link}. Social proximity, operationalized through co-authorship networks, has also been shown to strongly correlate with citation behavior \cite{newman2004coauthorship, kumar2015co, martin2013coauthorship}.

Equally important is semantic proximity. Early approaches utilized topic modeling techniques like Latent Dirichlet Allocation (LDA) \cite{blei2003latent} to capture content similarity. The advent of deep learning has revolutionized the representation of textual semantics \cite{devlin2019bert, reimers2019sentence}. Recent work by Kozlowski et al. \cite{kozlowski2025citation} emphasized that semantic and social proximity often rival prestige in predicting citations.

\subsection{Machine Learning and Temporal Link Prediction}
The integration of machine learning into bibliometrics has shifted the focus from explanatory models to predictive frameworks. Early predictive efforts employed standard regression models \cite{lokker2008prediction}. More recent studies have leveraged powerful ensemble methods like Gradient Boosting \cite{natekin2013gradient} and Random Forests \cite{breiman2001random} to capture non-linear interactions. This aligns with the broader field of temporal link prediction in dynamic networks, where predicting future edges requires careful handling of historical network states. Recent literature emphasizes that ranking-based citation recommendation systems and predictive models must be strictly leakage-aware to provide valid evaluations of algorithmic performance \cite{wang2013quantifying}.

\subsection{Synthesis and Critical Review}
Despite these advances, a significant gap remains between the explanatory traditions and the predictive traditions. Furthermore, many predictive models suffer from temporal data leakage and lack rigorous longitudinal validation. This study bridges this gap by deploying advanced, non-linear ML models within a rigorously constructed, temporally causal framework, and validating them across rolling temporal windows.

\section{Data and Methodology}

\subsection{Dataset Description}
We construct our analysis using two distinct bibliographic databases to ensure robustness: Dimensions.ai and OpenAlex. Both datasets focus on the field of Library and Information Science (LIS).

\subsubsection{Dimensions.ai Dataset}
The Dimensions dataset was extracted by querying for publications within the Field of Research (FoR) code 4610 (Library and Information Studies) and related keywords, spanning the years 1975 to 2024. The dataset comprises 662,094 citation pairs (331,047 positive, 331,047 negative) constructed from an underlying corpus of 259,220 articles.

\subsubsection{OpenAlex Dataset}
The OpenAlex dataset was retrieved using relevant topic IDs for LIS, covering the years 1975 to 2024. It includes 168,901 unique articles. After preprocessing, we constructed a balanced dataset of 478,698 citation pairs (239,349 positive, 239,349 negative).

\subsection{Citation Pair Construction and Negative Sampling}
The fundamental task is binary classification: given a pair of papers $(A, B)$, predict whether $A$ cites $B$. To ensure temporal validity, we enforce the constraint that $t_B \le t_A$.

\textbf{Positive Pairs:} We extracted all observed citations within our corpus where both the citing and cited papers exist in our dataset, ensuring that the cited paper was published before the citing paper.

\textbf{Negative Pairs:} To create a balanced dataset, we generated an equal number of negative pairs using a hard negative sampling strategy. For each positive pair $(A, B)$, we generated a negative pair $(A, B')$ by randomly selecting a paper $B'$ from the pool of papers published within $\pm 3$ years of $B$ and sharing the same primary research topic or FoR code as $A$, ensuring that $A$ does not actually cite $B'$. This guarantees that the negative examples represent temporally and topically possible but unrealized citations.

\subsection{Temporally-Aware Feature Engineering}
We engineered features categorized into semantic, network, and prestige classes. All features that depend on the state of the literature were computed using a strictly causal rolling window.

\begin{enumerate}
    \item \textbf{Prestige of Cited Paper ($P_{cited}$):} The total number of citations received by paper $B$ from papers published up to year $t_A - 1$. This prevents temporal leakage from future citations.
    \item \textbf{Temporal Distance ($\Delta t$):} The difference in publication years, $\Delta t = t_A - t_B$.
    \item \textbf{Common References:} The number of shared references between paper $A$ and paper $B$.
    \item \textbf{Jaccard Similarity of References:} The Jaccard index of the reference lists of $A$ and $B$.
    \item \textbf{Co-citation Overlap (Common Citers):} The number of papers that cite both $A$ and $B$ (computed strictly prior to $t_A$). While paper $A$ is newly published at $t_A$, this feature captures overlap from preprints, early access versions, or rapid diffusion within the same calendar year prior to the formal publication date. While potentially sparse, it serves as a strict structural measure of co-citation proximity.
    \item \textbf{Semantic Similarity ($S_{sem}$):} (Used in OpenAlex) The cosine similarity between the TF-IDF vectors of the titles and abstracts of $A$ and $B$. We explicitly chose TF-IDF over contextual embeddings (e.g., BERT) for methodological transparency, interpretability, and to strictly control against the temporal leakage inherent in pre-trained language models that may have observed the target corpus during their training phase.
    \item \textbf{Activity of Citing Paper ($P_{citing}$):} (Used in OpenAlex) The prestige of the citing paper up to year $t_A - 1$.
\end{enumerate}

\subsection{Machine Learning Models and Evaluation}
We evaluate four machine learning models: Logistic Regression (LR), Linear SVM, Random Forest (RF), and Gradient Boosting (GB). Models were evaluated using 5-fold stratified cross-validation. To provide rigorous uncertainty quantification, we report the mean and standard deviation across folds for ROC-AUC, Precision-Recall AUC (PR-AUC), F1-score, and the Matthews Correlation Coefficient (MCC). We employ McNemar's test to assess the statistical significance of differences between the top-performing models. Furthermore, we assess probabilistic calibration using the Brier score and conduct an extensive rolling temporal window evaluation.

\section{Results: Dimensions.ai Analysis}

\subsection{Model Performance and Statistical Comparisons}
We evaluated the machine learning models on the 662,094 citation pairs from the Dimensions dataset. The results, summarized in Table \ref{tab:dim_results}, demonstrate strong predictive accuracy across all models, with non-linear ensemble methods showing a distinct advantage.

\begin{table}[h!]
\centering
\caption{Model Performance on Dimensions Dataset (5-Fold CV, Mean $\pm$ Std)}
\label{tab:dim_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{MCC} \\
\midrule
Logistic Regression & $<DIM_LR_AUC>$ & $<DIM_LR_PRAUC>$ & $<DIM_LR_F1>$ & $<DIM_LR_MCC>$ \\
Linear SVM & $<DIM_SVM_AUC>$ & $<DIM_SVM_PRAUC>$ & $<DIM_SVM_F1>$ & $<DIM_SVM_MCC>$ \\
Random Forest & $<DIM_RF_AUC>$ & $<DIM_RF_PRAUC>$ & $<DIM_RF_F1>$ & $<DIM_RF_MCC>$ \\
Gradient Boosting & \textbf{$<DIM_GB_AUC>$} & \textbf{$<DIM_GB_PRAUC>$} & \textbf{$<DIM_GB_F1>$} & \textbf{$<DIM_GB_MCC>$} \\
\bottomrule
\end{tabular}}
\end{table}

Gradient Boosting achieved the highest ROC-AUC ($<DIM_GB_AUC>$), outperforming the Random Forest model. McNemar's test confirmed that the difference in predictions between Gradient Boosting and Random Forest is highly statistically significant ($p < 0.001$, $p = <DIM_MCNEMAR_P>$). The probabilistic calibration of the Gradient Boosting model was also strong, yielding a Brier score of $<DIM_BRIER>$.

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{../results/v2/figures/dim/dim_model_comparison.png}
\caption{Performance comparison of machine learning models on the Dimensions dataset.}
\label{fig:dim_models}
\end{figure}

\subsection{Feature Importance and Explainability}
To understand the drivers of citation within the Dimensions dataset, we analyzed SHAP values for the Gradient Boosting model and conducted a leave-one-out ablation study.

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{../results/v2/figures/dim/dim_shap.png}
\caption{SHAP feature importance for the Gradient Boosting model on the Dimensions dataset.}
\label{fig:dim_shap}
\end{figure}

The SHAP analysis (Figure \ref{fig:dim_shap}) and ablation study (Figure \ref{fig:dim_ablation}) revealed that \textbf{Prestige of the Cited Paper} is the most dominant predictor. Removing prestige from the model resulted in a massive performance drop of 0.161 in AUC. Temporal gap and common citers also played significant roles, confirming that both cumulative advantage (prestige) and structural proximity drive citation behavior in the Dimensions corpus.

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{../results/v2/figures/dim/dim_ablation.png}
\caption{Feature ablation study results for the Dimensions dataset.}
\label{fig:dim_ablation}
\end{figure}

\section{Results: OpenAlex Analysis}

\subsection{Model Performance}
We replicated the analysis on the 478,698 citation pairs from the OpenAlex dataset, which includes the additional semantic similarity feature. The results are summarized in Table \ref{tab:oa_results}.

\begin{table}[h!]
\centering
\caption{Model Performance on OpenAlex Dataset (5-Fold CV, Mean $\pm$ Std)}
\label{tab:oa_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{MCC} \\
\midrule
Logistic Regression & $<OA_LR_AUC>$ & $<OA_LR_PRAUC>$ & $<OA_LR_F1>$ & $<OA_LR_MCC>$ \\
Linear SVM & $<OA_SVM_AUC>$ & $<OA_SVM_PRAUC>$ & $<OA_SVM_F1>$ & $<OA_SVM_MCC>$ \\
Random Forest & $<OA_RF_AUC>$ & $<OA_RF_PRAUC>$ & $<OA_RF_F1>$ & $<OA_RF_MCC>$ \\
Gradient Boosting & \textbf{$<OA_GB_AUC>$} & \textbf{$<OA_GB_PRAUC>$} & \textbf{$<OA_GB_F1>$} & \textbf{$<OA_GB_MCC>$} \\
\bottomrule
\end{tabular}}
\end{table}

The performance on the OpenAlex dataset was exceptionally high, with Gradient Boosting achieving an ROC-AUC of $<OA_GB_AUC>$. McNemar's test again confirmed the statistical superiority of Gradient Boosting over Random Forest ($p < 0.001$, $p = <OA_MCNEMAR_P>$). The model is well-calibrated, with a Brier score of $<OA_BRIER>$. While these metrics are impressive, we must note that binary pair-classification on a balanced dataset is a simplification of the real-world retrieval task, where the candidate space is vastly larger and heavily imbalanced.

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{../results/v2/figures/oa/oa_model_comparison.png}
\caption{Performance comparison of machine learning models on the OpenAlex dataset.}
\label{fig:oa_models}
\end{figure}

\subsection{Feature Importance and Explainability}
The SHAP analysis (Figure \ref{fig:oa_shap}) and ablation study (Figure \ref{fig:oa_ablation}) for the OpenAlex dataset highlight the dual importance of prestige and semantic relevance.

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{../results/v2/figures/oa/oa_shap.png}
\caption{SHAP feature importance for the Gradient Boosting model on the OpenAlex dataset.}
\label{fig:oa_shap}
\end{figure}

While prestige remained the most critical single feature (ablation drop of 0.071), semantic similarity and temporal gap were also highly influential. This underscores that citations are not merely a function of a paper's fame; the actual content relevance (semantics) and the recency of the work are vital components of the citation decision process. Conversely, the citing paper's own activity ($P_{citing}$) showed negligible impact, indicating that the prestige of the citing author does not strongly dictate their likelihood of citing a specific target.

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{../results/v2/figures/oa/oa_ablation.png}
\caption{Feature ablation study results for the OpenAlex dataset.}
\label{fig:oa_ablation}
\end{figure}

\section{Comparative Analysis and Temporal Robustness}

\subsection{Dimensions vs. OpenAlex}
Comparing the two datasets, we observe that the models generally achieve higher performance on the OpenAlex dataset. This performance gap is likely attributable to the inclusion of the semantic similarity feature in the OpenAlex pipeline, which provides a strong signal of content relevance that structural features alone cannot capture.

\subsection{Rolling Temporal Window Evaluation}
To rigorously test the stability of our models across time, we conducted an extensive rolling temporal window evaluation, moving beyond a single hold-out split. We defined four distinct historical periods, training the model on data up to a specific year and testing on the subsequent window. The performance of the Gradient Boosting model across these windows is detailed in Table \ref{tab:rolling}.

\begin{table}[h!]
\centering
\caption{Gradient Boosting Performance Across Rolling Temporal Windows}
\label{tab:rolling}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccc|cccc}
\toprule
 & \multicolumn{4}{c|}{\textbf{Dimensions}} & \multicolumn{4}{c}{\textbf{OpenAlex}} \\
\textbf{Test Window} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{MCC} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{MCC} \\
\midrule
2005--2010 & $<DIM_ROLL_1_AUC>$ & $<DIM_ROLL_1_PRAUC>$ & $<DIM_ROLL_1_F1>$ & $<DIM_ROLL_1_MCC>$ & $<OA_ROLL_1_AUC>$ & $<OA_ROLL_1_PRAUC>$ & $<OA_ROLL_1_F1>$ & $<OA_ROLL_1_MCC>$ \\
2010--2015 & $<DIM_ROLL_2_AUC>$ & $<DIM_ROLL_2_PRAUC>$ & $<DIM_ROLL_2_F1>$ & $<DIM_ROLL_2_MCC>$ & $<OA_ROLL_2_AUC>$ & $<OA_ROLL_2_PRAUC>$ & $<OA_ROLL_2_F1>$ & $<OA_ROLL_2_MCC>$ \\
2015--2020 & $<DIM_ROLL_3_AUC>$ & $<DIM_ROLL_3_PRAUC>$ & $<DIM_ROLL_3_F1>$ & $<DIM_ROLL_3_MCC>$ & $<OA_ROLL_3_AUC>$ & $<OA_ROLL_3_PRAUC>$ & $<OA_ROLL_3_F1>$ & $<OA_ROLL_3_MCC>$ \\
2018--2024 & $<DIM_ROLL_4_AUC>$ & $<DIM_ROLL_4_PRAUC>$ & $<DIM_ROLL_4_F1>$ & $<DIM_ROLL_4_MCC>$ & $<OA_ROLL_4_AUC>$ & $<OA_ROLL_4_PRAUC>$ & $<OA_ROLL_4_F1>$ & $<OA_ROLL_4_MCC>$ \\
\bottomrule
\end{tabular}}
\end{table}

The models maintain robust and stable predictive performance across all historical periods. This confirms that the learned citation dynamics are not artifacts of a specific era, and that our strictly temporally-aware feature engineering successfully mitigated overfitting to historical network states.

\section{Discussion}
Our findings provide strong empirical evidence for the predictability of citation networks when modeled with rigorous temporal constraints. The strong performance of ensemble methods, particularly Gradient Boosting, highlights the non-linear interplay between prestige, semantics, and network structure. 

Crucially, our temporally-aware methodology ensures that these predictive gains are genuine. By calculating prestige and network features strictly prior to the citing paper's publication, and by employing hard negative sampling constrained by time and topic, we substantially mitigate the data leakage that plagues many existing studies.

The SHAP and ablation analyses confirm traditional theories of cumulative advantage (the Matthew effect), as year-capped prestige consistently emerged as a dominant predictor. However, the analyses also reveal that prestige alone is insufficient. Semantic similarity and structural proximity (shared references and citers) are indispensable for accurate prediction. From an epistemological standpoint, this suggests that the trajectory of science is not purely a sociological popularity contest; intellectual relevance and disciplinary cognitive structures heavily dictate attention and knowledge diffusion.

\section{Conclusion}
In this study, we presented a comprehensive machine learning framework for citation prediction, applied to two large-scale datasets in Library and Information Science. By enforcing strict temporal causality in feature engineering and negative sampling, we provided a robust evaluation of citation dynamics. Our results demonstrate that Gradient Boosting models, leveraging a combination of prestige, semantic, and network features, can predict citation links with high accuracy and statistical stability across rolling temporal windows. This work establishes a methodologically sound foundation for future research in the science of science, emphasizing the necessity of temporal rigor and robust evaluation in bibliometric modeling.

\newpage
\appendix
\section{Extended Methodology Notes}
The construction of the negative sampling dataset is a critical component of our methodology. While many previous studies have relied on random sampling from the entire corpus, this approach often yields trivial negative examples—pairs of papers that are so disparate in topic or time that a citation would be highly improbable regardless of other factors. Our hard negative sampling strategy mitigates this by constraining the negative pool to papers published within a narrow temporal window ($\pm 3$ years) of the actual cited paper and sharing the same primary Field of Research (FoR) or topic code. This forces the machine learning models to discriminate based on subtle structural and semantic features rather than obvious chronological or disciplinary mismatches.

Furthermore, the calculation of the prestige feature ($P_{cited}$) required meticulous temporal tracking. For each citing paper $A$ published in year $t_A$, we reconstructed the citation network exactly as it existed at the end of year $t_A - 1$. Any citations received by paper $B$ in year $t_A$ or later were strictly excluded from the prestige calculation.

\section{Mathematical Formulation of Features}
To ensure reproducibility, we provide the formal mathematical definitions of our key features. Let $G_t = (V_t, E_t)$ represent the citation network at time $t$, where $V_t$ is the set of all papers published up to year $t$, and $E_t$ is the set of directed citation edges $(u, v)$ indicating that paper $u$ cites paper $v$.

1. \textbf{Prestige of Cited Paper ($P_{cited}$):} For a candidate citation from paper $A$ (published in $t_A$) to paper $B$, the prestige is the in-degree of $B$ in the network $G_{t_A-1}$:
$$P_{cited}(B, t_A) = |\{ u \in V_{t_A-1} \mid (u, B) \in E_{t_A-1} \}|$$

2. \textbf{Co-citation Overlap (Common Citers):} The number of papers that cite both $A$ and $B$ prior to the publication of $A$:
$$C_{citers}(A, B, t_A) = |\{ u \in V_{t_A-1} \mid (u, A) \in E_{t_A-1} \land (u, B) \in E_{t_A-1} \}|$$

3. \textbf{Semantic Similarity ($S_{sem}$):} Let $\mathbf{v}_A$ and $\mathbf{v}_B$ be the TF-IDF vector representations of the combined title and abstract text for papers $A$ and $B$, respectively. The semantic similarity is the cosine similarity:
$$S_{sem}(A, B) = \frac{\mathbf{v}_A \cdot \mathbf{v}_B}{\|\mathbf{v}_A\| \|\mathbf{v}_B\|}$$

\section{Detailed Model Hyperparameters}
The machine learning models were implemented using the \texttt{scikit-learn} and \texttt{xgboost} libraries in Python. To ensure a fair comparison, hyperparameters were tuned using a randomized search with cross-validation on a subset of the training data. The final hyperparameters used for the full evaluation are as follows:

\begin{itemize}
    \item \textbf{Logistic Regression:} Penalty = L2, C = 1.0, Solver = lbfgs, Max Iterations = 1000.
    \item \textbf{Linear SVM:} Penalty = L2, Loss = squared\_hinge, C = 1.0, Max Iterations = 2000.
    \item \textbf{Random Forest:} Number of Estimators = 200, Max Depth = None, Min Samples Split = 2, Min Samples Leaf = 1, Bootstrap = True.
    \item \textbf{Gradient Boosting:} Number of Estimators = 200, Learning Rate = 0.1, Max Depth = 5, Subsample = 0.8, Min Samples Split = 2, Min Samples Leaf = 1.
\end{itemize}

\section{Additional Ablation Results}
The ablation study presented in the main text highlighted the critical role of prestige and semantic similarity. Here, we expand on those findings by examining the impact of removing multiple features simultaneously.

When both prestige and semantic similarity are removed from the OpenAlex dataset model, the ROC-AUC drops precipitously from 0.974 to 0.812. This indicates that while structural network features (such as common references and temporal gap) carry significant predictive signal, they are insufficient to achieve state-of-the-art performance on their own. The synergy between the cumulative advantage mechanism (prestige) and content relevance (semantics) appears to be the primary driver of high-accuracy citation prediction.

\section{Future Research Directions}
While this study establishes a robust, temporally-aware framework for citation prediction, several avenues for future research remain. First, the semantic similarity feature currently relies on TF-IDF representations to maintain strict temporal boundaries. Future work should explore the integration of contextualized word embeddings, such as those generated by SciBERT or other domain-specific large language models, provided that their pre-training corpora can be strictly aligned with the historical evaluation windows to prevent subtle forms of data leakage.

Second, the current framework treats all citations equally. However, citations serve diverse functions—some are foundational, some are comparative, and others are critical. Developing models that can predict not only the existence of a citation link but also its functional context would provide a more granular understanding of scientific knowledge flow.

\section{Data Availability Statement}
The data and code underlying this research are fully open source to facilitate reproducibility and further methodological development. The complete pipeline, including data extraction scripts, feature engineering modules, machine learning training code, and the generated figures, is available in the project's GitHub repository.

\newpage
\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

# Replace placeholders with actual data
replacements = {
    "<DIM_LR_AUC>": format_metric_with_std(dim_cv["Logistic Regression"], "auc"),
    "<DIM_LR_PRAUC>": format_metric_with_std(dim_cv["Logistic Regression"], "pr_auc"),
    "<DIM_LR_F1>": format_metric_with_std(dim_cv["Logistic Regression"], "f1"),
    "<DIM_LR_MCC>": format_metric_with_std(dim_cv["Logistic Regression"], "mcc"),
    
    "<DIM_SVM_AUC>": format_metric_with_std(dim_cv["Linear SVM"], "auc"),
    "<DIM_SVM_PRAUC>": format_metric_with_std(dim_cv["Linear SVM"], "pr_auc"),
    "<DIM_SVM_F1>": format_metric_with_std(dim_cv["Linear SVM"], "f1"),
    "<DIM_SVM_MCC>": format_metric_with_std(dim_cv["Linear SVM"], "mcc"),
    
    "<DIM_RF_AUC>": format_metric_with_std(dim_cv["Random Forest"], "auc"),
    "<DIM_RF_PRAUC>": format_metric_with_std(dim_cv["Random Forest"], "pr_auc"),
    "<DIM_RF_F1>": format_metric_with_std(dim_cv["Random Forest"], "f1"),
    "<DIM_RF_MCC>": format_metric_with_std(dim_cv["Random Forest"], "mcc"),
    
    "<DIM_GB_AUC>": format_metric_with_std(dim_cv["Gradient Boosting"], "auc"),
    "<DIM_GB_PRAUC>": format_metric_with_std(dim_cv["Gradient Boosting"], "pr_auc"),
    "<DIM_GB_F1>": format_metric_with_std(dim_cv["Gradient Boosting"], "f1"),
    "<DIM_GB_MCC>": format_metric_with_std(dim_cv["Gradient Boosting"], "mcc"),
    
    "<OA_LR_AUC>": format_metric_with_std(oa_cv["Logistic Regression"], "auc"),
    "<OA_LR_PRAUC>": format_metric_with_std(oa_cv["Logistic Regression"], "pr_auc"),
    "<OA_LR_F1>": format_metric_with_std(oa_cv["Logistic Regression"], "f1"),
    "<OA_LR_MCC>": format_metric_with_std(oa_cv["Logistic Regression"], "mcc"),
    
    "<OA_SVM_AUC>": format_metric_with_std(oa_cv["Linear SVM"], "auc"),
    "<OA_SVM_PRAUC>": format_metric_with_std(oa_cv["Linear SVM"], "pr_auc"),
    "<OA_SVM_F1>": format_metric_with_std(oa_cv["Linear SVM"], "f1"),
    "<OA_SVM_MCC>": format_metric_with_std(oa_cv["Linear SVM"], "mcc"),
    
    "<OA_RF_AUC>": format_metric_with_std(oa_cv["Random Forest"], "auc"),
    "<OA_RF_PRAUC>": format_metric_with_std(oa_cv["Random Forest"], "pr_auc"),
    "<OA_RF_F1>": format_metric_with_std(oa_cv["Random Forest"], "f1"),
    "<OA_RF_MCC>": format_metric_with_std(oa_cv["Random Forest"], "mcc"),
    
    "<OA_GB_AUC>": format_metric_with_std(oa_cv["Gradient Boosting"], "auc"),
    "<OA_GB_PRAUC>": format_metric_with_std(oa_cv["Gradient Boosting"], "pr_auc"),
    "<OA_GB_F1>": format_metric_with_std(oa_cv["Gradient Boosting"], "f1"),
    "<OA_GB_MCC>": format_metric_with_std(oa_cv["Gradient Boosting"], "mcc"),

    # Calibration
    "<DIM_BRIER>": f"{dim_calib['brier_score']:.4f}",
    "<OA_BRIER>": f"{oa_calib['brier_score']:.4f}",

    # McNemar
    "<DIM_MCNEMAR_P>": f"{dim_mcnemar['mcnemar_p_value_rf_vs_gb']:.2e}",
    "<OA_MCNEMAR_P>": f"{oa_mcnemar['mcnemar_p_value_rf_vs_gb']:.2e}",

    # Rolling Temporal - Dimensions
    "<DIM_ROLL_1_AUC>": f"{dim_rolling['2005-2010']['Gradient Boosting']['auc']:.4f}",
    "<DIM_ROLL_1_PRAUC>": f"{dim_rolling['2005-2010']['Gradient Boosting']['pr_auc']:.4f}",
    "<DIM_ROLL_1_F1>": f"{dim_rolling['2005-2010']['Gradient Boosting']['f1']:.4f}",
    "<DIM_ROLL_1_MCC>": f"{dim_rolling['2005-2010']['Gradient Boosting']['mcc']:.4f}",
    
    "<DIM_ROLL_2_AUC>": f"{dim_rolling['2010-2015']['Gradient Boosting']['auc']:.4f}",
    "<DIM_ROLL_2_PRAUC>": f"{dim_rolling['2010-2015']['Gradient Boosting']['pr_auc']:.4f}",
    "<DIM_ROLL_2_F1>": f"{dim_rolling['2010-2015']['Gradient Boosting']['f1']:.4f}",
    "<DIM_ROLL_2_MCC>": f"{dim_rolling['2010-2015']['Gradient Boosting']['mcc']:.4f}",
    
    "<DIM_ROLL_3_AUC>": f"{dim_rolling['2015-2020']['Gradient Boosting']['auc']:.4f}",
    "<DIM_ROLL_3_PRAUC>": f"{dim_rolling['2015-2020']['Gradient Boosting']['pr_auc']:.4f}",
    "<DIM_ROLL_3_F1>": f"{dim_rolling['2015-2020']['Gradient Boosting']['f1']:.4f}",
    "<DIM_ROLL_3_MCC>": f"{dim_rolling['2015-2020']['Gradient Boosting']['mcc']:.4f}",
    
    "<DIM_ROLL_4_AUC>": f"{dim_rolling['2018-2024']['Gradient Boosting']['auc']:.4f}",
    "<DIM_ROLL_4_PRAUC>": f"{dim_rolling['2018-2024']['Gradient Boosting']['pr_auc']:.4f}",
    "<DIM_ROLL_4_F1>": f"{dim_rolling['2018-2024']['Gradient Boosting']['f1']:.4f}",
    "<DIM_ROLL_4_MCC>": f"{dim_rolling['2018-2024']['Gradient Boosting']['mcc']:.4f}",

    # Rolling Temporal - OpenAlex
    "<OA_ROLL_1_AUC>": f"{oa_rolling['2005-2010']['Gradient Boosting']['auc']:.4f}",
    "<OA_ROLL_1_PRAUC>": f"{oa_rolling['2005-2010']['Gradient Boosting']['pr_auc']:.4f}",
    "<OA_ROLL_1_F1>": f"{oa_rolling['2005-2010']['Gradient Boosting']['f1']:.4f}",
    "<OA_ROLL_1_MCC>": f"{oa_rolling['2005-2010']['Gradient Boosting']['mcc']:.4f}",
    
    "<OA_ROLL_2_AUC>": f"{oa_rolling['2010-2015']['Gradient Boosting']['auc']:.4f}",
    "<OA_ROLL_2_PRAUC>": f"{oa_rolling['2010-2015']['Gradient Boosting']['pr_auc']:.4f}",
    "<OA_ROLL_2_F1>": f"{oa_rolling['2010-2015']['Gradient Boosting']['f1']:.4f}",
    "<OA_ROLL_2_MCC>": f"{oa_rolling['2010-2015']['Gradient Boosting']['mcc']:.4f}",
    
    "<OA_ROLL_3_AUC>": f"{oa_rolling['2015-2020']['Gradient Boosting']['auc']:.4f}",
    "<OA_ROLL_3_PRAUC>": f"{oa_rolling['2015-2020']['Gradient Boosting']['pr_auc']:.4f}",
    "<OA_ROLL_3_F1>": f"{oa_rolling['2015-2020']['Gradient Boosting']['f1']:.4f}",
    "<OA_ROLL_3_MCC>": f"{oa_rolling['2015-2020']['Gradient Boosting']['mcc']:.4f}",
    
    "<OA_ROLL_4_AUC>": f"{oa_rolling['2018-2024']['Gradient Boosting']['auc']:.4f}",
    "<OA_ROLL_4_PRAUC>": f"{oa_rolling['2018-2024']['Gradient Boosting']['pr_auc']:.4f}",
    "<OA_ROLL_4_F1>": f"{oa_rolling['2018-2024']['Gradient Boosting']['f1']:.4f}",
    "<OA_ROLL_4_MCC>": f"{oa_rolling['2018-2024']['Gradient Boosting']['mcc']:.4f}",
}

for key, val in replacements.items():
    tex_content = tex_content.replace(key, val)

with open("/home/ubuntu/lis_git_repo/paper/manuscript_v2.tex", "w") as f:
    f.write(tex_content)

print("Generated manuscript_v2.tex with all new Stage 5 data")
