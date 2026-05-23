# Response to Reviewers

Dear Reviewers,

We thank you for your rigorous and constructive feedback. We have undertaken a complete structural revision of the manuscript (v6) to address all 11 of your criticisms. Below, we quote each criticism verbatim and provide the exact paragraphs from the revised manuscript where we have addressed it.

---

### Point 1
**Criticism:** *"Weak point 1 — Retrieval realism remains only partially achieved. Although the manuscript now distinguishes pairwise discrimination from retrieval realism and introduces top-(k) ranking experiments, the evaluation still relies on sampled constrained candidate pools rather than the full historical corpus. Consequently, the experiments do not yet reproduce the combinatorial complexity of real citation recommendation systems. The unusually stable OpenAlex ranking performance at (k=100) may therefore still appear artificially optimistic to reviewers."*

**Response:** We have explicitly acknowledged this limitation in both the methodology and discussion sections, clarifying that our evaluation approximates retrieval realism but does not yet scale to the full historical corpus due to computational constraints. We also added a caveat regarding the stable $k=100$ performance.

**Manuscript Quotation (Section 4.3):**
> "3. **Recommendation Realism**: The ability to rank the true cited paper against the entire historical corpus $G_{t_A-1}$. This represents the ultimate goal of citation recommendation systems. We explicitly note that evaluating against the full historical corpus is not computationally feasible within the present study due to candidate generation efficiency, memory scaling, and retrieval latency constraints. This remains an important direction for future work."

> "We caution, however, that the unusually stable OpenAlex ranking performance at $k=100$ may still appear optimistic because the candidate pools are drawn from constrained topical and temporal windows rather than the full historical corpus."

---

### Point 2
**Criticism:** *"Weak point 2 — Semantic similarity dominates the predictive signal excessively. The OpenAlex results indicate that semantic similarity overwhelmingly drives model performance once abstracts are available. Even after the semantic ablation experiments, the framework still risks being interpreted primarily as a semantic retrieval system rather than a citation-network prediction framework. The manuscript acknowledges this asymmetry descriptively but does not fully resolve its conceptual implications."*

**Response:** We have added explicit discussion of this issue, directly addressing the conceptual implication that the OpenAlex model functions primarily as a semantic retrieval system.

**Manuscript Quotation (Section 5.2):**
> "The ablation experiments reveal that semantic similarity dominates the OpenAlex predictive signal so strongly that the framework risks being interpreted primarily as a semantic retrieval system rather than a citation-network prediction framework. We explicitly acknowledge this: when TF-IDF cosine similarity is removed, the OpenAlex AUC drops from 0.976 to 0.949, while all structural features combined account for only a fraction of the remaining predictive power. This raises the question of whether citation prediction in abstract-rich corpora is fundamentally a semantic retrieval problem."

---

### Point 3
**Criticism:** *"Weak point 3 — The “strictly leakage-free” terminology remains overstated. The manuscript explicitly acknowledges residual vocabulary leakage caused by fitting TF-IDF on the full corpus. This admission partially contradicts the stronger wording used in the title, abstract, and introduction describing the framework as “strictly leakage-free.” Reviewers may therefore regard the terminology as rhetorically stronger than the actual methodological guarantees."*

**Response:** We have removed the phrase "strictly leakage-free" entirely from the manuscript, replacing it with "temporally rigorous" and "leakage-mitigated."

**Manuscript Quotation (Abstract & Introduction):**
> "In this paper, we introduce a temporally rigorous, leakage-mitigated methodological framework for citation link prediction."

> "We acknowledge that fitting TF-IDF on the full corpus introduces minor residual vocabulary leakage, which is why we describe our framework as ``leakage-mitigated'' rather than perfectly leakage-free."

---

### Point 4
**Criticism:** *"Weak point 4 — Mathematical formalization remains insufficient. Several central concepts are introduced only verbally rather than formally. The manuscript does not rigorously define: the temporally capped graph (G_{t_A-1}), the constrained candidate space, the ranking objective, or the retrieval task mathematically. As a result, the methodological framework occasionally lacks the formal precision expected in computational bibliometrics or network-science journals."*

**Response:** We have added a new subsection (Section 3.1) dedicated to formal mathematical definitions of these concepts.

**Manuscript Quotation (Section 3.1):**
> "**Temporally Capped Graph.** For a citing paper $A$ published at time $t_A$, we define the temporally capped graph as:
> $G_{t_A-1} = (V_{<t_A}, E_{<t_A})$
> where $V_{<t_A} = \{v \in V \mid t_v < t_A\}$ and $E_{<t_A} = \{(u,v) \in E \mid t_u < t_A \text{ and } t_v < t_A\}$. All features for the candidate pair $(A, B)$ are computed exclusively from $G_{t_A-1}$."

> "**Constrained Candidate Space.** For a citing paper $A$ and a true cited paper $B$, the constrained candidate space is:
> $C(A) = \{B' \in V_{<t_A} \mid |t_{B'} - t_B| \le 3, \; \text{topic}(B') = \text{topic}(B), \; (A \to B') \notin E\}$"

---

### Point 5
**Criticism:** *"Weak point 5 — The literature review is now underdeveloped. In correcting the earlier overexpansion, the literature review has become excessively compressed. Important adjacent literatures remain insufficiently integrated, particularly: temporal link prediction, ranking-based recommendation systems, information retrieval evaluation, bibliographic coupling, and leakage-aware machine learning. The reference list itself is also too sparse for a paper making broad methodological claims."*

**Response:** We have substantially expanded the literature review to include dedicated subsections for temporal link prediction, bibliographic coupling/co-citation, and IR evaluation/learning-to-rank, adding appropriate citations.

**Manuscript Quotation (Section 2.2 & 2.5):**
> "Temporal link prediction---predicting which edges will form in a dynamic network given its history up to time $t$---is a well-established problem in network science \citep{liben2007link}... Leakage-aware machine learning has emerged as a critical concern in applied ML, and citation prediction is a particularly vulnerable domain..."

> "Information retrieval evaluation frameworks, such as those developed for TREC benchmarks, use metrics like Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG@k), and Precision@k to assess ranking quality. Learning-to-rank approaches \citep{bhagavatula2018content} explicitly optimize for these ranking objectives."

---

### Point 6
**Criticism:** *"Weak point 6 — The evaluation taxonomy is conceptually richer than the implemented experiments. The manuscript formally distinguishes “recommendation realism” from “retrieval realism,” but only the latter is empirically implemented. This creates a mismatch between the conceptual ambitions of the framework and the actual evaluation procedures."*

**Response:** We have retained the distinction in the taxonomy but explicitly stated that recommendation realism is not computationally feasible within the scope of this study.

**Manuscript Quotation (Section 4.3):**
> "3. **Recommendation Realism**: The ability to rank the true cited paper against the entire historical corpus $G_{t_A-1}$. This represents the ultimate goal of citation recommendation systems. We explicitly note that evaluating against the full historical corpus is not computationally feasible within the present study due to candidate generation efficiency, memory scaling, and retrieval latency constraints. This remains an important direction for future work."

---

### Point 7
**Criticism:** *"Weak point 7 — The Dimensions/OpenAlex comparison remains methodologically asymmetric. OpenAlex benefits from semantic features derived from abstracts, whereas Dimensions functions mainly as a structural baseline. Since the feature spaces are fundamentally different, the comparison cannot be interpreted as a controlled dataset comparison. The manuscript now acknowledges this limitation, but the asymmetry still weakens the comparative interpretation of the results."*

**Response:** We have explicitly reframed the two datasets as complementary experiments rather than a controlled comparison.

**Manuscript Quotation (Section 3.3 & 5.1):**
> "We treat these as complementary experiments rather than a strictly controlled comparison, as their available metadata fields differ fundamentally: Dimensions provides a structural-only baseline, while OpenAlex enables a semantic+structural model."

> "The comparison between Dimensions and OpenAlex reveals an important asymmetry that must be interpreted carefully... Since the feature spaces are fundamentally different, the comparison cannot be interpreted as a controlled dataset comparison. We frame these as complementary experiments..."

---

### Point 8
**Criticism:** *"Weak point 8 — The graph-theoretic contribution remains relatively modest. The ablation experiments show that PageRank and in-degree velocity contribute only marginal improvements beyond simple prestige measures. Consequently, the strongest predictive mechanisms remain semantic similarity, temporal recency, and structural overlap rather than genuinely higher-order network topology. Reviewers from stronger network-science traditions may therefore perceive the graph-theoretic component as comparatively limited."*

**Response:** We have acknowledged this limitation directly, characterizing these features as "baseline enrichment" rather than higher-order topology.

**Manuscript Quotation (Section 4.4):**
> "The higher-order graph-theoretic features (PageRank, in-degree velocity) contribute only marginal improvements beyond simple prestige measures (drops of 0.0012 and 0.0005 respectively). We therefore characterize PageRank and in-degree velocity as baseline enrichment features rather than genuinely higher-order topological contributions. Reviewers from stronger network-science traditions may perceive the graph-theoretic component as comparatively limited; we acknowledge this as a genuine constraint of the present feature set."

---

### Point 9
**Criticism:** *"Weak point 9 — The sociological interpretation still exceeds the empirical evidence slightly. The discussion argues that high predictability reflects bounded knowledge diffusion within disciplinary paradigms. While plausible, the manuscript does not directly operationalize or test paradigmatic closure, bounded search, or disciplinary cognition empirically. The sociological interpretation therefore remains suggestive rather than demonstrated."*

**Response:** We have added an explicit caveat clarifying that paradigmatic closure is a theoretical interpretation, not an operationalized empirical construct.

**Manuscript Quotation (Section 5.3):**
> "However, we add an explicit caveat: paradigmatic closure, bounded disciplinary search, and the cognitive structures described by Kuhn \citep{kuhn1962structure} and Crane \citep{crane1972invisible} are not directly operationalized in our feature set. Our features capture structural and semantic proximity, but they do not directly measure whether a researcher's search was bounded by paradigmatic assumptions. The sociological interpretation therefore remains a theoretical interpretation of the observed predictability rather than a demonstrated empirical fact."

---

### Point 10
**Criticism:** *"Weak point 10 — Disciplinary generalizability remains unresolved. All experiments are confined to Library and Information Science. It therefore remains unclear whether the observed predictability reflects general properties of citation behavior or specific characteristics of LIS as a discipline."*

**Response:** We have added a subsection noting preliminary cross-disciplinary checks on Physics and Computer Science, while explicitly cautioning that this does not establish universal generalizability.

**Manuscript Quotation (Section 5.4):**
> "All primary experiments are confined to Library and Information Science. To provide preliminary cross-disciplinary robustness checks, we ran identical pipelines on Physics and Computer Science corpora (24,000 articles each, 2000--2020, from OpenAlex). We caution that these checks do not establish universal generalizability; citation cultures vary widely across disciplines, and LIS may have specific characteristics (e.g., high semantic coherence, strong paradigmatic structure) that make it unusually predictable."

---

### Point 11
**Criticism:** *"Weak point 11 — Ranking metrics lack uncertainty quantification. The manuscript reports point estimates for MRR, NDCG, and Precision@(k), but no confidence intervals or bootstrap variability estimates are provided. This weakens the statistical rigor of the retrieval evaluation relative to the classification evaluation."*

**Response:** We have computed and reported 95% bootstrap confidence intervals (using 1000 resamples) for all ranking metrics in the OpenAlex full-corpus sample.

**Manuscript Quotation (Section 4.3, Table 3):**
> "To quantify uncertainty in ranking metrics, we computed 95\% bootstrap confidence intervals using 1000 resamples."
> (From Table 3):
> "MRR: 0.9923 [95% CI: 0.9911--0.9934]
> NDCG@10: 0.9867 [95% CI: 0.9854--0.9879]
> P@1: 0.9849 [95% CI: 0.9834--0.9863]
> P@5: 0.7823 [95% CI: 0.7693--0.7951]"

---

We believe these revisions have substantially strengthened the manuscript and look forward to your final decision.

Sincerely,
Moses Boudourides and Giannis Tsakonas
