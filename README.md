# Citation Dynamics

**Temporally-Aware Citation Prediction Across Disciplines: A Machine Learning Framework**

By Moses Boudourides

---

## Overview

This repository contains all scripts and tools for a comparative study of citation dynamics across 10 thematic bibliographic datasets spanning 5 broad disciplines. We apply a temporally-aware machine learning framework to predict whether one scientific paper will cite another, using keyword-defined corpora of approximately 50,000–70,000 peer-reviewed articles each, retrieved from [Dimensions.ai](https://www.dimensions.ai) (1975–2024).

A central design principle is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the data leakage that affects many prior citation prediction studies.

The key research question is: **do citation prediction patterns differ systematically across disciplines, and if so, which features drive those differences?**

---

## Datasets

Ten thematic bibliographic datasets are retrieved from Dimensions.ai using keyword searches in title and abstract, restricted to journal articles published 1975–2024. Datasets are grouped into 5 broad disciplines:

| Discipline | Keyword(s) | Records | With abstracts | With references |
|---|---|---|---|---|
| Science | dark matter | 55,338 | 53,358 (96.4%) | 50,508 (91.3%) |
| Science | information literacy + library information science | 66,679 | 63,254 (94.9%) | 41,944 (62.9%) |
| Engineering | fatigue crack | 57,166 | 53,787 (94.1%) | 45,730 (80.0%) |
| Engineering | environmental engineering | 53,394 | 51,471 (96.4%) | 40,595 (76.0%) |
| BioMed | neuroblastoma | 57,319 | 51,616 (90.1%) | 43,148 (75.3%) |
| BioMed | osteosarcoma + bone sarcoma | 61,187 | 54,528 (89.1%) | 44,170 (72.2%) |
| Social Science | political participation | 50,560 | 48,819 (96.6%) | 31,786 (62.9%) |
| Social Science | welfare state | 62,625 | 54,011 (86.3%) | 38,738 (61.9%) |
| Humanities | archaeology | 60,498 | 41,941 (69.3%) | 28,870 (47.7%) |
| Humanities | art history | 68,288 | 59,828 (87.6%) | 29,644 (43.4%) |

---

## Data Collection

All datasets are fetched using `scripts/data_collection/fetch_dimensions_keywords.py`, which iterates year by year (1975–2024) for each keyword and deduplicates across keywords within the same corpus.

**Requirements:** A valid [Dimensions.ai](https://www.dimensions.ai) API key stored in `data_collection_scripts/Dim_key.txt`.

```bash
cd computations/data_collection/data_collection_scripts
python fetch_dimensions_keywords.py
```

Individual per-dataset scripts are also provided for selective re-fetching.

---

## Citation Graph Construction

Citation graphs are built as directed acyclic graphs (DAGs) using NetworkX and pickled to `citation_analysis_scripts/nx_citation_graphs/`:

```bash
cd computations/citation_analysis_scripts
python build_citation_graphs.py
```

Each graph has:
- **Nodes**: paper IDs present in the corpus, with `year` attribute
- **Edges**: directed edge (citing → cited) for every internal reference, with the DAG constraint that `year(cited) < year(citing)`

Citation graph statistics (nodes, edges, density, weak components, citing/cited nodes, degree distributions, clustering coefficient) are computed by:

```bash
python citation_graph_stats.py
```

---

## Folder Structure

```
Citation Dynamics/
├── computations/
│   ├── data_collection/
│   │   ├── data/                              # raw json.gz + processed parquet files
│   │   └── data_collection_scripts/
│   │       ├── fetch_dimensions_keywords.py   # master fetch script (all 10 datasets)
│   │       ├── fetch_dimensions_*.py          # individual dataset scripts
│   │       ├── Dim_key.txt                    # Dimensions API key (not committed)
│   │       └── OA_key.txt                     # OpenAlex API key (not committed)
│   └── citation_analysis_scripts/
│       ├── build_citation_graphs.py           # builds NetworkX DAGs, pickles to nx_citation_graphs/
│       ├── citation_graph_stats.py            # computes and saves graph statistics table
│       └── nx_citation_graphs/                # pickled NetworkX graphs (not committed)
└── manuscript/                                # manuscript draft and figures
```

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
