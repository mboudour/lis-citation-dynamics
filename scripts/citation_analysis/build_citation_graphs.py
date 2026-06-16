"""
build_citation_graphs.py
-------------------------
Builds a directed citation graph (NetworkX DiGraph) for each of the 10
thematic bibliographic datasets and pickles each graph to:

    nx_citation_graphs/
        dark_matter_citation_graph.pkl
        LIS_citation_graph.pkl
        fatigue_crack_citation_graph.pkl
        environmental_engineering_citation_graph.pkl
        neuroblastoma_citation_graph.pkl
        osteosarcoma_citation_graph.pkl
        political_participation_citation_graph.pkl
        welfare_state_citation_graph.pkl
        archaeology_citation_graph.pkl
        art_history_citation_graph.pkl

Each graph:
  - Nodes  : paper IDs (strings) present in the corpus
  - Edges  : directed edge (citing -> cited) for every reference_id
             that resolves to another paper in the same corpus
  - Node attribute "year": publication year (int)

Citation graphs are DAGs by construction (only edges where
cited paper's year < citing paper's year are retained).

Run from citation_analysis_scripts/:
    python build_citation_graphs.py

Requirements:
    pip install pandas pyarrow networkx
"""

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path

HERE      = Path(__file__).parent
DATA_DIR  = HERE.parent / "data_collection" / "data"
GRAPH_DIR = HERE / "nx_citation_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "darkmatter",
    "LIS",
    "fatigue_crack",
    "environmental_engineering",
    "neuroblastoma",
    "osteosarcoma",
    "political_participation",
    "welfare_state",
    "archaeology",
    "art_history",
]


def build_graph(name):
    parquet_path = DATA_DIR / f"Dimensions_{name}_1975_2024.parquet"
    out_path     = GRAPH_DIR / f"{name}_citation_graph.pkl"

    if out_path.exists():
        print(f"[{name}] Graph already exists — skipping.")
        return

    print(f"\n[{name}] Loading parquet ...", flush=True)
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Build lookup: id -> year
    id_to_year = dict(zip(df["id"], df["year"]))
    corpus_ids = set(id_to_year.keys())

    G = nx.DiGraph()

    # Add all corpus nodes with year attribute
    for pid, yr in id_to_year.items():
        G.add_node(pid, year=int(yr))

    # Add edges: citing -> cited (DAG constraint: cited year < citing year)
    edges_added  = 0
    edges_skipped = 0
    for _, row in df.iterrows():
        citing_id   = row["id"]
        citing_year = int(row["year"])
        for cited_id in row["reference_ids"]:
            if cited_id in corpus_ids:
                cited_year = int(id_to_year[cited_id])
                if cited_year < citing_year:
                    G.add_edge(citing_id, cited_id)
                    edges_added += 1
                else:
                    edges_skipped += 1

    print(f"  Nodes          : {G.number_of_nodes():,}")
    print(f"  Edges (DAG)    : {G.number_of_edges():,}")
    print(f"  Edges skipped  : {edges_skipped:,}  (same/future year)")

    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for name in DATASETS:
        build_graph(name)
    print("\nAll citation graphs built.")
