"""
citation_graph_stats.py
------------------------
Loads each pickled NetworkX citation graph from nx_citation_graphs/
and computes the following statistics for the summary table:

  - Citation nodes       : total nodes in the graph
  - Citation edges       : total directed edges
  - Density              : edges / (nodes * (nodes-1))
  - Weak components      : number of weakly connected components
  - Citing nodes         : nodes with out-degree > 0
  - Average out-degree   : mean out-degree among citing nodes
  - Cited nodes          : nodes with in-degree > 0
  - Average in-degree    : mean in-degree among cited nodes
  - Avg clustering coeff : directed clustering coefficient (NetworkX default)

Prints a formatted table to stdout and saves it as:
    citation_graph_stats.csv
    citation_graph_stats.md

Run from citation_analysis_scripts/:
    python citation_graph_stats.py

Requirements:
    pip install networkx pandas tabulate
"""

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path

HERE      = Path(__file__).parent
GRAPH_DIR = HERE / "nx_citation_graphs"

DATASETS = [
    ("darkmatter",                 "Science",        "dark matter"),
    ("LIS",                        "Science",        "information literacy + LIS"),
    ("fatigue_crack",              "Engineering",    "fatigue crack"),
    ("environmental_engineering",  "Engineering",    "environmental engineering"),
    ("neuroblastoma",              "BioMed",         "neuroblastoma"),
    ("osteosarcoma",               "BioMed",         "osteosarcoma + bone sarcoma"),
    ("political_participation",    "Social Science", "political participation"),
    ("welfare_state",              "Social Science", "welfare state"),
    ("archaeology",                "Humanities",     "archaeology"),
    ("art_history",                "Humanities",     "art history"),
]

rows = []

for name, discipline, keyword in DATASETS:
    graph_path = GRAPH_DIR / f"{name}_citation_graph.pkl"
    if not graph_path.exists():
        print(f"[{name}] Graph not found — skipping.")
        continue

    print(f"[{name}] Computing stats ...", flush=True)
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    n = G.number_of_nodes()
    e = G.number_of_edges()

    density        = nx.density(G)
    weak_comp      = nx.number_weakly_connected_components(G)

    out_degrees    = dict(G.out_degree())
    in_degrees     = dict(G.in_degree())

    citing_nodes   = sum(1 for d in out_degrees.values() if d > 0)
    cited_nodes    = sum(1 for d in in_degrees.values() if d > 0)

    avg_out        = (sum(d for d in out_degrees.values() if d > 0) / citing_nodes
                      if citing_nodes > 0 else 0.0)
    avg_in         = (sum(d for d in in_degrees.values() if d > 0) / cited_nodes
                      if cited_nodes > 0 else 0.0)

    # Directed clustering coefficient
    # For a DAG this measures the fraction of pairs among a node's successors
    # that are also connected; averaged over all nodes.
    avg_clust      = nx.average_clustering(G)

    rows.append({
        "Discipline":          discipline,
        "Keyword":             keyword,
        "Citation nodes":      n,
        "Citation edges":      e,
        "Density":             round(density, 6),
        "Weak components":     weak_comp,
        "Citing nodes":        citing_nodes,
        "Avg out-degree":      round(avg_out, 2),
        "Cited nodes":         cited_nodes,
        "Avg in-degree":       round(avg_in, 2),
        "Avg clustering coeff": round(avg_clust, 4),
    })

df = pd.DataFrame(rows)

# Print table
try:
    from tabulate import tabulate
    print("\n" + tabulate(df, headers="keys", tablefmt="pipe", showindex=False))
except ImportError:
    print(df.to_string(index=False))

# Save CSV
csv_path = HERE / "citation_graph_stats.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV -> {csv_path}")

# Save Markdown
md_path = HERE / "citation_graph_stats.md"
try:
    from tabulate import tabulate
    md_table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    md_path.write_text(md_table + "\n")
    print(f"Saved Markdown -> {md_path}")
except ImportError:
    pass
