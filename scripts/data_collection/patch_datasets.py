"""
patch_datasets.py
-----------------
Run this ONCE from your citation_analysis_scripts/ directory.
It updates the DATASETS list (and related dicts) in all 6 stage scripts
from the old 10 datasets to the new 10 datasets.

Usage:
    cd /path/to/citation_analysis_scripts
    python patch_datasets.py
"""

from pathlib import Path

HERE = Path(__file__).parent

# ── Replacement strings ───────────────────────────────────────────────────────

OLD_DATASETS = '''DATASETS = [
    "darkmatter", "LIS", "fatigue_crack", "environmental_engineering",
    "neuroblastoma", "osteosarcoma", "political_participation", 
    "welfare_state", "archaeology", "art_history"
]'''

NEW_DATASETS = '''DATASETS = [
    "protein_folding", "CRISPR",
    "neuroblastoma", "osteosarcoma",
    "additive_manufacturing", "corrosion_protection",
    "income_inequality", "organizational_behavior",
    "film_studies", "memory_studies",
]'''

# stage5 has a trailing comma variant
OLD_DATASETS_STAGE5 = '''DATASETS = [
    "darkmatter", "LIS", "fatigue_crack", "environmental_engineering",
    "neuroblastoma", "osteosarcoma", "political_participation",
    "welfare_state", "archaeology", "art_history",
]'''

OLD_DISCIPLINES = '''DISCIPLINES = {
    "darkmatter": "Science",
    "LIS": "Science",
    "fatigue_crack": "Engineering",
    "environmental_engineering": "Engineering",
    "neuroblastoma": "BioMed",
    "osteosarcoma": "BioMed",
    "political_participation": "Social Science",
    "welfare_state": "Social Science",
    "archaeology": "Humanities",
    "art_history": "Humanities"
}'''

NEW_DISCIPLINES = '''DISCIPLINES = {
    "protein_folding":         "Science",
    "CRISPR":                  "Science",
    "neuroblastoma":           "BioMed",
    "osteosarcoma":            "BioMed",
    "additive_manufacturing":  "Engineering",
    "corrosion_protection":    "Engineering",
    "income_inequality":       "Social Science",
    "organizational_behavior": "Social Science",
    "film_studies":            "Humanities",
    "memory_studies":          "Humanities",
}'''

OLD_LABELS = '''LABELS = {
    "darkmatter":               "Dark Matter",
    "LIS":                      "Info Literacy (LIS)",
    "fatigue_crack":            "Fatigue Crack",
    "environmental_engineering":"Env. Engineering",
    "neuroblastoma":            "Neuroblastoma",
    "osteosarcoma":             "Osteosarcoma",
    "political_participation":  "Political Partic.",
    "welfare_state":            "Welfare State",
    "archaeology":              "Archaeology",
    "art_history":              "Art History",
}'''

NEW_LABELS = '''LABELS = {
    "protein_folding":          "Protein Folding",
    "CRISPR":                   "CRISPR",
    "neuroblastoma":            "Neuroblastoma",
    "osteosarcoma":             "Osteosarcoma",
    "additive_manufacturing":   "Additive Manuf.",
    "corrosion_protection":     "Corrosion Prot.",
    "income_inequality":        "Income Inequality",
    "organizational_behavior":  "Org. Behavior",
    "film_studies":             "Film Studies",
    "memory_studies":           "Memory Studies",
}'''

OLD_DISCIPLINE_COLORS = '''DISCIPLINE_COLORS = {
    "darkmatter":               "#1f77b4",   # Science \u2013 blue
    "LIS":                      "#1f77b4",
    "fatigue_crack":            "#ff7f0e",   # Engineering \u2013 orange
    "environmental_engineering":"#ff7f0e",
    "neuroblastoma":            "#2ca02c",   # BioMed \u2013 green
    "osteosarcoma":             "#2ca02c",
    "political_participation":  "#d62728",   # Social Science \u2013 red
    "welfare_state":            "#d62728",
    "archaeology":              "#9467bd",   # Humanities \u2013 purple
    "art_history":              "#9467bd",
}'''

NEW_DISCIPLINE_COLORS = '''DISCIPLINE_COLORS = {
    "protein_folding":          "#1f77b4",   # Science \u2013 blue
    "CRISPR":                   "#1f77b4",
    "neuroblastoma":            "#2ca02c",   # BioMed \u2013 green
    "osteosarcoma":             "#2ca02c",
    "additive_manufacturing":   "#ff7f0e",   # Engineering \u2013 orange
    "corrosion_protection":     "#ff7f0e",
    "income_inequality":        "#d62728",   # Social Science \u2013 red
    "organizational_behavior":  "#d62728",
    "film_studies":             "#9467bd",   # Humanities \u2013 purple
    "memory_studies":           "#9467bd",
}'''

OLD_FEATURES_STAGE5 = '''FEATURES = ["prestige_cited", "temporal_gap", "common_refs",
            "jaccard_refs", "common_citers", "semantic_similarity"]

FEATURE_LABELS = {
    "prestige_cited":    "Prestige",
    "temporal_gap":      "Temp. Gap",
    "common_refs":       "Common Refs",
    "jaccard_refs":      "Jaccard",
    "common_citers":     "Co-citers",
    "semantic_similarity": "Semantic",
}'''

NEW_FEATURES_STAGE5 = '''FEATURES = ["temporal_indegree", "citation_time_gap", "temporal_pagerank", "directional_similarity"]

FEATURE_LABELS = {
    "temporal_indegree":       "Temporal Indegree",
    "citation_time_gap":       "Citation Time Gap",
    "temporal_pagerank":       "Temporal PageRank",
    "directional_similarity":  "Directional Sim.",
}'''

# ── Patch functions ───────────────────────────────────────────────────────────

def patch(filepath, replacements):
    path = HERE / filepath
    if not path.exists():
        print(f"  [MISSING] {filepath}")
        return
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            changed = True
        else:
            print(f"  [NOT FOUND] pattern in {filepath}:")
            print(f"    {repr(old[:60])}...")
    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"  [OK] {filepath}")
    else:
        print(f"  [UNCHANGED] {filepath} — all patterns already applied or not found")

# ── Apply patches ─────────────────────────────────────────────────────────────

print("Patching stage scripts...\n")

patch("stage1_feature_engineering.py", [
    (OLD_DATASETS, NEW_DATASETS),
])

patch("stage2_sbert_semantic.py", [
    (OLD_DATASETS, NEW_DATASETS),
])

patch("stage2b_directional_semantic.py", [
    (OLD_DATASETS, NEW_DATASETS),
])

patch("stage3_ml_training.py", [
    (OLD_DATASETS, NEW_DATASETS),
])

patch("stage4_comparative_figures.py", [
    (OLD_DATASETS, NEW_DATASETS),
    (OLD_DISCIPLINES, NEW_DISCIPLINES),
])

patch("stage5_multiplots.py", [
    (OLD_DATASETS_STAGE5, NEW_DATASETS),
    (OLD_LABELS, NEW_LABELS),
    (OLD_DISCIPLINE_COLORS, NEW_DISCIPLINE_COLORS),
    (OLD_FEATURES_STAGE5, NEW_FEATURES_STAGE5),
])

print("\nDone. Verify with:")
print('  grep -A 12 "^DATASETS" stage1_feature_engineering.py')
