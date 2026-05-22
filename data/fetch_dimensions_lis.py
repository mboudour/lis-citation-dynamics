"""
fetch_dimensions_lis.py
-----------------------
Fetches Library and Information Science (LIS) journal articles from
Dimensions.ai (1975–2024) and saves the result as a pandas DataFrame pickle.

Requirements:
    pip install dimcli pandas

Usage:
    Place your Dimensions API key in a file ../key.txt (one line, no spaces),
    then run this script in a Jupyter notebook (%%time cell magic) or as a
    plain Python script.
"""

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
