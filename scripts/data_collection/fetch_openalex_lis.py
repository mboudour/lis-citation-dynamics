"""
fetch_openalex_lis.py
---------------------
Fetches Library and Information Science (LIS) journal articles from
OpenAlex (1975–2024) and saves the result as a pandas DataFrame pickle,
in the same format as the Dimensions.ai dataset used in the companion analysis.

Usage:
    python fetch_openalex_lis.py [--email your@email.com] [--api-key KEY] [--output OpenAlex_LIS_1975_2024.pkl]

Filter strategy
---------------
We use OpenAlex's primary_topic filter rather than the legacy concepts filter.
The concept C136764020 ("Library and Information Science") is assigned by an
automated ML tagger and returns ~4.2 million records — far too broad.
Instead we filter by five specific topic IDs whose primary_topic corresponds
to core LIS and scientometrics:

    T10712  Library Science and Information Literacy       (114,018 works)
    T14330  Library Science and Information Systems         (98,052 works)
    T13166  Information Science and Libraries               (55,834 works)
    T13673  Library Science and Information                 (25,971 works)
    T10102  Scientometrics and Bibliometrics Research       (99,916 works)

Combined (OR logic, deduplicated): ~168,901 works — a corpus comparable in
size and scope to the Dimensions.ai LIS dataset (259,220 works).

Output columns (matching Dimensions format):
    id              : OpenAlex work ID (e.g. W2741809807)
    doi             : DOI string (without https://doi.org/ prefix)
    title           : Paper title
    abstract        : Reconstructed plain-text abstract
    year            : Publication year (int)
    journal         : Source journal name
    authors         : List of author display names
    author_ids      : List of OpenAlex author IDs
    times_cited     : Cited-by count
    reference_ids   : List of referenced OpenAlex work IDs
    is_oa           : Boolean — is the paper Open Access?
"""

import argparse
import time
import pickle
import requests
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Five primary_topic IDs covering core LIS and scientometrics
TOPIC_IDS = ["T10712", "T14330", "T13166", "T13673", "T10102"]

YEAR_MIN  = 1975
YEAR_MAX  = 2024
PER_PAGE  = 200          # Max allowed by OpenAlex
SLEEP_S   = 0.1          # Polite delay between requests (seconds)
MAX_RETRIES = 5          # Retries on transient HTTP errors


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Convert OpenAlex inverted-index abstract to plain text."""
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def fetch_page(url: str, params: dict, headers: dict | None = None,
               retries: int = MAX_RETRIES) -> dict:
    """GET a single page from the OpenAlex API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers or {},
                                timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code} — retrying "
                      f"({attempt+1}/{retries}) …")
                time.sleep(5 * (attempt + 1))
        except requests.RequestException as e:
            print(f"  Request error: {e} — retrying ({attempt+1}/{retries}) …")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries.")


def fetch_all_works(email: str | None = None,
                    api_key: str | None = None) -> list[dict]:
    """
    Retrieve all LIS journal articles from OpenAlex using cursor-based
    pagination. Returns a list of raw work dictionaries.
    """
    base_url = "https://api.openalex.org/works"

    # OR-join topic IDs with pipe character
    topic_filter = "|".join(TOPIC_IDS)

    params = {
        "filter": (
            f"primary_topic.id:{topic_filter},"
            f"type:article,"
            f"publication_year:{YEAR_MIN}-{YEAR_MAX}"
        ),
        "select": (
            "id,doi,title,abstract_inverted_index,authorships,"
            "publication_year,primary_location,open_access,"
            "referenced_works,cited_by_count"
        ),
        "per-page": PER_PAGE,
        "cursor": "*",
    }
    if email:
        params["mailto"] = email   # Polite pool — higher rate limit

    # API key passed as Authorization header (never in URL/params)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    works = []
    page_num = 0

    # First request to get total count
    data = fetch_page(base_url, params, headers=headers)
    total = data["meta"]["count"]
    print(f"Total works to fetch: {total:,}")

    while True:
        page_num += 1
        results = data.get("results", [])
        if not results:
            break
        works.extend(results)
        fetched = len(works)
        if page_num % 50 == 0 or fetched >= total:
            pct = 100 * fetched / total
            print(f"  Page {page_num:4d} — {fetched:,}/{total:,} ({pct:.1f}%)")

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        params["cursor"] = next_cursor
        time.sleep(SLEEP_S)
        data = fetch_page(base_url, params, headers=headers)

    print(f"Fetched {len(works):,} works in {page_num} pages.")
    return works


def parse_works(raw_works: list[dict]) -> pd.DataFrame:
    """Parse raw OpenAlex work dicts into a clean DataFrame."""
    records = []
    for w in raw_works:
        # Authors
        authorships = w.get("authorships") or []
        authors    = [a["author"]["display_name"] for a in authorships
                      if a.get("author") and a["author"].get("display_name")]
        author_ids = [a["author"]["id"] for a in authorships
                      if a.get("author") and a["author"].get("id")]

        # Journal
        primary = w.get("primary_location") or {}
        source  = primary.get("source") or {}
        journal = source.get("display_name", "")

        # DOI — strip prefix
        doi_raw = w.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "").strip()

        # Abstract
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))

        # References (list of OpenAlex IDs)
        refs = w.get("referenced_works") or []

        # Open Access
        oa_info = w.get("open_access") or {}
        is_oa   = bool(oa_info.get("is_oa", False))

        records.append({
            "id":            w.get("id", ""),
            "doi":           doi,
            "title":         w.get("title") or "",
            "abstract":      abstract,
            "year":          w.get("publication_year"),
            "journal":       journal,
            "authors":       authors,
            "author_ids":    author_ids,
            "times_cited":   w.get("cited_by_count", 0),
            "reference_ids": refs,
            "is_oa":         is_oa,
        })

    df = pd.DataFrame(records)
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch LIS data from OpenAlex using primary_topic filter")
    parser.add_argument(
        "--email", default=None,
        help="Your email for OpenAlex polite pool (recommended)"
    )
    parser.add_argument(
        "--output", default="OpenAlex_LIS_1975_2024.pkl",
        help="Output pickle filename"
    )
    parser.add_argument(
        "--api-key", default=None, dest="api_key",
        help="OpenAlex API key (passed as Authorization header; never stored)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OpenAlex LIS Fetcher (topic-based filter)")
    print(f"  Topics  : {', '.join(TOPIC_IDS)}")
    print(f"  Years   : {YEAR_MIN}–{YEAR_MAX}")
    print(f"  Email   : {args.email or '(none — anonymous pool)'}")
    print(f"  API key : {'provided (Authorization header)' if args.api_key else '(none)'}")
    print(f"  Output  : {args.output}")
    print("=" * 60)

    raw = fetch_all_works(email=args.email, api_key=args.api_key)
    print("Parsing records …")
    df = parse_works(raw)

    print(f"\nDataset summary:")
    print(f"  Articles          : {len(df):,}")
    print(f"  Year range        : {df['year'].min()}–{df['year'].max()}")
    print(f"  With abstract     : {(df['abstract'].str.len() > 10).sum():,} "
          f"({100*(df['abstract'].str.len() > 10).mean():.1f}%)")
    print(f"  With references   : {(df['reference_ids'].apply(len) > 0).sum():,} "
          f"({100*(df['reference_ids'].apply(len) > 0).mean():.1f}%)")
    print(f"  Mean times cited  : {df['times_cited'].mean():.2f}")

    out_path = Path(args.output)
    df.to_pickle(out_path)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()