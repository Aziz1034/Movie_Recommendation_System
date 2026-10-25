"""Data loader and preprocessing pipeline for MovieLens dataset.

Downloads, extracts, cleans, and builds rich movie metadata combining titles,
genres, user-submitted tags, ratings, and IMDb identifiers.
"""

import os
import re
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import config


def download_dataset(
    url: str = config.MOVIELENS_URL,
    zip_path: Path = config.ZIP_PATH,
    extract_to: Path = config.DATA_DIR,
    force: bool = False
) -> Path:
    """Download and extract MovieLens Latest Small dataset if not already present.

    Args:
        url: Source download URL for the zip archive.
        zip_path: Destination path for the downloaded zip.
        extract_to: Directory to extract contents into.
        force: If True, redownload even if files exist.

    Returns:
        Path to the extracted dataset directory.
    """
    raw_dir = extract_to / "ml-latest-small"
    movies_file = raw_dir / "movies.csv"

    if movies_file.exists() and not force:
        print(f"[data_loader] Dataset already extracted at: {raw_dir}")
        return raw_dir

    extract_to.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists() or force:
        print(f"[data_loader] Downloading dataset from {url} ...")
        try:
            import requests
            try:
                import certifi
                verify_param = certifi.where()
            except ImportError:
                verify_param = True

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=60, verify=verify_param)
                resp.raise_for_status()
            except requests.exceptions.SSLError:
                print("[data_loader] SSL verification warning; retrying with unverified context...")
                resp = requests.get(url, headers=headers, stream=True, timeout=60, verify=False)
                resp.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            print(f"[data_loader] Saved archive to {zip_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MovieLens dataset from {url}: {e}\n"
                f"Please verify internet connection or manually place 'ml-latest-small.zip' into {extract_to}."
            ) from e

    print(f"[data_loader] Extracting {zip_path} into {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    if not movies_file.exists():
        raise FileNotFoundError(f"Expected file {movies_file} not found after extraction.")

    print(f"[data_loader] Extraction complete. Raw dataset in {raw_dir}")
    return raw_dir


def extract_year_and_clean_title(title: str) -> Tuple[str, Optional[int]]:
    """Extract release year from title string like 'Toy Story (1995)'.

    Returns:
        Tuple of (clean_title, year or None).
    """
    if not isinstance(title, str) or not title.strip():
        return "", None

    # Matches trailing four-digit year in parentheses: e.g., (1995) or (1995 )
    match = re.search(r"\((\d{4})\s*\)$", title.strip())
    if match:
        year = int(match.group(1))
        # Remove the year from the title for cleaner matching
        clean_title = re.sub(r"\s*\(\d{4}\s*\)$", "", title.strip()).strip()
        return clean_title, year

    return title.strip(), None


def compute_weighted_rating(
    df: pd.DataFrame,
    min_ratings: int = config.MIN_RATINGS_POPULAR
) -> pd.DataFrame:
    """Calculate IMDB/Bayesian weighted rating for ranking popular movies.

    Formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
      v = vote_count for the movie
      m = minimum votes required to be considered
      R = vote_average of the movie
      C = mean vote across the whole dataset (weighted by votes)
    """
    df = df.copy()
    total_votes = df["vote_count"].fillna(0).sum()
    if total_votes > 0:
        c_mean = float((df["vote_count"].fillna(0) * df["vote_average"].fillna(0)).sum() / total_votes)
    else:
        c_mean = 3.0

    m_threshold = min_ratings

    v = df["vote_count"].fillna(0)
    r = df["vote_average"].fillna(0)

    denominator = v + m_threshold
    # Avoid zero division
    denominator = np.where(denominator == 0, 1, denominator)

    df["weighted_rating"] = ((v / denominator) * r) + ((m_threshold / denominator) * c_mean)
    # If vote_count == 0, weighted_rating defaults to 0
    df.loc[df["vote_count"] == 0, "weighted_rating"] = 0.0
    return df


def prepare_movie_data(
    raw_dir: Path = config.RAW_DATA_DIR,
    output_path: Path = config.PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """Load raw MovieLens CSVs, clean, merge metadata, and construct feature soup.

    Returns:
        Processed pandas DataFrame containing all movie attributes and metadata soup.
    """
    movies_csv = raw_dir / "movies.csv"
    ratings_csv = raw_dir / "ratings.csv"
    tags_csv = raw_dir / "tags.csv"
    links_csv = raw_dir / "links.csv"

    if not movies_csv.exists():
        raise FileNotFoundError(f"movies.csv not found at {movies_csv}. Run download_dataset() first.")

    print("[data_loader] Loading raw files...")
    movies_df = pd.read_csv(movies_csv)
    ratings_df = pd.read_csv(ratings_csv) if ratings_csv.exists() else pd.DataFrame(columns=["movieId", "rating"])
    tags_df = pd.read_csv(tags_csv) if tags_csv.exists() else pd.DataFrame(columns=["movieId", "tag"])
    links_df = pd.read_csv(links_csv) if links_csv.exists() else pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])

    # 1. Deduplicate movies by movieId
    movies_df = movies_df.drop_duplicates(subset=["movieId"]).copy()

    # 2. Extract year and clean title
    extracted = movies_df["title"].apply(extract_year_and_clean_title)
    movies_df["clean_title"] = [item[0] for item in extracted]
    movies_df["year"] = [item[1] for item in extracted]

    # 3. Clean and normalize genres
    movies_df["genres"] = movies_df["genres"].fillna("").astype(str)
    # Replace (no genres listed) with empty
    movies_df["genres_clean"] = movies_df["genres"].apply(
        lambda g: "" if g.strip() == "(no genres listed)" else g.replace("|", " ").lower()
    )

    # 4. Process tags: aggregate unique tags per movieId
    if not tags_df.empty and "tag" in tags_df.columns:
        tags_df = tags_df.dropna(subset=["tag"]).copy()
        tags_df["tag_clean"] = tags_df["tag"].astype(str).str.lower().str.strip()
        tags_df["tag_clean"] = tags_df["tag_clean"].apply(lambda t: re.sub(r"[^\w\s-]", "", t))
        tags_grouped = tags_df.groupby("movieId")["tag_clean"].apply(
            lambda tags: " ".join(sorted(set(filter(None, tags))))
        ).reset_index().rename(columns={"tag_clean": "tags_soup"})
    else:
        tags_grouped = pd.DataFrame(columns=["movieId", "tags_soup"])

    # 5. Process ratings: compute mean rating and vote count
    if not ratings_df.empty and "rating" in ratings_df.columns:
        ratings_summary = ratings_df.groupby("movieId")["rating"].agg(
            vote_count="count",
            vote_average="mean"
        ).reset_index()
        ratings_summary["vote_average"] = ratings_summary["vote_average"].round(2)
    else:
        ratings_summary = pd.DataFrame(columns=["movieId", "vote_count", "vote_average"])

    # 6. Merge datasets
    merged = movies_df.merge(ratings_summary, on="movieId", how="left")
    merged = merged.merge(tags_grouped, on="movieId", how="left")
    if not links_df.empty and "imdbId" in links_df.columns:
        links_df = links_df.drop_duplicates(subset=["movieId"]).copy()
        # Format imdbId as 7-digit zero-padded string
        links_df["imdb_url"] = links_df["imdbId"].apply(
            lambda i: f"https://www.imdb.com/title/tt{str(int(i)).zfill(7)}/" if pd.notna(i) else ""
        )
        merged = merged.merge(links_df[["movieId", "imdbId", "imdb_url"]], on="movieId", how="left")
    else:
        merged["imdb_url"] = ""

    # 7. Fill missing values
    merged["vote_count"] = merged["vote_count"].fillna(0).astype(int)
    merged["vote_average"] = merged["vote_average"].fillna(0.0).astype(float)
    merged["tags_soup"] = merged["tags_soup"].fillna("").astype(str)

    # 8. Compute Bayesian weighted rating for popularity ranking
    merged = compute_weighted_rating(merged)

    # 9. Build content feature soup for TF-IDF Vectorization
    # Weight genres twice to balance with tags and title words
    def make_soup(row):
        genres_weighted = f"{row['genres_clean']} {row['genres_clean']}".strip()
        tags = row["tags_soup"].strip()
        title_tokens = re.sub(r"[^\w\s]", " ", row["clean_title"].lower()).strip()
        soup = f"{genres_weighted} {tags} {title_tokens}".strip()
        return soup if soup else "unknown"

    merged["metadata_soup"] = merged.apply(make_soup, axis=1)

    # 10. Save processed data for rapid app loading
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[data_loader] Processed {len(merged)} movies. Saved to {output_path}")

    return merged


def load_processed_data(
    processed_path: Path = config.PROCESSED_DATA_PATH,
    auto_download: bool = True
) -> pd.DataFrame:
    """Load the processed movies DataFrame. Automatically downloads and prepares if absent.

    Returns:
        DataFrame with processed movies metadata and features.
    """
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        # Ensure correct column types
        df["vote_count"] = df["vote_count"].fillna(0).astype(int)
        df["vote_average"] = df["vote_average"].fillna(0.0).astype(float)
        df["genres"] = df["genres"].fillna("").astype(str)
        df["tags_soup"] = df["tags_soup"].fillna("").astype(str)
        df["metadata_soup"] = df["metadata_soup"].fillna("").astype(str)
        return df

    if auto_download:
        print("[data_loader] Processed data not found. Preparing from raw dataset...")
        download_dataset()
        return prepare_movie_data(output_path=processed_path)

    raise FileNotFoundError(f"Processed data file not found at {processed_path}")


if __name__ == "__main__":
    print("[data_loader] Running standalone dataset setup...")
    download_dataset()
    df = prepare_movie_data()
    print(f"[data_loader] Successfully loaded {len(df)} movies!")
    print(df[["movieId", "title", "year", "genres", "vote_average", "vote_count"]].head(5))
