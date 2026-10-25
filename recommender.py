"""Content-Based Movie Recommendation Engine.

Uses Term Frequency-Inverse Document Frequency (TF-IDF) vectorization and
Cosine Similarity to recommend movies with similar content features
(genres, user tags, and title keywords).
"""

import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import config
from data_loader import load_processed_data


# ==============================================================================
# ALGORITHM EXPLANATION: TF-IDF & COSINE SIMILARITY
# ==============================================================================
# 1. Term Frequency - Inverse Document Frequency (TF-IDF):
#    - Term Frequency (TF): Quantifies how often a word occurs in a specific movie's
#      metadata representation (soup). Sublinear scaling (1 + log(tf)) prevents high-frequency
#      terms from overpowering the model.
#    - Inverse Document Frequency (IDF): Dampens weights of terms that appear ubiquitously
#      across all movies (e.g. 'movie', 'film') and boosts rare, descriptive terms
#      (e.g., 'noir', 'quentin tarantino', 'cyberpunk', 'studio ghibli').
#      Formula: IDF(t) = log((1 + N) / (1 + df(t))) + 1
#    - TF-IDF = TF(t, d) * IDF(t).
#
# 2. Cosine Similarity:
#    - Measures the angle between two multi-dimensional feature vectors:
#      CosineSimilarity(A, B) = (A . B) / (||A||_2 * ||B||_2)
#    - Since TfidfVectorizer normalizes all vectors to unit L2-norm (||v||_2 = 1),
#      the cosine similarity simplifies directly to the inner dot product (A . B).
#    - Values strictly range in [0.0, 1.0] for non-negative TF-IDF representations.
#    - NOTE: Cosine similarity measures content & metadata overlap; it does NOT
#      guarantee user satisfaction or recommendation accuracy on its own.
# ==============================================================================


def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned movie metadata dataframe.

    Args:
        data_path: Optional custom path to processed data CSV.

    Returns:
        pd.DataFrame containing movie metadata, ratings, and features.
    """
    path = config.PROCESSED_DATA_PATH if data_path is None else config.Path(data_path)
    return load_processed_data(processed_path=path, auto_download=True)


def prepare_movie_features(df: pd.DataFrame) -> pd.Series:
    """Extract and validate text feature representation (metadata soup) for movies.

    Args:
        df: DataFrame containing movie records.

    Returns:
        Series of text strings representing movie content.
    """
    if "metadata_soup" in df.columns:
        soup = df["metadata_soup"].fillna("").astype(str)
    else:
        # Fallback construction if metadata_soup column is missing
        genres = df.get("genres", pd.Series([""] * len(df))).fillna("").astype(str)
        tags = df.get("tags_soup", pd.Series([""] * len(df))).fillna("").astype(str)
        clean_title = df.get("clean_title", df.get("title", pd.Series([""] * len(df)))).fillna("").astype(str)
        genres_clean = genres.str.replace("|", " ", regex=False).str.lower()
        soup = (genres_clean + " " + genres_clean + " " + tags + " " + clean_title).str.strip()

    # Ensure no empty strings to keep vectorizer robust
    return soup.replace(r"^\s*$", "unknown", regex=True)


def build_recommendation_model(
    df: pd.DataFrame,
    force_recompute: bool = False,
    save_artifacts: bool = True
) -> Tuple[TfidfVectorizer, Any]:
    """Build and fit TF-IDF vectorizer over movie feature representations.

    Uses sparse representations to minimize memory consumption and maximize speed.

    Args:
        df: Movie DataFrame.
        force_recompute: If True, recomputes model even if cached files exist.
        save_artifacts: If True, saves vectorizer and sparse matrix to disk.

    Returns:
        Tuple of (fitted TfidfVectorizer, sparse tfidf_matrix).
    """
    if (
        not force_recompute
        and config.TFIDF_VECTORIZER_PATH.exists()
        and config.TFIDF_MATRIX_PATH.exists()
    ):
        try:
            vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
            matrix = joblib.load(config.TFIDF_MATRIX_PATH)
            # Verify matrix shape matches current dataframe
            if matrix.shape[0] == len(df):
                return vectorizer, matrix
        except Exception:
            # If load fails, fall through and recompute
            pass

    features = prepare_movie_features(df)

    vectorizer = TfidfVectorizer(
        ngram_range=config.TFIDF_NGRAM_RANGE,
        stop_words="english",
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(features)

    if save_artifacts:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
        joblib.dump(tfidf_matrix, config.TFIDF_MATRIX_PATH)

    return vectorizer, tfidf_matrix


def generate_recommendation_reason(
    query_row: pd.Series,
    candidate_row: pd.Series,
    similarity_score: float
) -> str:
    """Generate an honest, explainable rationale for why a movie was recommended.

    Args:
        query_row: Series for the seed movie.
        candidate_row: Series for the candidate recommendation.
        similarity_score: Cosine similarity score.

    Returns:
        Human-readable explanation of shared attributes.
    """
    # 1. Shared genres
    def parse_genres(g_str):
        if not isinstance(g_str, str) or not g_str or g_str == "(no genres listed)":
            return set()
        return set(part.strip() for part in g_str.split("|") if part.strip())

    q_genres = parse_genres(query_row.get("genres", ""))
    c_genres = parse_genres(candidate_row.get("genres", ""))
    shared_genres = sorted(list(q_genres.intersection(c_genres)))

    # 2. Shared tags
    def parse_tags(t_str):
        if not isinstance(t_str, str) or not t_str:
            return set()
        return set(tok.strip() for tok in t_str.split() if len(tok.strip()) > 2)

    q_tags = parse_tags(query_row.get("tags_soup", ""))
    c_tags = parse_tags(candidate_row.get("tags_soup", ""))
    shared_tags = sorted(list(q_tags.intersection(c_tags)))

    reasons = []
    if shared_genres:
        if len(shared_genres) == 1:
            reasons.append(f"Shares the **{shared_genres[0]}** genre")
        else:
            reasons.append(f"Shares {len(shared_genres)} genres: **{', '.join(shared_genres[:3])}**")

    if shared_tags:
        reasons.append(f"Common themes: *{', '.join(shared_tags[:3])}*")

    if not reasons:
        reasons.append("High overall textual content similarity in metadata")

    return " | ".join(reasons)


def recommend_movies(
    movie_identifier: Any,
    df: pd.DataFrame,
    tfidf_matrix: Any,
    top_n: int = config.DEFAULT_TOP_N,
    genre_filter: Optional[str] = None,
    exclude_selected: bool = True
) -> pd.DataFrame:
    """Recommend top N similar movies using Cosine Similarity on TF-IDF vectors.

    Args:
        movie_identifier: movieId (int) or exact/partial title (str).
        df: Processed movies DataFrame.
        tfidf_matrix: Sparse matrix of TF-IDF vectors for all movies in df.
        top_n: Number of recommendations to return.
        genre_filter: Optional genre to filter recommendations by (e.g. 'Action').
        exclude_selected: Whether to exclude the seed movie itself from results.

    Returns:
        DataFrame containing recommended movies, metadata, similarity scores, and explanations.
    """
    if df.empty or tfidf_matrix is None or top_n <= 0:
        return pd.DataFrame()

    if isinstance(movie_identifier, str) and not movie_identifier.strip():
        return pd.DataFrame()

    # Resolve movie index
    idx = None
    if isinstance(movie_identifier, (int, np.integer)):
        matches = df.index[df["movieId"] == movie_identifier].tolist()
        if matches:
            idx = matches[0]
    elif isinstance(movie_identifier, str):
        exact_matches = df.index[df["title"].str.lower() == movie_identifier.lower().strip()].tolist()
        if exact_matches:
            idx = exact_matches[0]
        else:
            # Partial match on clean_title or title
            partial_matches = df.index[
                df["clean_title"].str.lower().str.contains(re.escape(movie_identifier.lower().strip()), regex=True)
            ].tolist()
            if partial_matches:
                idx = partial_matches[0]

    if idx is None or idx < 0 or idx >= len(df):
        return pd.DataFrame()

    seed_row = df.iloc[idx]
    seed_vector = tfidf_matrix[idx]

    # Compute cosine similarity between seed vector and all movies in dataset
    # Because TF-IDF vectors are L2-normalized, linear_kernel computes exact cosine similarity
    sim_scores = linear_kernel(seed_vector, tfidf_matrix).flatten()

    # Make similarity values finite, clipped strictly to [0.0, 1.0]
    sim_scores = np.nan_to_num(sim_scores, nan=0.0, posinf=1.0, neginf=0.0)
    sim_scores = np.clip(sim_scores, 0.0, 1.0)

    # Sort all indices by descending similarity score
    sorted_indices = np.argsort(sim_scores)[::-1]

    recommendations = []
    for candidate_idx in sorted_indices:
        if exclude_selected and candidate_idx == idx:
            continue

        cand_row = df.iloc[candidate_idx]
        score = float(sim_scores[candidate_idx])

        # Optional genre filter
        if genre_filter and genre_filter.strip() and genre_filter != "All":
            cand_genres = str(cand_row.get("genres", "")).split("|")
            if genre_filter.strip().lower() not in [g.strip().lower() for g in cand_genres]:
                continue

        reason = generate_recommendation_reason(seed_row, cand_row, score)

        rec_item = {
            "movieId": cand_row["movieId"],
            "title": cand_row["title"],
            "clean_title": cand_row.get("clean_title", cand_row["title"]),
            "year": cand_row.get("year"),
            "genres": cand_row.get("genres", ""),
            "vote_average": cand_row.get("vote_average", 0.0),
            "vote_count": cand_row.get("vote_count", 0),
            "similarity_score": round(score, 4),
            "similarity_pct": f"{score * 100:.1f}%",
            "imdb_url": cand_row.get("imdb_url", ""),
            "tags_soup": cand_row.get("tags_soup", ""),
            "recommendation_reason": reason
        }
        recommendations.append(rec_item)

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


def search_movies(query: str, df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Search for movies matching a query string across titles.

    Args:
        query: User input query.
        df: Movies DataFrame.
        limit: Maximum number of search results to return.

    Returns:
        DataFrame with matching movies.
    """
    if not isinstance(query, str) or not query.strip() or df.empty:
        return pd.DataFrame()

    clean_q = query.strip().lower()
    escaped_q = re.escape(clean_q)

    # 1. Exact match on clean title
    exact_mask = df["clean_title"].str.lower() == clean_q
    # 2. Prefix match
    prefix_mask = df["clean_title"].str.lower().str.startswith(clean_q) & ~exact_mask
    # 3. Substring match
    sub_mask = df["clean_title"].str.lower().str.contains(escaped_q, regex=True) & ~exact_mask & ~prefix_mask
    # 4. Fallback search on full title including year
    title_mask = df["title"].str.lower().str.contains(escaped_q, regex=True) & ~exact_mask & ~prefix_mask & ~sub_mask

    # Combine in priority order
    ordered_indices = []
    for mask in [exact_mask, prefix_mask, sub_mask, title_mask]:
        ordered_indices.extend(df.index[mask].tolist())

    # Deduplicate while preserving order
    seen = set()
    final_indices = []
    for idx in ordered_indices:
        if idx not in seen:
            seen.add(idx)
            final_indices.append(idx)
            if len(final_indices) >= limit:
                break

    if not final_indices:
        return pd.DataFrame()

    return df.iloc[final_indices].copy()


def get_popular_movies(
    df: pd.DataFrame,
    top_n: int = 10,
    min_ratings: int = config.MIN_RATINGS_POPULAR,
    genre: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve top-rated popular movies using Bayesian weighted rating.

    Args:
        df: Movies DataFrame.
        top_n: Number of movies to return.
        min_ratings: Minimum number of user votes required.
        genre: Optional genre filter.

    Returns:
        DataFrame with top popular movies.
    """
    if df.empty:
        return pd.DataFrame()

    filtered = df[df["vote_count"] >= min_ratings].copy()

    if genre and genre != "All":
        filtered = filtered[
            filtered["genres"].str.contains(re.escape(genre), case=False, na=False)
        ]

    if "weighted_rating" not in filtered.columns:
        filtered["weighted_rating"] = filtered["vote_average"]

    sorted_df = filtered.sort_values(
        by=["weighted_rating", "vote_average", "vote_count"],
        ascending=[False, False, False]
    )

    return sorted_df.head(top_n).copy()


def get_all_genres(df: pd.DataFrame) -> List[str]:
    """Extract sorted unique list of all genres available in dataset.

    Args:
        df: Movies DataFrame.

    Returns:
        List of distinct genre names.
    """
    if df.empty or "genres" not in df.columns:
        return []

    genres_set = set()
    for g_str in df["genres"].dropna():
        for g in str(g_str).split("|"):
            clean_g = g.strip()
            if clean_g and clean_g != "(no genres listed)":
                genres_set.add(clean_g)

    return sorted(list(genres_set))
