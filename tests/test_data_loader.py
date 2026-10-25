"""Unit tests for data loading, parsing, and preprocessing."""

import pytest
import pandas as pd
import numpy as np

from data_loader import (
    extract_year_and_clean_title,
    compute_weighted_rating,
    load_processed_data
)


def test_extract_year_and_clean_title_standard():
    """Verify standard 'Title (YYYY)' extraction."""
    clean, year = extract_year_and_clean_title("Toy Story (1995)")
    assert clean == "Toy Story"
    assert year == 1995


def test_extract_year_and_clean_title_with_extra_spaces():
    """Verify title with space inside parentheses."""
    clean, year = extract_year_and_clean_title("The Matrix (1999 )")
    assert clean == "The Matrix"
    assert year == 1999


def test_extract_year_and_clean_title_no_year():
    """Verify title without year returns None for year and unmodified title."""
    clean, year = extract_year_and_clean_title("Babylon 5")
    assert clean == "Babylon 5"
    assert year is None


def test_extract_year_and_clean_title_edge_cases():
    """Verify edge cases like empty string, None, and title with numbers."""
    clean, year = extract_year_and_clean_title("")
    assert clean == ""
    assert year is None

    clean, year = extract_year_and_clean_title(None)
    assert clean == ""
    assert year is None

    clean, year = extract_year_and_clean_title("2001: A Space Odyssey (1968)")
    assert clean == "2001: A Space Odyssey"
    assert year == 1968


def test_compute_weighted_rating_bayesian():
    """Verify that Bayesian weighted rating dampens low vote counts."""
    sample_df = pd.DataFrame([
        {"movieId": 1, "title": "Obscure 5-Star", "vote_average": 5.0, "vote_count": 1},
        {"movieId": 2, "title": "Blockbuster 4.5", "vote_average": 4.5, "vote_count": 500},
        {"movieId": 3, "title": "Average Movie", "vote_average": 3.0, "vote_count": 150},
        {"movieId": 4, "title": "Unrated Movie", "vote_average": 0.0, "vote_count": 0},
    ])

    result = compute_weighted_rating(sample_df, min_ratings=25)

    assert "weighted_rating" in result.columns
    # The movie with 500 votes at 4.5 should rank higher than 1 vote at 5.0
    r_obscure = result.loc[result["movieId"] == 1, "weighted_rating"].values[0]
    r_blockbuster = result.loc[result["movieId"] == 2, "weighted_rating"].values[0]
    r_unrated = result.loc[result["movieId"] == 4, "weighted_rating"].values[0]

    assert r_blockbuster > r_obscure
    assert r_unrated == 0.0


def test_load_processed_data_schema():
    """Verify that loaded dataset conforms to expected schema."""
    df = load_processed_data(auto_download=False)
    assert not df.empty
    expected_cols = [
        "movieId",
        "title",
        "clean_title",
        "genres",
        "vote_average",
        "vote_count",
        "metadata_soup"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) >= 1000
    assert not df["metadata_soup"].isna().any()
    assert (df["vote_count"] >= 0).all()
    assert (df["vote_average"] >= 0.0).all()
