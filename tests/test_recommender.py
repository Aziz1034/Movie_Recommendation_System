"""Unit tests for content-based recommender engine, search, and ranking."""

import pytest
import numpy as np
import pandas as pd

import recommender


@pytest.fixture(scope="module")
def dataset_and_model():
    """Load dataset and build TF-IDF model once for testing."""
    df = recommender.load_data()
    vectorizer, tfidf_matrix = recommender.build_recommendation_model(df, save_artifacts=False)
    return df, vectorizer, tfidf_matrix


def test_build_recommendation_model(dataset_and_model):
    """Verify TF-IDF matrix dimensions and normalization."""
    df, vectorizer, tfidf_matrix = dataset_and_model
    assert tfidf_matrix.shape[0] == len(df)
    assert tfidf_matrix.shape[1] > 0
    # Verify row sparsity and non-emptiness
    assert tfidf_matrix.nnz > 0


def test_recommend_movies_by_id(dataset_and_model):
    """Verify recommendation generation using movieId (e.g., Toy Story ID 1)."""
    df, _, tfidf_matrix = dataset_and_model
    top_n = 5
    seed_id = 1  # Toy Story (1995)

    recs = recommender.recommend_movies(
        movie_identifier=seed_id,
        df=df,
        tfidf_matrix=tfidf_matrix,
        top_n=top_n,
        exclude_selected=True
    )

    assert not recs.empty
    assert len(recs) == top_n
    # Ensure seed movie is not recommended to itself
    assert seed_id not in recs["movieId"].values

    # Check similarity score bounds
    scores = recs["similarity_score"].values
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    # Check that results are sorted descending
    assert list(scores) == sorted(list(scores), reverse=True)


def test_recommend_movies_by_title(dataset_and_model):
    """Verify recommendation generation using movie title string."""
    df, _, tfidf_matrix = dataset_and_model
    recs = recommender.recommend_movies(
        movie_identifier="Toy Story",
        df=df,
        tfidf_matrix=tfidf_matrix,
        top_n=6
    )

    assert not recs.empty
    assert len(recs) == 6
    # Seed title shouldn't be the first recommendation if excluded
    assert "Toy Story (1995)" not in recs["title"].values


def test_recommend_movies_genre_filter(dataset_and_model):
    """Verify recommendations can be filtered by specific genre."""
    df, _, tfidf_matrix = dataset_and_model
    recs = recommender.recommend_movies(
        movie_identifier=1,  # Toy Story
        df=df,
        tfidf_matrix=tfidf_matrix,
        top_n=5,
        genre_filter="Animation"
    )

    assert not recs.empty
    for _, row in recs.iterrows():
        assert "Animation" in str(row["genres"])


def test_recommend_movies_unknown_identifier(dataset_and_model):
    """Verify unknown movie identifier returns empty DataFrame gracefully."""
    df, _, tfidf_matrix = dataset_and_model
    # Non-existent ID
    recs = recommender.recommend_movies(-99999, df, tfidf_matrix, top_n=5)
    assert recs.empty

    # Non-existent title
    recs = recommender.recommend_movies("NonExistentMovieXYZ123456", df, tfidf_matrix, top_n=5)
    assert recs.empty

    # Empty string
    recs = recommender.recommend_movies("", df, tfidf_matrix, top_n=5)
    assert recs.empty


def test_recommend_movies_deterministic(dataset_and_model):
    """Verify that recommendation outputs are deterministic across calls."""
    df, _, tfidf_matrix = dataset_and_model
    recs_1 = recommender.recommend_movies(1, df, tfidf_matrix, top_n=5)
    recs_2 = recommender.recommend_movies(1, df, tfidf_matrix, top_n=5)

    pd.testing.assert_frame_equal(recs_1, recs_2)


def test_search_movies_partial_and_exact(dataset_and_model):
    """Verify search handles exact, prefix, and partial string queries."""
    df, _, _ = dataset_and_model

    # Exact search
    res = recommender.search_movies("Matrix", df, limit=5)
    assert not res.empty
    assert any("Matrix" in t for t in res["title"])

    # Lowercase search
    res_lower = recommender.search_movies("toy story", df, limit=5)
    assert not res_lower.empty
    assert any("Toy Story" in t for t in res_lower["title"])

    # Empty query
    assert recommender.search_movies("", df).empty
    assert recommender.search_movies("   ", df).empty

    # Special characters that could break regex
    res_special = recommender.search_movies("[1995]", df, limit=5)
    # Should not crash
    assert isinstance(res_special, pd.DataFrame)


def test_get_popular_movies(dataset_and_model):
    """Verify popular movies ranking logic."""
    df, _, _ = dataset_and_model
    pop = recommender.get_popular_movies(df, top_n=10, min_ratings=30)

    assert not pop.empty
    assert len(pop) == 10
    # Every movie must meet the minimum vote count threshold
    assert (pop["vote_count"] >= 30).all()

    # Check genre filtering in popular movies
    pop_action = recommender.get_popular_movies(df, top_n=5, min_ratings=20, genre="Action")
    assert not pop_action.empty
    for _, row in pop_action.iterrows():
        assert "Action" in row["genres"]


def test_get_all_genres(dataset_and_model):
    """Verify extracted genre list is non-empty and contains common genres."""
    df, _, _ = dataset_and_model
    genres = recommender.get_all_genres(df)

    assert isinstance(genres, list)
    assert len(genres) >= 10
    assert "Action" in genres
    assert "Comedy" in genres
    assert "Drama" in genres
    assert "Sci-Fi" in genres
    assert "(no genres listed)" not in genres


def test_generate_recommendation_reason():
    """Verify that recommendation reason strings are generated correctly."""
    query_row = pd.Series({"genres": "Action|Adventure|Sci-Fi", "tags_soup": "space superhero alien"})
    cand_row = pd.Series({"genres": "Action|Sci-Fi|Thriller", "tags_soup": "alien space technology"})

    reason = recommender.generate_recommendation_reason(query_row, cand_row, 0.75)
    assert isinstance(reason, str)
    assert "Action" in reason or "Sci-Fi" in reason
    assert "space" in reason or "alien" in reason
