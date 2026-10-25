"""CineMatch - Movie Recommendation System Using Content-Based Filtering.

A polished, production-ready Streamlit web application providing intelligent
movie recommendations based on genres, user tags, and metadata using
TF-IDF vectorization and Cosine Similarity.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import config
import recommender


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="CineMatch | Content-Based Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM STYLING (Cinematic Dark Theme)
# ==============================================================================
CUSTOM_CSS = """
<style>
/* Global styles and typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 15% 10%, #151926 0%, #0c0e14 100%);
    color: #e2e8f0;
}

/* Header hero container */
.hero-container {
    background: linear-gradient(135deg, rgba(229, 9, 20, 0.12) 0%, rgba(99, 102, 241, 0.14) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #ff4b4b 0%, #f59e0b 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 12px;
}

.hero-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(229, 9, 20, 0.2);
    color: #fca5a5;
    border: 1px solid rgba(229, 9, 20, 0.3);
}

/* Selected movie hero card */
.selected-card {
    background: linear-gradient(145deg, #181d2c 0%, #111420 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

/* Recommendation Cards */
.rec-card {
    background: #151a27;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.rec-card:hover {
    transform: translateY(-3px);
    border-color: rgba(229, 9, 20, 0.5);
    box-shadow: 0 12px 28px rgba(229, 9, 20, 0.15);
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 4px;
    line-height: 1.3;
}

.card-meta {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 10px;
}

.rating-badge {
    background: rgba(245, 197, 24, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 197, 24, 0.3);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
    margin-right: 6px;
}

.sim-badge {
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
}

.genre-tag {
    background: rgba(255, 255, 255, 0.06);
    color: #cbd5e1;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    margin-right: 4px;
    margin-top: 4px;
    display: inline-block;
}

.reason-box {
    background: rgba(15, 23, 42, 0.75);
    border-left: 3px solid #e50914;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.8rem;
    color: #cbd5e1;
    margin-top: 10px;
    margin-bottom: 12px;
    line-height: 1.4;
}

.imdb-link {
    display: inline-flex;
    align-items: center;
    color: #f59e0b !important;
    text-decoration: none !important;
    font-size: 0.8rem;
    font-weight: 600;
    transition: opacity 0.2s;
}

.imdb-link:hover {
    opacity: 0.8;
}

/* Sidebar styling */
.sidebar-stat-box {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# DATA AND MODEL CACHING
# ==============================================================================
@st.cache_data(show_spinner="Loading MovieLens dataset...")
def get_movie_data() -> pd.DataFrame:
    """Load and cache the processed movie catalogue."""
    return recommender.load_data()


@st.cache_resource(show_spinner="Initializing TF-IDF vectorizer and matrix...")
def get_recommender_model(df_len: int):
    """Build and cache the TF-IDF recommendation artifacts."""
    df = get_movie_data()
    return recommender.build_recommendation_model(df, save_artifacts=True)


# Load data and model
df = get_movie_data()
vectorizer, tfidf_matrix = get_recommender_model(len(df))
available_genres = recommender.get_all_genres(df)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎬 CineMatch Recommender")
    st.caption("Content-Based Filtering via TF-IDF & Cosine Similarity")
    st.markdown("---")

    st.markdown("#### 📊 Dataset Statistics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Movies", f"{len(df):,}")
    with col_s2:
        total_ratings = int(df["vote_count"].sum())
        st.metric("Total Ratings", f"{total_ratings:,}")

    unique_genres_count = len(available_genres)
    st.markdown(
        f"""
        <div class="sidebar-stat-box">
            <small style="color: #94a3b8;">Source Dataset:</small><br>
            <strong>MovieLens Latest Small</strong><br>
            <small style="color: #64748b;">(GroupLens Research, Univ. of Minnesota)</small><br>
            <small style="color: #94a3b8; margin-top: 4px; display: block;">
                • {unique_genres_count} Distinct Genres<br>
                • Real User Tags & IMDb Keys
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### 🧠 Recommendation Architecture")
    with st.expander("How Recommendations Work", expanded=False):
        st.markdown(
            """
            1. **Feature Engineering**: Combines movie genres (weighted), crowdsourced user tags, and cleaned titles into a semantic metadata soup.
            2. **TF-IDF Vectorization**: Sublinear term frequency converts metadata words into a 10,000-dimensional sparse feature space with unigrams and bigrams.
            3. **Cosine Similarity**: Inner product between unit L2-norm vectors computes genuine content overlap in $[0.0, 1.0]$.
            4. **Explainability**: Identifies shared genres and common thematic keywords to provide clear transparency.
            """
        )

    st.markdown("---")
    st.markdown("#### 💡 Quick Pick Classic Movies")
    preset_choice = st.selectbox(
        "Load a popular title directly:",
        options=[
            "None",
            "Toy Story (1995)",
            "Matrix, The (1999)",
            "Pulp Fiction (1994)",
            "Fight Club (1999)",
            "Inception (2010)",
            "Interstellar (2014)",
            "Shawshank Redemption, The (1994)",
            "Dark Knight, The (2008)",
            "Forrest Gump (1994)"
        ],
        index=0
    )

    st.markdown("---")
    st.caption("Developed for ML Engineering Internship • 100% Offline & Open-Source")


# ==============================================================================
# HERO HEADER
# ==============================================================================
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">CineMatch</h1>
        <div class="hero-subtitle">Discover your next favorite movie through intelligent content-based recommendations.</div>
        <div>
            <span class="hero-badge">⚡ Real-time TF-IDF & Cosine Similarity</span>
            <span class="hero-badge" style="background: rgba(99, 102, 241, 0.2); color: #c7d2fe; border-color: rgba(99, 102, 241, 0.3);">
                🍿 9,742 Real Movies Loaded
            </span>
            <span class="hero-badge" style="background: rgba(16, 185, 129, 0.2); color: #a7f3d0; border-color: rgba(16, 185, 129, 0.3);">
                🛡️ Zero API Keys Required
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# MAIN TABS INTERFACE
# ==============================================================================
tab_recs, tab_discover, tab_metrics = st.tabs([
    "🎯 Movie Recommender",
    "🌟 Trending & Discovery",
    "📈 Evaluation & Technical Report"
])


# ==============================================================================
# TAB 1: MOVIE RECOMMENDER
# ==============================================================================
with tab_recs:
    st.markdown("### Search and Select a Movie")

    # Search Bar
    search_col, filter_col = st.columns([3, 1])

    with search_col:
        search_query = st.text_input(
            "Enter movie title to search:",
            placeholder="e.g. Inception, Toy Story, Matrix, Dark Knight...",
            help="Type any partial or full movie title."
        )

    # Resolve active movie selection
    selected_movie_id = None

    if preset_choice != "None":
        matched_preset = df[df["title"] == preset_choice]
        if not matched_preset.empty:
            selected_movie_id = int(matched_preset.iloc[0]["movieId"])

    if search_query.strip():
        search_results = recommender.search_movies(search_query, df, limit=12)
        if search_results.empty:
            st.warning(f"No movies found matching '{search_query}'. Try a different keyword.")
        else:
            options_dict = {}
            for _, row in search_results.iterrows():
                yr_str = f" ({int(row['year'])})" if pd.notna(row['year']) else ""
                key = f"{row['clean_title']}{yr_str} — {row['genres']} [ID: {row['movieId']}]"
                options_dict[key] = row["movieId"]

            chosen_label = st.selectbox(
                f"Matching Movies ({len(search_results)} found):",
                options=list(options_dict.keys()),
                index=0
            )
            selected_movie_id = int(options_dict[chosen_label])

    # If nothing searched and no preset, default to Toy Story (movieId=1)
    if selected_movie_id is None:
        selected_movie_id = 1

    # Selected Movie Details Hero
    selected_row = df[df["movieId"] == selected_movie_id].iloc[0]

    st.markdown("---")
    st.markdown("#### Currently Selected Movie")

    col_details, col_ctrl = st.columns([2.5, 1.5])

    with col_details:
        year_display = f"({int(selected_row['year'])})" if pd.notna(selected_row['year']) else ""
        genres_list = [g.strip() for g in str(selected_row['genres']).split("|") if g.strip()]
        tags_display = str(selected_row.get("tags_soup", "")).strip()

        st.markdown(
            f"""
            <div class="selected-card">
                <div style="font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-bottom: 6px;">
                    🎬 {selected_row['clean_title']} <span style="color: #94a3b8; font-weight: 500;">{year_display}</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="rating-badge">⭐ {selected_row['vote_average']:.1f} / 5.0</span>
                    <span style="color: #94a3b8; font-size: 0.85rem;">({int(selected_row['vote_count'])} verified user ratings)</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Genres:</strong> {' '.join([f'<span class="genre-tag">{g}</span>' for g in genres_list])}
                </div>
                {f'<div style="font-size: 0.85rem; color: #94a3b8;"><strong>Community Tags:</strong> <em>{tags_display}</em></div>' if tags_display else ''}
                {f'<div style="margin-top: 12px;"><a class="imdb-link" href="{selected_row["imdb_url"]}" target="_blank">↗ View Official IMDb Profile</a></div>' if selected_row.get('imdb_url') else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_ctrl:
        st.markdown("**Recommendation Settings**")
        top_n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=8, step=1)
        genre_filter = st.selectbox("Filter recommendations by genre:", options=["All"] + available_genres, index=0)
        generate_btn = st.button("✨ Generate Recommendations", type="primary", use_container_width=True)

    # Generate Recommendations
    st.markdown("---")
    st.markdown(f"### Recommendations Similar to *{selected_row['clean_title']}*")

    with st.spinner("Calculating cosine similarity matrix across movie vectors..."):
        recs = recommender.recommend_movies(
            movie_identifier=selected_movie_id,
            df=df,
            tfidf_matrix=tfidf_matrix,
            top_n=top_n,
            genre_filter=genre_filter,
            exclude_selected=True
        )

    if recs.empty:
        st.info("No matching movies found for the selected genre filter. Try changing the genre filter to 'All'.")
    else:
        st.caption(
            "📌 *Similarity scores indicate TF-IDF Cosine Similarity based on genres, user tags, and title keywords.*"
        )

        # Display recommendations in a responsive 2-column or 3-column card grid
        cols_per_row = 2
        rec_rows = [recs.iloc[i:i + cols_per_row] for i in range(0, len(recs), cols_per_row)]

        for row_df in rec_rows:
            card_cols = st.columns(cols_per_row)
            for c_idx, (_, rec_item) in enumerate(row_df.iterrows()):
                with card_cols[c_idx]:
                    rec_year = f"({int(rec_item['year'])})" if pd.notna(rec_item['year']) else ""
                    rec_genres = [g.strip() for g in str(rec_item['genres']).split("|") if g.strip()]
                    rec_imdb = rec_item.get("imdb_url", "")

                    genre_chips = " ".join([f'<span class="genre-tag">{g}</span>' for g in rec_genres[:4]])
                    imdb_link_html = (
                        f'<a class="imdb-link" href="{rec_imdb}" target="_blank">↗ IMDb</a>'
                        if rec_imdb else ""
                    )

                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div>
                                <div class="card-title">🎥 {rec_item['clean_title']} <span style="font-weight: 400; color: #94a3b8; font-size: 0.9rem;">{rec_year}</span></div>
                                <div class="card-meta">
                                    <span class="rating-badge">⭐ {rec_item['vote_average']:.1f}</span>
                                    <span class="sim-badge">Similarity: {rec_item['similarity_pct']}</span>
                                </div>
                                <div style="margin-bottom: 6px;">
                                    {genre_chips}
                                </div>
                                <div class="reason-box">
                                    💡 {rec_item['recommendation_reason']}
                                </div>
                            </div>
                            <div style="display: flex; justify-content: flex-end; margin-top: 6px;">
                                {imdb_link_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ==============================================================================
# TAB 2: DISCOVERY & CATALOGUE
# ==============================================================================
with tab_discover:
    st.markdown("### 🌟 Trending & Highest-Rated Movies")
    st.markdown(
        "Ranked using **Bayesian / IMDb Weighted Rating** formula to prevent titles with only 1 single rating from skewing the top list."
    )

    disc_c1, disc_c2, disc_c3 = st.columns([2, 1, 1])
    with disc_c1:
        disc_genre = st.selectbox("Filter Catalogue by Genre:", options=["All"] + available_genres, key="disc_genre")
    with disc_c2:
        disc_count = st.slider("Movies to display:", min_value=5, max_value=30, value=12, step=1)
    with disc_c3:
        min_v = st.number_input("Minimum Ratings Count:", min_value=5, max_value=200, value=30, step=5)

    popular_movies = recommender.get_popular_movies(
        df,
        top_n=disc_count,
        min_ratings=int(min_v),
        genre=disc_genre
    )

    if popular_movies.empty:
        st.warning("No movies matched the criteria. Try lowering the minimum ratings count.")
    else:
        # Display as tabular discovery list
        display_df = popular_movies[[
            "movieId", "clean_title", "year", "genres", "vote_average", "vote_count", "weighted_rating"
        ]].copy()
        display_df.columns = ["ID", "Title", "Year", "Genres", "Avg Rating", "Vote Count", "Bayesian Score"]
        display_df["Avg Rating"] = display_df["Avg Rating"].apply(lambda r: f"⭐ {r:.2f}")
        display_df["Bayesian Score"] = display_df["Bayesian Score"].apply(lambda s: f"{s:.2f}")

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )


# ==============================================================================
# TAB 3: TECHNICAL REPORT & EVALUATION
# ==============================================================================
with tab_metrics:
    st.markdown("### 📈 Scientific Methodology & Offline Evaluation")

    report_path = config.REPORTS_DIR / "evaluation_report.md"
    results_json_path = config.REPORTS_DIR / "evaluation_results.json"

    if results_json_path.exists():
        import json
        with open(results_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        m = eval_data.get("metrics", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Mean Metadata Cosine Sim", f"{m.get('mean_metadata_cosine_similarity', 0):.3f}")
            st.caption("Average TF-IDF inner product")
        with m2:
            st.metric("Genre Jaccard Index", f"{m.get('mean_genre_jaccard_similarity', 0):.3f}")
            st.caption("Semantic genre overlap (|A∩B| / |A∪B|)")
        with m3:
            st.metric("Intra-List Diversity", f"{m.get('mean_intra_list_diversity', 0):.3f}")
            st.caption("Content variation across list")
        with m4:
            st.metric("User Co-Rating Positive Ratio", f"{m.get('user_co_rating_positive_ratio', 0) * 100:.1f}%")
            st.caption(f"Baseline rating: {m.get('dataset_baseline_mean_rating', 0)} / 5.0")

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.markdown(report_text)
    else:
        st.info("Run `python evaluate.py` to generate the evaluation report.")
