"""Configuration constants and paths for the Movie Recommendation System."""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "ml-latest-small"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
PROCESSED_DATA_PATH = DATA_DIR / "movies_processed.csv"
RATINGS_SUMMARY_PATH = DATA_DIR / "ratings_summary.csv"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"

# Dataset Source
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Raw Files expected inside ml-latest-small
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
TAGS_FILE = RAW_DATA_DIR / "tags.csv"
LINKS_FILE = RAW_DATA_DIR / "links.csv"

# Model and Recommendation Defaults
DEFAULT_TOP_N = 8
MAX_TOP_N = 20
MIN_RATINGS_POPULAR = 20
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42

# Ensure core directories exist
for folder in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
