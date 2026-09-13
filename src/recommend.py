# recommend.py
import joblib
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# File Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------
# Setup logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / "recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# -------------------------------------------------
# Load Data
# -------------------------------------------------
logging.info("🔁 Loading data...")

try:
    df = joblib.load(BASE_DIR / "df_cleaned.pkl")
    tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.pkl")

    logging.info("✅ Data loaded successfully.")

except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise


# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def recommend_songs(song_name, top_n=5):

    logging.info("🎵 Recommending songs for: '%s'", song_name)

    # Find the selected song
    idx = df[df['song'].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None

    idx = idx[0]

    # Calculate similarity only for the selected song
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx:idx + 1],
        tfidf_matrix
    ).flatten()

    # Get highest similarity scores
    similar_indices = similarity_scores.argsort()[::-1]

    # Remove the selected song itself
    similar_indices = [
        i for i in similar_indices
        if i != idx
    ][:top_n]

    # Extract similarity percentages
    similarities = [
        round(similarity_scores[i] * 100, 2)
        for i in similar_indices
    ]

    logging.info(
        "✅ Top %d recommendations ready.",
        top_n
    )

    # Create result DataFrame
    result_df = df[['artist', 'song']].iloc[
        similar_indices
    ].reset_index(drop=True)

    # Add similarity percentage
    result_df["Match (%)"] = similarities

    # Start indexing from 1
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    return result_df