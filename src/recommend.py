import joblib
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
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
# Load Data & Generate TF-IDF Vector On-The-Fly
# -------------------------------------------------
logging.info("🔁 Loading data...")

try:
    # Load the lightweight text file from GitHub
    df = joblib.load(BASE_DIR / "df_cleaned.pkl")
    logging.info("✅ Dataset loaded successfully.")
    
    # Rebuild the matrix instantly in memory so we don't need a heavy file
    logging.info("🔠 Generating TF-IDF matrix dynamically...")
    tfidf = TfidfVectorizer(max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    logging.info("✅ TF-IDF matrix generated successfully.")

except Exception as e:
    logging.error("❌ Failed to initialize recommendation data: %s", str(e))
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

    # Calculate similarity only for the selected song against all others
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
