# preprocess.py

import pandas as pd
import re
import nltk
import joblib
import logging

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# Logging Configuration
# ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# ----------------------------------
# Download NLTK Resources
# ----------------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ----------------------------------
# Load Dataset
# ----------------------------------
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(
        n=20000,
        random_state=42
    )

    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))

except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise

# ----------------------------------
# Drop Unnecessary Column
# ----------------------------------
df = df.drop(columns=["link"], errors="ignore").reset_index(drop=True)

# ----------------------------------
# Text Cleaning
# ----------------------------------
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


logging.info("🧹 Cleaning lyrics...")

df["cleaned_text"] = df["text"].apply(preprocess_text)

logging.info("✅ Text cleaning completed.")

# ----------------------------------
# TF-IDF Vectorization
# ----------------------------------
logging.info("🔠 Creating TF-IDF matrix...")

tfidf = TfidfVectorizer(max_features=10000)

tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# ----------------------------------
# Cosine Similarity
# ----------------------------------
logging.info("📐 Computing cosine similarity...")

cosine_sim = cosine_similarity(tfidf_matrix)

logging.info("✅ Cosine similarity matrix created.")

# ----------------------------------
# Save Files
# ----------------------------------
joblib.dump(df, "df_cleaned.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
joblib.dump(cosine_sim, "cosine_sim.pkl")

logging.info("💾 Files saved successfully.")

logging.info("🎉 Preprocessing completed successfully.")