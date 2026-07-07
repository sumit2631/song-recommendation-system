# 🎵 Song Recommendation System

A content-based music recommendation system that suggests similar songs based on their lyrics. This project uses **Natural Language Processing (NLP)** techniques, **TF-IDF Vectorization**, and **Cosine Similarity** to identify songs with similar lyrical content. The application is built with **Python** and **Streamlit**.

---

## 📖 About the Project

Unlike collaborative recommendation systems that rely on user preferences, this project compares the lyrics of songs to find similar tracks.

The workflow is simple:

1. Song lyrics are cleaned by removing punctuation, converting text to lowercase, and removing common stopwords.
2. The cleaned lyrics are converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
3. **Cosine Similarity** is used to measure how similar two songs are.
4. When a user selects a song, the application returns the most lyrically similar songs along with their similarity scores.

This project demonstrates how Natural Language Processing (NLP) can be used to build a simple content-based recommendation engine.

---

## ✨ Features

- 🎵 Recommend songs based on lyrical similarity
- 🔍 TF-IDF Vectorization for text representation
- 📐 Cosine Similarity based recommendation engine
- ⚡ Fast recommendations using precomputed similarity scores
- 🎨 Interactive Streamlit web application
- 📊 Displays similarity percentage for each recommendation

---

## 📂 Project Structure

```text
song-recommendation-system/
│
├── src/
│   ├── main.py                     # Streamlit application
│   ├── preprocess.py               # Data preprocessing
│   ├── recommend.py                # Recommendation engine
│   └── spotify_millsongdata.csv    # Download separately
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web application |
| Pandas | Data manipulation |
| Scikit-learn | TF-IDF Vectorization & Cosine Similarity |
| NLTK | Text preprocessing |
| Joblib | Saving preprocessed data |

---

## 📋 Requirements

- Python 3.12 or later
- pip

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sumit2631/song-recommendation-system.git

cd song-recommendation-system
```

### Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv musicenv

musicenv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv musicenv

source musicenv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

This project uses the **Spotify Million Song Dataset**.

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

After downloading, place the file:

```text
spotify_millsongdata.csv
```

inside the **src/** folder.

---

## ⚙️ Preprocessing

Run the preprocessing script to clean the lyrics and generate the files required by the recommendation system.

```bash
python src/preprocess.py
```

The script creates:

- `df_cleaned.pkl`
- `tfidf_matrix.pkl`
- `cosine_sim.pkl`

> **Note:** These generated files are not included in the repository. They will be created automatically after running the preprocessing script.

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run src/main.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## ⚡ Quick Start

```bash
git clone https://github.com/sumit2631/song-recommendation-system.git

cd song-recommendation-system

pip install -r requirements.txt

python src/preprocess.py

streamlit run src/main.py
```

---

## 📸 Application Preview

*A screenshot of the application will be added here.*

---

## 🚀 Future Improvements

- Spotify API integration
- Display album artwork
- Artist and genre filtering
- Hybrid recommendation system
- Deploy the application using Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit and push your changes.
5. Open a Pull Request.

---

## 👨‍💻 Author

**Sumit Halder**

GitHub: https://github.com/sumit2631

LinkedIn: https://www.linkedin.com/in/sumit-halder-94859225a/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
