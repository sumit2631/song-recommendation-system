import streamlit as st
from recommend import df, recommend_songs

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to bottom, #191414, #121212);
}

/* Hide Streamlit Menu & Footer */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {background: transparent !important;} 


.main-title{
    font-size:48px;
    font-weight:700;
    color:#1DB954;
    text-align:center;
    margin-bottom:5px;
}

.sub-title{
    font-size:20px;
    color:#DDDDDD;
    text-align:center;
    margin-bottom:30px;
}

.song-card{
    background:#242424;
    padding:20px;
    border-radius:15px;
    margin-bottom:18px;
    border-left:6px solid #1DB954;
    box-shadow:0px 3px 10px rgba(0,0,0,0.3);
}

.song-title{
    color:white;
    font-size:24px;
    font-weight:bold;
}

.artist{
    color:#B3B3B3;
    font-size:18px;
    margin-top:8px;
}

.match{
    color:#1DB954;
    font-size:17px;
    font-weight:bold;
    margin-top:10px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:60px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("🎧  Music Recommender")

st.sidebar.markdown("---")

st.sidebar.info("""
### 🔍 How it Works

This project recommends songs using:

- 🎵 Song Lyrics
- 📝 TF-IDF Vectorization
- 📐 Cosine Similarity
- 🤖 Machine Learning
""")

st.sidebar.markdown("---")

st.sidebar.metric("🎵 Total Songs", len(df))
st.sidebar.metric("👤 Total Artists", df["artist"].nunique())

top_n = st.sidebar.slider(
    "🎼 Number of Recommendations",
    min_value=5,
    max_value=20,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.success("Made by **Sumit Halder** ❤️")

# ---------------------------------
# Main Page
# ---------------------------------
st.markdown(
    "<div class='main-title'>🎵 Music Recommendation System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Discover similar songs using Machine Learning & NLP</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------
# Song Selection
# ---------------------------------
song_list = sorted(df["song"].dropna().unique())

selected_song = st.selectbox(
    "🎼 Search and Select a Song",
    song_list
)

# ---------------------------------
# Recommend Button
# ---------------------------------
if st.button("🚀 Recommend Similar Songs", use_container_width=True):

    with st.spinner("🎧 Searching for similar songs..."):

        recommendations = recommend_songs(
            selected_song,
            top_n
        )

    if recommendations is None:

        st.error("❌ Song not found.")

    else:

        st.success(f"🎉 Top {top_n} Recommended Songs")

        for _, row in recommendations.iterrows():

            st.markdown(f"""
            <div class="song-card">

            <div class="song-title">
            🎵 {row['song']}
            </div>

            <div class="artist">
            👤 {row['artist']}
            </div>

            <div class="match">
            ⭐ Match Score: {row['Match (%)']}%
            </div>

            </div>
            """, unsafe_allow_html=True)

# ---------------------------------
# Footer
# ---------------------------------
st.divider()

st.markdown("""
<div class="footer">

🎵 AI Music Recommendation System <br>

Built with ❤️ using <b>Streamlit</b>, <b>Scikit-Learn</b>, <b>Machine Learning</b> and <b>Natural Language Processing</b>.

<br><br>

Developed by <b>Sumit Halder</b>

</div>
""", unsafe_allow_html=True)