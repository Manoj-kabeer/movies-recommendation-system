import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
    }
    .movie-title {
        color: #e0e0ff;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    .movie-genre {
        color: #a0a0c0;
        font-size: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .movie-rating {
        color: #ffd700;
        font-size: 1rem;
        font-weight: 600;
    }
    .genre-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        color: #a0b4ff;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        margin: 2px 2px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load & Process Data (cached — runs only once) ────────────────
@st.cache_resource(show_spinner="Loading movie database & computing similarities...")
def load_and_build_model():
    """Load MovieLens CSVs, build content features, store sparse vectors.

    Instead of computing the full NxN similarity matrix (~720 MB for 9,700
    movies), we keep the sparse term-frequency vectors (~5 MB) and compute
    cosine similarity on-the-fly per query.  This lets the app run within
    Render's free-tier 512 MB RAM limit.
    """
    DATA_DIR = "ml-latest-small"

    # Load CSVs
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
    tags = pd.read_csv(f"{DATA_DIR}/tags.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    links = pd.read_csv(f"{DATA_DIR}/links.csv")

    # Average rating per movie
    avg_ratings = ratings.groupby("movieId")["rating"].mean().round(2).reset_index()
    avg_ratings.columns = ["movieId", "avg_rating"]

    # Merge links & ratings
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    movies = movies.merge(avg_ratings, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0.0)
    movies["tmdbId"] = movies["tmdbId"].fillna(0).astype(int)

    # Parse genres
    movies["genres_clean"] = movies["genres"].apply(
        lambda x: x.replace("|", " ").replace("-", "") if x != "(no genres listed)" else ""
    )

    # Aggregate tags per movie
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    tags_agg.columns = ["movieId", "tags_combined"]
    movies = movies.merge(tags_agg, on="movieId", how="left")
    movies["tags_combined"] = movies["tags_combined"].fillna("")

    # Combined content feature
    movies["content"] = (movies["genres_clean"] + " " + movies["tags_combined"]).str.strip()
    movies = movies[movies["content"] != ""].reset_index(drop=True)

    # Vectorize — keep sparse vectors instead of dense NxN matrix
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["content"])

    # Final DataFrame
    movies_final = movies[["movieId", "title", "genres", "avg_rating", "tmdbId"]].copy()

    return movies_final, vectors


movies, vectors = load_and_build_model()


# ── Helper Functions ─────────────────────────────────────────────
def get_star_rating(rating):
    """Convert numeric rating to star display."""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.3 else 0
    empty_stars = 5 - full_stars - half_star
    return "★" * full_stars + ("½" if half_star else "") + "☆" * empty_stars


def format_genres(genres_str):
    """Convert pipe-separated genres to styled badges."""
    if genres_str == "(no genres listed)":
        return '<span class="genre-badge">Unknown</span>'
    genres = genres_str.split("|")
    return " ".join([f'<span class="genre-badge">{g}</span>' for g in genres[:4]])


def recommend(movie):
    """Find top 5 most similar movies (on-the-fly similarity)."""
    movie_index = movies[movies['title'] == movie].index[0]
    # Compute similarity only for this movie against all others
    distances = cosine_similarity(vectors[movie_index], vectors).flatten()
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    recommended = []
    for i in movies_list:
        row = movies.iloc[i[0]]
        recommended.append({
            'title': row['title'],
            'genres': row['genres'],
            'avg_rating': row['avg_rating'],
            'similarity': round(i[1] * 100, 1),
        })
    return recommended


# ── UI ───────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🎬 Movie Recommender</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Discover your next favorite movie — powered by content-based filtering on 9,700+ films</p>',
    unsafe_allow_html=True,
)

# Movie selector
col_select, col_btn = st.columns([4, 1])
with col_select:
    selected_movie = st.selectbox(
        "Pick a movie you like",
        movies['title'].values,
        index=None,
        placeholder="Type or search for a movie...",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("🔍 Recommend", use_container_width=True)

# Show recommendations
if recommend_btn and selected_movie:
    results = recommend(selected_movie)

    st.markdown("---")
    st.markdown("### 🎯 Top 5 Recommendations")

    cols = st.columns(5)
    for idx, (col, movie) in enumerate(zip(cols, results)):
        with col:
            stars = get_star_rating(movie['avg_rating'])
            genres_html = format_genres(movie['genres'])
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="movie-genre">{genres_html}</div>
                    <div class="movie-rating">{stars} {movie['avg_rating']:.1f}</div>
                    <div style="color:#8888bb; font-size:0.75rem; margin-top:8px;">
                        {movie['similarity']}% match
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Show details of the selected movie
    st.markdown("---")
    selected_data = movies[movies['title'] == selected_movie].iloc[0]
    st.markdown("### 📋 Selected Movie Details")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    with detail_col1:
        st.metric("Title", selected_data['title'][:40])
    with detail_col2:
        st.metric("Average Rating", f"⭐ {selected_data['avg_rating']:.2f}")
    with detail_col3:
        st.metric("Genres", selected_data['genres'].replace("|", ", ")[:40])

elif recommend_btn and not selected_movie:
    st.warning("⚠️ Please select a movie first!")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#666; font-size:0.8rem;">'
    'Built with Streamlit • Dataset: MovieLens Latest Small (9,742 movies) • '
    'Content-based filtering with cosine similarity'
    '</p>',
    unsafe_allow_html=True,
)