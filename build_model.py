"""
build_model.py
--------------
Processes the MovieLens Latest-Small dataset and generates
pickle files for the movie recommendation Streamlit app.

Content-based approach:
  1. Parse genres (pipe-separated → individual tokens)
  2. Aggregate user tags per movie
  3. Combine genres + tags into a single "content" column
  4. Vectorize with CountVectorizer
  5. Compute cosine similarity
  6. Save movies DataFrame + similarity matrix as pickle files
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "ml-latest-small"

# ── 1. Load raw CSVs ─────────────────────────────────────────────
print("Loading CSVs...")
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
tags = pd.read_csv(f"{DATA_DIR}/tags.csv")
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
links = pd.read_csv(f"{DATA_DIR}/links.csv")

print(f"  Movies : {len(movies)}")
print(f"  Tags   : {len(tags)}")
print(f"  Ratings: {len(ratings)}")
print(f"  Links  : {len(links)}")

# ── 2. Compute average rating per movie ──────────────────────────
avg_ratings = ratings.groupby("movieId")["rating"].mean().round(2).reset_index()
avg_ratings.columns = ["movieId", "avg_rating"]

# ── 3. Merge links (tmdbId for poster fetching) ─────────────────
movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
movies = movies.merge(avg_ratings, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0.0)

# ── 4. Parse genres into space-separated tokens ─────────────────
# "Adventure|Animation|Children" → "Adventure Animation Children"
movies["genres_clean"] = movies["genres"].apply(
    lambda x: x.replace("|", " ").replace("-", "") if x != "(no genres listed)" else ""
)

# ── 5. Aggregate tags per movie ──────────────────────────────────
# Combine all user tags for each movie into a single string
tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
tags_agg.columns = ["movieId", "tags_combined"]

movies = movies.merge(tags_agg, on="movieId", how="left")
movies["tags_combined"] = movies["tags_combined"].fillna("")

# ── 6. Build combined content feature ────────────────────────────
# genres + tags give us a rich text representation per movie
movies["content"] = movies["genres_clean"] + " " + movies["tags_combined"]
movies["content"] = movies["content"].str.strip()

# Drop movies with no content at all (very rare edge case)
movies = movies[movies["content"] != ""].reset_index(drop=True)

print(f"\nMovies after cleaning: {len(movies)}")
print(f"Sample content:\n  {movies.iloc[0]['title']}: {movies.iloc[0]['content'][:100]}...")

# ── 7. Vectorize with CountVectorizer ────────────────────────────
print("\nVectorizing content...")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["content"])
print(f"  Feature matrix shape: {vectors.shape}")

# ── 8. Compute cosine similarity ─────────────────────────────────
print("Computing cosine similarity...")
similarity = cosine_similarity(vectors)
print(f"  Similarity matrix shape: {similarity.shape}")

# ── 9. Prepare final DataFrame ───────────────────────────────────
movies_final = movies[["movieId", "title", "genres", "avg_rating", "tmdbId"]].copy()
movies_final["tmdbId"] = movies_final["tmdbId"].fillna(0).astype(int)

print(f"\nFinal DataFrame columns: {list(movies_final.columns)}")
print(movies_final.head(10).to_string())

# ── 10. Save pickle files ────────────────────────────────────────
print("\nSaving pickle files...")

with open("movies_new.pkl", "wb") as f:
    pickle.dump(movies_final, f)
    print(f"  Saved movies_new.pkl")

with open("similarity_new.pkl", "wb") as f:
    pickle.dump(similarity, f)
    print(f"  Saved similarity_new.pkl")

print("\n✅ Done! New model files generated successfully.")
print(f"   Total movies: {len(movies_final)}")
print(f"   Similarity matrix: {similarity.shape[0]}x{similarity.shape[1]}")
