# 🎬 Movie Recommender System

A content-based movie recommendation engine built with **Streamlit** and **scikit-learn**, powered by the MovieLens dataset (9,700+ films).

### 🌐 [Live Demo →](https://movies-recommendation-system-3-kqcc.onrender.com/)

> **Note:** The app is hosted on Render's free tier. The first load may take ~30–60 seconds if the server has been idle.

---

## ✨ Features

- **Content-Based Filtering** — Recommends movies based on genres and user-generated tags using cosine similarity
- **9,700+ Movies** — Covers a wide range of films from the MovieLens Latest Small dataset
- **Star Ratings** — Displays average community ratings for each movie
- **Similarity Scores** — Shows how closely each recommendation matches your pick
- **Beautiful UI** — Gradient cards, genre badges, and a responsive modern design

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **ML / NLP** | scikit-learn (CountVectorizer, Cosine Similarity) |
| **Data** | Pandas, MovieLens Latest Small |
| **Deployment** | Docker, Render |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+

### Setup

```bash
# Clone the repo
git clone https://github.com/Manoj-kabeer/movies-recommendation-system.git
cd movies-recommendation-system

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

---

## 📂 Project Structure

```
movies-recommendation-system/
├── app.py                 # Streamlit app — UI + recommendation logic
├── build_model.py         # Standalone model builder (offline/experimentation)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image for deployment
├── render.yaml            # Render deployment blueprint
├── .dockerignore          # Files excluded from Docker build
├── .gitignore             # Files excluded from Git
└── ml-latest-small/       # MovieLens dataset
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    ├── links.csv
    └── README.txt
```

---

## 🧠 How It Works

1. **Data Loading** — Reads movies, ratings, tags, and links from CSV files
2. **Feature Engineering** — Combines genres and user tags into a single text feature per movie
3. **Vectorization** — Converts text features into sparse vectors using `CountVectorizer`
4. **Similarity** — Computes cosine similarity on-the-fly for the selected movie against all others
5. **Ranking** — Returns the top 5 most similar movies with ratings and match percentages

---

## 📊 Dataset

This project uses the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset by GroupLens Research:

- **9,742 movies** • **100,836 ratings** • **3,683 tags**
- Collected from 610 users between 1996–2018

---

## 📄 License

This project is for educational and personal use. The MovieLens dataset is provided by [GroupLens Research](https://grouplens.org/) under their [terms of use](https://grouplens.org/datasets/movielens/).

---

<p align="center">
  Built with ❤️ using Streamlit
</p>
