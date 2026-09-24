# 🎞️ Movie Recommendation System

A content-based movie recommender with a Streamlit web interface. Search for a movie you like and the app suggests the most similar movies from the TMDB 5000 dataset.

## Dataset
**TMDB 5000 Movie Dataset** (Kaggle): https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Files needed: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` (4,803 movies).

## Project Description
The app builds a "tags" text profile for every movie by combining its **keywords, plot overview, top-3 cast, director and genres**. The text is cleaned (lower-cased, stop words removed, stemmed), converted to a bag-of-words matrix (`CountVectorizer`, 5,000 features), and movies are compared with **cosine similarity**. The six most similar movies are shown as recommendations, each with a match percentage.

**Features**
- Search by title, with a dropdown when several movies match (e.g. the two "Batman" films)
- Movie details: rating, runtime, genres, overview, director, stars, trailer link
- Six recommendations with match score; click any to explore further
- Watchlist in the sidebar
- Optional posters via the TMDB API (the app works without them)

## Technologies Used
Python 3.10+, Streamlit, pandas, NumPy, scikit-learn, NLTK (PorterStemmer), requests, TMDB API (optional, posters)

## Setup & Run
1. Install Python 3.10 or newer.
2. Put these files in one folder:
   - `YourName_MovieRecommender.py`
   - `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` (in the same folder or in a `data/` sub-folder)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run YourName_MovieRecommender.py
   ```
5. *(Optional)* For posters, get a free API key from https://www.themoviedb.org/settings/api and paste it into the sidebar box, or set the `TMDB_API_KEY` environment variable.

The first launch takes around 20 seconds while the model is built; after that it is cached.

## Key Information
- Everything (data preparation, model, and UI) is in a single `.py` file. No pre-built pickle files are needed.
- Movies are merged on their unique `id`, so there are no duplicate rows.
- The movie itself is never recommended to itself.
- This product uses the TMDB API but is not endorsed or certified by TMDB.

## Author
Your Name