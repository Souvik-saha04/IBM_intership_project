import ast
import os
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎞️", layout="wide")

N_RECS = 6                                   
MAX_MATCHES = 25                                             
BASE_DIR = Path(__file__).resolve().parent


                                                                               
                                                      
                                                                               
def find_csv(name):
    """Look for a dataset file next to this script or inside ./data."""
    for folder in (BASE_DIR, BASE_DIR / "data"):
        if (folder / name).exists():
            return folder / name
    return None


def parse_names(raw):
    """'[{"id": 28, "name": "Action"}, ...]' -> ['Action', ...]"""
    return [item["name"] for item in ast.literal_eval(raw)]


def get_director(raw):
    for person in ast.literal_eval(raw):
        if person["job"] == "Director":
            return person["name"]
    return ""


def squash(names):
    """Remove spaces so 'Sam Worthington' becomes one token, 'SamWorthington'."""
    return [n.replace(" ", "") for n in names if n]


@st.cache_resource(show_spinner="Preparing data and building the model (first run only)...")
def build_model(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)[["movie_id", "cast", "crew"]]

                                                                       
    df = movies.merge(credits, left_on="id", right_on="movie_id").drop(columns="movie_id")
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["year"] = df["release_date"].fillna("").str[:4]

    genres = df["genres"].apply(parse_names)
    keywords = df["keywords"].apply(parse_names)
    cast = df["cast"].apply(parse_names)
    df["director"] = df["crew"].apply(get_director)

                                             
    df["genre_list"] = genres
    df["cast_list"] = cast.apply(lambda c: c[:5])

                                                                   
    tags = (
        keywords.apply(squash)
        + df["overview"].str.split()
        + cast.apply(lambda c: squash(c[:3]))
        + df["director"].apply(lambda d: squash([d]))
        + genres.apply(squash)
    )

                                                                     
                                                                          
                                                                            
    stemmer = PorterStemmer()
    df["tags"] = tags.apply(
        lambda words: " ".join(
            stemmer.stem(w)
            for w in " ".join(words).lower().split()
            if w not in ENGLISH_STOP_WORDS
        )
    )

                                                                      
    matrix = CountVectorizer(max_features=5000, stop_words="english").fit_transform(df["tags"])

    keep = [
        "id", "title", "year", "vote_average", "runtime", "overview",
        "tagline", "popularity", "genre_list", "cast_list", "director",
    ]
    return df[keep].reset_index(drop=True), matrix


                                                                               
                                                   
                                                                               
def recommend(matrix, movie_idx, n=N_RECS):
    """Return [(row_index, similarity), ...] for the n movies most like movie_idx."""
    scores = cosine_similarity(matrix[movie_idx], matrix).ravel()
    scores[movie_idx] = -1                                    
    top = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in top]


def search_movies(df, query):
    """Case-insensitive title search; exact matches first, then by popularity."""
    hits = df[df["title"].str.contains(query, case=False, regex=False, na=False)].copy()
    hits["exact"] = hits["title"].str.lower() == query.lower()
    return hits.sort_values(["exact", "popularity"], ascending=False)


                                                                               
                                                    
                                                                               
def default_api_key():
    key = os.getenv("TMDB_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("TMDB_API_KEY", "")
        except Exception:
            key = ""
    return key


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(tmdb_id, api_key):
    """Return a poster URL for a TMDB movie id, or None if unavailable."""
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=5,
        )
        path = resp.json().get("poster_path") if resp.ok else None
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except (requests.RequestException, ValueError):
        return None


def show_poster(url):
    if url:
        st.image(url, use_container_width=True)
    else:
        st.markdown("<div style='text-align:center;font-size:4rem'>🎬</div>", unsafe_allow_html=True)


                                                                               
                                                                      
                                                                               
def header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎞️ Movie Recommender")
        st.caption("Search. Explore. Discover.")
    with col2:
        st.write("")
        st.write("Find your next favorite movie 🍿")
    st.divider()


def select_movie(idx, title):
    """Button callback: jump to a movie (also fills the search box with its title)."""
    st.session_state.query = title
    st.session_state.selected = idx


def reset_selection():
    """Search text changed -> forget the previously selected movie."""
    st.session_state.pop("selected", None)


def add_to_watchlist(idx):
    watchlist = st.session_state.setdefault("watchlist", [])
    if idx not in watchlist:
        watchlist.append(idx)


def clear_watchlist():
    st.session_state.watchlist = []


def label(df, idx):
    year = df.at[idx, "year"] or "N/A"
    return f"{df.at[idx, 'title']} ({year})"


def movie_card(df, idx, api_key, section, match=None):
    movie = df.loc[idx]
    show_poster(fetch_poster(int(movie["id"]), api_key))
    st.write(f"**{movie['title']}**")
    st.caption(movie["year"] or "N/A")
    st.write(f"⭐ {movie['vote_average']:.1f}")
    st.caption(", ".join(movie["genre_list"][:3]))
    if match is not None:
        st.caption(f"Match: {match:.0%}")
    st.button(
        "View details",
        key=f"view_{section}_{idx}",
        on_click=select_movie,
        args=(idx, movie["title"]),
        use_container_width=True,
    )


def movie_details(df, idx, api_key):
    movie = df.loc[idx]
    poster_col, info_col = st.columns([1, 3])
    with poster_col:
        show_poster(fetch_poster(int(movie["id"]), api_key))
    with info_col:
        st.subheader(f"{movie['title']} ({movie['year'] or 'N/A'})")
        runtime = f"{int(movie['runtime'])} min" if pd.notna(movie["runtime"]) and movie["runtime"] else "N/A"
        st.write(f"⭐ {movie['vote_average']:.1f}/10   🕐 {runtime}")
        st.write(", ".join(movie["genre_list"]))
        if movie["tagline"]:
            st.caption(f"*{movie['tagline']}*")
        st.write(movie["overview"] or "No overview available.")
        st.write(f"**Director:** {movie['director'] or 'N/A'}")
        st.write(f"**Stars:** {', '.join(movie['cast_list']) or 'N/A'}")

        trailer_col, watchlist_col = st.columns(2)
        with trailer_col:
            query = quote_plus(f"{movie['title']} {movie['year']} trailer")
            st.link_button(
                "▶ Watch Trailer",
                f"https://www.youtube.com/results?search_query={query}",
                use_container_width=True,
            )
        with watchlist_col:
            st.button(
                "+ Add to Watchlist",
                key=f"watch_{idx}",
                on_click=add_to_watchlist,
                args=(idx,),
                use_container_width=True,
            )


                                                                               
        
                                                                               
def main():
    header()

    movies_path = find_csv("tmdb_5000_movies.csv")
    credits_path = find_csv("tmdb_5000_credits.csv")
    if movies_path is None or credits_path is None:
        st.error(
            "Dataset files not found. Download **tmdb_5000_movies.csv** and "
            "**tmdb_5000_credits.csv** from "
            "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place them next to this file (or in a `data` folder)."
        )
        st.stop()

    df, matrix = build_model(str(movies_path), str(credits_path))

                                                        
    with st.sidebar:
        api_key = st.text_input(
            "TMDB API key (optional, for posters)",
            value=default_api_key(),
            type="password",
            help="Free key from themoviedb.org. Without it the app still works, just without posters.",
        )
        st.header("📌 My Watchlist")
        watchlist = st.session_state.get("watchlist", [])
        if watchlist:
            for i in watchlist:
                st.write(f"• {label(df, i)}")
            st.button("Clear watchlist", on_click=clear_watchlist)
        else:
            st.caption("Nothing saved yet.")

                      
    st.text_input(
        "🔍 Search for a movie",
        key="query",
        placeholder="e.g. Avatar, Batman, Inception",
        on_change=reset_selection,
    )
    query = st.session_state.query.strip()

    if not query:
        st.subheader("Popular picks")
        popular = df.sort_values("popularity", ascending=False).head(N_RECS).index
        for column, idx in zip(st.columns(N_RECS), popular):
            with column:
                movie_card(df, idx, api_key, "popular")
    else:
        matches = search_movies(df, query)
        if matches.empty:
            st.warning(f'No movies found for "{query}". Try a different spelling or a shorter title.')
        else:
            st.subheader("Search Results")
            st.caption(f'Found {len(matches)} result(s) for "{query}"')
            selected = st.selectbox(
                "Select a movie",
                matches.index.tolist()[:MAX_MATCHES],
                key="selected",
                format_func=lambda i: label(df, i),
            )

            movie_details(df, selected, api_key)
            st.divider()

            st.subheader("Recommended Movies")
            recs = recommend(matrix, selected)
            for column, (idx, score) in zip(st.columns(N_RECS), recs):
                with column:
                    movie_card(df, idx, api_key, "rec", match=score)

    st.divider()
    st.caption(
        "Data: TMDB 5000 Movie Dataset."
    )


main()