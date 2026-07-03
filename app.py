"""
MovieLens Ratings Analytics — Streamlit app
Rebuilds the notebook's SQL marts, charts, and simple recommender/predictor
as a live, filterable web app.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import zipfile
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieLens Ratings Analytics",
    page_icon="🎬",
    layout="wide",
)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

NAVY = "#1E2761"
GOLD = "#E8B94E"
ICE = "#CADCFC"


# --------------------------------------------------------------------------
# Data loading (cached — only runs once per session / until cache cleared)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading and loading MovieLens data…")
def load_data():
    data = urllib.request.urlopen(MOVIELENS_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    with z.open("ml-latest-small/ratings.csv") as f:
        ratings = pd.read_csv(f)
    with z.open("ml-latest-small/movies.csv") as f:
        movies = pd.read_csv(f)
    return ratings, movies


@st.cache_resource(show_spinner=False)
def get_connection(ratings: pd.DataFrame, movies: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("ratings", ratings)
    con.register("movies", movies)
    return con


with st.spinner("Setting up…"):
    ratings_df, movies_df = load_data()
    con = get_connection(ratings_df, movies_df)

all_genres = sorted(
    {g for genres in movies_df["genres"].dropna() for g in genres.split("|") if g != "(no genres listed)"}
)

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("🎬 Filters")
min_ratings = st.sidebar.slider("Minimum number of ratings", 5, 300, 50, step=5)
genre_filter = st.sidebar.selectbox("Genre", ["All genres"] + all_genres, index=0)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: MovieLens `ml-latest-small` — {len(ratings_df):,} ratings, "
    f"{movies_df['movieId'].nunique():,} movies, {ratings_df['userId'].nunique():,} users."
)
st.sidebar.caption("Source: GroupLens Research (files.grouplens.org)")

genre_clause = ""
if genre_filter != "All genres":
    genre_clause = f"AND list_contains(string_split(m.genres, '|'), '{genre_filter}')"

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🎬 MovieLens Ratings Analytics")
st.caption("From raw ratings to recommendations — an interactive version of the analysis notebook.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ratings loaded", f"{len(ratings_df):,}")
c2.metric("Movies", f"{movies_df['movieId'].nunique():,}")
c3.metric("Users", f"{ratings_df['userId'].nunique():,}")
c4.metric("Avg rating (all)", f"{ratings_df['rating'].mean():.2f}")

# --------------------------------------------------------------------------
# Data quality panel
# --------------------------------------------------------------------------
with st.expander("✅ Data quality checks"):
    dup = con.sql("""
        SELECT COUNT(*) total, COUNT(DISTINCT userId||'-'||movieId||'-'||timestamp) unique_keys
        FROM ratings
    """).df()
    nulls = con.sql("""
        SELECT SUM(userId IS NULL) null_user, SUM(movieId IS NULL) null_movie,
               SUM(timestamp IS NULL) null_ts
        FROM ratings
    """).df()
    qa1, qa2 = st.columns(2)
    qa1.write("**Duplicate-key check**")
    qa1.dataframe(dup, use_container_width=True, hide_index=True)
    qa2.write("**Null-key check**")
    qa2.dataframe(nulls, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_top, tab_genre, tab_users, tab_trend, tab_recommend, tab_predict = st.tabs(
    ["🏆 Top Movies", "🎭 Genre Leaderboard", "👥 User Activity",
     "📈 Trends", "🔗 Recommender", "🤖 Predict"]
)

# ---- Tab: Top movies -------------------------------------------------
with tab_top:
    st.subheader(f"Highest-rated movies (min. {min_ratings} ratings)")
    query = f"""
        SELECT m.title, COUNT(*) AS num_ratings, ROUND(AVG(r.rating),2) AS avg_rating
        FROM ratings r JOIN movies m USING (movieId)
        WHERE 1=1 {genre_clause}
        GROUP BY m.title
        HAVING COUNT(*) >= {min_ratings}
        ORDER BY avg_rating DESC, num_ratings DESC
        LIMIT 20
    """
    top_movies = con.sql(query).df()
    if top_movies.empty:
        st.info("No movies match this filter combination — try lowering the minimum ratings.")
    else:
        fig = px.bar(
            top_movies.sort_values("avg_rating"),
            x="avg_rating", y="title", orientation="h",
            text="avg_rating", color_discrete_sequence=[GOLD],
            labels={"avg_rating": "Average rating", "title": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=550, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_movies, use_container_width=True, hide_index=True)

# ---- Tab: Genre leaderboard -------------------------------------------
with tab_genre:
    st.subheader("Best-rated movie within each genre")
    genre_rank_query = """
        WITH movie_ratings AS (
            SELECT m.title, genre, COUNT(*) AS rating_count, AVG(r.rating) AS avg_rating
            FROM ratings r
            JOIN movies m USING (movieId),
                 UNNEST(STRING_SPLIT(m.genres, '|')) AS g(genre)
            WHERE genre != '(no genres listed)'
            GROUP BY m.title, genre
            HAVING COUNT(*) >= 20
        )
        SELECT genre, title, rating_count, ROUND(avg_rating,2) AS avg_rating,
               RANK() OVER (PARTITION BY genre ORDER BY avg_rating DESC) AS genre_rank
        FROM movie_ratings
        QUALIFY genre_rank <= 5
        ORDER BY genre, genre_rank
    """
    genre_rankings = con.sql(genre_rank_query).df()
    genres_available = sorted(genre_rankings["genre"].unique())
    picked = st.multiselect("Show genres", genres_available, default=genres_available[:6])
    view = genre_rankings[genre_rankings["genre"].isin(picked)] if picked else genre_rankings
    st.dataframe(view, use_container_width=True, hide_index=True)

# ---- Tab: User activity -------------------------------------------
with tab_users:
    st.subheader("Rating activity by user")
    user_activity = con.sql("""
        SELECT userId, COUNT(*) AS ratings_given, ROUND(AVG(rating),2) AS avg_rating_given,
               RANK() OVER (ORDER BY COUNT(*) DESC) AS activity_rank,
               NTILE(4) OVER (ORDER BY COUNT(*)) AS activity_quartile
        FROM ratings GROUP BY userId
    """).df()

    colA, colB = st.columns([1, 1])
    with colA:
        st.write("**Top raters**")
        st.dataframe(user_activity.sort_values("activity_rank").head(10),
                     use_container_width=True, hide_index=True)
    with colB:
        quartile_counts = user_activity["activity_quartile"].value_counts().sort_index()
        fig2 = px.pie(
            names=[f"Q{i}" for i in quartile_counts.index],
            values=quartile_counts.values,
            color_discrete_sequence=[ICE, "#9AB6E8", "#5B7FCB", NAVY],
            hole=0.5,
        )
        fig2.update_layout(title="Users by activity quartile", height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ---- Tab: Trends -------------------------------------------
with tab_trend:
    st.subheader("Ratings over time")
    monthly = con.sql("""
        WITH monthly AS (
            SELECT DATE_TRUNC('month', TO_TIMESTAMP(timestamp)) AS month, COUNT(*) AS ratings_count
            FROM ratings GROUP BY month
        )
        SELECT month, ratings_count,
               SUM(ratings_count) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_ratings
        FROM monthly ORDER BY month
    """).df()
    fig3 = px.line(monthly, x="month", y="cumulative_ratings",
                    labels={"month": "Month", "cumulative_ratings": "Cumulative ratings"},
                    color_discrete_sequence=[NAVY])
    fig3.update_layout(height=450, plot_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

# ---- Tab: Recommender -------------------------------------------
with tab_recommend:
    st.subheader("\"More like this\" — co-rated movies")
    movie_choice = st.selectbox("Pick a movie", sorted(movies_df["title"].unique()))
    movie_id_row = movies_df.loc[movies_df["title"] == movie_choice, "movieId"]
    if not movie_id_row.empty:
        mid = int(movie_id_row.iloc[0])
        similar = con.sql(f"""
            SELECT m.title, COUNT(*) AS shared_raters
            FROM ratings a
            JOIN ratings b ON a.userId = b.userId AND a.movieId != b.movieId
            JOIN movies m ON m.movieId = b.movieId
            WHERE a.movieId = {mid}
            GROUP BY m.title
            ORDER BY shared_raters DESC
            LIMIT 10
        """).df()
        st.dataframe(similar, use_container_width=True, hide_index=True)

# ---- Tab: Predict -------------------------------------------
with tab_predict:
    st.subheader("Can rating volume predict rating quality?")
    st.caption("Random Forest vs. a mean-baseline — trained live on the current dataset.")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    feats = con.sql("""
        SELECT m.movieId, COUNT(*) AS n_ratings,
               LENGTH(m.genres) - LENGTH(REPLACE(m.genres,'|','')) + 1 AS n_genres,
               AVG(r.rating) AS avg_rating
        FROM ratings r JOIN movies m USING (movieId)
        GROUP BY m.movieId, m.genres
        HAVING COUNT(*) >= 10
    """).df()

    X = feats[["n_ratings", "n_genres"]]
    y = feats["avg_rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    m1, m2 = st.columns(2)
    m1.metric("Baseline MAE", f"{mean_absolute_error(y_test, baseline_pred):.3f}")
    m1.metric("Baseline R²", f"{r2_score(y_test, baseline_pred):.3f}")
    m2.metric("Random Forest MAE", f"{mean_absolute_error(y_test, pred):.3f}")
    m2.metric("Random Forest R²", f"{r2_score(y_test, pred):.3f}")

    st.info(
        "In this dataset, rating count and genre count alone rarely beat a simple mean-baseline — "
        "a useful, honest finding: don't use rating volume as a stand-in for quality."
    )

    fig4 = px.scatter(x=y_test, y=pred, labels={"x": "Actual avg rating", "y": "Predicted avg rating"})
    fig4.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                    line=dict(dash="dash", color="gray"))
    fig4.update_layout(height=450)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption("Built with DuckDB, pandas, scikit-learn, Plotly and Streamlit — data © GroupLens Research.")
