# MovieLens Ratings Analytics — Streamlit App

Interactive version of the notebook: live SQL marts (DuckDB), a genre leaderboard,
a user-activity view, a ratings trend chart, a co-rated "more like this" recommender,
and a live baseline-vs-Random-Forest prediction check.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the MovieLens `ml-latest-small` dataset (~1MB) from
GroupLens and caches it — subsequent runs are instant.

## Deploy for free

**Streamlit Community Cloud** (fastest path):
1. Push `app.py` and `requirements.txt` to a public GitHub repo.
2. Go to share.streamlit.io → "New app" → point it at the repo/`app.py`.
3. Done — you get a shareable URL.

## Using Power BI or Looker Studio instead

This app is the "real, live" option, but the same marts work as static
dashboards too. To use Power BI or Looker Studio:

1. Run the app once, or run the notebook, to materialize the marts.
2. Export each mart to CSV, e.g. in a Python/DuckDB session:
   ```python
   top_movies.to_csv("top_movies.csv", index=False)
   genre_rankings.to_csv("genre_rankings.csv", index=False)
   user_activity.to_csv("user_activity.csv", index=False)
   monthly_trends.to_csv("monthly_trends.csv", index=False)
   ```
3. **Power BI**: Home → Get Data → Text/CSV → import each file → build
   visuals (bar chart for top movies, matrix for genre leaderboard, pie
   for activity quartiles, line chart for cumulative trend).
4. **Looker Studio**: upload the CSVs to a Google Sheet, then in Looker
   Studio choose Create → Report → Google Sheets as the data source, and
   build the same chart set.

Power BI and Looker Studio both live in your own Microsoft/Google account,
so building the dashboard there has to happen in your browser — this
export step is the bridge from the analysis to either tool.
