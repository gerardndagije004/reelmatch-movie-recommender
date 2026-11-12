# ReelMatch: movie recommender

Hybrid SVD + XGBoost classifier trained on MovieLens 25M. Posters from TMDB.

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

The TMDB key goes in `.streamlit/secrets.toml` as `TMDB_API_KEY = "..."` (not committed).
