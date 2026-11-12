import gzip
import html
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="ReelMatch | Movie picks", page_icon=":material/movie:", layout="wide")

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
RED, GREEN, MUTED = "#E50914", "#1F8A3B", "#5F5F5F"


# ---------- data and model ----------
@st.cache_resource(show_spinner="Loading the model...")
def load_everything():
    with open(os.path.join(ART, "meta.json")) as f:
        meta = json.load(f)
    catalog = pd.read_csv(os.path.join(ART, "catalog.csv.gz"))
    viewers = pd.read_csv(os.path.join(ART, "viewers.csv.gz")).set_index("userId")
    lookup = pd.read_csv(os.path.join(ART, "movies_lookup.csv.gz")).set_index("movieId")
    item_vecs = np.load(os.path.join(ART, "item_vectors.npy"))
    viewer_vecs = np.load(os.path.join(ART, "viewer_vectors.npy"))
    with gzip.open(os.path.join(ART, "viewer_history.json.gz"), "rt") as f:
        history = {int(k): v for k, v in json.load(f).items()}
    model = XGBClassifier()
    model.load_model(os.path.join(ART, "recommender_xgb.json"))

    # content vector for "more like this": each block scaled to unit length, then stacked
    blocks = []
    for prefix in ["genre_svd_", "genome_svd_", "tag_svd_"]:
        cols = [c for c in catalog.columns if c.startswith(prefix)]
        if cols:
            blocks.append(catalog[cols].to_numpy(dtype=float))
    blocks.append(item_vecs.astype(float))
    normed = []
    for b in blocks:
        n = np.linalg.norm(b, axis=1, keepdims=True)
        normed.append(np.divide(b, n, out=np.zeros_like(b), where=n > 0))
    content = np.hstack(normed)
    n = np.linalg.norm(content, axis=1, keepdims=True)
    content = np.divide(content, n, out=np.zeros_like(content), where=n > 0)

    catalog["genre_list"] = catalog["genres"].fillna("").str.split("|")
    return meta, catalog, viewers, lookup, item_vecs, viewer_vecs, history, model, content


meta, catalog, viewers, lookup, item_vecs, viewer_vecs, history, model, content = load_everything()
FEATURES = meta["feature_cols"]


def tmdb_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.environ.get("TMDB_API_KEY", "")


def _one_poster(tmdb_id, key):
    if not key or pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
                         params={"api_key": key}, timeout=6)
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def posters_for(tmdb_ids: tuple):
    key = tmdb_key()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda t: _one_poster(t, key), tmdb_ids))


# ---------- scoring ----------
def score_viewer(user_id):
    """Run the hybrid SVD + XGBoost classifier for one viewer against the whole catalog."""
    v = viewers.loc[user_id]
    X = catalog.copy()
    X["userId"] = user_id
    for c in ["user_rating_count", "user_avg_rating", "user_bias"]:
        X[c] = v[c]
    for c in [c for c in viewers.columns if c.startswith("svd_user_")]:
        X[c] = v[c]
    X["svd_affinity"] = item_vecs @ viewer_vecs[int(v["row"])]
    for c in FEATURES:
        if c not in X.columns:
            X[c] = 0.0
    proba = model.predict_proba(X[FEATURES].fillna(0))[:, 1]
    out = catalog[["movieId", "title", "year", "genres", "genre_list", "tmdbId",
                   "movie_rating_count", "movie_avg_rating", "hidden_gem"]].copy()
    out["match"] = proba
    seen = set(history.get(user_id, {}).get("seen", []))
    return out[~out["movieId"].isin(seen)].sort_values("match", ascending=False)


def similar_to(movie_id, n):
    i = catalog.index[catalog["movieId"] == movie_id][0]
    sims = content @ content[i]
    sims[i] = -1
    top = np.argsort(sims)[::-1][:n]
    out = catalog.iloc[top][["movieId", "title", "year", "genres", "tmdbId"]].copy()
    out["sim"] = sims[top]
    return out


# ---------- look ----------
st.markdown(f"""
<style>
  .block-container {{padding-top: 2.2rem; max-width: 1400px;}}
  html, body, [class*="st-"] {{font-size: 17px;}}
  h1, h2, h3 {{letter-spacing: -0.3px;}}
  .rm-banner {{background:{RED}; padding:28px 32px; border-radius:6px; margin-bottom:24px; text-align:center;}}
  .rm-banner .kicker {{color:#FFFFFF; opacity:.85; font-weight:700; letter-spacing:2px; font-size:14px;}}
  .rm-banner .title {{color:#fff; font-size:40px; font-weight:800; margin:4px 0 6px 0; line-height:1.1;}}
  .rm-banner .sub {{color:#FFFFFF; font-size:18px;}}
  .rm-grid {{display:grid; grid-template-columns:repeat(auto-fill, minmax(160px, 1fr)); gap:18px; margin:8px 0 26px 0;}}
  .rm-grid.small {{grid-template-columns:repeat(8, minmax(0, 1fr)); gap:12px;}}
  .rm-card {{background:#FFFFFF; border:1px solid #E2E2E2; border-radius:6px; overflow:hidden; transition:transform .15s ease, box-shadow .15s ease;}}
  .rm-card:hover {{transform:scale(1.03); box-shadow:0 6px 16px rgba(0,0,0,.15);}}
  .rm-card img {{width:100%; aspect-ratio:2/3; object-fit:cover; display:block;}}
  .rm-ph {{width:100%; aspect-ratio:2/3; background:#221F1F; border-top:4px solid {RED};
           display:flex; align-items:center; justify-content:center; text-align:center;
           padding:12px; color:#E5E5E5; font-weight:600; font-size:14px;}}
  .rm-meta {{padding:8px 10px 11px 10px;}}
  .rm-name {{color:#221F1F; font-size:15px; font-weight:600; line-height:1.25;
             overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}}
  .rm-line {{font-size:13.5px; color:{MUTED}; margin-top:4px;}}
  .rm-match {{color:{GREEN}; font-weight:700;}}
  .rm-rank {{color:{RED}; font-weight:800; margin-right:4px;}}
  .rm-step {{background:#FFFFFF; border:1px solid #E2E2E2; border-left:4px solid {RED}; border-radius:6px; padding:16px 18px; height:100%; color:#221F1F;}}
  .rm-step b {{color:{RED}; font-size:22px; display:block;}}
  section[data-testid="stSidebar"] {{background:#221F1F;}}
  section[data-testid="stSidebar"] * {{color:#F5F5F1;}}
  .rm-logo {{color:{RED} !important; font-weight:900; font-size:26px; letter-spacing:1px;}}
</style>
""", unsafe_allow_html=True)


def banner(kicker, title, sub):
    st.markdown(f'<div class="rm-banner"><div class="kicker">{kicker}</div>'
                f'<div class="title">{title}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def nice_title(t):
    t = str(t)
    for art in [", The", ", A", ", An"]:
        if t.endswith(art):
            return art[2:] + " " + t[: -len(art)]
    return t


def card_grid(df, line_fn, small=False, ranked=False):
    urls = posters_for(tuple(df["tmdbId"].tolist()))
    cards = []
    for k, ((_, r), url) in enumerate(zip(df.iterrows(), urls), start=1):
        name = html.escape(nice_title(r["title"]))
        img = (f'<img src="{url}" loading="lazy" alt="{name}">' if url
               else f'<div class="rm-ph">{name}</div>')
        rank = f'<span class="rm-rank">{k}</span>' if ranked else ""
        year = "" if pd.isna(r.get("year")) else f" ({int(r['year'])})"
        cards.append(f'<div class="rm-card">{img}<div class="rm-meta">'
                     f'<div class="rm-name" title="{name}{year}">{rank}{name}</div>'
                     f'{line_fn(r)}</div></div>')
    cls = "rm-grid small" if small else "rm-grid"
    st.markdown(f'<div class="{cls}">{"".join(cards)}</div>', unsafe_allow_html=True)


def viewer_label(uid):
    return f"Viewer {uid}  ·  {int(viewers.loc[uid, 'user_rating_count'])} ratings in the sample"


# ---------- pages ----------
def home():
    banner("MSBA 315  ·  SECTION 1  ·  GROUP 5", "What should we watch tonight?",
           "Movie picks from our hybrid SVD + XGBoost model, trained on MovieLens 25M")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Movies in the catalog", f"{len(catalog):,}"),
        (c2, "Viewers you can try", f"{len(viewers):,}"),
        (c3, "Model AUC (validation)", f"{meta['metrics'].get('AUC', float('nan')):.3f}"),
        (c4, "Ratings the model learned from", f"{meta['n_ratings_sample']:,}"),
    ]:
        with col.container(border=True):
            st.metric(label, value)

    st.subheader("How to use it")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="rm-step"><b>1</b>Open <b style="display:inline;font-size:inherit">Picks for a viewer</b> '
                'and choose a viewer from the list, or press Random viewer.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="rm-step"><b>2</b>Look at what that viewer rated highly, then at the movies the '
                'model thinks they will like next, ranked by match.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="rm-step"><b>3</b>Narrow the list by genre or switch to hidden gems. '
                'Use <b style="display:inline;font-size:inherit">More like this</b> to start from one movie instead.</div>',
                unsafe_allow_html=True)

    st.subheader("Most loved in the catalog")
    top = catalog[catalog["movie_rating_count"] >= catalog["movie_rating_count"].quantile(0.75)]
    top = top.sort_values("movie_avg_rating", ascending=False).head(12)
    card_grid(top, lambda r: f'<div class="rm-line">★ {r["movie_avg_rating"]:.2f}  ·  '
                             f'{int(r["movie_rating_count"]):,} ratings</div>')


def picks():
    banner("PICKS FOR A VIEWER", "Recommended for you",
           "Pick a viewer and the model scores every movie they have not rated yet")

    ids = viewers.sort_values("user_rating_count", ascending=False).index.tolist()
    if "viewer" not in st.session_state:
        st.session_state.viewer = ids[0]

    def random_viewer():
        st.session_state.viewer = random.choice(ids)

    a, b, c = st.columns([3, 1, 2], vertical_alignment="bottom")
    uid = a.selectbox("Viewer", ids, key="viewer", format_func=viewer_label)
    b.button("Random viewer", on_click=random_viewer, use_container_width=True)
    n = c.slider("How many picks", 5, 20, 10)

    all_genres = sorted({g for gl in catalog["genre_list"] for g in gl if g and g != "(no genres listed)"})
    f1, f2 = st.columns([3, 1], vertical_alignment="bottom")
    genres = f1.multiselect("Only these genres (optional)", all_genres)
    gems = f2.toggle("Hidden gems only", help="Movies in the least-rated third of the catalog")

    loved = [m for m in history.get(uid, {}).get("loved", []) if m in lookup.index][:8]
    if loved:
        st.markdown(f"#### What viewer {uid} rated highly")
        lv = lookup.loc[loved].reset_index()
        card_grid(lv, lambda r: f'<div class="rm-line">{html.escape(str(r["genres"]).split("|")[0])}</div>',
                  small=True)

    with st.spinner("Scoring the catalog..."):
        recs = score_viewer(uid)
    if genres:
        recs = recs[recs["genre_list"].apply(lambda gl: any(g in gl for g in genres))]
    if gems:
        recs = recs[recs["hidden_gem"] == 1]
    recs = recs.head(n)

    st.markdown(f"#### Top {len(recs)} picks for viewer {uid}")
    if recs.empty:
        st.info("Nothing left after those filters. Try fewer genres or turn hidden gems off.")
        return
    card_grid(recs, lambda r: f'<div class="rm-line"><span class="rm-match">{min(round(r["match"]*100), 99)}% match</span>'
                              f'  ·  {html.escape(" / ".join(r["genre_list"][:2]))}</div>', ranked=True)

    with st.expander("See the numbers behind these picks"):
        show = recs[["title", "year", "genres", "match", "movie_avg_rating", "movie_rating_count"]].copy()
        show.columns = ["Title", "Year", "Genres", "P(will like)", "Avg rating", "Ratings"]
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"P(will like)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                    "Year": st.column_config.NumberColumn(format="%d"),
                                    "Avg rating": st.column_config.NumberColumn(format="%.2f")})


def more_like_this():
    banner("MORE LIKE THIS", "Start from a movie you already love",
           "Similar titles by genre, genome tags, user tags and how viewers rated them")
    order = catalog.sort_values("movie_rating_count", ascending=False)
    labels = {r.movieId: f"{nice_title(r.title)}" + ("" if pd.isna(r.year) else f" ({int(r.year)})")
              for r in order.itertuples()}
    mid = st.selectbox("Type a title", list(labels), format_func=labels.get)
    n = st.slider("How many", 5, 20, 10, key="n_like")

    left, right = st.columns([1, 4])
    with left:
        sel = catalog[catalog["movieId"] == mid]
        card_grid(sel, lambda r: f'<div class="rm-line">★ {r["movie_avg_rating"]:.2f}  ·  '
                                 f'{html.escape(str(r["genres"]).replace("|", ", "))}</div>')
    with right:
        st.markdown("#### Because you liked it")
        card_grid(similar_to(mid, n), lambda r: f'<div class="rm-line"><span class="rm-match">'
                                                f'{max(r["sim"], 0)*100:.0f}% similar</span></div>')


def how_it_works():
    banner("HOW IT WORKS", "Method and data", "What sits behind each pick")
    m = meta["metrics"]
    st.markdown(f"""
**Data.** MovieLens 25M ratings ({meta['n_ratings_sample']:,} ratings in our working sample), with genome tag
scores, user tags and TMDB links for posters.

**Model.** A hybrid classifier. Truncated SVD (K={meta['svd_k']}) on the full rating matrix gives every viewer
and movie a latent vector. We add the dot product of the two (svd_affinity), compressed SVD factors,
viewer and movie statistics, genre, genome and tag features, and train XGBoost to answer one question:
will this viewer rate this movie 4 stars or more?

**Ranking.** For the chosen viewer, the app scores every catalog movie they have not rated and sorts by the
predicted probability. That probability is the match % on each card.

**Catalog.** Movies with at least {meta['min_ratings']} ratings in the sample, so every pick has enough
history behind it. Hidden gems are the least-rated third of that catalog.
""")
    c1, c2, c3, c4 = st.columns(4)
    for col, k in zip([c1, c2, c3, c4], ["Accuracy", "F1_Weighted", "F1_Macro", "AUC"]):
        with col.container(border=True):
            st.metric(k.replace("_", " "), f"{m.get(k, float('nan')):.3f}")
    st.caption("Movie posters come from TMDB. "
               "This product uses the TMDB API but is not endorsed or certified by TMDB.")


# ---------- navigation ----------
with st.sidebar:
    st.markdown('<div class="rm-logo">REELMATCH</div>', unsafe_allow_html=True)
    st.caption("Section 1 · Group 5")

nav = st.navigation([
    st.Page(home, title="1. Home", icon=":material/home:", default=True),
    st.Page(picks, title="2. Picks for a viewer", icon=":material/recommend:", url_path="picks"),
    st.Page(more_like_this, title="3. More like this", icon=":material/movie:", url_path="more-like-this"),
    st.Page(how_it_works, title="4. How it works", icon=":material/menu_book:", url_path="how-it-works"),
])

with st.sidebar:
    st.divider()
    if not tmdb_key():
        st.caption("No TMDB key found, so cards show titles instead of posters. "
                   "Add TMDB_API_KEY in the app secrets.")

nav.run()