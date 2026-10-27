"""Streamlit dashboard for the movie recommendation project.

Reads the saved pipeline results (outputs/) and the MovieLens files, so it runs
without Spark or Java.

  streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "ml-latest-small"

st.set_page_config(page_title="Movie Recommender · Spark", page_icon="🎬", layout="wide")


@st.cache_data
def load():
    ratings = pd.read_csv(DATA / "ratings.csv")
    movies = pd.read_csv(DATA / "movies.csv")
    recs = pd.read_csv(OUT / "dashboard_recommendations.csv")
    metrics = json.loads((OUT / "metrics.json").read_text())
    tables = {name: pd.read_csv(OUT / f"{name}.csv") for name in (
        "graph_pagerank_movies", "graph_popular_movies", "graph_user_communities",
        "graph_movies_liked_together", "genre_stats", "split_distribution")}
    return ratings, movies, recs, metrics, tables


def bar(df, x, y, title, color="#2E86AB", horizontal=True, fmt=",.0f"):
    if horizontal:
        enc = dict(x=alt.X(f"{x}:Q", title=None),
                   y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)))
    else:
        enc = dict(x=alt.X(f"{y}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{x}:Q", title=None))
    base = alt.Chart(df, title=title).encode(**enc, tooltip=[y, alt.Tooltip(x, format=fmt)])
    # Mid-grey labels stay readable on both light and dark themes.
    labels = base.mark_text(align="left" if horizontal else "center", dx=4 if horizontal else 0,
                            dy=0 if horizontal else -8, color="#8D99AE").encode(text=alt.Text(f"{x}:Q", format=fmt))
    return (base.mark_bar(color=color) + labels).properties(height=max(260, 34 * len(df)) if horizontal else 320)


ratings, movies, recs, M, T = load()
info = M["data"]
a, b = M["model_a_rating_prediction"], M["model_b_top10"]

st.title("🎬 Distributed Movie Recommendation and User Analysis")
st.caption("Apache Spark 4.2 · PySpark · GraphFrames · Spark MLlib (ALS) · MovieLens  —  "
           "Nitesh Kumar · Roll 43 · Reg. 12306652 · Cluster Computing (INT315)")

overview, recommend, graph, models, split = st.tabs(
    ["Overview", "Recommend movies", "Graph insights", "Models & scalability", "Train / test split"])

with overview:
    c = st.columns(4)
    c[0].metric("Ratings", f"{info['ratings']:,}")
    c[1].metric("Users", f"{info['users']:,}")
    c[2].metric("Rated movies", f"{info['movies']:,}")
    c[3].metric("Sparsity", f"{info['sparsity_pct']}%", help="Share of the user × movie grid with no rating")

    c = st.columns(4)
    c[0].metric("Model A RMSE", a["rmse"], f"{a['rmse'] - a['baseline_global_mean_rmse']:.3f} vs average baseline", delta_color="inverse")
    p, pop = b["implicit_als"]["precision@10"], b["popularity_baseline"]["precision@10"]
    c[1].metric("Model B precision@10", f"{p:.3f}", f"+{(p / pop - 1) * 100:.0f}% vs popularity")
    c[2].metric("User communities", M["graph_user_similarity"]["communities"])
    c[3].metric("Pipeline run time", f"{sum(M['stage_seconds'].values()) / 60:.1f} min")

    left, right = st.columns(2)
    dist = ratings.rating.value_counts().sort_index().rename_axis("rating").reset_index(name="count")
    dist["rating"] = dist.rating.astype(str)
    left.altair_chart(bar(dist, "count", "rating", "Ratings per star value", horizontal=False))
    right.altair_chart(bar(T["genre_stats"].head(10), "numRatings", "genre", "Ratings per genre (top 10)"))

with recommend:
    st.subheader("Top-10 recommendations from Model B (implicit ALS)")
    user = st.number_input("User ID", min_value=int(ratings.userId.min()), max_value=int(ratings.userId.max()), value=1, step=1)
    mine = ratings[ratings.userId == user].merge(movies, on="movieId")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**User {user} rated {len(mine)} movies.** Their favourites:")
        fav = mine.sort_values(["rating", "timestamp"], ascending=False).head(10)
        st.dataframe(fav[["title", "genres", "rating"]], hide_index=True, width="stretch")
    with right:
        st.markdown("**Recommended** (unseen, each with ≥ 20 ratings):")
        st.dataframe(recs[recs.userId == user][["rank", "title", "genres"]], hide_index=True, width="stretch")
    liked = mine[mine.rating >= 4].genres.str.split("|").explode().value_counts().head(5)
    if len(liked):
        st.caption("Genres this user likes most: " + ", ".join(liked.index))

with graph:
    g, s = M["graph_user_movie"], M["graph_user_similarity"]
    c = st.columns(4)
    c[0].metric("User–movie graph", f"{g['vertices']:,} vertices", f"{g['edges']:,} edges", delta_color="off")
    c[1].metric("User k-NN graph", f"{s['user_vertices']} users", f"{s['similarity_edges']:,} edges", delta_color="off")
    c[2].metric("Communities", s["communities"])
    c[3].metric("Connected components", s["components"])

    left, right = st.columns(2)
    left.altair_chart(bar(T["graph_pagerank_movies"].head(10), "pagerank", "title", "Most influential movies (PageRank)",
                          color="#3A9E5F", fmt=".1f"))
    left.markdown("**Movies most often liked together**")
    left.dataframe(T["graph_movies_liked_together"].head(10), hide_index=True, width="stretch")
    comm = T["graph_user_communities"].assign(label=lambda d: "C" + d.community.astype(str) + " " + d.topGenre)
    right.altair_chart(bar(comm, "users", "label", "Largest user communities (label = most over-liked genre)",
                           horizontal=False))
    right.dataframe(comm[["community", "users", "topGenre", "genreLift"]], hide_index=True, width="stretch")
    right.caption("Lift = how much more a community likes the genre than the average user.")

with models:
    left, right = st.columns(2)
    rmse = pd.DataFrame({"model": ["Global average", "Movie average", "ALS (Model A)"],
                         "rmse": [a["baseline_global_mean_rmse"], a["baseline_movie_mean_rmse"], a["rmse"]]})
    left.altair_chart(bar(rmse, "rmse", "model", "Model A: test RMSE (lower is better)", horizontal=False, fmt=".3f"))
    rank = pd.DataFrame([{"model": name, "metric": k, "value": v}
                         for name, key in (("Popularity baseline", "popularity_baseline"), ("Explicit ALS (A)", "explicit_als"),
                                           ("Implicit ALS (B)", "implicit_als"))
                         for k, v in b[key].items()])
    right.altair_chart(alt.Chart(rank, title="Model B: top-10 quality (higher is better)").mark_bar().encode(
        x=alt.X("model:N", title=None, sort=["Popularity baseline", "Explicit ALS (A)", "Implicit ALS (B)"],
                axis=alt.Axis(labels=False, ticks=False)), y=alt.Y("value:Q", title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=["Popularity baseline", "Explicit ALS (A)", "Implicit ALS (B)"],
                                                   range=["#8D99AE", "#2E86AB", "#3A9E5F"]), legend=alt.Legend(orient="top", title=None)),
        column=alt.Column("metric:N", title=None), tooltip=["model", "metric", alt.Tooltip("value", format=".3f")]
    ).properties(width=120, height=280))

    left, right = st.columns(2)
    cores = pd.DataFrame(M["scaling_cores"]).melt(id_vars="cores", value_vars=["load_clean_s", "graph_s", "als_train_s"],
                                                 var_name="stage", value_name="seconds")
    cores["stage"] = cores.stage.map({"load_clean_s": "Load + clean", "graph_s": "Graph", "als_train_s": "ALS training"})
    left.altair_chart(alt.Chart(cores, title="Execution time by stage vs cores").mark_bar().encode(
        x=alt.X("cores:O", title="Local cores", axis=alt.Axis(labelAngle=0)), y=alt.Y("seconds:Q", title="Seconds"),
        color=alt.Color("stage:N", scale=alt.Scale(domain=["Load + clean", "Graph", "ALS training"], range=["#8D99AE", "#2E86AB", "#E8A33D"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["cores", "stage", alt.Tooltip("seconds", format=".1f")]).properties(height=320))
    data = pd.DataFrame(M["scaling_data"]).assign(share=lambda d: (d.fraction * 100).astype(int).astype(str) + "%")
    right.altair_chart(alt.Chart(data, title="Test RMSE vs share of data used").mark_line(point=True, color="#C8553D").encode(
        x=alt.X("share:N", sort=None, title="Share of ratings used", axis=alt.Axis(labelAngle=0)), y=alt.Y("rmse:Q", scale=alt.Scale(zero=False), title="RMSE"),
        tooltip=["share", "ratings", "rmse", "train_seconds"]).properties(height=320))

    st.markdown("**Seconds per pipeline stage**")
    stages = pd.DataFrame(list(M["stage_seconds"].items()), columns=["stage", "seconds"])
    st.altair_chart(bar(stages, "seconds", "stage", "", fmt=".1f"))

with split:
    st.markdown("80 / 20 random split with seed 42, the same split used to train and test both models.")
    left, right = st.columns([3, 2])
    left.image(str(OUT / "split_01_counts.png"), caption="Console output of show_train_test_split.py")
    right.image(str(OUT / "split_02_samples.png"), caption="First rows of each set (df.show())")
    sd = T["split_distribution"].melt(id_vars="rating", var_name="set", value_name="pct")
    sd["set"] = sd.set.map({"train_pct": "Train", "test_pct": "Test"})
    sd["rating"] = sd.rating.astype(str)
    st.altair_chart(alt.Chart(sd, title="% of ratings per star value: train vs test").mark_bar().encode(
        x=alt.X("set:N", title=None, sort=["Train", "Test"], axis=alt.Axis(labels=False, ticks=False)), y=alt.Y("pct:Q", title="% of ratings"),
        color=alt.Color("set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#2E86AB", "#E8A33D"]), legend=alt.Legend(orient="top", title=None)),
        column=alt.Column("rating:N", title="Rating"), tooltip=["rating", "set", "pct"]).properties(width=95, height=260))
