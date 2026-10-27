"""Charts for the report. Every function writes one PNG into the output folder."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN = "#2a6fdb", "#e8833a", "#3a9e5f"


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def rating_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["rating"].astype(str), df["count"], color=BLUE)
    ax.set(title="Rating distribution", xlabel="Rating", ylabel="Number of ratings")
    return _save(fig, out_dir, "01_rating_distribution.png")


def genre_popularity(df, out_dir):
    df = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["genre"], df["numRatings"], color=BLUE)
    ax.set(title="Ratings per genre", xlabel="Number of ratings")
    return _save(fig, out_dir, "02_genre_popularity.png")


def popular_movies(df, out_dir):
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(df["title"], df["numRatings"], color=BLUE, label="All ratings (in-degree)")
    ax.barh(df["title"], df["likes"], color=ORANGE, label="Liked (rating >= 4)")
    ax.set(title="Most connected movies in the user-movie graph", xlabel="Edges")
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "03_graph_popular_movies.png")


def pagerank_movies(df, out_dir):
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(df["title"], df["pagerank"], color=GREEN)
    ax.set(title="Most influential movies (PageRank)", xlabel="PageRank score")
    return _save(fig, out_dir, "04_graph_pagerank_movies.png")


def communities(df, out_dir):
    labels = [f"C{c}\n{g}" for c, g in zip(df["community"], df["topGenre"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, df["users"], color=BLUE)
    ax.set(title="Largest user communities (Label Propagation) and their most over-liked genre",
           ylabel="Users")
    return _save(fig, out_dir, "05_graph_communities.png")


def tuning(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    for reg in sorted({r["regParam"] for r in results}):
        pts = sorted((r["rank"], r["val_rmse"]) for r in results if r["regParam"] == reg)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"regParam={reg}")
    ax.set(title="ALS hyper-parameter tuning", xlabel="Rank (latent factors)", ylabel="Validation RMSE")
    ax.legend()
    return _save(fig, out_dir, "06_als_tuning.png")


def model_vs_baselines(metrics, out_dir):
    names = ["Global mean", "Movie mean", "ALS"]
    vals = [metrics["baseline_global_mean_rmse"], metrics["baseline_movie_mean_rmse"], metrics["rmse"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=[ORANGE, ORANGE, GREEN])
    ax.bar_label(bars, fmt="%.3f")
    ax.set(title="Test RMSE (lower is better)", ylabel="RMSE")
    return _save(fig, out_dir, "07_rmse_vs_baselines.png")


def ranking_comparison(pop, explicit, implicit, out_dir):
    names = list(implicit.keys())
    x = range(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (label, m, color) in enumerate((("Popularity baseline", pop, ORANGE),
                                           ("Explicit ALS (model A)", explicit, BLUE),
                                           ("Implicit ALS (model B)", implicit, GREEN))):
        bars = ax.bar([p + (i - 1) * w for p in x], [m[n] for n in names], w, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(list(x), names)
    ax.set(title="Top-10 recommendation quality on the test set (higher is better)", ylabel="Score")
    ax.legend()
    return _save(fig, out_dir, "08_top10_ranking_quality.png")


def predicted_vs_actual(pdf, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = pdf.groupby("rating")["prediction"]
    ratings = sorted(groups.groups.keys())
    ax.boxplot([groups.get_group(r) for r in ratings], tick_labels=[str(r) for r in ratings], showfliers=False)
    ax.plot(range(1, len(ratings) + 1), ratings, "r--", label="Perfect prediction")
    ax.set(title="Predicted vs actual rating (test set)", xlabel="Actual rating", ylabel="Predicted rating")
    ax.legend()
    return _save(fig, out_dir, "08b_predicted_vs_actual.png")


def data_scaling(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [r["ratings"] for r in rows]
    ax.plot(x, [r["train_seconds"] for r in rows], marker="o", color=BLUE)
    ax.set(title="ALS training time vs data size", xlabel="Number of ratings", ylabel="Training time (s)")
    ax2 = ax.twinx()
    ax2.plot(x, [r["rmse"] for r in rows], marker="s", color=ORANGE)
    ax2.set_ylabel("Test RMSE", color=ORANGE)
    return _save(fig, out_dir, "09_scaling_data_size.png")


def core_scaling(rows, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    cores = [str(r["cores"]) for r in rows]
    bottom = [0] * len(rows)
    for key, label, color in (("load_clean_s", "Load + clean", BLUE), ("graph_s", "Graph", GREEN),
                              ("als_train_s", "ALS training", ORANGE)):
        vals = [r[key] for r in rows]
        a1.bar(cores, vals, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    a1.set(title="Execution time by stage", xlabel="Local cores", ylabel="Seconds")
    a1.legend()
    a2.plot([r["cores"] for r in rows], [r["speedup"] for r in rows], marker="o", label="Measured")
    a2.plot([r["cores"] for r in rows], [r["cores"] for r in rows], "--", color="grey", label="Ideal (linear)")
    a2.set(title="Speed-up vs 1 core", xlabel="Local cores", ylabel="Speed-up")
    a2.legend()
    return _save(fig, out_dir, "10_scaling_cores.png")
