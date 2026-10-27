"""Shows the 80/20 train/test split used by the models and saves screenshots of it to outputs/.

  python show_train_test_split.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src import ingest, recommender  # noqa: E402
from src.spark_session import PROJECT_DIR, get_spark  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "outputs")
DATA = os.path.join(PROJECT_DIR, "data", "ml-latest-small")


def show_string(df, n=8):
    """The exact text Spark's df.show() prints, captured as a string."""
    return df._jdf.showString(n, 40, False)


def console_png(text, title, path):
    """Saves console text as a terminal-style image."""
    lines = text.rstrip("\n").split("\n")
    width = max(len(line) for line in lines)
    fig_w = max(8, width * 0.085)
    fig_h = 0.6 + len(lines) * 0.2
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1e1e1e")
    fig.text(0.012, 1 - 0.18 / fig_h, title, color="#9cdcfe", fontsize=10, family="monospace",
             va="top", weight="bold")
    fig.text(0.012, 1 - 0.5 / fig_h, "\n".join(lines), color="#d4d4d4", fontsize=9.5,
             family="monospace", va="top", linespacing=1.25)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    spark = get_spark()
    ratings, movies = ingest.load_raw(spark, DATA)
    ratings, movies, _ = ingest.clean(ratings, movies)
    train, test = recommender.split(ratings)

    total = ratings.count()
    n_train, n_test = train.count(), test.count()
    tr_users, te_users = train.select("userId").distinct(), test.select("userId").distinct()
    tr_movies, te_movies = train.select("movieId").distinct(), test.select("movieId").distinct()
    cold_movies = te_movies.join(tr_movies, "movieId", "left_anti").count()
    cold_users = te_users.join(tr_users, "userId", "left_anti").count()
    cold_rows = test.join(tr_movies, "movieId", "left_anti").count()
    avg_tr = train.agg(F.avg("rating")).first()[0]
    avg_te = test.agg(F.avg("rating")).first()[0]
    liked_te = test.filter("rating >= 4").count()

    counts = f"""$ python show_train_test_split.py
Spark {spark.version} | master {spark.sparkContext.master}

Split method : DataFrame.randomSplit([0.8, 0.2], seed=42)

                     TRAIN          TEST         TOTAL
Ratings        {n_train:>10,}    {n_test:>10,}    {total:>10,}
Share          {n_train / total:>10.1%}    {n_test / total:>10.1%}    {1:>10.1%}
Users          {tr_users.count():>10,}    {te_users.count():>10,}
Movies         {tr_movies.count():>10,}    {te_movies.count():>10,}
Avg rating     {avg_tr:>10.3f}    {avg_te:>10.3f}

Overlap check
  train and test share no (user, movie) pair   : {train.join(test, ["userId", "movieId"]).count() == 0}
  test users not seen in training              : {cold_users}
  test movies not seen in training             : {cold_movies:,}
  test ratings dropped by ALS (cold start)     : {cold_rows:,}  -> {n_test - cold_rows:,} ratings evaluated
  test ratings >= 4 (used for Precision@10)    : {liked_te:,}
"""
    samples = ("TRAIN set - first 8 rows\n" + show_string(train.orderBy("userId", "movieId"))
               + "\nTEST set - first 8 rows\n" + show_string(test.orderBy("userId", "movieId")))

    print(counts)
    print(samples)
    with open(os.path.join(OUT, "split_summary.txt"), "w", encoding="utf-8") as f:
        f.write(counts + "\n" + samples)
    console_png(counts, "Train / test split - sizes and checks", os.path.join(OUT, "split_01_counts.png"))
    console_png(samples, "Train / test split - sample rows (df.show())", os.path.join(OUT, "split_02_samples.png"))

    # Rating distribution in each split, as percentages, to show the split is balanced.
    tr = train.groupBy("rating").count().toPandas().set_index("rating")["count"] / n_train * 100
    te = test.groupBy("rating").count().toPandas().set_index("rating")["count"] / n_test * 100
    keys = sorted(set(tr.index) | set(te.index))
    pd.DataFrame({"rating": keys, "train_pct": [round(tr.get(k, 0), 2) for k in keys],
                  "test_pct": [round(te.get(k, 0), 2) for k in keys]}).to_csv(
        os.path.join(OUT, "split_distribution.csv"), index=False)
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], [tr.get(k, 0) for k in keys], 0.4, label=f"Train ({n_train:,})", color="#2a6fdb")
    ax.bar([i + 0.2 for i in x], [te.get(k, 0) for k in keys], 0.4, label=f"Test ({n_test:,})", color="#e8833a")
    ax.set_xticks(list(x), [str(k) for k in keys])
    ax.set(title="Rating distribution: train vs test (80/20 split, seed 42)", xlabel="Rating", ylabel="% of ratings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "split_03_distribution.png"), dpi=130)
    plt.close(fig)

    spark.stop()
    print(f"Saved split screenshots to {OUT}")


if __name__ == "__main__":
    main()
