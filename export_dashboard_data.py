"""Saves top-10 recommendations for every user so the dashboard can run without Spark.

Retrains Model B with the tuned parameters from outputs/metrics.json on the same
seeded split, so the results match the pipeline.

  python export_dashboard_data.py
"""
import json
import os
import sys

from pyspark.sql import functions as F

from src import ingest, recommender
from src.spark_session import PROJECT_DIR, get_spark

OUT = os.path.join(PROJECT_DIR, "outputs")
DATA = os.path.join(PROJECT_DIR, "data", "ml-latest-small")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(OUT, "metrics.json")) as f:
        best = json.load(f)["model_b_top10"]["params"]

    spark = get_spark()
    ratings, movies = ingest.load_raw(spark, DATA)
    ratings, movies, _ = ingest.clean(ratings, movies)
    ratings.cache()
    train, _ = recommender.split(ratings)
    model, _ = recommender.train(train, best["rank"], best["regParam"], implicit=True, alpha=best["alpha"])

    # "Seen" uses all of a user's ratings, as in recommend_for_user.
    top = recommender._top_k_unseen(model.recommendForAllUsers(3000), ratings,
                                    recommender.eligible_movies(ratings), 10)
    recs = (top.join(movies.select("movieId", "title", "genres"), "movieId")
            .select("userId", F.col("pos").alias("rank"), "movieId", "title", "genres")
            .orderBy("userId", "rank").toPandas())
    path = os.path.join(OUT, "dashboard_recommendations.csv")
    recs.to_csv(path, index=False)
    print(f"Saved {len(recs):,} recommendations for {recs.userId.nunique()} users to {path}")
    spark.stop()


if __name__ == "__main__":
    main()
