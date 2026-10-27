"""ALS models: A (explicit) predicts ratings, B (implicit) ranks top-10 lists."""
import time

from pyspark.ml.evaluation import RankingEvaluator, RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import Window
from pyspark.sql import functions as F

SEED = 42
# Only movies with at least this many training ratings are recommended. Without
# it, ALS puts obscure movies with a single 5-star rating at the top.
MIN_MOVIE_RATINGS = 20


def split(ratings, test_fraction=0.2):
    train, test = ratings.select("userId", "movieId", "rating").randomSplit([1 - test_fraction, test_fraction], seed=SEED)
    return train.cache(), test.cache()


def clip(df, col="prediction"):
    """ALS can predict outside the 0.5-5 rating scale; clip to valid ratings."""
    return df.withColumn(col, F.least(F.greatest(F.col(col), F.lit(0.5)), F.lit(5.0)))


def make_als(rank=10, reg=0.1, max_iter=10, implicit=False, alpha=1.0):
    return ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=rank, regParam=reg,
               maxIter=max_iter, implicitPrefs=implicit, alpha=alpha, nonnegative=not implicit,
               coldStartStrategy="drop", seed=SEED)


def train(train_df, rank, reg, implicit=False, alpha=1.0, max_iter=10):
    start = time.perf_counter()
    model = make_als(rank, reg, max_iter, implicit, alpha).fit(train_df)
    return model, time.perf_counter() - start


# ---- Model A: rating prediction ---------------------------------------------

def tune_explicit(train_df):
    """Grid search over rank and regParam with an inner train/validation split."""
    als = make_als()
    grid = (ParamGridBuilder()
            .addGrid(als.rank, [10, 20, 50])
            .addGrid(als.regParam, [0.05, 0.1, 0.2])
            .build())
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    tvs = TrainValidationSplit(estimator=als, estimatorParamMaps=grid, evaluator=evaluator,
                               trainRatio=0.8, seed=SEED, parallelism=2)
    start = time.perf_counter()
    model = tvs.fit(train_df)
    seconds = time.perf_counter() - start

    results = [
        {"rank": p[als.rank], "regParam": p[als.regParam], "val_rmse": round(m, 4)}
        for p, m in zip(grid, model.validationMetrics)
    ]
    best = min(results, key=lambda r: r["val_rmse"])
    return {"rank": best["rank"], "regParam": best["regParam"]}, sorted(results, key=lambda r: r["val_rmse"]), seconds


def evaluate_ratings(model, train_df, test_df):
    """RMSE / MAE / R2 on the test set, compared with two simple baselines."""
    preds = clip(model.transform(test_df)).cache()

    def score(df, metric):
        return round(RegressionEvaluator(metricName=metric, labelCol="rating",
                                         predictionCol="prediction").evaluate(df), 4)

    metrics = {"rmse": score(preds, "rmse"), "mae": score(preds, "mae"), "r2": score(preds, "r2"),
               "test_predictions": preds.count()}

    # Baseline 1: predict the global average rating for everything.
    mean = train_df.agg(F.avg("rating")).first()[0]
    metrics["baseline_global_mean_rmse"] = score(test_df.withColumn("prediction", F.lit(mean)), "rmse")

    # Baseline 2: predict each movie's average rating.
    movie_avg = train_df.groupBy("movieId").agg(F.avg("rating").alias("prediction"))
    metrics["baseline_movie_mean_rmse"] = score(test_df.join(movie_avg, "movieId"), "rmse")
    return metrics, preds


# ---- Model B: top-K recommendation ------------------------------------------

def eligible_movies(train_df, min_ratings=MIN_MOVIE_RATINGS):
    return train_df.groupBy("movieId").count().filter(F.col("count") >= min_ratings).select("movieId")


def relevant_items(test_df):
    """Movies each user liked (rating >= 4) in the held-out set: the 'right answers'."""
    return (test_df.filter("rating >= 4").groupBy("userId")
            .agg(F.collect_list(F.col("movieId").cast("double")).alias("label")))


def _top_k_unseen(recs_df, train_df, eligible, k):
    """Explodes (userId, recommendations) rows, drops seen / ineligible movies, keeps top K per user."""
    exploded = (recs_df.select("userId", F.explode("recommendations").alias("r"))
                .select("userId", F.col("r.movieId").alias("movieId"), F.col("r.rating").alias("score"))
                .join(eligible, "movieId")
                .join(train_df.select("userId", "movieId"), ["userId", "movieId"], "left_anti"))
    w = Window.partitionBy("userId").orderBy(F.desc("score"))
    return exploded.withColumn("pos", F.row_number().over(w)).filter(F.col("pos") <= k)


def _ranking_scores(top_k, relevant, k):
    """top_k: (userId, movieId, pos) rows. Returns precision / recall / NDCG at K."""
    recs = (top_k.groupBy("userId")
            .agg(F.sort_array(F.collect_list(F.struct("pos", "movieId"))).alias("s"))
            .select("userId", F.col("s.movieId").cast("array<double>").alias("prediction")))
    joined = recs.join(relevant, "userId")
    out = {}
    for metric in ("precisionAtK", "recallAtK", "ndcgAtK"):
        ev = RankingEvaluator(metricName=metric, k=k, labelCol="label", predictionCol="prediction")
        out[f"{metric.replace('AtK', '')}@{k}"] = round(ev.evaluate(joined), 4)
    return out


def ranking_metrics(model, train_df, test_df, k=10):
    """Precision / Recall / NDCG @K for one ALS model's top-K lists."""
    relevant = relevant_items(test_df)
    top = _top_k_unseen(model.recommendForUserSubset(relevant.select("userId"), 1000),
                        train_df, eligible_movies(train_df), k)
    return _ranking_scores(top, relevant, k)


def popularity_ranking_metrics(train_df, test_df, k=10):
    """Baseline: recommend the most-liked movies the user has not seen yet."""
    relevant = relevant_items(test_df)
    popular = (train_df.filter("rating >= 4").groupBy("movieId").count()
               .orderBy(F.desc("count")).limit(k + 300)
               .select(F.struct(F.col("movieId"), F.col("count").cast("float").alias("rating")).alias("r")))
    recs = relevant.select("userId").crossJoin(popular).groupBy("userId") \
        .agg(F.collect_list("r").alias("recommendations"))
    return _ranking_scores(_top_k_unseen(recs, train_df, eligible_movies(train_df), k), relevant, k)


def tune_implicit(train_df, k=10):
    """Picks rank / alpha / regParam by NDCG@K on a validation split carved out of
    the training data (the test set is never used for tuning)."""
    inner_train, val = train_df.randomSplit([0.8, 0.2], seed=SEED + 1)
    inner_train.cache()
    results = []
    start = time.perf_counter()
    for rank in (20, 50):
        for alpha in (1.0, 10.0):
            for reg in (0.1, 1.0):
                model, _ = train(inner_train, rank, reg, implicit=True, alpha=alpha)
                s = ranking_metrics(model, inner_train, val, k)
                results.append({"rank": rank, "alpha": alpha, "regParam": reg, f"val_ndcg@{k}": s[f"ndcg@{k}"]})
    seconds = time.perf_counter() - start
    results.sort(key=lambda r: -r[f"val_ndcg@{k}"])
    best = {key: results[0][key] for key in ("rank", "alpha", "regParam")}
    inner_train.unpersist()
    return best, results, seconds


def recommend_for_user(model, user_id, ratings, movies, k=10):
    """Top-K unseen movies for one user, with the user's own favourites for context."""
    spark = ratings.sparkSession
    user = spark.createDataFrame([(user_id,)], ["userId"])
    top = _top_k_unseen(model.recommendForUserSubset(user, 1000), ratings, eligible_movies(ratings), k)
    recs = (top.join(movies.select("movieId", "title", "genres"), "movieId")
            .select("pos", "title", "genres").orderBy("pos").withColumnRenamed("pos", "rank").toPandas())
    favourites = (ratings.filter(F.col("userId") == user_id)
                  .join(movies.select("movieId", "title", "genres"), "movieId")
                  .orderBy(F.desc("rating"), F.desc("timestamp")).limit(k)
                  .select("title", "genres", "rating").toPandas())
    return recs, favourites
