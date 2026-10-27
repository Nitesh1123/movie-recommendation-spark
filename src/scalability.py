"""Scalability experiments: ALS on growing data sizes, and the workload on 1..N cores."""
import os
import time

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

from src.ingest import clean, load_raw
from src.recommender import SEED, clip, make_als
from src.spark_session import get_spark


def _als_run(ratings, rank, reg):
    train, test = ratings.select("userId", "movieId", "rating").randomSplit([0.8, 0.2], seed=SEED)
    train.cache().count()
    start = time.perf_counter()
    model = make_als(rank, reg).fit(train)
    train_s = time.perf_counter() - start
    rmse = RegressionEvaluator(metricName="rmse", labelCol="rating").evaluate(clip(model.transform(test)))
    return train_s, rmse


def data_scaling(ratings, rank, reg, fractions=(0.25, 0.5, 0.75, 1.0)):
    rows = []
    for frac in fractions:
        sample = ratings if frac == 1.0 else ratings.sample(fraction=frac, seed=SEED)
        n = sample.cache().count()
        train_s, rmse = _als_run(sample, rank, reg)
        rows.append({"fraction": frac, "ratings": n, "train_seconds": round(train_s, 2), "rmse": round(rmse, 4)})
        sample.unpersist()
        print(f"    data {int(frac * 100):>3}%  {n:>9,} ratings  train {train_s:6.2f}s  RMSE {rmse:.4f}")
    return rows


def core_scaling(data_dir, rank, reg, core_counts=None):
    """Full pipeline (load -> clean -> graph degree + PageRank -> ALS) per core count."""
    from graphframes import GraphFrame

    if core_counts is None:
        max_cores = os.cpu_count() or 4
        core_counts = [c for c in (1, 2, 4, 8, 16) if c <= max_cores]

    rows = []
    for cores in core_counts:
        spark = get_spark(cores=cores, app_name=f"scale-{cores}", shuffle_partitions=max(cores * 2, 4))
        start = time.perf_counter()

        ratings, movies = load_raw(spark, data_dir)
        ratings, movies, _ = clean(ratings, movies)
        ratings.cache().count()
        t_load = time.perf_counter() - start

        t0 = time.perf_counter()
        edges = ratings.select(F.concat(F.lit("u"), "userId").alias("src"), F.concat(F.lit("m"), "movieId").alias("dst"))
        verts = edges.select(F.col("src").alias("id")).union(edges.select(F.col("dst").alias("id"))).distinct()
        g = GraphFrame(verts, edges)
        g.inDegrees.count()
        g.pageRank(resetProbability=0.15, maxIter=5).vertices.count()
        t_graph = time.perf_counter() - t0

        t_als, _ = _als_run(ratings, rank, reg)
        total = time.perf_counter() - start
        rows.append({"cores": cores, "load_clean_s": round(t_load, 2), "graph_s": round(t_graph, 2),
                     "als_train_s": round(t_als, 2), "total_s": round(total, 2)})
        print(f"    {cores:>2} cores  load {t_load:5.1f}s  graph {t_graph:5.1f}s  ALS {t_als:5.1f}s  total {total:5.1f}s")
        spark.stop()

    base = rows[0]["total_s"]
    for r in rows:
        r["speedup"] = round(base / r["total_s"], 2)
    return rows
