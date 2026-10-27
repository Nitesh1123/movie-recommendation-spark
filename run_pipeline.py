"""Runs the whole pipeline and writes tables, charts and metrics.json to outputs/.

  python run_pipeline.py
  python run_pipeline.py --skip-scaling --skip-tuning
"""
import argparse
import json
import os
import sys
import time

from src import graph_analysis as ga
from src import ingest, plots, recommender, scalability
from src.spark_session import PROJECT_DIR, get_spark

timings = {}


def stage(name):
    """Prints a stage header and records how long the stage took."""
    class _Stage:
        def __enter__(self):
            print(f"\n=== {name} ===")
            self.t = time.perf_counter()

        def __exit__(self, *exc):
            timings[name] = round(time.perf_counter() - self.t, 2)
            print(f"    ({timings[name]}s)")
    return _Stage()


def save_csv(df, out_dir, name):
    df.to_csv(os.path.join(out_dir, name), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(PROJECT_DIR, "data", "ml-latest-small"))
    p.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "outputs"))
    p.add_argument("--cores", default="*", help="Local cores for the main run, e.g. 4 or *")
    p.add_argument("--user", type=int, default=1, help="User to show sample recommendations for")
    p.add_argument("--knn", type=int, default=10,
                   help="Neighbours per user in the user similarity graph")
    p.add_argument("--skip-tuning", action="store_true", help="Use rank=10, regParam=0.1 instead of grid search")
    p.add_argument("--skip-scaling", action="store_true", help="Skip the scalability experiments")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # movie titles contain accented characters

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    spark = get_spark(cores=args.cores)
    print(f"Spark {spark.version} | master {spark.sparkContext.master} | "
          f"parallelism {spark.sparkContext.defaultParallelism}")
    report = {"dataset": os.path.basename(os.path.normpath(args.data_dir))}

    # ---- Task 2: ingestion and preprocessing --------------------------------
    with stage("1. Data ingestion and cleaning"):
        ratings, movies = ingest.load_raw(spark, args.data_dir)
        ratings, movies, raw = ingest.clean(ratings, movies)
        ratings.cache()
        movies.cache()
        report["raw_ratings"] = raw
        print(f"    raw ratings: {raw:,} -> clean ratings: {ratings.count():,}")

    with stage("2. Exploratory distributed analysis"):
        stats, rating_dist, top_rated, genre_stats = ingest.explore(ratings, movies)
        report["data"] = stats
        print(f"    {stats['users']:,} users, {stats['movies']:,} movies, avg rating {stats['avgRating']}, "
              f"sparsity {stats['sparsity_pct']}%")
        save_csv(top_rated, out, "top_rated_movies.csv")
        save_csv(genre_stats, out, "genre_stats.csv")
        plots.rating_distribution(rating_dist, out)
        plots.genre_popularity(genre_stats, out)

    # ---- Task 3: graph analysis ---------------------------------------------
    with stage("3. Graph analysis - user-movie graph"):
        g = ga.build_user_movie_graph(ratings, movies)
        report["graph_user_movie"] = {"vertices": g.vertices.count(), "edges": g.edges.count()}
        print(f"    vertices {report['graph_user_movie']['vertices']:,}, edges {report['graph_user_movie']['edges']:,}")
        popular = ga.popular_movies(g)
        pr_movies, pr_users = ga.influential_vertices(g)
        pairs = ga.similar_movie_pairs(ratings, movies)
        save_csv(popular, out, "graph_popular_movies.csv")
        save_csv(pr_movies, out, "graph_pagerank_movies.csv")
        save_csv(pr_users, out, "graph_pagerank_users.csv")
        save_csv(pairs, out, "graph_movies_liked_together.csv")
        plots.popular_movies(popular, out)
        plots.pagerank_movies(pr_movies, out)
        print("    most-rated movie:", popular.iloc[0]["title"])
        print("    top PageRank movie:", pr_movies.iloc[0]["title"])

    with stage("4. Graph analysis - user communities"):
        sim = ga.build_user_similarity_graph(ratings, k=args.knn)
        conn = ga.connectivity(sim)
        comms, n_comms = ga.user_communities(sim, ratings, movies)
        conn["communities"] = n_comms
        report["graph_user_similarity"] = conn
        save_csv(comms, out, "graph_user_communities.csv")
        plots.communities(comms, out)
        print(f"    {conn['similarity_edges']:,} similarity edges, {n_comms} communities, "
              f"{conn['components']} connected components")

    # ---- Task 4: ALS recommender --------------------------------------------
    train, test = recommender.split(ratings)
    print(f"\n    train {train.count():,} ratings | test {test.count():,} ratings")

    # Model A: explicit ALS -> rating prediction
    if args.skip_tuning:
        best = {"rank": 10, "regParam": 0.1}
    else:
        with stage("5. Model A (rating prediction) - tuning"):
            best, grid, secs = recommender.tune_explicit(train)
            report["tuning_model_a"] = {"best": best, "grid": grid, "seconds": round(secs, 2)}
            plots.tuning(grid, out)
            print(f"    best rank={best['rank']} regParam={best['regParam']}")

    with stage("6. Model A (rating prediction) - train and evaluate"):
        model_a, secs_a = recommender.train(train, best["rank"], best["regParam"])
        metrics_a, preds = recommender.evaluate_ratings(model_a, train, test)
        metrics_a.update({"train_seconds": round(secs_a, 2), "params": best})
        report["model_a_rating_prediction"] = metrics_a
        plots.model_vs_baselines(metrics_a, out)
        plots.predicted_vs_actual(preds.select("rating", "prediction").toPandas(), out)
        for k, v in metrics_a.items():
            print(f"    {k}: {v}")

    # Model B: implicit ALS -> top-10 recommendation lists
    if args.skip_tuning:
        best_b = {"rank": 20, "alpha": 1.0, "regParam": 0.1}
    else:
        with stage("7. Model B (top-10 recommendation) - tuning"):
            best_b, grid_b, secs = recommender.tune_implicit(train)
            report["tuning_model_b"] = {"best": best_b, "grid": grid_b, "seconds": round(secs, 2)}
            print(f"    best rank={best_b['rank']} alpha={best_b['alpha']} regParam={best_b['regParam']}")

    with stage("8. Model B (top-10 recommendation) - train and evaluate"):
        model_b, secs_b = recommender.train(train, best_b["rank"], best_b["regParam"],
                                            implicit=True, alpha=best_b["alpha"])
        metrics_b = recommender.ranking_metrics(model_b, train, test)
        metrics_b_explicit = recommender.ranking_metrics(model_a, train, test)
        metrics_pop = recommender.popularity_ranking_metrics(train, test)
        report["model_b_top10"] = {"implicit_als": metrics_b, "explicit_als": metrics_b_explicit,
                                   "popularity_baseline": metrics_pop,
                                   "train_seconds": round(secs_b, 2), "params": best_b}
        plots.ranking_comparison(metrics_pop, metrics_b_explicit, metrics_b, out)
        for name, m in (("popularity baseline", metrics_pop), ("explicit ALS (model A)", metrics_b_explicit),
                        ("implicit ALS (model B)", metrics_b)):
            print(f"    {name:<24} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))

    with stage("9. Sample recommendations"):
        recs, favs = recommender.recommend_for_user(model_b, args.user, ratings, movies)
        save_csv(recs, out, f"recommendations_user_{args.user}.csv")
        save_csv(favs, out, f"favourites_user_{args.user}.csv")
        print(f"    user {args.user} liked:", "; ".join(favs["title"].head(4)))
        print(f"    recommended:", "; ".join(recs["title"].head(5)))

    report["stage_seconds"] = dict(timings)
    spark.stop()

    # ---- Task 5: scalability ------------------------------------------------
    if not args.skip_scaling:
        with stage("10. Scalability - data size"):
            spark = get_spark(cores=args.cores)
            r, m = ingest.load_raw(spark, args.data_dir)
            r, _, _ = ingest.clean(r, m)
            report["scaling_data"] = scalability.data_scaling(r.cache(), best["rank"], best["regParam"])
            plots.data_scaling(report["scaling_data"], out)
            spark.stop()
        with stage("11. Scalability - cores"):
            report["scaling_cores"] = scalability.core_scaling(args.data_dir, best["rank"], best["regParam"])
            plots.core_scaling(report["scaling_cores"], out)
        report["stage_seconds"] = dict(timings)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
