"""Checks that the pipeline's outputs are correct and prints PASS/FAIL for each check.

The Spark results in ./outputs are compared against an independent count done
with plain pandas (no Spark), plus sanity checks on the models and recommendations.

  python verify.py            # check the existing outputs
  python verify.py --run      # run the pipeline (quick mode) first, then check
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, passed, detail=""):
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n{title}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "ml-latest-small"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--user", type=int, default=1, help="User whose recommendations to check")
    p.add_argument("--run", action="store_true", help="Run the pipeline (--skip-scaling --skip-tuning) first")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = args.out_dir

    if args.run:
        print("Running the pipeline in quick mode (about 3-4 minutes)...")
        cmd = [sys.executable, os.path.join(ROOT, "run_pipeline.py"), "--skip-scaling", "--skip-tuning",
               "--user", str(args.user), "--data-dir", args.data_dir, "--out-dir", out]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        check("Pipeline finished without errors", proc.returncode == 0 and "Done." in proc.stdout,
              f"exit code {proc.returncode}")
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            return 1

    # ---- 1. Outputs exist ---------------------------------------------------
    section("1. Outputs exist")
    metrics_path = os.path.join(out, "metrics.json")
    check("metrics.json exists", os.path.exists(metrics_path))
    if not os.path.exists(metrics_path):
        print("\nNo outputs found. Run: python verify.py --run")
        return 1
    with open(metrics_path) as f:
        m = json.load(f)
    charts = glob.glob(os.path.join(out, "*.png"))
    check("Charts were created", len(charts) >= 8, f"{len(charts)} PNG files")
    needed = ["graph_popular_movies.csv", "graph_pagerank_movies.csv", "graph_user_communities.csv",
              f"recommendations_user_{args.user}.csv", "genre_stats.csv"]
    missing = [n for n in needed if not os.path.exists(os.path.join(out, n))]
    check("Result tables were created", not missing, f"missing: {missing}" if missing else f"{len(needed)} checked")
    if missing:
        return 1

    # ---- 2. Spark numbers match an independent pandas count ------------------
    section("2. Spark results match an independent pandas count")
    ratings = pd.read_csv(os.path.join(args.data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(args.data_dir, "movies.csv"))
    d = m["data"]
    check("Number of ratings", d["ratings"] == len(ratings), f"Spark {d['ratings']:,} vs pandas {len(ratings):,}")
    check("Number of users", d["users"] == ratings.userId.nunique(),
          f"Spark {d['users']:,} vs pandas {ratings.userId.nunique():,}")
    check("Number of rated movies", d["movies"] == ratings.movieId.nunique(),
          f"Spark {d['movies']:,} vs pandas {ratings.movieId.nunique():,}")
    avg = round(ratings.rating.mean(), 3)
    check("Average rating", abs(d["avgRating"] - avg) < 0.001, f"Spark {d['avgRating']} vs pandas {avg}")

    popular = pd.read_csv(os.path.join(out, "graph_popular_movies.csv"))
    counts = ratings.movieId.value_counts()
    title_of = movies.set_index("movieId").title
    expected_top3 = [title_of[i] for i in counts.index[:3]]
    check("Top 3 most-rated movies (graph in-degree)", list(popular.title[:3]) == expected_top3,
          "; ".join(expected_top3))
    check("In-degree counts are exact", list(popular.numRatings[:3]) == list(counts.values[:3]),
          f"{list(popular.numRatings[:3])}")
    g = m["graph_user_movie"]
    n_vertices = ratings.userId.nunique() + movies.movieId.nunique()
    check("Graph vertices = users + movies", g["vertices"] == n_vertices, f"{g['vertices']:,} vs {n_vertices:,}")
    check("Graph edges = ratings", g["edges"] == len(ratings), f"{g['edges']:,}")

    # ---- 3. Graph results make sense ---------------------------------------
    section("3. Graph results make sense")
    pr = pd.read_csv(os.path.join(out, "graph_pagerank_movies.csv"))
    top20 = set(title_of[i] for i in counts.index[:20])
    check("Top PageRank movie is among the 20 most-rated", pr.title[0] in top20, pr.title[0])
    check("PageRank scores are sorted high to low", pr.pagerank.is_monotonic_decreasing)
    sim = m["graph_user_similarity"]
    # Users who never rated anything >= 4 have no liked movies, so they can't be linked.
    likers = ratings[ratings.rating >= 4].userId.nunique()
    check("Every user with a liked movie is in the similarity graph", sim["user_vertices"] == likers,
          f"{sim['user_vertices']} vs {likers} users who rated something >= 4")
    check("Label Propagation found several communities", 2 <= sim["communities"] < sim["user_vertices"],
          f"{sim['communities']} communities")
    comms = pd.read_csv(os.path.join(out, "graph_user_communities.csv"))
    check("Community sizes add up to at most the user count", comms.users.sum() <= sim["user_vertices"],
          f"top {len(comms)} communities hold {comms.users.sum()} users")

    # ---- 4. Models beat simple baselines -----------------------------------
    section("4. Models beat simple baselines")
    a = m["model_a_rating_prediction"]
    check("Model A RMSE beats 'predict the global average'", a["rmse"] < a["baseline_global_mean_rmse"],
          f"{a['rmse']} < {a['baseline_global_mean_rmse']}")
    check("Model A RMSE beats 'predict the movie average'", a["rmse"] < a["baseline_movie_mean_rmse"],
          f"{a['rmse']} < {a['baseline_movie_mean_rmse']}")
    check("Model A RMSE is in the normal range for MovieLens (0.80-0.95)", 0.80 <= a["rmse"] <= 0.95,
          f"{a['rmse']}")
    check("Model A MAE is below RMSE (as it must be)", a["mae"] <= a["rmse"], f"MAE {a['mae']}")
    b = m["model_b_top10"]
    for metric in ("precision@10", "recall@10", "ndcg@10"):
        check(f"Model B {metric} beats the popularity baseline",
              b["implicit_als"][metric] > b["popularity_baseline"][metric],
              f"{b['implicit_als'][metric]} > {b['popularity_baseline'][metric]}")

    # ---- 5. Recommendations are valid --------------------------------------
    section(f"5. Recommendations for user {args.user} are valid")
    recs = pd.read_csv(os.path.join(out, f"recommendations_user_{args.user}.csv"))
    seen = set(title_of[ratings[ratings.userId == args.user].movieId])
    check("Exactly 10 recommendations", len(recs) == 10, f"{len(recs)}")
    check("No duplicate recommendations", recs.title.is_unique)
    already = set(recs.title) & seen
    check("No movie the user already rated", not already, f"already rated: {already}" if already else "")
    eligible = set(title_of[counts[counts >= 20].index])
    rare = set(recs.title) - eligible
    check("Every recommended movie has at least 20 ratings", not rare, f"too rare: {rare}" if rare else "")

    # ---- 6. Timing and scalability (only after a full run) ------------------
    if "scaling_cores" in m:
        section("6. Scalability results")
        sc = m["scaling_cores"]
        check("ALS training is faster on more cores", sc[-1]["als_train_s"] < sc[0]["als_train_s"],
              f"{sc[0]['als_train_s']}s on {sc[0]['cores']} core -> {sc[-1]['als_train_s']}s on {sc[-1]['cores']} cores")
        sd = m["scaling_data"]
        check("More training data gives lower RMSE", sd[-1]["rmse"] < sd[0]["rmse"],
              f"{sd[0]['rmse']} at {int(sd[0]['fraction'] * 100)}% -> {sd[-1]['rmse']} at 100%")
    else:
        print("\n6. Scalability results: skipped (run without --skip-scaling to include)")

    passed, total = sum(results), len(results)
    print(f"\n{'=' * 50}\n{passed}/{total} checks passed" + ("" if passed == total else "  <-- see FAIL lines"))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
