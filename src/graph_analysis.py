"""Graph analysis with GraphFrames: a user-movie graph and a user-user k-NN graph."""
from graphframes import GraphFrame
from pyspark.sql import Window
from pyspark.sql import functions as F

LIKE_THRESHOLD = 4.0


def build_user_movie_graph(ratings, movies):
    users = ratings.select(F.concat(F.lit("u"), F.col("userId")).alias("id")).distinct() \
        .withColumn("type", F.lit("user")).withColumn("title", F.lit(None).cast("string"))
    movie_v = movies.select(F.concat(F.lit("m"), F.col("movieId")).alias("id"),
                            F.lit("movie").alias("type"), "title")
    edges = ratings.select(
        F.concat(F.lit("u"), F.col("userId")).alias("src"),
        F.concat(F.lit("m"), F.col("movieId")).alias("dst"),
        "rating",
    )
    return GraphFrame(users.unionByName(movie_v), edges)


def popular_movies(g, top_n=15):
    """Movies with the most incoming rating edges, and the most 'liked' edges."""
    liked = g.edges.filter(F.col("rating") >= LIKE_THRESHOLD).groupBy(F.col("dst").alias("id")).count() \
        .withColumnRenamed("count", "likes")
    return (
        g.inDegrees.join(g.vertices, "id")
        .join(liked, "id", "left")
        .select("id", "title", F.col("inDegree").alias("numRatings"), F.coalesce("likes", F.lit(0)).alias("likes"))
        .orderBy(F.desc("numRatings")).limit(top_n).toPandas()
    )


def influential_vertices(g, top_n=15, max_iter=10):
    """PageRank on the user-movie graph, with edges in both directions."""
    both = g.edges.select("src", "dst").unionByName(
        g.edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
    pr = GraphFrame(g.vertices, both).pageRank(resetProbability=0.15, maxIter=max_iter).vertices
    top_movies = pr.filter("type = 'movie'").orderBy(F.desc("pagerank")).limit(top_n) \
        .select("id", "title", F.round("pagerank", 3).alias("pagerank")).toPandas()
    top_users = pr.filter("type = 'user'").orderBy(F.desc("pagerank")).limit(top_n) \
        .select("id", F.round("pagerank", 3).alias("pagerank")).toPandas()
    return top_movies, top_users


def build_user_similarity_graph(ratings, k=10):
    """Links each user to their k most similar users (Jaccard on liked movies).

    k-NN instead of a shared-likes threshold, which turned heavy raters into hubs.
    """
    liked = ratings.filter(F.col("rating") >= LIKE_THRESHOLD).select("userId", "movieId").cache()
    n_liked = liked.groupBy("userId").agg(F.count("*").alias("n"))
    pairs = (
        liked.alias("a").join(liked.alias("b"), "movieId")
        .filter(F.col("a.userId") < F.col("b.userId"))
        .groupBy(F.col("a.userId").alias("u1"), F.col("b.userId").alias("u2"))
        .agg(F.count("*").alias("shared"))
        .join(n_liked.select(F.col("userId").alias("u1"), F.col("n").alias("n1")), "u1")
        .join(n_liked.select(F.col("userId").alias("u2"), F.col("n").alias("n2")), "u2")
        .withColumn("jaccard", F.col("shared") / (F.col("n1") + F.col("n2") - F.col("shared")))
    )
    both = pairs.select("u1", "u2", "jaccard").unionByName(
        pairs.select(F.col("u2").alias("u1"), F.col("u1").alias("u2"), "jaccard"))
    nearest = (both.withColumn("rank", F.row_number().over(Window.partitionBy("u1").orderBy(F.desc("jaccard"))))
               .filter(F.col("rank") <= k))
    edges = nearest.select(F.col("u1").cast("string").alias("src"), F.col("u2").cast("string").alias("dst"),
                           F.round("jaccard", 4).alias("jaccard")).cache()
    vertices = edges.select(F.col("src").alias("id")).union(edges.select(F.col("dst").alias("id"))).distinct()
    return GraphFrame(vertices, edges)


def user_communities(sim_graph, ratings, movies, max_iter=5, top_n=8):
    """Label Propagation communities, each tagged with its highest-lift genre."""
    labels = sim_graph.labelPropagation(maxIter=max_iter)
    sizes = labels.groupBy("label").count().orderBy(F.desc("count")).cache()

    top_labels = sizes.limit(top_n)
    members = labels.join(top_labels, "label").select(F.col("id").cast("int").alias("userId"), "label")
    liked_genres = (ratings.filter(F.col("rating") >= LIKE_THRESHOLD)
                    .join(movies.select("movieId", F.explode("genreList").alias("genre")), "movieId")
                    .filter("genre NOT IN ('(no genres listed)', 'IMAX')"))  # IMAX is a format, not a genre

    total_likes = liked_genres.count()
    overall = liked_genres.groupBy("genre").agg((F.count("*") / total_likes).alias("overallShare"))
    comm = liked_genres.join(members, "userId").groupBy("label", "genre").count()
    comm = comm.withColumn("share", F.col("count") / F.sum("count").over(Window.partitionBy("label")))
    lift = (comm.filter(F.col("count") >= 20).join(overall, "genre")
            .withColumn("lift", F.col("share") / F.col("overallShare")))
    best = lift.groupBy("label").agg(
        F.max_by("genre", "lift").alias("topGenre"), F.round(F.max("lift"), 2).alias("genreLift"))

    summary = top_labels.join(best, "label", "left").orderBy(F.desc("count")) \
        .withColumnRenamed("count", "users").toPandas()
    summary.insert(0, "community", range(1, len(summary) + 1))
    return summary.drop(columns=["label"]), sizes.count()


def connectivity(sim_graph):
    """Connected components and degree stats of the user similarity graph."""
    cc = sim_graph.connectedComponents(algorithm="graphx")
    comp_sizes = cc.groupBy("component").count().orderBy(F.desc("count"))
    largest = comp_sizes.first()
    deg = sim_graph.degrees.agg(F.avg("degree").alias("avg"), F.max("degree").alias("max")).first()
    return {
        "user_vertices": sim_graph.vertices.count(),
        "similarity_edges": sim_graph.edges.count(),
        "components": comp_sizes.count(),
        "largest_component_users": largest["count"] if largest else 0,
        "avg_degree": round(deg["avg"] or 0, 2),
        "max_degree": deg["max"] or 0,
    }


def similar_movie_pairs(ratings, movies, min_ratings=50, top_n=15):
    """Movies most often liked together (item co-occurrence on the bipartite graph)."""
    liked = ratings.filter(F.col("rating") >= LIKE_THRESHOLD).select("userId", "movieId")
    popular = liked.groupBy("movieId").count().filter(F.col("count") >= min_ratings).select("movieId")
    liked = liked.join(popular, "movieId")
    pairs = (
        liked.alias("a").join(liked.alias("b"), "userId")
        .filter(F.col("a.movieId") < F.col("b.movieId"))
        .groupBy(F.col("a.movieId").alias("m1"), F.col("b.movieId").alias("m2"))
        .agg(F.count("*").alias("usersLikedBoth"))
        .orderBy(F.desc("usersLikedBoth")).limit(top_n)
    )
    t = movies.select("movieId", "title")
    return (
        pairs.join(t.withColumnRenamed("movieId", "m1").withColumnRenamed("title", "movie1"), "m1")
        .join(t.withColumnRenamed("movieId", "m2").withColumnRenamed("title", "movie2"), "m2")
        .select("movie1", "movie2", "usersLikedBoth").orderBy(F.desc("usersLikedBoth")).toPandas()
    )
