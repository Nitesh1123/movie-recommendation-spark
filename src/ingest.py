"""Task 2: data ingestion, cleaning, preprocessing and distributed analysis."""
import os

from pyspark.sql import functions as F
from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType,
                               StructField, StructType)

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType()),
    StructField("movieId", IntegerType()),
    StructField("rating", DoubleType()),
    StructField("timestamp", LongType()),
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType()),
    StructField("title", StringType()),
    StructField("genres", StringType()),
])


def load_raw(spark, data_dir):
    """Reads ratings.csv and movies.csv (MovieLens CSV format: small, 25m, 32m)."""
    ratings = spark.read.csv(os.path.join(data_dir, "ratings.csv"), header=True, schema=RATINGS_SCHEMA)
    movies = spark.read.csv(os.path.join(data_dir, "movies.csv"), header=True, schema=MOVIES_SCHEMA,
                            quote='"', escape='"')
    return ratings, movies


def clean(ratings, movies):
    """Drops nulls, duplicates and out-of-range ratings; adds year and a date column."""
    raw_count = ratings.count()

    ratings = (
        ratings.dropna()
        .filter((F.col("rating") >= 0.5) & (F.col("rating") <= 5.0))
        .dropDuplicates(["userId", "movieId"])
        .withColumn("ratedAt", F.to_timestamp(F.from_unixtime("timestamp")))
    )

    movies = (
        movies.dropna(subset=["movieId", "title"])
        .dropDuplicates(["movieId"])
        # Some titles have no year; nullif keeps the cast from failing under ANSI mode.
        .withColumn("year", F.nullif(F.regexp_extract("title", r"\((\d{4})\)\s*$", 1), F.lit("")).cast("int"))
        .withColumn("genreList", F.split(F.col("genres"), r"\|"))
    )

    # Keep only ratings for movies we have metadata for.
    ratings = ratings.join(movies.select("movieId"), "movieId", "inner")
    return ratings, movies, raw_count


def explore(ratings, movies):
    """Distributed aggregations used in the report. Returns small pandas tables."""
    stats = ratings.agg(
        F.count("*").alias("ratings"),
        F.countDistinct("userId").alias("users"),
        F.countDistinct("movieId").alias("movies"),
        F.round(F.avg("rating"), 3).alias("avgRating"),
    ).first().asDict()
    stats["sparsity_pct"] = round(100 * (1 - stats["ratings"] / (stats["users"] * stats["movies"])), 3)

    rating_dist = ratings.groupBy("rating").count().orderBy("rating").toPandas()

    movie_stats = ratings.groupBy("movieId").agg(
        F.count("*").alias("numRatings"), F.round(F.avg("rating"), 2).alias("avgRating"))
    top_rated = (
        movie_stats.filter("numRatings >= 50")
        .join(movies.select("movieId", "title"), "movieId")
        .orderBy(F.desc("avgRating"), F.desc("numRatings"))
        .limit(15).toPandas()
    )

    genre_stats = (
        ratings.join(movies.select("movieId", F.explode("genreList").alias("genre")), "movieId")
        .filter("genre != '(no genres listed)'")
        .groupBy("genre")
        .agg(F.count("*").alias("numRatings"), F.round(F.avg("rating"), 2).alias("avgRating"))
        .orderBy(F.desc("numRatings")).toPandas()
    )

    user_activity = ratings.groupBy("userId").count()
    activity = user_activity.agg(
        F.min("count").alias("min"), F.expr("percentile_approx(count, 0.5)").alias("median"),
        F.max("count").alias("max")).first().asDict()
    stats["ratings_per_user"] = activity

    return stats, rating_dist, top_rated, genre_stats
