"""Creates the SparkSession used by every stage of the pipeline."""
import glob
import os
import sys

from pyspark.sql import SparkSession

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JARS_DIR = os.path.join(PROJECT_DIR, "jars")


def get_spark(cores="*", app_name="MovieRecommender", shuffle_partitions=8):
    """Local Spark session with the GraphFrames jars from ./jars.

    extraClassPath avoids spark.jars.packages, which needs winutils.exe on Windows.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    jars = glob.glob(os.path.join(JARS_DIR, "*.jar"))
    if not jars:
        raise FileNotFoundError(f"No GraphFrames jars found in {JARS_DIR}. See README.md, step 2.")

    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.extraClassPath", os.pathsep.join(jars))
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
