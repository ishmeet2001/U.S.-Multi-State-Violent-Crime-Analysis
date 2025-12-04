from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType
import functools

spark = SparkSession.builder.appName("USA Violence Cleaning").getOrCreate()

file_path = "/Users/arsh/Downloads/USA Violence/combined_2020_2024.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True).cache()

critical_columns = [
    "Offense ID", "County", "Location Name",
    "Offender Sex", "Victim Sex",
    "Offender Age", "Victim Age Group",
    "Offense Name", "Offense Category"
]

MIN_RECORDS_PER_STATE = 600_000

states = [r["State"] for r in df.select("State").distinct().collect()]
cleaned_dfs = []

for state in states:
    group = df.filter(F.col("State") == state)
    cleaned = group.dropna(subset=critical_columns)

    if cleaned.count() < MIN_RECORDS_PER_STATE:
        null_cols = [F.col(c).isNull().cast("int") for c in df.columns]
        null_sum = functools.reduce(lambda a, b: a + b, null_cols, F.lit(0))

        cleaned = (
            group.withColumn("null_count", null_sum)
                 .filter(F.col("null_count") <= 2)
                 .drop("null_count")
        )

    final_count = cleaned.count()

    if final_count < MIN_RECORDS_PER_STATE:
        diff = MIN_RECORDS_PER_STATE / max(final_count, 1)
        cleaned = cleaned.sample(withReplacement=True, fraction=diff, seed=42) \
                         .limit(MIN_RECORDS_PER_STATE)

    cleaned_dfs.append(cleaned)

if not cleaned_dfs:
    raise RuntimeError("No cleaned dataframes created.")

df = functools.reduce(lambda a, b: a.unionByName(b), cleaned_dfs)

# Fill missing categorical values
def fill_with_mode(df_in, col):
    mode_row = (
        df_in.filter(F.col(col).isNotNull())
             .groupBy(col)
             .count()
             .orderBy(F.desc("count"))
             .first()
    )
    if mode_row:
        return df_in.na.fill({col: mode_row[0]})
    return df_in.na.fill({col: f"Unknown_{col.replace(' ','_')}"})

if df.filter(F.col("Weapon Name").isNull()).count() > 0:
    df = fill_with_mode(df, "Weapon Name")

df = df.na.fill({
    "Victim Race": "Unknown",
    "Victim Resident Status": "Unknown",
    "Offender Race": "Unknown"
})

# Fill population using per-state median
median_pop = (
    df.groupBy("State")
      .agg(F.expr("percentile_approx(Population, 0.5)").alias("median_pop"))
)

df = (
    df.join(median_pop, on="State", how="left")
      .withColumn("Population", F.coalesce("Population", "median_pop"))
      .drop("median_pop")
)

# Fill Incident Hour
median_hour = df.approxQuantile("Incident Hour", [0.5], 0.01)[0]
df = df.na.fill({"Incident Hour": median_hour})

output_dir = "/Users/arsh/Downloads/USA Violence/final_combined_2020_2024_cleaned"
df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_dir)

df.groupBy("State").count().orderBy(F.desc("count")).show()
