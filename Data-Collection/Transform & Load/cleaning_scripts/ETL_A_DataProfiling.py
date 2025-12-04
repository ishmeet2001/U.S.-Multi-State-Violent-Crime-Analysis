import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import NumericType

def main():
    os.makedirs("reports", exist_ok=True)
    spark = SparkSession.builder.appName("ComprehensiveCrimeDataProfiling").getOrCreate()

    #HDFS path for dataset
    path = "hdfs:///user/raa126/dataset_local/combined_2020_2024.csv"
    df = spark.read.csv(path, header=True, inferSchema=True)

    print("\nPART A: DATA PROFILING & ANALYSIS :")
    print("Objective: To explore and understand the combined crime dataset (2020–2024)")
    print("Dataset Path:", path)

    #1. Dataset Overview :
    print("\n=== 1. DATASET OVERVIEW ===")
    print(f"Total Rows: {df.count():,}")
    print(f"Total Columns: {len(df.columns)}")
    print("Column Names:", df.columns)
    df.printSchema()

    #2. Missing Value Analysis :
    print("\n=== 2. MISSING VALUE SUMMARY ===")
    nulls = df.select([
        count(when(col(c).isNull() | (col(c) == "") | isnan(c), c)).alias(c)
        for c in df.columns
    ])
    nulls.show(truncate=False)

    #3. Data Type Classification :
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    print("\nNumeric Columns:", numeric_cols)
    print("Categorical Columns:", cat_cols)

    #4. Descriptive Statistics :
    print("\n=== 4. BASIC STATISTICS (NUMERIC COLUMNS) ===")
    df.select(numeric_cols).describe().show(truncate=False)

    #5. Outlier Analysis (Z-score > 3) :
    print("\n=== 5. OUTLIER ANALYSIS ===")
    for c in numeric_cols:
        stats = df.select(mean(col(c)).alias("mean"), stddev(col(c)).alias("std")).collect()[0]
        mean_val, std_val = stats["mean"], stats["std"]
        if std_val:
            outlier_count = df.filter(abs((col(c) - mean_val) / std_val) > 3).count()
            print(f"{c}: {outlier_count} potential outliers")

    #6. Unique Value Counts & Cardinality :
    print("\n=== 6. UNIQUENESS & CARDINALITY ===")
    for c in cat_cols:
        approx_unique = df.select(approx_count_distinct(c)).collect()[0][0]
        total = df.count()
        ratio = approx_unique / total
        print(f"{c}: ~{approx_unique} unique values ({ratio:.4f} of total rows)")

    #7. Top 10 Frequent Values :
    print("\n=== 7. TOP 10 FREQUENT VALUES (CATEGORICAL COLUMNS) ===")
    for c in cat_cols:
        print(f"\nColumn: {c}")
        df.groupBy(c).count().orderBy(desc("count")).show(10, truncate=False)

    #8. Date Range and Temporal Coverage :
    print("\n=== 8. INCIDENT DATE VALIDATION ===")
    df.select(min("Incident Date").alias("Earliest Date"),
              max("Incident Date").alias("Latest Date")).show()

    df.groupBy("Year").count().orderBy("Year").show()

    #9. Logical Sanity Checks :
    print("\n=== 9. LOGICAL VALIDATION ===")
    invalid_age = df.filter((col("Victim Type") == "Individual") & col("Victim Age Group").isNull()).count()
    print(f"→ Individuals missing Victim Age Group: {invalid_age}")

    invalid_weapon = df.filter((col("Weapon Name") == "None") & (col("Offense Category") == "Homicide")).count()
    print(f"→ 'Homicide' offenses reported with no weapon: {invalid_weapon}")

    #10. Analytical Insights :
    print("\n=== 10. ANALYTICAL INSIGHTS ===")
    print("\n→ TOP 10 COUNTIES BY INCIDENTS")
    df.groupBy("County").count().orderBy(desc("count")).show(10, truncate=False)

    print("\n→ INCIDENT DISTRIBUTION BY OFFENSE CATEGORY")
    df.groupBy("Offense Category").count().orderBy(desc("count")).show(10, truncate=False)

    print("\n→ WEAPON USAGE DISTRIBUTION")
    df.groupBy("Weapon Name").count().orderBy(desc("count")).show(10, truncate=False)

    print("\n→ VICTIM DEMOGRAPHICS (SEX × RACE)")
    df.groupBy("Victim Sex", "Victim Race").count().orderBy(desc("count")).show(10, truncate=False)

    print("\n→ YEARLY TREND OF CRIMES")
    df.groupBy("Year", "Offense Category").count().orderBy("Year", desc("count")).show(20, truncate=False)

    #11. Correlation Matrix (Numeric Columns) :
    if len(numeric_cols) >= 2:
        print("\n=== 11. CORRELATION MATRIX (NUMERIC COLUMNS) ===")
        pairs = [(a, b) for idx, a in enumerate(numeric_cols) for b in numeric_cols[idx + 1:]]
        for a, b in pairs:
            corr_val = df.stat.corr(a, b)
            print(f"Correlation({a}, {b}) = {corr_val:.3f}")

    spark.stop()

if __name__ == "__main__":
    main()