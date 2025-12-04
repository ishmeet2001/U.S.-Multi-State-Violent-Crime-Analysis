from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, when, sum as _sum, max as _max
from functools import reduce

spark = SparkSession.builder.appName("CrimeAnalysisB").getOrCreate()

#Loading CSV from HDFS
csv_path = "hdfs:///user/raa126/crime_data/std_final_combined_2020_2024.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)

#B1: Crime rate per 100k population
county_population = df.groupBy("State", "County") \
                      .agg(_max("Population").alias("CountyPopulation"),
                           count("*").alias("CountyIncidents"))

state_stats = county_population.groupBy("State") \
                               .agg(
                                   _sum("CountyIncidents").alias("TotalIncidents"),
                                   _sum("CountyPopulation").alias("TotalPopulation")
                               )

state_stats = state_stats.withColumn("CrimeRatePer100k",
                                     col("TotalIncidents") / col("TotalPopulation") * 100000)

print("B1: Crime rate per 100,000 population per state :")
state_stats.orderBy(desc("CrimeRatePer100k")).show(truncate=False)

#B2: Top violent counties within each state
violent_categories = [
    "Assault Offenses",
    "Homicide Offenses",
    "Kidnapping/Abduction",
    "Robbery",
    "Sex Offenses"
]

violent_df = df.filter(col("Offense Category").isin(violent_categories))

violent_by_county = (
    violent_df.groupBy("State", "County")
              .agg(count("*").alias("ViolentIncidents"))
              .orderBy("State", desc("ViolentIncidents"))
)

print("\nB2: Top counties for violent crimes in each state :")
violent_by_county.show(20, truncate=False)

#B3: Counties with the highest weapon-involved crimes
df_weapon_flag = df.withColumn(
    "WeaponInvolved",
    when(col("Weapon Name").isin("Unknown", "Other"), "No").otherwise("Yes")
)

weapon_by_county = (
    df_weapon_flag.filter(col("WeaponInvolved") == "Yes")
                  .groupBy("State", "County")
                  .agg(count("*").alias("WeaponCrimes"))
                  .orderBy("State", desc("WeaponCrimes"))
)

print("\n B3: Counties with highest weapon-involved crimes :")
weapon_by_county.show(20, truncate=False)

#B4: Agencies with most violent incidents
violent_by_agency = (
    violent_df.groupBy("State", "Agency Name")
              .agg(count("*").alias("ViolentIncidents"))
              .orderBy("State", desc("ViolentIncidents"))
)

print("\nB4: Agencies with most violent incidents :")
violent_by_agency.show(20, truncate=False)

#B5: States with highest proportion of unknown fields
unknown_columns = ["Victim Sex", "Victim Race", "Offender Sex", "Offender Race"]

unknown_expr = reduce(lambda a, b: a + b, [when(col(c) == "Unknown", 1).otherwise(0) for c in unknown_columns])
df_unknown = df.withColumn("UnknownCount", unknown_expr)

unknown_stats = (
    df_unknown.groupBy("State")
              .agg(
                  _sum("UnknownCount").alias("TotalUnknowns"),
                  count("*").alias("TotalRows")
              )
              .withColumn("UnknownRate", col("TotalUnknowns") / col("TotalRows"))
              .orderBy(desc("UnknownRate"))
)

print("\nB5: States with highest proportion of UNKNOWN fields :")
unknown_stats.show(20, truncate=False)
spark.stop()