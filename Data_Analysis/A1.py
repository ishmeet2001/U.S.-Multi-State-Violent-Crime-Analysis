from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, when

#Initializing Spark Session
spark = SparkSession.builder.appName("CrimeAnalysisA1").getOrCreate()

#Loading CSV from hdfs
csv_path = "hdfs:///user/raa126/crime_data/std_final_combined_2020_2024.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)

#Defining crime categories
violent_categories = [
    "Assault Offenses",
    "Homicide Offenses",
    "Kidnapping/Abduction",
    "Robbery",
    "Sex Offenses",
    "Sex Offenses, Non-forcible",
    "Human Trafficking"
]

df = df.withColumn(
    "CrimeType",
    when(col("Offense Category").isin(violent_categories), "Violent").otherwise("Non-Violent")
)

#Violent vs Non-Violent Crimes
print("\nViolent vs Non-Violent Offense Categories:")
crime_type_table = df.select("Offense Category", "CrimeType").distinct().orderBy("CrimeType", "Offense Category")
crime_type_table.show(truncate=False)

#Question 1: States with highest violent incidents:
print("\nQ1: Top states by number of violent incidents")
violent_states = (
    df.filter(col("CrimeType") == "Violent")
      .groupBy("State")
      .agg(count("*").alias("IncidentCount"))
      .orderBy(desc("IncidentCount"))
)
violent_states.show(10, truncate=False)

#Question 2: Top 10 Offense Categories:
print("\nQ2: Top 10 Offense Categories")
top_categories = (
    df.groupBy("Offense Category")
      .agg(count("*").alias("Count"))
      .orderBy(desc("Count"))
)
top_categories.show(10, truncate=False)

#Question 3: Distribution of Offense Names per Offense Category:
print("\nQ3: Distribution of Offense Names per Offense Category")
offense_dist = (
    df.groupBy("Offense Category", "Offense Name")
      .agg(count("*").alias("Count"))
      .orderBy("Offense Category", desc("Count"))
)
offense_dist.show(20, truncate=False)

#Question 4: Crimes with and without weapons: 
print("\nQ4: Crimes with vs without weapons:")
df_with_weapon = df.withColumn(
    "HasWeapon",
    when(~col("Weapon Name").isin(["Other", "Unknown"]), "With Weapon")
    .otherwise("Without Weapon")
)

weapon_summary = df_with_weapon.groupBy("HasWeapon").agg(count("*").alias("Count")).orderBy(desc("Count"))
weapon_summary.show(truncate=False)

#Question 5: Most frequent Weapon Types:
print("\nQ5: Most frequent Weapon Types :")
top_weapons = (
    df.filter(~col("Weapon Name").isin(["Other", "Unknown"]))
      .groupBy("Weapon Name")
      .agg(count("*").alias("Count"))
      .orderBy(desc("Count"))
)
top_weapons.show(10, truncate=False)
spark.stop()