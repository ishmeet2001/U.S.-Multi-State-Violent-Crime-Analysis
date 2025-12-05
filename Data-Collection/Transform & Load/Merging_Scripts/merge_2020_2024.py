import pandas as pd
import os

csv_files = [f for f in os.listdir() if f.startswith("new_mexico_") and f.endswith(".csv")]

if not csv_files:
    print(" No CSV files found in the current folder.")
    exit()

print(f" Found {len(csv_files)} CSV files: {csv_files}")

all_data = []

for file in csv_files:
    parts = file.replace(".csv", "").split("_")
    state = " ".join(parts[:-1]).title()  
    year = parts[-1]                      

    print(f"Processing file: {file}")

    
    df = pd.read_csv(file)

    df["State"] = state

    if year.isdigit():
        df["Year"] = int(year)
    else:
        print(f" '{year}' is not a valid year in filename '{file}', assigning 'merged'")
        df["Year"] = "merged"

    all_data.append(df)

merged_df = pd.concat(all_data, ignore_index=True)

output_file = "mexico_2020_2024.csv"
merged_df.to_csv(output_file, index=False)

print(f"\n Merged CSV saved as '{output_file}'")
print(f" Total rows: {len(merged_df):,}")
print(f" Columns: {len(merged_df.columns)}")

print("\n--- Summary by Year ---")
print(merged_df["Year"].value_counts())
