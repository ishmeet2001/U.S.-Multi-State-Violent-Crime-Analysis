import pandas as pd
import os

csv_files = [
    "colorado_2020_2024.csv",
    "mexico_2020_2024.csv",
    "new_york_2020_2024.csv",
    "texas_2020_2024.csv",
    "washington_2020_2024.csv"
]


dataframes = []
print(" Individual file row counts:\n")

for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        row_count = len(df)
        print(f"{file}: {row_count} rows")
        dataframes.append(df)
    else:
        print(f" File not found: {file}")


if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    total_rows = len(combined_df)
    print("\n Combined dataset created successfully")
    print(f"Total combined rows: {total_rows}")

  
    combined_df.to_csv("combined_2020_2024.csv", index=False)
    print(" Saved as 'combined_2020_2024.csv'")
else:
    print("\n No valid CSV loaded.")
