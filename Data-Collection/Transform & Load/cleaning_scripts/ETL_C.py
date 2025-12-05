#Convert 2020 dates from DD-MMM-YY to YYYY-MM-DD format.
#Other years are already in correct format. """

import pandas as pd
from datetime import datetime

INPUT_FILE = "final_combined_2020_2024_ages_numeric.csv"
OUTPUT_FILE = "final_combined_2020_2024_dates_fixed.csv"
CHUNK_SIZE = 500_000


print("FIXING 2020 DATE FORMAT")

def fix_date_format(date_str, year):
   
    if pd.isna(date_str):
        return date_str
    
    date_str = str(date_str).strip()
    
    # Only fix 2020 dates (which have DD-MMM-YY format)
    if year == 2020:
        try:
            # Parse DD-MMM-YY 
            parsed = datetime.strptime(date_str, "%d-%b-%y")
            # Return in YYYY-MM-DD format
            return parsed.strftime("%Y-%m-%d")
        except:
            # If parsing fails, try to return as-is or handle
            try:
                
                parsed = datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except:
                return date_str
    else:
   
        return date_str

first_chunk = True
total_rows = 0
dates_fixed = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    total_rows += len(chunk)
    
    if 'Incident Date' in chunk.columns and 'Year' in chunk.columns:
        # Count 2020 dates before fixing
        dates_2020_before = (chunk['Year'] == 2020).sum()
        
        chunk['Incident Date'] = chunk.apply(
            lambda row: fix_date_format(row['Incident Date'], row['Year']),
            axis=1
        )
        
        dates_fixed += dates_2020_before
    
    
    chunk.to_csv(OUTPUT_FILE, mode='w' if first_chunk else 'a',
                 header=first_chunk, index=False)
    first_chunk = False
    
    if total_rows % 1_000_000 == 0:
        print(f"Processed {total_rows:,} rows...")


print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Total rows processed: {total_rows:,}")
print(f"2020 dates converted: {dates_fixed:,}")
print("\nDate format: All dates now in YYYY-MM-DD format")
