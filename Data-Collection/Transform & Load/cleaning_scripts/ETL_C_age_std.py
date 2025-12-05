"""
Drop rows where Victim Age Group == 185 (invalid age).

"""

import pandas as pd
from datetime import datetime

INPUT_FILE = "final_combined_2020_2024_std.csv"
OUTPUT_FILE = "final_combined_2020_2024_std_age_dropped.csv"
CHUNK_SIZE = 500_000


print("DROP INVALID VICTIM AGE (== 185)")

print(f"Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

first = True
rows_in = 0
rows_out = 0
rows_dropped = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    rows_in += len(chunk)

    if 'Victim Age Group' not in chunk.columns:
        raise KeyError("Column 'Victim Age Group' not found in input file")

    mask_drop = (chunk['Victim Age Group'] == 185)
    dropped = int(mask_drop.sum())
    rows_dropped += dropped

    filtered = chunk.loc[~mask_drop]
    rows_out += len(filtered)

    filtered.to_csv(OUTPUT_FILE, mode='w' if first else 'a', header=first, index=False)
    first = False

    print(f"Processed: {rows_in:,} | Dropped this chunk: {dropped:,} | Written total: {rows_out:,}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total input rows:   {rows_in:,}")
print(f"Total rows dropped: {rows_dropped:,}")
print(f"Total rows written: {rows_out:,}")
print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
