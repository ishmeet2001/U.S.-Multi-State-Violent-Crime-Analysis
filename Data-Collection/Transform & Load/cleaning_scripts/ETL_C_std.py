import re
import pandas as pd

INPUT_FILE = "final_combined_2020_2024_cleaned.csv"
OUTPUT_FILE = "final_combined_2020_2024_ages_numeric.csv"
CHUNK_SIZE = 500_000


def extract_numeric_age(age_value):
    if pd.isna(age_value):
        return -1

    age_str = str(age_value).strip()

    range_match = re.match(r"(\d+)-(\d+)", age_str)
    if range_match:
        low, high = map(int, range_match.groups())
        return (low + high) // 2

    num_match = re.search(r"\d+", age_str)
    if num_match:
        return int(num_match.group(0))

    return -1


def find_age_columns(columns):
    victim_col = None
    offender_col = None

    for col in columns:
        lower = col.lower()
        if victim_col is None and "victim" in lower and "age" in lower:
            victim_col = col
        if offender_col is None and "offender" in lower and "age" in lower:
            offender_col = col

    return victim_col, offender_col


def convert_chunk(chunk, victim_col, offender_col):
    victim_count = 0
    offender_count = 0

    if victim_col:
        victim_count = chunk[victim_col].notna().sum()
        chunk[victim_col] = chunk[victim_col].apply(extract_numeric_age).astype(int)

    if offender_col:
        offender_count = chunk[offender_col].notna().sum()
        chunk[offender_col] = chunk[offender_col].apply(extract_numeric_age).astype(int)

    return victim_count, offender_count


def convert_file(input_path=INPUT_FILE, output_path=OUTPUT_FILE, chunk_size=CHUNK_SIZE):
    first_chunk = True
    total_rows = 0
    victim_age_converted = 0
    offender_age_converted = 0

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)

        victim_col, offender_col = find_age_columns(chunk.columns)
        v_count, o_count = convert_chunk(chunk, victim_col, offender_col)
        victim_age_converted += v_count
        offender_age_converted += o_count

        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    return total_rows, victim_age_converted, offender_age_converted


if __name__ == "__main__":
    rows, victim_count, offender_count = convert_file()

    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total rows processed: {rows:,}")
    print(f"Victim age values processed: {victim_count:,}")
    print(f"Offender age values processed: {offender_count:,}")
    print("\nAge columns converted to int (with -1 for unknown ages)")
