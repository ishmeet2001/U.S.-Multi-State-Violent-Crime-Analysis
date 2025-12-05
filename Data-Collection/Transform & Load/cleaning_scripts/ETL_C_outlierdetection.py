"""Lightweight rules-only outlier check for NIBRS (2020-2024)."""

import pandas as pd
from datetime import datetime

INPUT_FILE = "final_combined_2020_2024_dates_fixed.csv"
OUTPUT_REPORT = "outlier_analysis_report_rules.txt"

# Validation rules
VALID_YEAR_RANGE = (2020, 2024)
VALID_HOUR_RANGE = (0, 23)
MAX_REASONABLE_AGE = 120

report = open(OUTPUT_REPORT, 'w')

def w(line: str = ""):
    print(line)
    report.write(line + "\n")

w("=" * 80)
w("RULES-ONLY OUTLIER REPORT")
w("Dataset: NIBRS Crime Data 2020–2024")
w("=" * 80)

w(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"Input File: {INPUT_FILE}")
w("=" * 80)

sample = pd.read_csv(INPUT_FILE, nrows=1000, low_memory=False)
columns = sample.columns.tolist()
dtypes = {c: str(sample[c].dtype) for c in columns}

w("\nColumns (name | dtype):")
for i, c in enumerate(columns, 1):
    w(f"{i:2d}. {c} | {dtypes[c]}")

w("\n" + "=" * 80)
w("PROCESSING")
w("=" * 80)

row_count = 0
null_counts = {c: 0 for c in columns}

# Age metrics
age_metrics = {
    'Victim Age Group': {
        'unknown_minus1': 0,
        'valid_count': 0,
        'lt0_excl_minus1': 0,
        'gt100': 0,
        'gt120': 0,
        'eq185': 0,
    },
    'Offender Age': {
        'unknown_minus1': 0,
        'valid_count': 0,
        'lt0_excl_minus1': 0,
        'gt100': 0,
        'gt120': 0,
    },
}

# Population metrics
pop_zeros = 0
pop_negatives = 0

# Year and hour metrics
invalid_year_count = 0
invalid_year_values = set()
invalid_hour_count = 0
invalid_hour_values = set()

for chunk in pd.read_csv(INPUT_FILE, chunksize=500_000, low_memory=False):
    row_count += len(chunk)

    # Null counts
    for c in columns:
        null_counts[c] += chunk[c].isna().sum()

    # Age checks (Victim Age Group)
    if 'Victim Age Group' in chunk.columns:
        col = chunk['Victim Age Group']
        age_metrics['Victim Age Group']['unknown_minus1'] += int((col == -1).sum())
        valid_mask = col.notna() & (col != -1)
        valid = col[valid_mask]
        age_metrics['Victim Age Group']['valid_count'] += int(valid_mask.sum())
        age_metrics['Victim Age Group']['lt0_excl_minus1'] += int((valid < 0).sum())
        age_metrics['Victim Age Group']['gt100'] += int((valid > 100).sum())
        age_metrics['Victim Age Group']['gt120'] += int((valid > MAX_REASONABLE_AGE).sum())
        age_metrics['Victim Age Group']['eq185'] += int((col == 185).sum())

    # Age checks (Offender Age)
    if 'Offender Age' in chunk.columns:
        col = chunk['Offender Age']
        age_metrics['Offender Age']['unknown_minus1'] += int((col == -1).sum())
        valid_mask = col.notna() & (col != -1)
        valid = col[valid_mask]
        age_metrics['Offender Age']['valid_count'] += int(valid_mask.sum())
        age_metrics['Offender Age']['lt0_excl_minus1'] += int((valid < 0).sum())
        age_metrics['Offender Age']['gt100'] += int((valid > 100).sum())
        age_metrics['Offender Age']['gt120'] += int((valid > MAX_REASONABLE_AGE).sum())

    # Population checks
    if 'Population' in chunk.columns:
        pop = chunk['Population']
        pop_zeros += int((pop == 0).sum())
        pop_negatives += int((pop < 0).sum())
        # Intentionally ignore high outliers

    # Year checks
    if 'Year' in chunk.columns:
        year = chunk['Year']
        mask_invalid_year = year.notna() & ((year < VALID_YEAR_RANGE[0]) | (year > VALID_YEAR_RANGE[1]))
        invalid_year_count += int(mask_invalid_year.sum())
        if mask_invalid_year.any():
            invalid_year_values.update(set(year[mask_invalid_year].unique().tolist()))

    # Incident Hour checks
    if 'Incident Hour' in chunk.columns:
        hour = chunk['Incident Hour']
        mask_invalid_hour = hour.notna() & ((hour < VALID_HOUR_RANGE[0]) | (hour > VALID_HOUR_RANGE[1]))
        invalid_hour_count += int(mask_invalid_hour.sum())
        if mask_invalid_hour.any():
            invalid_hour_values.update(set(hour[mask_invalid_hour].unique().tolist()))

w("\n" + "=" * 80)
w("SUMMARY (RULES-ONLY)")
w("=" * 80)

w(f"\nTotal rows processed: {row_count:,}")

# Nulls
w("\nNULL COUNTS BY COLUMN:")
for c in columns:
    w(f"- {c}: {null_counts[c]:,} ({(null_counts[c]/row_count)*100:.2f}%)")

# Population summary (rules only)
w("\nPOPULATION CHECKS:")
w(f"- Zero population: {pop_zeros:,}")
w(f"- Negative population: {pop_negatives:,}")

# Year and hour
w("\nYEAR CHECKS:")
w(f"- Valid range: {VALID_YEAR_RANGE[0]}–{VALID_YEAR_RANGE[1]}")
w(f"- Invalid year values count: {invalid_year_count:,}")
w(f"- Invalid year unique values: {sorted(list(invalid_year_values)) if invalid_year_values else 'None'}")

w("\nINCIDENT HOUR CHECKS:")
w(f"- Valid range: {VALID_HOUR_RANGE[0]}–{VALID_HOUR_RANGE[1]}")
w(f"- Invalid hour values count: {invalid_hour_count:,}")
w(f"- Invalid hour unique values: {sorted(list(invalid_hour_values)) if invalid_hour_values else 'None'}")

# Age checks
w("\nAGE CHECKS (EXCLUDING -1 UNKNOWN):")
for col in ['Victim Age Group', 'Offender Age']:
    if col in age_metrics:
        m = age_metrics[col]
        w(f"\n{col}:")
        w(f"- Unknown (-1): {m['unknown_minus1']:,}")
        w(f"- Valid ages: {m['valid_count']:,}")
        w(f"- Ages < 0 (excluding -1): {m['lt0_excl_minus1']:,}")
        w(f"- Ages > 100: {m['gt100']:,}")
        w(f"- Ages > {MAX_REASONABLE_AGE}: {m['gt120']:,}")
        if col == 'Victim Age Group':
            w(f"- Specific invalid age=185: {m['eq185']:,}")

w("\n" + "=" * 80)
w("KEY FINDINGS")
w("=" * 80)

#  key findings
if age_metrics['Victim Age Group']['eq185'] > 0:
    w(f"- Victim Age Group contains {age_metrics['Victim Age Group']['eq185']:,} records with age=185 (impossible)")
if age_metrics['Victim Age Group']['gt120'] > 0:
    w(f"- Victim Age Group has ages > {MAX_REASONABLE_AGE}: {age_metrics['Victim Age Group']['gt120']:,}")
if age_metrics['Offender Age']['gt120'] > 0:
    w(f"- Offender Age has ages > {MAX_REASONABLE_AGE}: {age_metrics['Offender Age']['gt120']:,}")
if invalid_year_count > 0:
    w(f"- Year contains values outside {VALID_YEAR_RANGE[0]}–{VALID_YEAR_RANGE[1]}: {invalid_year_count:,}")
if invalid_hour_count > 0:
    w(f"- Incident Hour contains invalid values outside {VALID_HOUR_RANGE[0]}–{VALID_HOUR_RANGE[1]}: {invalid_hour_count:,}")
if pop_negatives > 0:
    w(f"- Negative population values found: {pop_negatives:,}")

w("\nReport saved to: " + OUTPUT_REPORT)
report.close()
