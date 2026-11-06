import os
import re
import shutil
from pathlib import Path
import pandas as pd

# =============================================================================
# Combines the MDClone split tables into one table, using bounded memory.
#
# The tables are too large to hold in RAM together (3.2M+ rows, hundreds of
# columns), so nothing here ever loads a whole file:
#   Pass 1: stream each file in chunks and split its rows into N_BUCKETS files
#           by a hash of the patient id. A given patient lands in the same
#           bucket number in every table.
#   Pass 2: merge one bucket at a time and append it to the output.
# Peak memory depends on N_BUCKETS and CHUNK_ROWS, not on the cohort size.
# If memory still runs out, raise N_BUCKETS and lower CHUNK_ROWS.
#
# Every value is read as raw text, so it reaches the output exactly as it
# appears in the source (no dtype guessing, and no DtypeWarning). The one
# exception: the time is stripped from date columns, "23/10/2021 17:30:04"
# becomes "23/10/2021". Dates stay as day-first text, so parse them later with
# dayfirst=True.
# Output rows are grouped by bucket, so they are NOT in the primary file's order.
#
# Any failure stops the run: the output is only put in place once every file
# has merged and the row count has been checked.
# =============================================================================

# 1. Define source path
base_path = Path(
    r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\RawData\MDclone15 and older"
)

# 2. Define output path
output_dir = Path(
    r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Temporary bucket files stay on the project share, next to the output, so
# patient data is not copied to a local disk. Needs free space about equal to
# the size of the raw files; deleted after a successful run.
work_dir = output_dir / "_merge_buckets_tmp"

primary_file_name = "PostWarPychosis_slim_Drugs.csv"
merge_key = "cohort reference event-patient id"

# Try 'utf-8' first. If the characters still look broken, change this to 'windows-1255'
HEBREW_ENCODING = "utf-8"

N_BUCKETS = 64
CHUNK_ROWS = 50_000

DATE_COLUMN_PATTERN = re.compile(r"\bdate\b", re.IGNORECASE)
# Only matches a value that starts with a date, so free text is left alone
DATE_WITH_TIME_PATTERN = r"^(\d[\d/.\-]*)\s+\d{1,2}:\d{2}.*$"


def read_header(path):
    return list(pd.read_csv(path, encoding=HEBREW_ENCODING, nrows=0).columns)


def is_wanted_column(col):
    """Drops "demographics*" and "charlson*" columns unless "2023" is in the name."""
    col_lower = col.lower()
    return "2023" in col_lower or not col_lower.startswith(("demographics", "charlson"))


def read_wanted_columns(path):
    header = read_header(path)
    wanted = [col for col in header if is_wanted_column(col)]
    print(f"{path.name}: {len(header)} columns, keeping {len(wanted)} after the demographics/charlson filter")
    return wanted


def bucket_path(table_index, bucket):
    return work_dir / f"table{table_index}_bucket{bucket}.csv"


def split_into_buckets(path, table_index, columns):
    """Stream one table into per-bucket files. Returns its row count."""
    n_rows = 0
    date_cols = [col for col in columns if DATE_COLUMN_PATTERN.search(col)]
    print(f"  Stripping time from {len(date_cols)} date columns: {date_cols}")
    reader = pd.read_csv(
        path,
        encoding=HEBREW_ENCODING,
        usecols=columns,
        dtype=str,
        na_filter=False,
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        chunk = chunk[columns].copy()
        for col in date_cols:
            chunk[col] = chunk[col].str.replace(DATE_WITH_TIME_PATTERN, r"\1", regex=True)
        n_rows += len(chunk)
        buckets = pd.util.hash_pandas_object(chunk[merge_key], index=False) % N_BUCKETS
        for bucket, rows in chunk.groupby(buckets.to_numpy()):
            target = bucket_path(table_index, bucket)
            rows.to_csv(target, mode="a", header=not target.exists(), index=False, encoding="utf-8")
        print(f"  {n_rows:,} rows split...", end="\r")
    print(f"  {n_rows:,} rows split.     ")
    return n_rows


def read_bucket(table_index, bucket, columns):
    target = bucket_path(table_index, bucket)
    if not target.exists():
        return pd.DataFrame({col: pd.Series(dtype=str) for col in columns})
    return pd.read_csv(target, encoding="utf-8", dtype=str, na_filter=False)


# --- STEP 1: Work out which columns to take from each file ---
primary_file_path = base_path / primary_file_name
secondary_paths = sorted(
    item for item in base_path.glob("*.csv")
    if item.is_file() and item.name != primary_file_name
)

# Filter out columns containing "medication" from the primary file
primary_columns = [col for col in read_wanted_columns(primary_file_path) if "medication" not in col]
tables = [(primary_file_path, primary_columns)]
seen_columns = set(primary_columns)

# A column name that already exists (except the merge key) is dropped from the
# later file, to prevent suffix duplicates (_x, _y)
for item in secondary_paths:
    header = read_wanted_columns(item)
    duplicate_cols = [col for col in header if col in seen_columns and col != merge_key]
    if duplicate_cols:
        print(f"Dropping duplicate columns from {item.name}: {duplicate_cols}")
    columns = [col for col in header if col == merge_key or col not in seen_columns]
    seen_columns.update(columns)
    tables.append((item, columns))

# --- STEP 2: Split every file into buckets ---
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir(parents=True)

table_rows = []
for table_index, (path, columns) in enumerate(tables):
    print(f"\nSplitting file: {path.name} ({len(columns)} columns)")
    table_rows.append(split_into_buckets(path, table_index, columns))

# --- STEP 3: Merge bucket by bucket, saving with UTF-8 BOM ---
output_path = output_dir / "matched_cohort_full.csv"
partial_path = output_dir / "matched_cohort_full.partial.csv"

rows_written = 0
matched_rows = [0] * len(tables)

print(f"\nMerging {N_BUCKETS} buckets...")
with open(partial_path, "w", encoding="utf-8-sig", newline="") as output_file:
    for bucket in range(N_BUCKETS):
        merged_df = read_bucket(0, bucket, tables[0][1])

        for table_index in range(1, len(tables)):
            df_secondary = read_bucket(table_index, bucket, tables[table_index][1])
            matched_rows[table_index] += int(merged_df[merge_key].isin(df_secondary[merge_key]).sum())
            # validate: a repeated id in a secondary file would silently add rows
            merged_df = pd.merge(
                merged_df, df_secondary, on=merge_key, how="left", validate="many_to_one"
            )

        merged_df.to_csv(output_file, header=(bucket == 0), index=False)
        rows_written += len(merged_df)
        print(f"  Bucket {bucket + 1}/{N_BUCKETS} done ({rows_written:,} rows)", end="\r")

n_columns = len(seen_columns)
print(f"\n\nMerged data: {rows_written:,} rows, {n_columns} columns")

# --- STEP 4: Check the result, then put it in place ---
print("\nPatients matched per file:")
for table_index in range(1, len(tables)):
    share = matched_rows[table_index] / table_rows[0] if table_rows[0] else 0
    print(f"  {tables[table_index][0].name}: {matched_rows[table_index]:,} of "
          f"{table_rows[0]:,} ({share:.1%}), file has {table_rows[table_index]:,} rows")

if rows_written != table_rows[0]:
    raise RuntimeError(f"Row count changed: primary has {table_rows[0]:,}, output has {rows_written:,}")

unmatched_files = [
    tables[table_index][0].name
    for table_index in range(1, len(tables))
    if matched_rows[table_index] == 0
]
if unmatched_files:
    raise RuntimeError(f"No patient ids matched in: {unmatched_files}. Check the id format in these files.")

os.replace(partial_path, output_path)
shutil.rmtree(work_dir)
print(f"\nSuccess! Merged dataset saved to:\n{output_path}")
