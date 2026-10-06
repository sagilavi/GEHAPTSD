# Quick look at the membership / follow-up columns, to decide which ones should define
# the start and end of person-time at risk.
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

INPUT_FILE = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar\matched_cohort_FirstEverOutcomes.csv"
OUTPUT_CSV = r"C:\Users\sagil\MyCode\PTSD\followup_columns_samples.csv"
N_SAMPLES = 10

DATE_COLS = {
    "cohort reference event-membership start date": "cohort_start",
    "cohort reference event-membership end date":   "cohort_end",
    "clalit membership-membership start date":      "clalit_start",
    "clalit membership-membership end date":        "clalit_end",
    "birth date":                                   "birth",
    "immigration date":                             "immigration",
}
OTHER_COLS = {
    "cohort reference event-membership period": "cohort_period",
    "clalit membership-membership period":      "clalit_period",
}

# --- Load only the columns we need ---
header = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", nrows=0).columns
wanted = ["patient id"] + list(DATE_COLS) + list(OTHER_COLS)
missing = [c for c in wanted if c not in header]
if missing:
    print("MISSING COLUMNS:", missing)
# Chunked load (same pattern as load_frame() in presentation_summary_step6.py): read as text,
# parse dates per chunk, and keep only the parsed dates -- raw text is kept just as counts of
# values that failed to parse, so memory stays small.
bad_raw = {col: [] for col in DATE_COLS.values()}
parts = []
for chunk in pd.read_csv(INPUT_FILE, encoding="utf-8-sig", dtype=str,
                         usecols=[c for c in wanted if c in header], chunksize=100_000):
    chunk = chunk.rename(columns={**DATE_COLS, **OTHER_COLS})
    for col in DATE_COLS.values():
        if col in chunk:
            parsed = pd.to_datetime(chunk[col], errors="coerce")
            bad_raw[col].append(chunk.loc[parsed.isna() & chunk[col].notna(), col].value_counts())
            chunk[col] = parsed
    parts.append(chunk)
df = pd.concat(parts, ignore_index=True)
del parts

df["age18"] = df["birth"] + pd.DateOffset(years=18)

# --- 1. Overall summary per date column ---
print(f"\nRows: {len(df):,}\n")
print("=== Date column summary ===")
summary = pd.DataFrame({
    "missing":   df[list(DATE_COLS.values())].isna().sum(),
    "min":       df[list(DATE_COLS.values())].min(),
    "median":    df[list(DATE_COLS.values())].median(),
    "max":       df[list(DATE_COLS.values())].max(),
})
print(summary)

# Raw values that failed to parse as dates (e.g. sentinel values like 9999 or empty strings)
print("\n=== Raw values that did NOT parse as a date (top 5 per column) ===")
for col, counts in bad_raw.items():
    if counts:
        bad = pd.concat(counts).groupby(level=0).sum().sort_values(ascending=False)
        if len(bad):
            print(f"{col}: {bad.head(5).to_dict()}")

print("\n=== Period columns (first values) ===")
for col in OTHER_COLS.values():
    if col in df:
        print(f"{col}: {df[col].value_counts(dropna=False).head(5).to_dict()}")

# --- 2. Consistency checks between the columns ---
print("\n=== Consistency checks ===")
checks = {
    "cohort_start == clalit_start":       df["cohort_start"] == df["clalit_start"],
    "cohort_start > clalit_start (gap?)": df["cohort_start"] > df["clalit_start"],
    "cohort_start < clalit_start (odd)":  df["cohort_start"] < df["clalit_start"],
    "cohort_end is empty":                df["cohort_end"].isna(),
    "clalit_end is empty":                df["clalit_end"].isna(),
    "cohort_end > 2026-06-30 (future)":   df["cohort_end"] > pd.Timestamp("2026-06-30"),
    "cohort_end < 2017-01-01":            df["cohort_end"] < pd.Timestamp("2017-01-01"),
    "cohort_start after 2017-01-01":      df["cohort_start"] > pd.Timestamp("2017-01-01"),
    "immigration after cohort_start":     df["immigration"] > df["cohort_start"],
    "age18 after cohort_start":           df["age18"] > df["cohort_start"],
}
for name, mask in checks.items():
    print(f"{name:40s} {int(mask.sum()):>10,}  ({mask.mean():.1%})")

# --- 3. Sample rows from each interesting group ---
show_cols = ["patient id", "birth", "age18", "immigration",
             "cohort_start", "cohort_end", "clalit_start", "clalit_end",
             "cohort_period", "clalit_period"]
show_cols = [c for c in show_cols if c in df]

groups = {
    "random":                          pd.Series(True, index=df.index),
    "cohort_start != clalit_start":    df["cohort_start"] != df["clalit_start"],
    "cohort_end empty":                df["cohort_end"].isna(),
    "cohort_end filled (left/died?)":  df["cohort_end"].notna() & (df["cohort_end"] <= pd.Timestamp("2026-06-30")),
    "cohort_end != clalit_end":        df["cohort_end"] != df["clalit_end"],
    "joined after 2017":               df["cohort_start"] > pd.Timestamp("2017-01-01"),
    "immigrants":                      df["immigration"].notna(),
    "turned 18 after 2017":            df["age18"] > pd.Timestamp("2017-01-01"),
}

samples = []
for name, mask in groups.items():
    sub = df.loc[mask.fillna(False), show_cols]
    sample = sub.sample(min(N_SAMPLES, len(sub)), random_state=0)
    print(f"\n=== {name}  (total {len(sub):,}) ===")
    print(sample.to_string(index=False))
    samples.append(sample.assign(group=name))

pd.concat(samples).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nSamples saved to {OUTPUT_CSV}")
