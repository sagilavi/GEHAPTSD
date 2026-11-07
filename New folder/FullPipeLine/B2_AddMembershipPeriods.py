import os
import sys
import numpy as np
import pandas as pd

# =============================================================================
# B2 -- ADD MEMBERSHIP PERIODS + REMOVE DUPLICATE COLUMNS
#
# Runs between step B and step C. Patients leave Clalit and rejoin, so one
# membership start/end pair can't describe them. This step:
#
#   1. Removes columns whose content is identical to an earlier column, keeping
#      the first. Never removed, even when identical:
#        - columns starting with "diagnosis"
#        - columns that are entirely empty (all empty columns look identical)
#        - columns script C reads by name, and every "<outcome>_..." column
#        - the patient id and membership columns this step writes
#   2. Reads the membership file (one row per membership period, at most the
#      first 150 per patient), merges periods that overlap or are back to back
#      (gap of 1 day or less), and keeps the ones that overlap the cohort time.
#   3. Adds MultipleMemberships:
#        0  -> exactly 1 period; membership start/end are set to that period
#        "start-end:start-end:..." -> more than 1 period; membership start/end
#              are set to the first start and the last end
#        -1 -> no period in the cohort time (or no rows); start/end unchanged
#
# NOTE: script C ends follow-up at the membership end only, so for patients
# with more than 1 period the gaps between periods still count as follow-up
# until script C reads MultipleMemberships.
#
# Dates are written as DD/MM/YYYY, the same format as the rest of the file;
# 06/06/2079 means still a member.
# =============================================================================

COHORT_START = pd.Timestamp("2017-01-01")
COHORT_END   = pd.Timestamp("2026-06-30")   # = STUDY_END in script C
MAX_MERGE_GAP_DAYS = 1
EXTRACTION_LIMIT   = 150   # the membership extract holds at most this many periods per patient

CHUNK_ROWS = 100_000
ENCODING   = "utf-8-sig"

PATIENT_ID_COL       = "patient id"
MEMBERSHIP_START_COL = "cohort reference event-membership start date"
MEMBERSHIP_END_COL   = "cohort reference event-membership end date"
MULTIPLE_COL         = "MultipleMemberships"

SOURCE_ID_COL    = "mdcinternalpatientid"
SOURCE_START_COL = "clalit membership-membership start date"
SOURCE_END_COL   = "clalit membership-membership end date"

DATE_FORMAT = "%d/%m/%Y"   # written to the output
# The membership file holds "2000-01-06 00:00:00"; DD/MM/YYYY is accepted too. Only the date part is read.
MEMBERSHIP_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y"]


def parse_membership_dates(values):
    text = values.str.strip().str[:10]
    parsed = pd.to_datetime(text, format=MEMBERSHIP_DATE_FORMATS[0], errors="coerce")
    for fmt in MEMBERSHIP_DATE_FORMATS[1:]:
        parsed = parsed.fillna(pd.to_datetime(text, format=fmt, errors="coerce"))
    return parsed


def columns_script_c_reads():
    """Column names step C loads, and its outcome names, read from step C itself."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import C_ChangeInIRRByExposre as step_c
    return set(step_c.NEEDED_COLS), list(step_c.ALL_OUTCOMES)


# --- STEP 1: Find columns whose content repeats an earlier column ---
def find_duplicate_columns(path, protected_names, outcomes):
    header = list(pd.read_csv(path, encoding=ENCODING, nrows=0).columns)
    n_cols = len(header)
    # Two independent 64-bit hashes per column, summed over rows. The row number is part
    # of each hash, so the result depends on order as well as values.
    hash_a = np.zeros(n_cols, dtype=np.uint64)
    hash_b = np.zeros(n_cols, dtype=np.uint64)
    non_empty = np.zeros(n_cols, dtype=bool)
    n_rows = 0

    reader = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False,
                         chunksize=CHUNK_ROWS)
    for chunk in reader:
        chunk.index = pd.RangeIndex(n_rows, n_rows + len(chunk))
        n_rows += len(chunk)
        for i in range(n_cols):
            values = chunk.iloc[:, i]
            hash_a[i] += pd.util.hash_pandas_object(values, index=True).to_numpy().sum(dtype=np.uint64)
            hash_b[i] += pd.util.hash_pandas_object(
                values, index=True, hash_key="membershipcheck1").to_numpy().sum(dtype=np.uint64)
            non_empty[i] |= bool((values != "").any())
        print(f"  {n_rows:,} rows hashed...", end="\r")
    print(f"  {n_rows:,} rows hashed.     ")

    def is_protected(i):
        col = header[i]
        return (col.startswith("diagnosis") or not non_empty[i] or col in protected_names
                or any(col.startswith(f"{o}_") for o in outcomes))

    first_seen = {}
    duplicates = {}
    for i, col in enumerate(header):
        key = (int(hash_a[i]), int(hash_b[i]))
        if key not in first_seen:
            first_seen[key] = col
        elif not is_protected(i):
            duplicates[col] = first_seen[key]
    return header, duplicates, n_rows


def normalize_id(values):
    """'0x' + upper-case hex, whatever surrounds it or its letter case; NaN when there is none."""
    return "0x" + values.str.extract(r"0[xX]([0-9A-Fa-f]+)", expand=False).str.upper()


# --- STEP 2: Membership periods per patient, merged and cut to the cohort time ---
def read_membership_periods(path):
    header = list(pd.read_csv(path, encoding=ENCODING, nrows=0).columns)
    missing_cols = [c for c in (SOURCE_ID_COL, SOURCE_START_COL, SOURCE_END_COL) if c not in header]
    if missing_cols:
        raise RuntimeError(f"Membership file has no column(s) {missing_cols}. Its columns: {header}")

    parts = []
    n_rows = 0
    failures = {"no 0x id": [0, []], "bad start date": [0, []], "bad end date": [0, []]}
    reader = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False,
                         usecols=[SOURCE_ID_COL, SOURCE_START_COL, SOURCE_END_COL],
                         chunksize=1_000_000)
    for chunk in reader:
        n_rows += len(chunk)
        period = pd.DataFrame({
            "id": normalize_id(chunk[SOURCE_ID_COL]),
            "start": parse_membership_dates(chunk[SOURCE_START_COL]),
            "end": parse_membership_dates(chunk[SOURCE_END_COL]),
        })
        for reason, field, raw_col in [("no 0x id", "id", SOURCE_ID_COL),
                                       ("bad start date", "start", SOURCE_START_COL),
                                       ("bad end date", "end", SOURCE_END_COL)]:
            failed = period[field].isna()
            failures[reason][0] += int(failed.sum())
            if len(failures[reason][1]) < 3:
                failures[reason][1] += chunk.loc[failed, raw_col].head(3).tolist()
        parts.append(period.dropna())
    periods = pd.concat(parts, ignore_index=True)
    print(f"Membership file: {n_rows:,} rows, {periods['id'].nunique():,} patients")
    for reason, (n_failed, examples) in failures.items():
        if n_failed:
            print(f"  WARNING: {n_failed:,} rows skipped, {reason}. "
                  f"Examples: {examples[:3]}")
    if periods.empty:
        raise RuntimeError("No usable membership rows. See the skipped-row examples above.")

    rows_per_patient = periods.groupby("id").size()
    n_at_limit = int((rows_per_patient >= EXTRACTION_LIMIT).sum())
    if n_at_limit:
        print(f"  WARNING: {n_at_limit:,} patients have {EXTRACTION_LIMIT}+ periods, "
              f"so later periods may be missing from the extract")

    # Merge periods that overlap or are back to back
    periods = periods.sort_values(["id", "start", "end"]).reset_index(drop=True)
    running_end = periods.groupby("id")["end"].cummax()
    previous_end = running_end.groupby(periods["id"]).shift()
    starts_new = previous_end.isna() | (periods["start"] > previous_end + pd.Timedelta(days=MAX_MERGE_GAP_DAYS))
    periods["block"] = starts_new.cumsum()
    merged = periods.groupby("block").agg(id=("id", "first"), start=("start", "min"), end=("end", "max"))

    in_cohort = (merged["start"] <= COHORT_END) & (merged["end"] >= COHORT_START)
    print(f"  {merged.loc[in_cohort, 'id'].nunique():,} patients have a period overlapping "
          f"{COHORT_START:%d/%m/%Y}-{COHORT_END:%d/%m/%Y}")
    return merged[in_cohort].reset_index(drop=True)


def check_ids_match(summary):
    """Stops before the long passes if the cohort's patient ids don't match the membership ids."""
    raw_ids = pd.read_csv(INPUT_FILE, encoding=ENCODING, dtype=str, keep_default_na=False,
                          usecols=[PATIENT_ID_COL])[PATIENT_ID_COL]
    matched = int(normalize_id(raw_ids).isin(summary.index).sum())
    print(f"Cohort patients found in the membership periods: {matched:,} of {len(raw_ids):,} "
          f"({matched / len(raw_ids):.1%})")
    if matched == 0:
        print(f"  cohort '{PATIENT_ID_COL}' examples:   {raw_ids.head(3).tolist()}")
        print(f"  membership id examples (cleaned): {summary.index[:3].tolist()}")
        raise RuntimeError("No patient matched the membership file. Compare the id examples above.")


def summarize_periods(periods):
    """One row per patient: MultipleMemberships, first start, last end."""
    periods = periods.sort_values(["id", "start"])
    text = (periods["start"].dt.strftime(DATE_FORMAT) + "-" + periods["end"].dt.strftime(DATE_FORMAT))
    grouped = text.groupby(periods["id"])
    summary = pd.DataFrame({
        "count": grouped.size(),
        "all_periods": grouped.agg(":".join),
        "first_start": periods.groupby("id")["start"].min().dt.strftime(DATE_FORMAT),
        "last_end": periods.groupby("id")["end"].max().dt.strftime(DATE_FORMAT),
    })
    summary[MULTIPLE_COL] = np.where(summary["count"] == 1, "0", summary["all_periods"])
    return summary[[MULTIPLE_COL, "first_start", "last_end"]]


def add_memberships(chunk, summary):
    found = summary.reindex(normalize_id(chunk[PATIENT_ID_COL]))
    has_period = found[MULTIPLE_COL].notna().to_numpy()
    for col, value_col in [(MEMBERSHIP_START_COL, "first_start"), (MEMBERSHIP_END_COL, "last_end")]:
        if col not in chunk.columns:
            chunk[col] = ""
        chunk.loc[has_period, col] = found[value_col].to_numpy()[has_period]
    chunk[MULTIPLE_COL] = found[MULTIPLE_COL].fillna("-1").to_numpy()
    return chunk


# --- STEP 3: Write the output ---
def main():
    protected_names, outcomes = columns_script_c_reads()
    protected_names |= {PATIENT_ID_COL, MEMBERSHIP_START_COL, MEMBERSHIP_END_COL}

    # Membership first: a mismatch stops the run in minutes, not after the full passes
    print(f"Reading membership periods from {MEMBERSHIP_FILE}...")
    summary = summarize_periods(read_membership_periods(MEMBERSHIP_FILE))
    check_ids_match(summary)

    print(f"\nFinding duplicate columns in {INPUT_FILE}...")
    header, duplicates, n_input_rows = find_duplicate_columns(INPUT_FILE, protected_names, outcomes)
    for col, same_as in duplicates.items():
        print(f"  Dropping '{col}' (same content as '{same_as}')")
    print(f"Dropping {len(duplicates)} of {len(header)} columns")
    kept_columns = [col for col in header if col not in duplicates]
    if PATIENT_ID_COL not in kept_columns:
        raise RuntimeError(f"'{PATIENT_ID_COL}' not found in {INPUT_FILE}")
    for col in (MEMBERSHIP_START_COL, MEMBERSHIP_END_COL):
        if col not in kept_columns:
            print(f"  NOTE: '{col}' not in the input, it will be added")

    partial_path = OUTPUT_FILE + ".partial"
    counts = {"0": 0, "multiple": 0, "-1": 0}
    n_written = 0
    print(f"\nWriting {OUTPUT_FILE}...")
    with open(partial_path, "w", encoding=ENCODING, newline="") as output_file:
        reader = pd.read_csv(INPUT_FILE, encoding=ENCODING, dtype=str, keep_default_na=False,
                             usecols=kept_columns, chunksize=CHUNK_ROWS)
        for chunk in reader:
            chunk = add_memberships(chunk[kept_columns], summary)
            chunk.to_csv(output_file, header=(n_written == 0), index=False)
            n_written += len(chunk)
            values = chunk[MULTIPLE_COL]
            counts["0"] += int((values == "0").sum())
            counts["-1"] += int((values == "-1").sum())
            counts["multiple"] += int((~values.isin(["0", "-1"])).sum())
            print(f"  {n_written:,} rows written...", end="\r")

    if n_written != n_input_rows:
        raise RuntimeError(f"Row count changed: input {n_input_rows:,}, output {n_written:,}")
    os.replace(partial_path, OUTPUT_FILE)

    print(f"\n\nMultipleMemberships for {n_written:,} patients:")
    print(f"   0 (1 period):            {counts['0']:,}")
    print(f"   2+ periods:              {counts['multiple']:,}")
    print(f"  -1 (no period 2017-2026): {counts['-1']:,}")
    print(f"\nSaved to: {OUTPUT_FILE}")


# =============================================================================
# PATHS
# =============================================================================
FOLDER          = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar"
INPUT_FILE      = os.path.join(FOLDER, "matched_cohort_PeriodOutComes.csv")
OUTPUT_FILE     = os.path.join(FOLDER, "matched_cohort_PeriodOutComes_Membership.csv")
MEMBERSHIP_FILE = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\RawData\MetaData\Memberships_Full.csv"

if __name__ == "__main__":
    main()
