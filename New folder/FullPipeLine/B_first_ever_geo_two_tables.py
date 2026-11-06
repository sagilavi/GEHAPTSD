import pandas as pd
import numpy as np
import time
import os
import re
import unicodedata

# =============================================================================
# PART 2 — DERIVED VARIABLES (CRASH-SAFE CHUNKED VERSION, TWO OUTPUT TABLES)
#
# Processes 100,000 rows at a time.
# After each chunk: appends results to BOTH output CSVs and updates progress file.
# On restart: reads progress file to resume from last completed chunk.
#
# Progress file: same folder as main output, named "progress.txt"
#   - Empty or missing -> start from row 0
#   - Contains a number N -> resume from row N
#
# TWO OUTPUT TABLES:
#   Table 1 (OUTPUT_FILE_MAIN):      earliest diagnosis per outcome-period,
#                                     regardless of facility (same logic as before)
#   Table 2 (OUTPUT_FILE_INPATIENT): earliest diagnosis per outcome-period,
#                                     restricted to inpatient psychiatric facility only
#
# Both tables use 36 outcome-period column sets (9 outcomes x 4 periods),
# each with 9 metadata fields + diagnosis_vector string, and both have the
# incident case definition enforced independently.
#
# PERIOD BOUNDARIES:
#   Period 0 (Pre-baseline): any date before 2017-01-01 (2016 and earlier)
#   Period 1 (Baseline):     2017-01-01 - 2019-12-31
#   Period 2 (COVID):        2020-01-01 - 2023-10-06
#   Period 3 (War):          2023-10-07 - 2026-06-30
#
# INCIDENT CASE DEFINITION (applied independently to each table):
#   Period 0 clears Period 1, 2, 3
#   Period 1 clears Period 2, 3
#   Period 2 clears Period 3
#
# INPATIENT MATCHING (Table 2 only) — TWO RULES, either is sufficient:
#   Rule A (old list): facility contains "INP"; Hebrew-only name (after stripping
#     INP/underscores/spaces/non-Hebrew/nikud) matches known psychiatric hospital set.
#     פלימן and בניציון removed from that set.
#   Rule B (new):  facility_type == 'בי״ח נפש ממשלתי'  AND
#     raw facility string ends with '_INP' or '_AD' (case-insensitive).
#
# OUTCOME GROUPS (9 total, updated):
#   OCD added (F42).
#   Personality expanded to the union of F60 subtypes (F60.0-F60.9) AND
#   the broader F61-F69 range (mixed / enduring change / other / unspecified).
# =============================================================================

CHUNK_SIZE      = 100_000
INPUT_ENCODING  = "utf-8"
OUTPUT_ENCODING = "utf-8-sig"

# ── PERIOD BOUNDARIES ────────────────────────────────────────────────────────
PERIOD_0_START = pd.Timestamp("1900-01-01")
PERIOD_0_END   = pd.Timestamp("2016-12-31")
PERIOD_1_START = pd.Timestamp("2017-01-01")
PERIOD_1_END   = pd.Timestamp("2019-12-31")
PERIOD_2_START = pd.Timestamp("2020-01-01")
PERIOD_2_END   = pd.Timestamp("2023-10-06")
PERIOD_3_START = pd.Timestamp("2023-10-07")
PERIOD_3_END   = pd.Timestamp("2026-06-30")

PERIODS = {
    "period0": (PERIOD_0_START, PERIOD_0_END),
    "period1": (PERIOD_1_START, PERIOD_1_END),
    "period2": (PERIOD_2_START, PERIOD_2_END),
    "period3": (PERIOD_3_START, PERIOD_3_END),
}

OUTCOME_GROUPS = {
    "psychosis":   ["f20", "f20 only schizophrenic", "f21", "f22",
                    "f23", "f24", "f25", "f28", "f29"],
    "bipolar":     ["f30", "f31"],
    "ptsd":        ["f43.1"],
    "asr":         ["f43.0"],
    "anxiety":     ["f40", "f41"],
    "depression":  ["f32", "f33"],
    "OCD":         ["f42"],
    "substance":   ["f10", "f11", "f12", "f13", "f14",
                    "f15", "f16", "f17", "f18", "f19"],
    "personality": ["f60.0", "f60.1", "f60.2", "f60.3", "f60.4",
                    "f60.5", "f60.6", "f60.7", "f60.8", "f60.9",
                    "f61", "f62", "f63", "f64", "f65", "f66", "f68", "f69"],
}

META_FIELDS = [
    "age at event",
    "diagnosis",
    "start date",
    "department",
    "profession",
    "facility",
    "facility type",
    "data source",
    "original description",
]

# --- Inpatient hospital set: Hebrew-only names (non-Hebrew + nikud stripped before match) ---
# פלימן and בניציון intentionally removed
INPATIENT_HOSPITAL_NAMES = frozenset([
    'אברבנל',
    'המרכזלבריאותהנפשמעלההכרמל',
    'כפרשאול',
    'מזרע',
    'פרדסיהלבהשרון',
    'פסיכבאריעקב',
    'פסיכבש',
    'שערמנשה',
    'שלוותה',
    'גהה',
])

# --- Pre-stripped target for Rule B facility_type comparison (strip nikud once at load time) ---
_FACILITY_TYPE_GOVT_PSY = ''.join(
    c for c in unicodedata.normalize('NFD', 'בי״ח נפש ממשלתי')
    if unicodedata.category(c) != 'Mn'
).replace(' ', '')


# --- PART 2, STEP 1a: Strip nikud (Hebrew diacritics) from a string ---
# Decomposes to NFD then drops all Unicode combining characters (category Mn)
def strip_nikud(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# --- PART 2B, STEP 8: Normalize Hebrew text by keeping only Hebrew letters ---
# Strips spaces, dashes, parentheses, dots, and all non-Hebrew chars before matching
_STRIP_NON_HEBREW = re.compile(r'[^א-ת]')


def normalize_hebrew(text):
    if pd.isna(text):
        return None
    return _STRIP_NON_HEBREW.sub("", str(text))


# --- PART 2B, STEP 9: Bucket a distance-to-Gaza value into a proximity zone ---
# Boundaries are upper-inclusive: 0-7 / 7-20 / 20-30 / 30+ (km)
def assign_proximity_zone(dist):
    if pd.isna(dist):
        return None
    if dist <= 7:
        return "0-7km"
    elif dist <= 20:
        return "7-20km"
    elif dist <= 30:
        return "20-30km"
    else:
        return "30+km"


# --- PART 2B, STEP 10: Load Gaza border reference CSV and build clean lookup dict ---
# Loaded ONCE at startup; dict: clean_hebrew_city -> {distance, zone, shelter time}
def load_gaza_reference(ref_path):
    ref = pd.read_csv(ref_path, encoding=REF_ENCODING)
    ref.columns = ref.columns.str.strip()

    lookup = {}
    for _, row in ref.iterrows():
        key = normalize_hebrew(row["City"])
        if not key:
            continue

        dist = row["distance"]
        lookup[key] = {
            "distance_to_gaza_km": dist,
            "time_to_shelter":     row["Time to get to shelter"],
            "proximity_zone":      assign_proximity_zone(dist),
        }

    print(f"Gaza reference loaded: {len(lookup)} cities.")
    return lookup


# --- PART 2B, STEP 11: Add 4 geographic columns to one chunk via exact Hebrew match ---
# Unmatched residencies get NaN distance/zone/shelter; matched_city kept for QC
def add_geographic_exposure_chunk(chunk, lookup):
    chunk = chunk.copy()
    empty = {"distance_to_gaza_km": None, "time_to_shelter": None, "proximity_zone": None}

    # Per-chunk cache of unique residency values (avoids re-matching per row)
    cache = {}
    for res in chunk[RESIDENCY_COL].dropna().unique():
        clean = normalize_hebrew(res)
        cache[res] = lookup.get(clean, empty)

    chunk["distance_to_gaza_km"] = chunk[RESIDENCY_COL].map(
        lambda x: cache.get(x, empty).get("distance_to_gaza_km"))
    chunk["time_to_shelter"] = chunk[RESIDENCY_COL].map(
        lambda x: cache.get(x, empty).get("time_to_shelter"))
    chunk["proximity_zone"] = chunk[RESIDENCY_COL].map(
        lambda x: cache.get(x, empty).get("proximity_zone"))
    chunk["matched_city"] = chunk[RESIDENCY_COL].map(
        lambda x: x if (pd.notna(x) and normalize_hebrew(x) in lookup) else None)

    return chunk


# --- PART 2, STEP 1b: Check if a facility value is an inpatient psychiatric hospital ---
# Rule A: facility contains "INP" or ends with "_ NP" / "_NP" (typo for _INP);
#         Hebrew-only name (nikud-stripped) matches known hospital set.
#         Spaces between underscore and suffix are tolerated (e.g. 'הלל יפה_ INP').
# Rule B: facility_type == 'בי״ח נפש ממשלתי' AND raw facility ends with
#         _INP, _ INP, _NP, _ NP, _AD, or _ AD (spaces tolerated, case-insensitive).
def is_inpatient_psychiatric(facility_value, facility_type_value=None):
    s = str(facility_value)

    # Collapse "_ INP" / "_ NP" / "_ AD" -> "_INP" / "_INP" / "_AD"
    # so the rest of the logic never sees spaces next to the suffix
    s_norm = re.sub(r'_\s+NP\b', '_INP', s,  flags=re.IGNORECASE)  # _NP typo + spaces
    s_norm = re.sub(r'_\s+INP\b', '_INP', s_norm, flags=re.IGNORECASE)  # _ INP with space
    s_norm = re.sub(r'_\s+AD\b',  '_AD',  s_norm, flags=re.IGNORECASE)  # _ AD  with space

    # Rule A — old list match (nikud stripped before comparison)
    if 'INP' in s_norm:
        name_part   = s_norm.replace('INP', '').replace('_', '').replace(' ', '')
        hebrew_only = strip_nikud(re.sub(r'[^\u05d0-\u05ea\u05f0-\u05f4\ufb1d-\ufb4e]', '', name_part))
        if hebrew_only in INPATIENT_HOSPITAL_NAMES:
            return True

    # Rule B — government psychiatric hospital by facility type + suffix (spaces already collapsed)
    if facility_type_value is not None:
        ft_stripped = strip_nikud(str(facility_type_value).strip()).replace(' ', '')
        if ft_stripped == _FACILITY_TYPE_GOVT_PSY:
            if re.search(r'_(INP|NP|AD)$', s_norm, flags=re.IGNORECASE):
                return True

    return False


# --- PART 2, STEP 2: Read progress file — returns row index to resume from ---
# Returns 0 if file missing or empty; otherwise returns last completed row count
def read_progress(progress_file):
    if not os.path.exists(progress_file):
        return 0
    with open(progress_file, "r") as f:
        content = f.read().strip()
    if not content:
        return 0
    return int(content)


# --- PART 2, STEP 3: Write progress file — saves how many rows are done so far ---
# Overwrites the file with the current completed row count
def write_progress(progress_file, rows_done):
    with open(progress_file, "w") as f:
        f.write(str(rows_done))


# --- PART 2, STEP 4: Map each f-code to its actual column names in the dataframe ---
# Returns dict of meta-field -> column name (or None if column missing in df)
def get_fcode_columns(df, fcode):
    col_map = {}
    for field in META_FIELDS:
        col_name = f"diagnosis {fcode}-{field}"
        col_map[field] = col_name if col_name in df.columns else None
    return col_map


# --- PART 2, STEP 4b: Parse a diagnosis date column written by step A as DD/MM/YYYY text ---
# errors="coerce" would otherwise lose bad dates silently, so each column that
# has unparsed values is reported once
_reported_date_cols = set()


def parse_diagnosis_dates(values):
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    n_unparsed = int((values.notna() & parsed.isna()).sum())
    if n_unparsed and values.name not in _reported_date_cols:
        _reported_date_cols.add(values.name)
        print(f"\n  WARNING: {n_unparsed:,} unparsed dates in '{values.name}' (this chunk)")
    return parsed


# --- PART 2, STEP 5: For one outcome group and one period, find earliest diagnosis per patient ---
# inpatient_only=True restricts candidate dates to rows whose facility passes is_inpatient_psychiatric
# Returns df with earliest-date metadata + diagnosis_vector of all f-codes that fired in window
def first_diagnosis_in_period(df, fcode_list, period_start, period_end, inpatient_only=False):
    date_cols          = {}
    facility_cols      = {}
    facility_type_cols = {}
    for fcode in fcode_list:
        col_map            = get_fcode_columns(df, fcode)
        date_col           = col_map["start date"]
        facility_col       = col_map["facility"]
        facility_type_col  = col_map["facility type"]
        if date_col is not None and date_col in df.columns:
            date_cols[fcode]          = date_col
            facility_cols[fcode]      = facility_col
            facility_type_cols[fcode] = facility_type_col

    if not date_cols:
        empty = pd.DataFrame(index=df.index)
        for field in META_FIELDS:
            empty[field] = None
        empty["start date"]       = pd.NaT
        empty["diagnosis_vector"] = None
        return empty

    masked_dates = pd.DataFrame(index=df.index)
    for fcode, date_col in date_cols.items():
        s = parse_diagnosis_dates(df[date_col])
        s = s.where(s.between(period_start, period_end), other=pd.NaT)

        if inpatient_only:
            facility_col      = facility_cols.get(fcode)
            facility_type_col = facility_type_cols.get(fcode)

            fac_series  = (df[facility_col].astype(str)
                           if facility_col and facility_col in df.columns
                           else pd.Series('', index=df.index))
            ftype_series = (df[facility_type_col].astype(str)
                            if facility_type_col and facility_type_col in df.columns
                            else pd.Series('', index=df.index))

            fac_mask = pd.Series(
                [is_inpatient_psychiatric(fac, ftype)
                 for fac, ftype in zip(fac_series, ftype_series)],
                index=df.index,
            )
            s = s.where(fac_mask, other=pd.NaT)

        masked_dates[fcode] = s

    earliest_date  = masked_dates.min(axis=1)
    has_any_date   = masked_dates.notna().any(axis=1)
    earliest_fcode = pd.Series(index=df.index, dtype=object)
    if has_any_date.any():
        earliest_fcode[has_any_date] = masked_dates[has_any_date].idxmin(axis=1)
    earliest_fcode[~has_any_date] = None

    def build_vector(row):
        fired = row.dropna()
        if fired.empty:
            return None
        return ",".join(f.upper() for f in fired.sort_values().index.tolist())

    diagnosis_vector = masked_dates.apply(build_vector, axis=1)

    result = pd.DataFrame(index=df.index)
    for field in META_FIELDS:
        fcode_to_col = {
            fcode: get_fcode_columns(df, fcode)[field]
            for fcode in fcode_list
            if get_fcode_columns(df, fcode)[field] is not None
        }
        field_values = pd.Series(index=df.index, dtype=object)
        for fcode, col in fcode_to_col.items():
            mask = (earliest_fcode == fcode)
            field_values[mask] = df.loc[mask, col].values
        result[field] = field_values

    result["start date"]       = earliest_date
    result["diagnosis_vector"] = diagnosis_vector
    return result


# --- PART 2, STEP 6: Build all 36 outcome-period columns for one chunk ---
# inpatient_only passed straight through to first_diagnosis_in_period for every outcome-period
def build_outcome_period_columns_chunk(chunk, inpatient_only=False):
    all_results = []
    for outcome_name, fcode_list in OUTCOME_GROUPS.items():
        for period_name, (p_start, p_end) in PERIODS.items():
            period_df = first_diagnosis_in_period(
                chunk, fcode_list, p_start, p_end, inpatient_only=inpatient_only)
            period_df.columns = [
                f"{outcome_name}_{period_name}_{col}" for col in period_df.columns
            ]
            all_results.append(period_df)

    return pd.concat(all_results, axis=1)


# --- PART 2, STEP 7: Replace raw diagnosis columns with the 36 computed period columns ---
# Drops all original "diagnosis f*" columns and attaches computed period outcome sets
def build_clean_table(chunk, outcome_cols):
    diagnosis_cols = [col for col in chunk.columns if col.startswith("diagnosis")]
    chunk_clean    = chunk.drop(columns=diagnosis_cols)
    return pd.concat([chunk_clean, outcome_cols], axis=1)


# --- PART 2, STEP 8: Enforce incident case definition — one diagnosis per outcome group ever ---
# Period 0 clears 1+2+3; Period 1 clears 2+3; Period 2 clears 3; Period 3 never cleared
def enforce_incident_cases(df):
    df = df.copy()
    for outcome in OUTCOME_GROUPS.keys():
        date_p0 = f"{outcome}_period0_start date"
        date_p1 = f"{outcome}_period1_start date"
        date_p2 = f"{outcome}_period2_start date"

        fields  = META_FIELDS + ["diagnosis_vector"]
        cols_p1 = [f"{outcome}_period1_{f}" for f in fields]
        cols_p2 = [f"{outcome}_period2_{f}" for f in fields]
        cols_p3 = [f"{outcome}_period3_{f}" for f in fields]

        mask_p0 = df[date_p0].notna()
        df.loc[mask_p0, cols_p1] = None
        df.loc[mask_p0, cols_p2] = None
        df.loc[mask_p0, cols_p3] = None

        mask_p1 = df[date_p1].notna() & ~mask_p0
        df.loc[mask_p1, cols_p2] = None
        df.loc[mask_p1, cols_p3] = None

        mask_p2 = df[date_p2].notna() & ~mask_p0 & ~mask_p1
        df.loc[mask_p2, cols_p3] = None

    return df


# --- PART 2, STEP 8b: Add one first-ever-diagnosis-date column per outcome (all periods combined) ---
# Periods are contiguous and span the full dataset range, so the first-ever date
# per outcome is just the row-wise min of that outcome's 4 period start-date columns
def add_first_ever_columns(df):
    df = df.copy()
    for outcome in OUTCOME_GROUPS.keys():
        period_cols = [f"{outcome}_period{p}_start date" for p in range(4)]
        df[f"{outcome}_start date"] = df[period_cols].min(axis=1)
    return df


# --- PART 2, STEP 9: Process one chunk into both table variants ---
# Returns (chunk_clean_main, chunk_clean_inpatient) — both incident-case enforced independently
def process_chunk(chunk, gaza_lookup):
    chunk = add_geographic_exposure_chunk(chunk, gaza_lookup)

    outcome_cols_main = build_outcome_period_columns_chunk(chunk, inpatient_only=False)
    chunk_main = build_clean_table(chunk, outcome_cols_main)
    chunk_main = enforce_incident_cases(chunk_main)
    chunk_main = add_first_ever_columns(chunk_main)

    outcome_cols_inp = build_outcome_period_columns_chunk(chunk, inpatient_only=True)
    chunk_inp = build_clean_table(chunk, outcome_cols_inp)
    chunk_inp = enforce_incident_cases(chunk_inp)
    chunk_inp = add_first_ever_columns(chunk_inp)

    return chunk_main, chunk_inp


# --- PART 2, STEP 9b: Stop if an outcome has no diagnosis columns in the input ---
# Without this, a missing outcome would come out as an empty group with no error
def check_input_has_outcomes():
    header = set(pd.read_csv(INPUT_FILE, encoding=INPUT_ENCODING, nrows=0).columns)
    for outcome, fcode_list in OUTCOME_GROUPS.items():
        missing = [f for f in fcode_list if f"diagnosis {f}-start date" not in header]
        if len(missing) == len(fcode_list):
            raise RuntimeError(f"No 'start date' columns in the input for outcome '{outcome}': {missing}")
        if missing:
            print(f"WARNING: outcome '{outcome}' has no input columns for: {missing}")


# --- PART 2, STEP 9c: Stop a resume onto output files built with a different outcome list ---
# Appending would put the new columns under the old header
def check_resume_matches_outcomes():
    for output_file in [OUTPUT_FILE_MAIN, OUTPUT_FILE_INPATIENT]:
        if not os.path.exists(output_file):
            continue
        header = set(pd.read_csv(output_file, encoding=OUTPUT_ENCODING, nrows=0).columns)
        missing = [o for o in OUTCOME_GROUPS if f"{o}_start date" not in header]
        if missing:
            raise RuntimeError(
                f"{output_file} was built without outcomes {missing}. "
                f"To rerun from scratch: delete or empty {PROGRESS_FILE}"
            )


# --- PART 2, STEP 10: Main chunked processing loop — resume-safe, writes 2 output tables ---
# Both output files are written per chunk before the progress file is updated,
# so a crash mid-chunk never leaves the two tables out of sync
def main():
    t_total = time.time()

    check_input_has_outcomes()

    rows_done = read_progress(PROGRESS_FILE)
    if rows_done == 0:
        print("Starting from the beginning.")
        for f in [OUTPUT_FILE_MAIN, OUTPUT_FILE_INPATIENT]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Deleted old output file: {f}")
    else:
        check_resume_matches_outcomes()
        print(f"Resuming from row {rows_done:,} (found in progress file).")

    gaza_lookup = load_gaza_reference(REF_PATH)

    print(f"\nCounting input rows...")
    with open(INPUT_FILE, "r", encoding=INPUT_ENCODING) as f:
        total_rows = sum(1 for _ in f) - 1
    print(f"Input has {total_rows:,} rows.")

    if rows_done >= total_rows:
        print("All rows already processed. Nothing to do.")
        return

    chunk_num    = rows_done // CHUNK_SIZE + 1
    write_header = not os.path.exists(OUTPUT_FILE_MAIN)

    reader = pd.read_csv(
        INPUT_FILE, encoding=INPUT_ENCODING, low_memory=False,
        chunksize=CHUNK_SIZE,
        skiprows=range(1, rows_done + 1) if rows_done else None,
    )

    for chunk in reader:
        start = rows_done
        end   = min(start + len(chunk), total_rows)

        t0 = time.time()
        print(f"\n-- Chunk {chunk_num} | rows {start:,}-{end:,} of {total_rows:,} --")

        print(f"  Building both table variants (main + inpatient)...", end=" ")
        chunk_main, chunk_inp = process_chunk(chunk, gaza_lookup)
        print("done")

        print(f"  Writing main table...", end=" ")
        chunk_main.to_csv(
            OUTPUT_FILE_MAIN, index=False, encoding=OUTPUT_ENCODING,
            mode="w" if write_header else "a", header=write_header,
        )
        print("done")

        print(f"  Writing inpatient-only table...", end=" ")
        chunk_inp.to_csv(
            OUTPUT_FILE_INPATIENT, index=False, encoding=OUTPUT_ENCODING,
            mode="w" if write_header else "a", header=write_header,
        )
        print("done")

        write_header = False

        # Only update progress after BOTH files are safely written for this chunk
        rows_done = end
        write_progress(PROGRESS_FILE, rows_done)

        elapsed = time.time() - t0
        print(f"  Chunk {chunk_num} complete in {elapsed:.1f}s | "
              f"Progress: {rows_done:,}/{total_rows:,} rows "
              f"({rows_done/total_rows*100:.1f}%)")

        chunk_num += 1

    total_elapsed = time.time() - t_total
    print(f"\nAll {total_rows:,} rows complete in {total_elapsed/60:.1f} min.")
    print(f"Main output:      {OUTPUT_FILE_MAIN}")
    print(f"Inpatient output: {OUTPUT_FILE_INPATIENT}")
    print(f"Progress file:    {PROGRESS_FILE}")
    print("To rerun from scratch: delete or empty the progress file.")


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE            = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar\matched_cohort_full.csv"


OUTPUT_FILE_MAIN       = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar\matched_cohort_PeriodOutComes.csv"
OUTPUT_FILE_INPATIENT  = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar\matched_cohort_PeriodOutComes_Inpatient.csv"
PROGRESS_FILE          = os.path.join(os.path.dirname(OUTPUT_FILE_MAIN), "progress.txt")

REF_PATH        = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\info\GazaBorderSettlement.csv"
REF_ENCODING    = "windows-1255"
RESIDENCY_COL   = "demographics_2023-residency"

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
main()

# =============================================================================
# SANITY CHECKS — run after completion:
#
# df_main = pd.read_csv(OUTPUT_FILE_MAIN,      encoding=OUTPUT_ENCODING, low_memory=False)
# df_inp  = pd.read_csv(OUTPUT_FILE_INPATIENT, encoding=OUTPUT_ENCODING, low_memory=False)
#
# --- CHECK 1: Row counts match input and each other ---
# n_input = len(pd.read_csv(INPUT_FILE, encoding=INPUT_ENCODING))
# assert len(df_main) == n_input and len(df_inp) == n_input, "Row count mismatch!"
# print(f"CHECK 1 PASSED: {n_input:,} rows in input, main, and inpatient tables.")
#
# --- CHECK 2: Inpatient table has <= diagnoses than main table per outcome-period ---
# for outcome in OUTCOME_GROUPS.keys():
#     for period in ["period0","period1","period2","period3"]:
#         n_main = df_main[f"{outcome}_{period}_start date"].notna().sum()
#         n_inp  = df_inp[f"{outcome}_{period}_start date"].notna().sum()
#         assert n_inp <= n_main, f"FAIL {outcome}_{period}: inpatient ({n_inp}) > main ({n_main})!"
#         print(f"{outcome}_{period}: main={n_main}, inpatient={n_inp}")
# print("CHECK 2 PASSED.")
#
# --- CHECK 3: פלימן and בניציון are absent from inpatient facility columns ---
# banned = ['פלימן', 'בניציון']
# for outcome in OUTCOME_GROUPS.keys():
#     for period in ["period0","period1","period2","period3"]:
#         col = f"{outcome}_{period}_facility"
#         if col in df_inp.columns:
#             vals = df_inp[col].dropna().astype(str)
#             for b in banned:
#                 hits = vals[vals.str.contains(b, na=False)]
#                 assert len(hits) == 0, f"FAIL: {b} found in {col}! ({len(hits)} rows)"
# print("CHECK 3 PASSED: פלימן and בניציון absent from inpatient table.")
#
# --- CHECK 4: Inpatient diagnosis counts broken down by facility (all outcomes, all periods) ---
# rows = []
# for outcome in OUTCOME_GROUPS.keys():
#     for period in ["period0","period1","period2","period3"]:
#         fac_col  = f"{outcome}_{period}_facility"
#         ftype_col = f"{outcome}_{period}_facility type"
#         date_col = f"{outcome}_{period}_start date"
#         if fac_col not in df_inp.columns or date_col not in df_inp.columns:
#             continue
#         sub = df_inp[[fac_col, ftype_col, date_col]].dropna(subset=[date_col])
#         counts = sub[fac_col].fillna('MISSING').value_counts()
#         for fac_name, n in counts.items():
#             rows.append({'outcome': outcome, 'period': period,
#                          'facility': fac_name, 'n': n})
# sanity_df = pd.DataFrame(rows)
# print("\n=== INPATIENT DIAGNOSES PER FACILITY (all outcomes + periods) ===")
# print(sanity_df.groupby('facility')['n'].sum().sort_values(ascending=False).to_string())
# print("\n=== INPATIENT DIAGNOSES PER OUTCOME x FACILITY ===")
# print(sanity_df.groupby(['outcome','facility'])['n'].sum()
#       .sort_values(ascending=False).to_string())
#
# --- CHECK 5: NEW — OCD and expanded personality columns exist and are populated ---
# for outcome in ["OCD", "personality"]:
#     for period in ["period0","period1","period2","period3"]:
#         col = f"{outcome}_{period}_start date"
#         assert col in df_main.columns, f"Missing expected column: {col}"
#         print(f"{col}: n_non_null={df_main[col].notna().sum()}")
# print("CHECK 5 PASSED.")
#
# print("\nAll sanity checks passed.")
# =============================================================================