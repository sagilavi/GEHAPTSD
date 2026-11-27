# C_IRR_Core.py -- LOGIC 1/4: run-time state, loading and the person-time core.
# Nothing to edit here: the run settings are in the USER SETTINGS block of
# C_ChangeInIRRByExposre.py. set_window_mode() / set_trauma_mode() / set_diagnoser_mode() /
# set_followup() change the state the other modules read as core.NAME.
# Part of the pipeline run by C_ChangeInIRRByExposre.py (all files in the same directory):
#   C_IRR_Core.py                  -- run-time state, loading, membership / follow-up /
#                                     person-time core, population variants
#   C_IRR_Analyses.py              -- the computations behind Statements 1-7 and the descriptive PDF
#   C_IRR_Models.py                -- adjusted IRR (Poisson) and the era comparisons
#   C_IRR_Checks.py                -- automated sanity checks
#   C_IRR_Tables.py                -- table formatting
#   C_IRR_Graphs_Statement1.py     -- graphs: colors, Statement 1 + its supporting tests
#   C_IRR_Graphs_Statements2to6.py -- graphs: Statements 2, 3, 5, 6 and comorbidity
#   C_IRR_Graphs_Demog_Models.py   -- graphs: demographic / relative-year PDFs, adjusted IRR
#                                     forests, era comparisons, section dividers, descriptive charts
#   C_ChangeInIRRByExposre.py      -- MAIN: USER SETTINGS (everything you edit), PDF assembly
#                                     (export_all*), __main__

import os
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

OCT7 = pd.Timestamp("2023-10-07")

ALL_OUTCOMES = ["psychosis", "bipolar", "ptsd", "asr",
                "anxiety", "depression", "substance", "personality", "OCD"]

AGE_BANDS = {
    "age15_40":   ("Age 15-40", 15, 40),
    "age15_plus": ("Age 15+",   15, 120),
}

# --- Trauma exposure mode: PTSD/ASR merged (default), PTSD only, or ASR only ----------------
# TRAUMA_EXPOSURE_OUTCOMES / TRAUMA_EXPOSURE_LABEL are read by every function below that deals
# with "the trauma exposure" -- Statement 1 (H2 bars, pre/post heatmap, TEST 2b/5/6/9) and the
# "trauma" entry in Statement 2's alternate-exposure set. export_all_trauma_modes() sets these
# once per run, before calling export_all() for that mode -- see TRAUMA_MODES / bottom of file.
TRAUMA_MODES = {
    "ptsd_asr":  (["ptsd", "asr"], "PTSD/ASR"),   # merged (matches the rest of the pipeline)
    "ptsd_only": (["ptsd"],        "PTSD"),
    "asr_only":  (["asr"],         "ASR"),
}
TRAUMA_EXPOSURE_OUTCOMES, TRAUMA_EXPOSURE_LABEL = TRAUMA_MODES["ptsd_asr"]   # default/initial

# --- Diagnoser mode: any diagnoser (default), psychiatrist only, or non-psychiatrist only ----
# Applies to EVERY exposure definition in the script (trauma in Statement 1, and all 8
# alternate exposures in Statement 2's B1 analysis) -- filters "exposed" to only those whose
# EARLIEST diagnosis of that exposure was given by a psychiatrist / non-psychiatrist / anyone.
# Implemented inside first_of() below (the single function every exposure-date lookup goes
# through), so no other function needs to change to respect this filter. set_diagnoser_mode()
# sets these once per run -- see export_all_diagnoser_modes() at the bottom of the file.
DIAGNOSER_MODES = {
    "any_diagnoser":         None,
    "psychiatrist_only":     "psychiatrist",
    "non_psychiatrist_only": "non_psychiatrist",
}
DIAGNOSER_LABELS = {
    "any_diagnoser":         "Any diagnoser",
    "psychiatrist_only":     "Psychiatrist-diagnosed only",
    "non_psychiatrist_only": "Non-psychiatrist-diagnosed only",
}
CURRENT_DIAGNOSER_MODE = "any_diagnoser"
CURRENT_DIAGNOSER_LABEL = DIAGNOSER_LABELS[CURRENT_DIAGNOSER_MODE]
_current_outcome_profession_meta = None   # set by set_diagnoser_mode(); needed to look up who
                                          # diagnosed each outcome's earliest occurrence

# --- Set from the USER SETTINGS block of C_ChangeInIRRByExposre.py when it starts -----------
# Do not edit them here (main overwrites them); other modules read them as core.NAME.
SELECTED_TRAUMA_MODE = None   # trauma mode used when a function is not given one
LAG_DAYS = None               # protopathic-bias exclusion window, in days (0 = off)

# --- Statement 1 & 2: exposure definitions -----------------------------------
# 8 possible entry exposures (every outcome group can be an "entry" diagnosis). The trauma
# entry is built dynamically from TRAUMA_EXPOSURE_OUTCOMES/LABEL so it follows the active
# trauma mode -- call current_alt_exposures() rather than a static module dict.
def current_alt_exposures():
    return {
        TRAUMA_EXPOSURE_LABEL: TRAUMA_EXPOSURE_OUTCOMES,
        "Depression":  ["depression"],
        "Anxiety":     ["anxiety"],
        "Substance":   ["substance"],
        "Bipolar":     ["bipolar"],
        "Psychosis":   ["psychosis"],
        "Personality": ["personality"],
        "OCD":         ["OCD"],
    }
FOCUS_OUTCOMES = ["psychosis", "bipolar"]     # statement 1 & 2 downstream targets
S1_DOWNSTREAM  = ["psychosis", "bipolar", "anxiety",
                  "depression", "substance", "personality", "OCD"]  # H2's original 6 + OCD

# --- Window mode: WHOLE-YEAR windows only (12 or 24 months) -- deliberately not 18. A window
# that isn't a whole number of years is NOT seasonally symmetric: a patient entering in January
# sees their window end in summer, one entering in July sees it end in winter -- so cohorts
# entering at different times of year get different seasonal mixes inside the same "months"
# label, which can bias a pre/post comparison if the disorder itself has any seasonality.
# 12 or 24 months means every entry month is followed by exactly one full cycle of every
# calendar month, so this cancels out regardless of when someone entered.
WINDOW_MODES = {"12mo": 365, "24mo": 730}
FIXED_WINDOW_DAYS = WINDOW_MODES["12mo"]   # current active window; set by set_window_mode()

def set_window_mode(mode_key):
    global FIXED_WINDOW_DAYS
    FIXED_WINDOW_DAYS = WINDOW_MODES[mode_key]

STUDY_END = pd.Timestamp("2026-06-30")   # for window-reachability censoring, matches Step 5
REGISTRY_START = pd.Timestamp("2017-01-01")   # earliest point with data -- start of continuous
                                              # general-population tracking (see PERSON-TIME
                                              # CORE below). Replaces the old 5-anchor design.

# --- FOLLOW-UP END: person-time stops when the patient leaves Clalit -------------------------
# A diagnosis can only be recorded while a patient is an active member, so after membership ends
# we don't know what happened -- counting that time would inflate the denominator.
# Per patient (built in build_frame()):
#   fu_end = earliest of ("cohort reference event-membership end date", STUDY_END)
#            (2079-06-06 = still a member, capped at STUDY_END; missing -> STUDY_END)
#   fu_start = later of (REGISTRY_START, 15th birthday) -- only the END comes from membership.
#   Days between two membership periods are not person-time -- see MEMBERSHIP GAPS below.
# Rules, applied to EVERY person-time calculation below:
#   - EXPOSED: at risk from the exposure (PTSD/ASR) date until the EARLIEST of -- the downstream
#     diagnosis, entry + FIXED_WINDOW_DAYS, fu_end. A first-ever exposure before the 15th
#     birthday is not an entry, and that patient is not general either -- excluded from both
#     (enforced in first_of()). An exposure before 2017 at age 15+ is still an entry.
#     Membership start / immigration date play no role.
#   - GENERAL: tracked from fu_start until the earliest of exposure / downstream / fu_end.
#   - PREVALENT CASES: a patient whose first-ever outcome is dated on/before the start of their
#     at-risk time (entry for exposed, fu_start for general) is not at risk of a FIRST-ever
#     outcome and contributes no person-time for that outcome -- incl. an outcome before age 15.
#   - STATEMENT 3: diagnoses and person-years both counted only from fu_start (age 15).
MEMBERSHIP_END_COL   = "cohort reference event-membership end date"
MARKETING_SECTOR_COL = "marketing sector"
FOLLOWUP_MIN_AGE     = 15
_followup_end = None   # per-patient fu_end Series (full-population index), set by set_followup()

# --- MEMBERSHIP GAPS: only days as a member count as person-time ----------------------------
# Patients leave Clalit and come back. Step B2 lists each patient's membership periods in
# 2017-2026 in MultipleMemberships ("0" = one period, "start-end:start-end:..." = several,
# "-1" = none). Every person-time calculation goes through member_days(), which removes the
# days between two periods: member 1.1.2017-1.1.2018 and 1.1.2019-1.1.2020 -> 2 years, not 3.
# Follow-up still starts at fu_start and ends at fu_end (= the last membership end) as above;
# windows stay calendar windows (a gap inside one is not counted, the window is not extended).
# A diagnosis dated inside a gap is still counted as an event (see the sanity checks).
MEMBERSHIP_PERIODS_COL = "MultipleMemberships"
_membership_gaps = None      # DataFrame indexed by frame row: gap_start, gap_end
_gaps_by_patient = {}        # same gaps, {frame row: [(gap_start, gap_end), ...]}


def parse_membership_gaps(periods_text):
    """Gaps between consecutive membership periods, one row per gap, indexed by frame row."""
    multi = periods_text.dropna().astype(str)
    multi = multi[multi.str.contains(":")]
    if multi.empty:
        return pd.DataFrame({"gap_start": pd.Series(dtype="datetime64[ns]"),
                             "gap_end": pd.Series(dtype="datetime64[ns]")})
    pieces = multi.str.split(":").explode()
    bounds = pieces.str.split("-", expand=True)
    periods = pd.DataFrame({"start": pd.to_datetime(bounds[0], format="%d/%m/%Y", errors="coerce"),
                            "end": pd.to_datetime(bounds[1], format="%d/%m/%Y", errors="coerce")},
                           index=pieces.index).dropna()
    periods = periods.reset_index().sort_values(["index", "start"]).set_index("index")
    next_start = periods.groupby(level=0)["start"].shift(-1)
    gaps = pd.DataFrame({"gap_start": periods["end"], "gap_end": next_start}).dropna()
    gaps.index.name = None
    return gaps[gaps["gap_end"] > gaps["gap_start"]]


def set_followup(frame):
    global _followup_end, _membership_gaps, _gaps_by_patient
    _followup_end = frame["fu_end"]
    if MEMBERSHIP_PERIODS_COL in frame.columns:
        _membership_gaps = parse_membership_gaps(frame[MEMBERSHIP_PERIODS_COL])
    else:
        print(f"WARNING: '{MEMBERSHIP_PERIODS_COL}' not found -- gaps between membership periods "
              f"are counted as follow-up (run step B2 first)")
        _membership_gaps = parse_membership_gaps(pd.Series(dtype=object))
    _gaps_by_patient = {}
    for row, gap_start, gap_end in zip(_membership_gaps.index, _membership_gaps["gap_start"],
                                       _membership_gaps["gap_end"]):
        _gaps_by_patient.setdefault(row, []).append((gap_start, gap_end))
    n_patients = _membership_gaps.index.nunique()
    print(f"Membership gaps: {len(_membership_gaps):,} gaps in {n_patients:,} patients "
          f"are left out of person-time")


def member_days(start, end, gaps=None):
    """Days from start to end (per patient, index-aligned Series) minus the days that fall in a
    membership gap. Negative/NaT behave as end - start does, so callers clip as before.
    `gaps` defaults to the loaded patients' gaps (the sanity checks pass their own)."""
    days = (end - start).dt.days
    gaps = _membership_gaps if gaps is None else gaps
    if gaps is None or gaps.empty:
        return days
    gaps = gaps[gaps.index.isin(start.index)]
    if gaps.empty:
        return days
    s, e = start.reindex(gaps.index), end.reindex(gaps.index)
    overlap_start = s.where(s > gaps["gap_start"], gaps["gap_start"])
    overlap_end = e.where(e < gaps["gap_end"], gaps["gap_end"])
    lost = (overlap_end - overlap_start).dt.days.clip(lower=0).groupby(level=0).sum()
    return days - lost.reindex(days.index, fill_value=0)


def member_days_one(row, start, end):
    """member_days() for a single patient (for the per-patient loops)."""
    days = (end - start).days
    for gap_start, gap_end in _gaps_by_patient.get(row, ()):
        days -= max(0, (min(end, gap_end) - max(start, gap_start)).days)
    return days


def membership_pieces(start, end):
    """Splits each patient's [start, end] at their membership gaps. Returns (row, start, end,
    is_last) arrays, one entry per piece; patients without a gap keep one piece."""
    rows, starts, ends, last = [], [], [], []
    for row, s, e in zip(start.index, start, end):
        pieces = [(s, e)]
        for gap_start, gap_end in _gaps_by_patient.get(row, ()):
            split = []
            for p_start, p_end in pieces:
                if gap_end <= p_start or gap_start >= p_end:
                    split.append((p_start, p_end))
                    continue
                if gap_start > p_start:
                    split.append((p_start, gap_start))
                if gap_end < p_end:
                    split.append((gap_end, p_end))
            pieces = split or [(s, s)]
        for i, (p_start, p_end) in enumerate(pieces):
            rows.append(row); starts.append(p_start); ends.append(p_end); last.append(i == len(pieces) - 1)
    return (np.array(rows), pd.to_datetime(pd.Series(starts)), pd.to_datetime(pd.Series(ends)),
            np.array(last, dtype=bool))


def followup_end_for(index):
    """fu_end for the given patients; STUDY_END for everyone if set_followup() wasn't called."""
    if _followup_end is None:
        return pd.Series(STUDY_END, index=index)
    return _followup_end.reindex(index)

# LAG_DAYS (protopathic-bias exclusion window) is set in the USER SETTINGS block of
# C_ChangeInIRRByExposre.py -- see the explanation there.

# =============================================================================
# PERSON-TIME CORE -- restricted (fixed-window) person-time, replacing binary cumulative risk
#
# Old design: "did this person reach the outcome within window_days?" (yes/no) -- anyone
# without window_days of observed follow-up before STUDY_END was DROPPED entirely
# (eligible_reachable_mask). New design: every person contributes exactly however much of the
# window they were actually observed for, whether or not they had an event -- nobody is
# dropped for lacking follow-up. A person followed for only 8 of the 12 window-months without
# an event contributes 8/12 person-years at risk and zero events; a person who converts after 3
# months contributes 3/12 person-years and one event. The rate is events / total person-years,
# not events / headcount -- this is an incidence rate, not a cumulative risk, so titles/axes
# below say "per 1,000 person-years" and RR becomes IRR (incidence rate ratio).
# =============================================================================

def person_time_events_and_years(entry_dates, event_dates, window_days=None, lag_days=None):
    """For a cohort with `entry_dates` (index-aligned Series, NaT = not in this cohort) and
    `event_dates` (first occurrence of the outcome, NaT = never), returns (n_events,
    total_person_years) using a window capped at `window_days` (default: current
    FIXED_WINDOW_DAYS) and STUDY_END, whichever is reached first. An event within `lag_days` of
    entry is NOT counted as a hit (protopathic-bias exclusion, off by default) -- that person's
    person-time then simply runs to the window/STUDY_END boundary as if no event had occurred
    (this script only has first-ever dates, not recurrences, so there is no later event to
    still count for them)."""
    entry, end_time, is_event, _ = exposed_intervals(entry_dates, event_dates, window_days, lag_days)
    days_at_risk = member_days(entry, end_time).clip(lower=0)
    person_years = days_at_risk.sum() / 365.25
    n_events = int(is_event.sum())
    return n_events, person_years


def exposed_intervals(entry_dates, event_dates, window_days=None, lag_days=None):
    """Per at-risk exposed patient: (entry, end_time, is_event, event) -- the at-risk interval
    behind person_time_events_and_years(); also used by the adjusted Poisson model so both
    count exactly the same person-time."""
    window_days = FIXED_WINDOW_DAYS if window_days is None else window_days
    lag_days = LAG_DAYS if lag_days is None else lag_days
    valid = entry_dates.notna()
    entry = entry_dates[valid]
    event = event_dates.reindex(entry.index)

    # prevalent: already had the outcome on/before entry -- not at risk of a first-ever one
    prevalent = event.notna() & (event <= entry)
    entry, event = entry[~prevalent], event[~prevalent]

    window_end = entry + pd.to_timedelta(window_days, unit="D")
    boundary = window_end.where(window_end <= STUDY_END, STUDY_END)   # earlier of window/STUDY_END
    fu_end = followup_end_for(entry.index)
    boundary = boundary.where(boundary <= fu_end, fu_end)              # ...and of leaving Clalit

    is_event = event.notna() & (event > entry) & (event <= boundary)
    if lag_days > 0:
        too_early = is_event & ((event - entry).dt.days < lag_days)
        is_event = is_event & ~too_early   # excluded events don't count as hits...

    # person-time: up to the event (if counted) or up to the boundary otherwise
    end_time = pd.Series(boundary, index=entry.index)
    end_time = end_time.where(~is_event, event)   # counted-event rows end exactly at the event
    return entry, end_time, is_event, event


def person_time_rate(entry_dates, event_dates, window_days=None, lag_days=None):
    """Rate per 1,000 person-years -- wraps person_time_events_and_years()."""
    n_events, person_years = person_time_events_and_years(entry_dates, event_dates, window_days, lag_days)
    rate = (n_events / person_years * 1000) if person_years > 0 else np.nan
    return rate, n_events, person_years


# --- General-population reference: CONTINUOUS dynamic-exposure Lexis split, not anchors -------
# Replaces the old "5 fixed pseudo-entry anchors, averaged" design entirely. Rationale: an
# anchor + window (e.g. 2023-01-01 + 24mo) doesn't actually stay inside the calendar
# period it's meant to represent -- it bleeds into the following year(s), so "the general rate
# for 2023" computed that way was never really just 2023. The correct fix is to drop the
# window/anchor concept for the general side altogether: every patient is tracked CONTINUOUSLY
# as "general" (never yet exposed to `exposure_outcomes`) from fu_start (later of 2017 / age 15)
# until the EARLIEST of -- they get exposed (raw exposure date, regardless of diagnoser mode --
# see note below), they get `downstream_outcome`, or fu_end (left Clalit / died / STUDY_END --
# see FOLLOW-UP END above). That continuous interval is then split
# across whatever calendar periods the caller asks for (whole years, split-2023 years,
# quarters, ...) via ordinary Lexis splitting, giving a genuine period-specific rate with no
# anchor choice involved at all.
# NOTE on diagnoser mode: this uses first_of_raw() (UNfiltered exposure date), not first_of()
# (diagnoser-filtered) -- whether someone has left the "genuinely never exposed" pool depends
# on whether they were EVER diagnosed with the exposure, regardless of who diagnosed it. The
# diagnoser filter only decides who qualifies on the EXPOSED side of a comparison; it doesn't
# make someone "still general" just because their real trauma diagnosis doesn't match the
# current diagnoser filter.
def compute_general_lexis_split(frame, exposure_outcomes, downstream_outcome, periods):
    """periods: list of (label, start, end) tuples, end exclusive. Returns a DataFrame indexed
    by label with columns n_events/person_years/rate (per 1,000 person-years)."""
    gen_start, end_time, is_event, down = general_intervals(frame, exposure_outcomes, downstream_outcome)

    rows = []
    for label, p_start, p_end in periods:
        seg_start = gen_start.clip(lower=p_start)
        overlap_end = end_time.clip(upper=p_end)
        days = member_days(seg_start, overlap_end).clip(lower=0)
        person_years = (days / 365.25).sum()
        n_events = int((is_event & (down >= p_start) & (down < p_end)).sum())
        rate = (n_events / person_years * 1000) if person_years > 0 else np.nan
        rows.append({"label": label, "n_events": n_events, "person_years": person_years, "rate": rate})
    return pd.DataFrame(rows).set_index("label")


def general_intervals(frame, exposure_outcomes, downstream_outcome):
    """Per patient: (gen_start, end_time, is_event, down) -- the "general" tracking interval
    behind compute_general_lexis_split(); also used by the adjusted Poisson model."""
    exposure_date = first_of_raw(frame, exposure_outcomes)
    down = frame[f"first_{downstream_outcome}"]

    # per-patient tracking window: from fu_start (later of 2017 / age 15) to the earliest of
    # exposure / fu_end (fu_end is already capped at STUDY_END)
    gen_start = frame["fu_start"].clip(lower=REGISTRY_START)
    fu_end = frame["fu_end"]
    boundary = exposure_date.where(exposure_date.notna() & (exposure_date < fu_end), fu_end)
    prevalent = down.notna() & (down <= gen_start)   # already had the outcome when tracking starts
    is_event = down.notna() & (down > gen_start) & (down <= boundary)
    end_time = boundary.where(~is_event, down)   # exit exactly at the event if it's counted
    end_time = end_time.where(~prevalent, gen_start)   # prevalent -> zero person-time
    return gen_start, end_time, is_event, down


def periods_from_years(years):
    return [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in years]


def compute_general_rate_overall(frame, exposure_outcomes, downstream_outcome):
    """Single whole-study-period general rate -- for callers that need ONE constant reference
    (H2, the pre/post heatmap, Statement 2's B1), not a per-year breakdown."""
    result = compute_general_lexis_split(
        frame, exposure_outcomes, downstream_outcome, [("all", REGISTRY_START, STUDY_END)])
    row = result.loc["all"]
    return row["rate"], row["n_events"], row["person_years"]

# --- Statement 1 supporting tests (TEST 2b/5/6/9 from Trauma_specificity_probe_Step_5.py,
# reframed here around the PRE vs POST Oct7 split specifically, since that's the question
# being asked -- the originals split by ALT_EXPOSURES instead of by time) -------------------
META_ENTRY_OUTCOMES      = ["ptsd", "asr"]                       # TEST 5/9 need entry metadata
META_FIELDS_ENTRY        = ["profession", "department", "facility type"]
META_DOWNSTREAM_OUTCOMES = ["psychosis", "bipolar"]               # TEST 6 needs downstream vector
PERIODS_RANGE            = range(4)

PSYCHOSIS_SPECTRUM_CODES = {"F20", "F20 ONLY SCHIZOPHRENIC", "F21", "F22",
                            "F24", "F25", "F28", "F29"}
PSYCHOSIS_TRANSIENT_CODE = {"F23"}

# --- Statement 5 (who diagnoses what): profession breakdown for EVERY outcome's own first-ever
# diagnosis (not just the trauma entry), split pre/post Oct7 by that outcome's own date. Needs
# "profession" metadata for all 9 outcomes, not just the 2 entry ones above -- and period DATES
# for the 5 outcomes (anxiety/depression/substance/personality/OCD) not already covered by
# META_ENTRY_OUTCOMES/META_DOWNSTREAM_OUTCOMES.
# "profession" is a NUMERIC CODE field, not free text (confirmed from real data -- values like
# "71.0"/"71" appear for the same code due to float-vs-int formatting upstream, so codes are
# always normalized through pd.to_numeric() before comparison, never matched as raw strings).
# Decoder as provided:
PROFESSION_CODE_LABELS = {
    71:  "Psychiatry (Adult)",
    73:  "Child/Adol. Psychiatry",
    10:  "Family Medicine",
    11:  "GP / General Practice",
    32:  "Internal Medicine",
    63:  "Neurology",
    999: "Unknown/Admin",
}
PSYCHIATRIST_PROFESSION_CODES = {71, 73}   # adult + child/adolescent psychiatry
MISSING_PROFESSION_CODES      = {999}      # "Unknown/Unassigned/Admin" -- treated as missing,
                                           # excluded from both numerator AND denominator
                                           # everywhere profession is used (not counted as
                                           # "non-psychiatrist" -- it isn't data at all).

# --- Normalize a profession field (may arrive as '71', '71.0', 71.0, NaN, ...) to a clean code ---
def normalize_profession_code(series):
    return pd.to_numeric(series, errors="coerce")


def profession_code_label(code):
    if pd.isna(code):
        return "Missing"
    return PROFESSION_CODE_LABELS.get(int(code), f"Code {int(code)}")


# ptsd/asr already get "profession" via META_FIELDS_ENTRY above -- every OTHER outcome
# (including psychosis/bipolar, which only had diagnosis_vector pulled for TEST 6) still
# needs its own profession columns requested for Statement 5.
OUTCOMES_NEEDING_PROFESSION_COLS = [o for o in ALL_OUTCOMES if o not in META_ENTRY_OUTCOMES]

# --- Statement 3: zone scheme (Alarms_Step5_1.py, verbatim) ------------------
ZONE_ORDER  = ["Near Gaza", "Northern Border", "Rest of Israel"]
ZONE_LABELS = {"Near Gaza": "Near Gaza", "Northern Border": "Northern Border",
               "Rest of Israel": "Rest of Israel"}

STATEMENT3_ROSE         = ["total", "ptsd", "asr", "anxiety", "depression"]
STATEMENT3_DID_NOT_RISE = ["psychosis", "bipolar", "substance", "personality"]
# The two lists above state a finding. OCD has no result yet: once its post/pre ratio is known,
# move it into one of them and this section disappears.
STATEMENT3_NOT_CLASSIFIED = ["OCD"]
STATEMENT3_SECTIONS = [(label, items) for label, items in [
    ("ROSE AFTER OCT 7", STATEMENT3_ROSE),
    ("DID NOT RISE", STATEMENT3_DID_NOT_RISE),
    ("NOT YET CLASSIFIED", STATEMENT3_NOT_CLASSIFIED)] if items]
OCT_DEC_YEARS = list(range(2017, 2026))

BASE_COLS = ["patient id", "birth date", "gender",
             "demographics_2023-residency border", "distance_to_gaza_km",
             "matched_city",
             # --- demographic comparison (Statement 7) -- not needed by any earlier statement,
             # added specifically for the pre/post-Oct7 demographic profile comparison. ---
             "socioeconomic score five level scale", "socioeconomic score three level scale",
             "demographics_2023-sector",
             "demographics_2023-district", "country of birth", "immigration date",
             "mailing language", "demographics_2023-insurance type",
             "charlson - lastuntil2023-charlson score total",
             # --- follow-up (membership) window -- see FOLLOW-UP END below ---
             MARKETING_SECTOR_COL,   # Sector in the adjusted models
             MEMBERSHIP_END_COL, MEMBERSHIP_PERIODS_COL]
# Period-level columns needed for the Statement-1 supporting tests (2b uses the collapsed
# "_start date" cols already in BASE; 5/9 need entry meta, 6 needs the downstream
# diagnosis_vector) PLUS Statement 5, which needs "profession" for every outcome and
# therefore period DATES for every outcome too (not just the 4 already needed above).
PERIOD_META_COLS = [f"{o}_period{p}_{f}"
                    for o in META_ENTRY_OUTCOMES for p in PERIODS_RANGE
                    for f in META_FIELDS_ENTRY]
PERIOD_VECTOR_COLS = [f"{o}_period{p}_diagnosis_vector"
                      for o in META_DOWNSTREAM_OUTCOMES for p in PERIODS_RANGE]
PERIOD_PROFESSION_COLS_ALL = [f"{o}_period{p}_profession"
                              for o in OUTCOMES_NEEDING_PROFESSION_COLS for p in PERIODS_RANGE]
PERIOD_DATE_COLS = [f"{o}_period{p}_start date"
                    for o in ALL_OUTCOMES for p in PERIODS_RANGE]   # now covers all 9

NEEDED_COLS = (BASE_COLS + [f"{o}_start date" for o in ALL_OUTCOMES]
              + PERIOD_DATE_COLS + PERIOD_META_COLS + PERIOD_VECTOR_COLS
              + PERIOD_PROFESSION_COLS_ALL)


# =============================================================================
# PART 0 -- LOAD & FRAME CONSTRUCTION
# =============================================================================

# --- PART 0, STEP 1: Two-pass, usecols-filtered, chunked load ---
# Mirrors the loading pattern used across Step 4 / Step 5 / Alarms_Step5_1 to
# avoid tokenizing the full ~360-column wide file for every row.
def load_frame(input_file, chunksize=100_000):
    header = pd.read_csv(input_file, encoding="utf-8-sig", nrows=0)
    available = set(header.columns)
    if MARKETING_SECTOR_COL not in available:
        raise RuntimeError(f"'{MARKETING_SECTOR_COL}' (Sector in the adjusted models) is not in {input_file}. "
                           f"Sector-like columns there: {[c for c in header.columns if 'sector' in c.lower()]}")
    usecols = [c for c in NEEDED_COLS if c in available]
    missing = [c for c in NEEDED_COLS if c not in available]
    if missing:
        print(f"WARNING: {len(missing)} expected columns not found (will be skipped): {missing}")
    missing_outcomes = [o for o in ALL_OUTCOMES if f"{o}_start date" not in available]
    if missing_outcomes:
        raise RuntimeError(f"{input_file} has no first-ever date for outcomes {missing_outcomes}")
    chunks = pd.read_csv(input_file, encoding="utf-8-sig", usecols=usecols,
                         chunksize=chunksize, low_memory=False)
    return pd.concat(chunks, ignore_index=True)


# --- PART 0, STEP 2: Assign 3-category zone -- Near Gaza > Northern Border > Rest ---
# Verbatim logic from Alarms_Step5_1.py's assign_zone(), so "north border" here
# is consistent with the zone scheme already used elsewhere in this project.
def assign_zone(df):
    dist = pd.to_numeric(df.get("distance_to_gaza_km"), errors="coerce")
    is_near = dist <= 30
    is_north = (
        df.get("demographics_2023-residency border", pd.Series(index=df.index))
        .astype(str).str.strip() == "גבול הצפון"
    )
    return np.select([is_near, is_north & ~is_near],
                     ["Near Gaza", "Northern Border"],
                     default="Rest of Israel")


# --- PART 0, STEP 3: Build the shared per-patient analysis frame ---
def build_frame(df):
    frame = pd.DataFrame({
        "patient id": df["patient id"].values,
        "gender":     df.get("gender", pd.Series(index=df.index)).values,
    })
    birth = pd.to_datetime(df["birth date"], errors="coerce")
    frame["age_at_oct7"] = (OCT7 - birth).dt.days / 365.25
    frame["birth"] = birth.values   # attained age in the adjusted Poisson model
    frame["zone"] = assign_zone(df)
    # raw Northern Border flag (zone gives Near Gaza priority, so it can't be read back from zone)
    frame["is_north_border"] = (
        df.get("demographics_2023-residency border", pd.Series(index=df.index))
        .astype(str).str.strip() == "גבול הצפון").values

    # --- Demographic comparison fields (Statement 7) -- raw, not derived/recoded, since we
    # don't have a decoder for most of these (unlike profession) -- displayed as whatever
    # categories the real data actually contains. ---
    frame["ses_5level"]      = df.get("socioeconomic score five level scale", pd.Series(index=df.index)).values
    frame["ses_3level"]      = df.get("socioeconomic score three level scale", pd.Series(index=df.index)).values
    frame["distance_km"]     = pd.to_numeric(df.get("distance_to_gaza_km", pd.Series(index=df.index)), errors="coerce").values
    frame["sector"]          = df.get("demographics_2023-sector", pd.Series(index=df.index)).values
    frame["marketing_sector"] = df.get(MARKETING_SECTOR_COL, pd.Series(index=df.index)).values
    frame["district"]        = df.get("demographics_2023-district", pd.Series(index=df.index)).values
    frame["country_of_birth"] = df.get("country of birth", pd.Series(index=df.index)).values
    frame["immigration_date"] = pd.to_datetime(
        df.get("immigration date", pd.Series(index=df.index)), errors="coerce").values
    frame["is_immigrant"]    = pd.notna(frame["immigration_date"])
    frame["mailing_language"] = df.get("mailing language", pd.Series(index=df.index)).values
    frame["insurance_type"]  = df.get("demographics_2023-insurance type", pd.Series(index=df.index)).values
    frame["charlson_score"]  = pd.to_numeric(
        df.get("charlson - lastuntil2023-charlson score total", pd.Series(index=df.index)), errors="coerce").values

    # --- Follow-up window (see FOLLOW-UP END near the top of the file) ---
    if MEMBERSHIP_END_COL in df.columns:
        # DD/MM/YYYY text (steps A and B2) -- without dayfirst, 27/11/2023 would become NaT
        membership_end = pd.to_datetime(df[MEMBERSHIP_END_COL], errors="coerce", dayfirst=True)
    else:
        print(f"WARNING: '{MEMBERSHIP_END_COL}' not found -- follow-up runs to STUDY_END for "
              f"everyone (membership end NOT taken into account)")
        membership_end = pd.Series(STUDY_END, index=df.index)
    # follow-up starts at the later of REGISTRY_START and the 15th birthday (missing birth date
    # -> REGISTRY_START); only the END of follow-up comes from membership
    age15_date = pd.Series((birth + pd.DateOffset(years=FOLLOWUP_MIN_AGE)).values, index=frame.index)
    frame["age15_date"] = age15_date   # NaT = unknown birth date
    frame["fu_start"] = age15_date.where(age15_date > REGISTRY_START, REGISTRY_START)
    frame["fu_end"] = membership_end.fillna(STUDY_END).clip(upper=STUDY_END).values   # 2079-06-06 = active
    frame[MEMBERSHIP_PERIODS_COL] = df.get(MEMBERSHIP_PERIODS_COL, pd.Series(index=df.index)).values

    for outcome in ALL_OUTCOMES:
        col = f"{outcome}_start date"
        frame[f"first_{outcome}"] = (
            pd.to_datetime(df[col], errors="coerce").values
            if col in df.columns else pd.NaT
        )

    entry_meta = {o: get_first_ever_with_meta(df, o, META_FIELDS_ENTRY)
                 for o in META_ENTRY_OUTCOMES}
    downstream_meta = {o: get_first_ever_with_meta(df, o, ["diagnosis_vector"])
                       for o in META_DOWNSTREAM_OUTCOMES}
    # Statement 5: profession at first-ever diagnosis, for ALL 9 outcomes (not just the trauma
    # entry ones) -- who diagnoses psychosis/bipolar vs who diagnoses everything else.
    outcome_profession_meta = {o: get_first_ever_with_meta(df, o, ["profession"])
                               for o in ALL_OUTCOMES}
    return frame, entry_meta, downstream_meta, outcome_profession_meta


# --- PART 0, STEP 3b: Get first-ever date PLUS matching metadata from the SAME period ---
# Ported from Trauma_specificity_probe_Step_5.py's get_first_ever_with_meta(): finds which
# period column holds each patient's earliest date for `outcome`, then pulls `meta_fields`
# from that same period (so e.g. "profession" reflects who made THAT specific diagnosis).
def get_first_ever_with_meta(df, outcome, meta_fields):
    period_date_cols = {p: f"{outcome}_period{p}_start date" for p in PERIODS_RANGE
                        if f"{outcome}_period{p}_start date" in df.columns}
    result = pd.DataFrame(index=df.index)
    if not period_date_cols:
        result["date"] = pd.to_datetime(df.get(f"{outcome}_start date"), errors="coerce")
        for f in meta_fields:
            result[f] = np.nan
        return result

    dates = pd.DataFrame({p: pd.to_datetime(df[col], errors="coerce")
                          for p, col in period_date_cols.items()})
    earliest_date = dates.min(axis=1)
    has_any = dates.notna().any(axis=1)
    earliest_period = pd.Series(index=df.index, dtype=object)
    earliest_period[has_any] = dates.loc[has_any].idxmin(axis=1)

    result["date"] = earliest_date
    for f in meta_fields:
        field_values = pd.Series(index=df.index, dtype=object)
        for p in period_date_cols:
            meta_col = f"{outcome}_period{p}_{f}"
            if meta_col not in df.columns:
                continue
            mask = (earliest_period == p)
            field_values[mask] = df.loc[mask, meta_col].values
        result[f] = field_values
    return result


# --- PART 0, STEP 3c: Pull one metadata field for a (possibly pooled) exposure ---
# For pooled exposures (e.g. PTSD/ASR) picks the field from whichever component fired first.
def get_exposure_meta_field(entry_meta, outcome_list, field):
    if len(outcome_list) == 1:
        return entry_meta[outcome_list[0]][field]
    dates = pd.concat({o: entry_meta[o]["date"] for o in outcome_list}, axis=1)
    has_any = dates.notna().any(axis=1)
    earliest_outcome = pd.Series(index=dates.index, dtype=object)
    if has_any.any():
        earliest_outcome[has_any] = dates.loc[has_any].idxmin(axis=1)
    result = pd.Series(index=dates.index, dtype=object)
    for o in outcome_list:
        mask = (earliest_outcome == o)
        result[mask] = entry_meta[o][field][mask].values
    return result


# --- PART 0, STEP 4: Restrict frame + metadata dicts to an age band (measured at Oct 7) ---
def apply_age(frame, entry_meta, downstream_meta, outcome_profession_meta, lo, hi):
    mask = frame["age_at_oct7"].between(lo, hi)
    f2   = frame[mask].copy()
    em2  = {o: d.loc[mask].copy() for o, d in entry_meta.items()}
    dm2  = {o: d.loc[mask].copy() for o, d in downstream_meta.items()}
    opm2 = {o: d.loc[mask].copy() for o, d in outcome_profession_meta.items()}
    return f2, em2, dm2, opm2


# --- PART 0, STEP 5: Earliest date across a list of outcomes (e.g. ptsd+asr) ---
# RAW version: no diagnoser filtering. Used internally by first_of()/diagnoser_excluded_mask()
# below, and safe to call directly if a raw (unfiltered) exposure date is ever needed.
def first_of_raw(frame, outcome_list):
    cols = [f"first_{o}" for o in outcome_list if f"first_{o}" in frame.columns]
    if not cols:
        return pd.Series(pd.NaT, index=frame.index)
    return frame[cols].min(axis=1)


# --- Who diagnosed the EARLIEST occurrence across outcome_list, for the diagnoser filter ---
# Mirrors get_exposure_meta_field()'s pooling logic (earliest component wins) but works off
# outcome_profession_meta (built for all 9 outcomes, not just the 2 trauma ones), so it applies
# to any of Statement 2's 8 alternate exposures too, not just trauma.
def _diagnoser_match_mask(frame, outcome_list, raw_dates):
    diagnoser_filter = DIAGNOSER_MODES.get(CURRENT_DIAGNOSER_MODE)
    if diagnoser_filter is None or _current_outcome_profession_meta is None:
        return pd.Series(True, index=frame.index)   # no filter active -- everyone matches

    valid_outcomes = [o for o in outcome_list if o in _current_outcome_profession_meta]
    if not valid_outcomes:
        return pd.Series(True, index=frame.index)   # no profession metadata for this exposure

    per_outcome_dates = pd.concat(
        {o: _current_outcome_profession_meta[o]["date"].reindex(frame.index) for o in valid_outcomes}, axis=1)
    has_any = per_outcome_dates.notna().any(axis=1)
    earliest_outcome = pd.Series(index=frame.index, dtype=object)
    earliest_outcome[has_any] = per_outcome_dates.loc[has_any].idxmin(axis=1)

    profession_at_first = pd.Series(np.nan, index=frame.index, dtype=object)
    for o in valid_outcomes:
        mask = (earliest_outcome == o)
        profession_at_first[mask] = _current_outcome_profession_meta[o]["profession"].reindex(frame.index)[mask].values

    codes = normalize_profession_code(profession_at_first)
    is_psychiatrist = codes.isin(PSYCHIATRIST_PROFESSION_CODES)
    is_known = codes.notna() & ~codes.isin(MISSING_PROFESSION_CODES)   # 999 = can't classify

    if diagnoser_filter == "psychiatrist":
        return is_known & is_psychiatrist
    else:   # "non_psychiatrist"
        return is_known & ~is_psychiatrist


# Exposure date used EVERYWHERE in this script to define "exposed" -- applies the active
# diagnoser filter. A patient who had the exposure but not via the matching diagnoser type
# gets NaT here (looks "unexposed" to every function downstream) -- but must ALSO be removed
# from any "general/unexposed" population those functions build, or they'd be wrongly folded
# in as if they never had the exposure at all. See diagnoser_excluded_mask() below -- every
# place in this file that constructs a "general" population intersects it with
# ~diagnoser_excluded_mask(...) for exactly this reason.
def first_of(frame, outcome_list):
    raw_dates = first_of_raw(frame, outcome_list)
    # a first-ever exposure before age 15 is not an entry -- and since first_of_raw() still sees
    # it, the patient is not "general" either: excluded from both. Compared to the 15th birthday
    # only (not fu_start), so an exposure before 2017 at age 15+ stays an entry as before.
    raw_dates = raw_dates.where(~(raw_dates < frame["age15_date"]))
    if CURRENT_DIAGNOSER_MODE == "any_diagnoser":
        return raw_dates
    keep = _diagnoser_match_mask(frame, outcome_list, raw_dates)
    return raw_dates.where(keep)


def diagnoser_excluded_mask(frame, outcome_list):
    """True for patients who HAD this exposure but not via the currently active diagnoser
    filter -- must be dropped from 'general' too, not just from 'exposed'."""
    if CURRENT_DIAGNOSER_MODE == "any_diagnoser":
        return pd.Series(False, index=frame.index)
    raw_dates = first_of_raw(frame, outcome_list)
    keep = _diagnoser_match_mask(frame, outcome_list, raw_dates)
    return raw_dates.notna() & ~keep


def set_diagnoser_mode(mode_key, outcome_profession_meta):
    global CURRENT_DIAGNOSER_MODE, CURRENT_DIAGNOSER_LABEL, _current_outcome_profession_meta
    CURRENT_DIAGNOSER_MODE = mode_key
    CURRENT_DIAGNOSER_LABEL = DIAGNOSER_LABELS[mode_key]
    _current_outcome_profession_meta = outcome_profession_meta



# --- Load once, reused across all 3 trauma-mode runs (avoids re-reading the large CSV 3x) ---
def load_and_build(input_file):
    print("Loading input file...")
    df = load_frame(input_file)
    print(f"Loaded {len(df):,} rows.")
    built = build_frame(df)
    set_followup(built[0])
    global _FULL_COHORT_N
    _FULL_COHORT_N = len(built[0])
    return built


_FULL_COHORT_N = None   # patients in the input file, set by load_and_build()


# --- Write the 2 age-band PDFs for the CURRENTLY ACTIVE trauma mode + diagnoser mode (module
# globals TRAUMA_EXPOSURE_OUTCOMES/LABEL, CURRENT_DIAGNOSER_MODE) -- call set_trauma_mode() /
# set_diagnoser_mode() first, or use export_all_diagnoser_modes() below to run all 3 diagnoser
# modes automatically for one trauma mode. ---
# --- Population variants: the whole file re-run on a geographic sub-population (BOTH the exposed
# and the general group are restricted -- every rate and IRR is computed inside that area only).
# Distance = distance_to_gaza_km; patients with no distance value are in neither variant.
# key -> (label for titles, file-name suffix, mask function or None for everyone)
POPULATIONS = {
    "all":             ("", "", None),
    "gaza20km":        ("<=20 km from Gaza", "_gaza20km",
                        lambda f: f["distance_km"] <= 20),
    "over20km_nonorth": (">20 km from Gaza, no Northern Border", "_over20km_nonorth",
                         lambda f: (f["distance_km"] > 20) & ~f["is_north_border"]),
    # missing distance is KEPT here (no distance = not near a border)
    "no7km_nonorth":   ("no 0-7 km from Gaza, no Northern Border (missing distance kept)", "_no7km_nonorth",
                        lambda f: ~(f["distance_km"] <= 7) & ~f["is_north_border"].astype(bool)),
}
# which variants get their own extra PDF: EXTRA_POPULATIONS in the USER SETTINGS of main


def apply_population(frame, entry_meta, downstream_meta, outcome_profession_meta, population_key):
    mask_fn = POPULATIONS[population_key][2]
    if mask_fn is None:
        return frame, entry_meta, downstream_meta, outcome_profession_meta
    mask = mask_fn(frame).fillna(False).astype(bool)
    return (frame[mask].copy(),
            {o: d.loc[mask].copy() for o, d in entry_meta.items()},
            {o: d.loc[mask].copy() for o, d in downstream_meta.items()},
            {o: d.loc[mask].copy() for o, d in outcome_profession_meta.items()})


def set_trauma_mode(mode_key):
    global TRAUMA_EXPOSURE_OUTCOMES, TRAUMA_EXPOSURE_LABEL
    TRAUMA_EXPOSURE_OUTCOMES, TRAUMA_EXPOSURE_LABEL = TRAUMA_MODES[mode_key]
