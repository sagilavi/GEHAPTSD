# C_IRR_Analyses.py -- LOGIC 2/4: the computations behind Statements 1-7 (rates, IRRs, trends,
# Kaplan-Meier, comorbidity, zones, latency, demographics, relative year) and the helpers
# behind the descriptive companion PDF.
# Part of the pipeline run by C_ChangeInIRRByExposre.py (all files in the same directory):
#   C_IRR_Core.py                  -- run settings (SLIM_RUN, SELECTED_TRAUMA_MODE, ...), run-time
#                                     state, loading, membership / follow-up / person-time core,
#                                     population variants
#   C_IRR_Analyses.py              -- the computations behind Statements 1-7 and the descriptive PDF
#   C_IRR_Models.py                -- adjusted IRR (Poisson) and the era comparisons
#   C_IRR_Checks.py                -- automated sanity checks
#   C_IRR_Tables.py                -- table formatting
#   C_IRR_Graphs_Statement1.py     -- graphs: colors, Statement 1 + its supporting tests
#   C_IRR_Graphs_Statements2to6.py -- graphs: Statements 2, 3, 5, 6 and comorbidity
#   C_IRR_Graphs_Demog_Models.py   -- graphs: demographic / relative-year PDFs, adjusted IRR
#                                     forests, era comparisons, section dividers, descriptive charts
#   C_ChangeInIRRByExposre.py      -- MAIN: PDF assembly (export_all*), INPUT_FILE / OUTPUT_DIR,
#                                     __main__
# Run-time state (active window / trauma / diagnoser mode, follow-up end) lives in C_IRR_Core and
# is always read as core.NAME outside it (the two print-once diagnostic flags: analyses.NAME),
# so every module sees the current value.

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

import C_IRR_Core as core   # run-time state -- read as core.NAME
from C_IRR_Core import (  # noqa: F401
    ALL_OUTCOMES, FOCUS_OUTCOMES, LAG_DAYS, MISSING_PROFESSION_CODES, OCT7,
    PSYCHIATRIST_PROFESSION_CODES, PSYCHOSIS_SPECTRUM_CODES, PSYCHOSIS_TRANSIENT_CODE, STUDY_END,
    ZONE_ORDER, compute_general_lexis_split, compute_general_rate_overall, current_alt_exposures,
    first_of, first_of_raw, followup_end_for, get_exposure_meta_field, member_days,
    member_days_one, membership_pieces, normalize_profession_code, person_time_rate,
    profession_code_label,
)


# =============================================================================
# PART 1 -- STATEMENT 1: TRAUMA -> PSYCHOSIS/BIPOLAR
# =============================================================================

# --- S1, STEP 1: Exposed (post-exposure) vs general-population rate + RR per outcome ---
# UPDATED: this now applies the SAME fixed-window + censoring design as
# compute_prepost_window_rr()/TEST 2b, everywhere a "second illness after the first" is
# being counted (H2 bars, B1 bars in Statement 2, TEST 5/6/9 below) -- not just the
# pre/post-war split. Two things changed from a plain "any time after exposure" RR:
#   1. EXPOSED patients are only counted if their window (exposure_date + window_days)
#      fully elapsed by STUDY_END -- otherwise a late-diagnosed patient could ONLY ever
#      show up as "no second illness" for lack of observation time, which would bias the
#      rate down for reasons that have nothing to do with biology.
#   2. A downstream ("second") diagnosis only counts if it falls within window_days of the
#      exposure date -- otherwise late follow-up time inflates the "exposed" rate relative
#      to the general population, whose own comparable window is capped the same way.
# UPDATED: this now applies the SAME fixed-window + censoring design as
# compute_prepost_window_rr()/TEST 2b, everywhere a "second illness after the first" is
# being counted (H2 bars, B1 bars in Statement 2, TEST 5/6/9 below) -- not just the
# pre/post-war split. Two things changed from a plain "any time after exposure" RR:
#   1. EXPOSED patients are only counted if their window (exposure_date + window_days)
#      fully elapsed by STUDY_END -- otherwise a late-diagnosed patient could ONLY ever
#      show up as "no second illness" for lack of observation time, which would bias the
#      rate down for reasons that have nothing to do with biology.
#   2. A downstream ("second") diagnosis only counts if it falls within window_days of the
#      exposure date -- otherwise late follow-up time inflates the "exposed" rate relative
#      to the general population.
#   3. The GENERAL (never-exposed) population is measured via CONTINUOUS dynamic-exposure
#      Lexis tracking (compute_general_rate_overall()) -- not a fixed window from an anchor --
#      over the WHOLE study period, so it's directly comparable as a stable reference rate.
# NOTE: this means Statement 1 page 1 / Statement 2 page 1 no longer reproduce the
# original H2/B1 images' raw numbers (RR 2.9/4.7 etc, which used unrestricted follow-up
# time) -- that's the intended effect of this change, not a bug. See sanity check note.
def compute_exposed_vs_general(frame, exposure_outcomes, downstream_outcomes,
                               window_days=None):
    window_days = core.FIXED_WINDOW_DAYS if window_days is None else window_days
    exposure_date = first_of(frame, exposure_outcomes)
    n_exp = int(exposure_date.notna().sum())   # headcount, for display only -- NOT the rate denominator
    n_gen = int(first_of_raw(frame, exposure_outcomes).isna().sum())   # headcount, for display only

    rows = []
    for outcome in downstream_outcomes:
        down = frame[f"first_{outcome}"]
        exp_rate, a, exp_person_years = person_time_rate(exposure_date, down, window_days)
        n_at_risk = int((exposure_date.notna() & ~(down <= exposure_date)).sum())   # excl. prevalent cases
        gen_rate, c, gen_person_years = compute_general_rate_overall(frame, exposure_outcomes, outcome)
        irr = exp_rate / gen_rate if (pd.notna(gen_rate) and gen_rate > 0 and pd.notna(exp_rate)) else np.nan

        # Poisson exact test via the conditional-binomial reformulation: given a+c total events
        # across the two person-time totals, a ~ Binomial(a+c, T_exp/(T_exp+T_gen)) under
        # H0: equal rates. This is the standard exact test for comparing two Poisson rates and
        # needs only the two (events, person-time) pairs -- no individual-level vectors, which
        # person-time analysis doesn't naturally produce the way the old binary design did.
        pval_str = "ns"
        a_r, c_r = round(a), round(c)
        total_py = exp_person_years + gen_person_years
        if (a_r + c_r) > 0 and total_py > 0:
            try:
                p = stats.binomtest(a_r, a_r + c_r, exp_person_years / total_py, alternative="two-sided").pvalue
                pval_str = f"{p:.2e}" if p < 0.05 else "ns"
            except ValueError:
                pval_str = "ns"

        rows.append({"outcome": outcome, "n_exp_cases": a, "n_at_risk": n_at_risk,
                    "exp_rate": exp_rate, "gen_rate": gen_rate,
                    "rr": irr, "mw_p": pval_str})
    return n_exp, n_gen, pd.DataFrame(rows)


# --- S1, STEP 3: Fixed-window RR, split by whether entry was pre- or post-Oct7 ---
# Generalized TEST 2c: works for any single exposure and any downstream list.
# FIX 1 (censoring): a patient whose window (entry_date + window_days) extends past STUDY_END
# hasn't had the full follow-up period observed -- counting them as "no conversion" would
# bias the rate down. Excluded from the denominator entirely, matching TEST 2a/2b's
# "reachable" filter in Trauma_specificity_probe_Step_5.py. This matters MORE for the
# "post" (on/after Oct7) group here specifically, since those patients have systematically
# less time between entry and STUDY_END than the "pre" group.
# FIX 2 (general-rate consistency): the general-population reference rate must be measured
# over the SAME window_days as the exposed side (pseudo-entry at PERIOD_1_START), not over
# unrestricted lifetime follow-up -- otherwise every RR here comes out deflated relative to
# compute_exposed_vs_general()'s pooled RR for the same exposure/outcome (this was caught
# after a real-data run showed H2 reporting RR 11.5 for an exposure while this heatmap
# reported 1.6/1.2 for the same exposure split pre/post -- see sanity check #11).
def compute_prepost_window_rr(frame, exposure_outcomes, downstream_outcomes, window_days):
    exp_date = first_of(frame, exposure_outcomes)
    pre_entry  = exp_date.where(exp_date < OCT7)
    post_entry = exp_date.where(exp_date >= OCT7)
    n_pre, n_post = int(pre_entry.notna().sum()), int(post_entry.notna().sum())

    def _rr_row(entry_dates):
        out = {}
        for outcome in downstream_outcomes:
            rate, _, _ = person_time_rate(entry_dates, frame[f"first_{outcome}"], window_days)
            out[outcome] = rate
        return out

    rates_pre  = _rr_row(pre_entry)
    rates_post = _rr_row(post_entry)
    # RR here expressed relative to the "general" (never-exposed) rate, so the two rows are
    # comparable to TEST 2c's RR heatmap, not just raw rates. Uses the continuous dynamic-
    # exposure general rate (compute_general_rate_overall()) -- whole-study-period, no anchor.
    n_gen = int(first_of_raw(frame, exposure_outcomes).isna().sum())
    gen_rates = {o: compute_general_rate_overall(frame, exposure_outcomes, o)[0] for o in downstream_outcomes}

    def _to_rr(rates):
        return {o: (rates[o] / gen_rates[o] if gen_rates.get(o) else np.nan)
               for o in downstream_outcomes}

    return (n_pre, _to_rr(rates_pre)), (n_post, _to_rr(rates_post)), n_gen


# =============================================================================
# PART 1B -- STATEMENT 1 SUPPORTING TESTS (pre/post PTSD/ASR variance)
#
# Ports of TEST 2b / 5 / 6 / 9 from Trauma_specificity_probe_Step_5.py. The
# originals split by ALT_EXPOSURES (which of 4 diagnoses was the entry); here
# they're reframed around PRE-Oct7 vs POST-Oct7 PTSD/ASR entry specifically,
# since that split IS the question -- each test probes a different mechanical
# explanation for why the post-war RR comes out lower than the pre-war RR:
#   2b -- is the gap real once follow-up time is properly censored?
#   5  -- is post-war trauma diagnosed more often by non-psychiatric staff?
#   6  -- is the post-war psychosis/bipolar that DOES appear a "weaker" subtype?
#   9  -- does inpatient vs outpatient trauma entry explain any of it?
# =============================================================================

# --- T2b, STEP 1: Direct pre-vs-post rate + RR (not normalized against "general"), censored ---
def compute_t2b_rates(frame, downstream_outcome):
    down = frame[f"first_{downstream_outcome}"]
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    pre_entry  = first_trauma.where(first_trauma < OCT7)
    post_entry = first_trauma.where(first_trauma >= OCT7)

    rate_pre,  a_pre,  py_pre  = person_time_rate(pre_entry,  down, core.FIXED_WINDOW_DAYS)
    rate_post, a_post, py_post = person_time_rate(post_entry, down, core.FIXED_WINDOW_DAYS)
    n_pre, n_post = int(pre_entry.notna().sum()), int(post_entry.notna().sum())

    rr = rate_post / rate_pre if rate_pre and rate_pre > 0 else np.nan

    # Poisson exact test (conditional-binomial form) -- see compute_exposed_vs_general() for
    # the same construction; replaces the old two-sample Mann-Whitney, which needed individual-
    # level binary vectors that person-time analysis doesn't produce directly.
    pval_str = "ns"
    total_py = py_pre + py_post
    if (a_pre + a_post) > 0 and total_py > 0:
        try:
            p = stats.binomtest(a_pre, a_pre + a_post, py_pre / total_py, alternative="two-sided").pvalue
            pval_str = f"{p:.2e}" if p < 0.05 else "ns"
        except ValueError:
            pval_str = "ns"
    return {"rate_pre": rate_pre, "n_pre": n_pre, "rate_post": rate_post,
           "n_post": n_post, "rr": rr, "pval": pval_str}


# --- Shared helper: reachability (full follow-up) + within-window conversion, used by
# TEST 5/6/9 below so all three apply the exact same window+censoring rule as TEST 2b
# and compute_exposed_vs_general() above. ---
def eligible_reachable_mask(exposure_date, window_days=None):
    window_days = core.FIXED_WINDOW_DAYS if window_days is None else window_days
    return (exposure_date + pd.to_timedelta(window_days, unit="D")) <= STUDY_END


def converted_within_window(exposure_date, downstream_date, window_days=None):
    window_days = core.FIXED_WINDOW_DAYS if window_days is None else window_days
    return (downstream_date.notna() & (downstream_date > exposure_date) &
           (downstream_date <= exposure_date + pd.to_timedelta(window_days, unit="D")))


# --- T5, STEP 1: Entry-diagnosis profession breakdown among converters, pre vs post Oct7 ---
# Windowed + censored: only trauma-exposed patients with a full 18mo follow-up window count,
# and "converted" means the downstream dx fell within that same 18mo window.
def compute_t5_breakdown(frame, entry_meta, downstream_outcome, side):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    eligible = first_trauma.notna() & eligible_reachable_mask(first_trauma)
    side_mask = eligible & ((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    converted_mask = side_mask & converted_within_window(first_trauma, down)

    profession = get_exposure_meta_field(entry_meta, core.TRAUMA_EXPOSURE_OUTCOMES, "profession")
    codes = normalize_profession_code(profession[converted_mask])
    codes = codes[codes.notna() & ~codes.isin(MISSING_PROFESSION_CODES)]
    if codes.empty:
        return pd.Series(dtype=float), 0
    prof_conv = codes.apply(profession_code_label)
    top = prof_conv.value_counts().head(4)
    other_n = len(prof_conv) - top.sum()
    if other_n > 0:
        top["Other"] = other_n
    return top / top.sum() * 100, len(prof_conv)


# --- T6, STEP 1: Classify psychosis / bipolar subtype from the period diagnosis_vector ---
def classify_psychosis_subtype(vector_str):
    if pd.isna(vector_str):
        return None
    codes = set(str(vector_str).split(","))
    if codes & PSYCHOSIS_SPECTRUM_CODES:
        return "F20-F29 spectrum"
    if codes & PSYCHOSIS_TRANSIENT_CODE:
        return "F23 only (transient)"
    return "Other"


def classify_bipolar_subtype(vector_str):
    if pd.isna(vector_str):
        return None
    codes = set(str(vector_str).split(","))
    if "F31" in codes:
        return "F31 (bipolar disorder)"
    if "F30" in codes:
        return "F30 only (manic episode)"
    return "Other"


def compute_t6_breakdown(frame, downstream_meta, downstream_outcome, classifier, side):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    eligible = first_trauma.notna() & eligible_reachable_mask(first_trauma)
    side_mask = eligible & ((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    converted_mask = side_mask & converted_within_window(first_trauma, down)

    vectors = downstream_meta[downstream_outcome]["diagnosis_vector"][converted_mask]
    subtypes = vectors.apply(classifier).dropna()
    if subtypes.empty:
        return pd.Series(dtype=float), 0
    counts = subtypes.value_counts()
    return counts / counts.sum() * 100, len(subtypes)


# --- T9, STEP 1: Classify trauma entry as "heavy" (inpatient/hospital) vs "light" (other) ---
# "facility type" turned out to be Hebrew free text describing care SETTING, not an English
# INP/OUT code as originally guessed (confirmed from a real-data run) -- e.g. "ראשונית"
# (primary care), "כפרית" (rural), "קופ\"ח-כללי" (HMO-general), vs "בי\"ח נפש ממשלתי"
# (government psychiatric HOSPITAL). Matches any of several Hebrew spellings/punctuation
# variants of "hospital"/"hospitalization" -- PROVISIONAL: only ONE hospital-type category was
# visible in the first real-data run's top-8 printout; there may be others not yet seen.
# print_facility_type_diagnostic() now shows the top 20 (not 8) specifically so this can be
# checked -- if the real category list has other hospital-type entries this doesn't catch
# (private hospitals, a different wording, etc.), add them to HOSPITAL_FACILITY_PATTERNS below.
HOSPITAL_FACILITY_PATTERNS = ['בי"ח', 'בי״ח', 'בית חולים', 'אשפוז']

def classify_trauma_engagement(entry_meta):
    facility_type = get_exposure_meta_field(entry_meta, core.TRAUMA_EXPOSURE_OUTCOMES, "facility type").astype(str)
    is_heavy = pd.Series(False, index=facility_type.index)
    for pattern in HOSPITAL_FACILITY_PATTERNS:
        is_heavy |= facility_type.str.contains(pattern, na=False, regex=False)
    return is_heavy


_facility_type_diagnostic_printed = False

def print_facility_type_diagnostic(entry_meta):
    global _facility_type_diagnostic_printed
    if _facility_type_diagnostic_printed:
        return
    _facility_type_diagnostic_printed = True
    facility_type = get_exposure_meta_field(entry_meta, core.TRAUMA_EXPOSURE_OUTCOMES, "facility type")
    vc = facility_type.dropna().astype(str).value_counts()
    print("\n  [TEST 9 diagnostic] Real 'facility type' values found in the data (top 20):")
    print(f"    {dict(vc.head(20))}")
    matched_categories = [v for v in vc.index if any(p in v for p in HOSPITAL_FACILITY_PATTERNS)]
    if not matched_categories:
        print(f"    *** WARNING: none of the hospital patterns {HOSPITAL_FACILITY_PATTERNS} "
             "matched ANY value above -- TEST 9's 'Heavy' (inpatient) bars will all read ~0. "
             "Send the full category list (this printout) and I'll update "
             "HOSPITAL_FACILITY_PATTERNS to match the real values. ***")
    else:
        plural = "y" if len(matched_categories) == 1 else "ies"
        print(f"    OK: matched {len(matched_categories)} hospital-type categor{plural}: "
             f"{matched_categories}")
        print("    NOTE: this is a PROVISIONAL match based on Hebrew hospital-related "
             "keywords, not a confirmed decoder (unlike profession's "
             "PSYCHIATRIST_PROFESSION_CODES, which came from an explicit decoder table) -- "
             "review the full category list above and confirm these are the only categories "
             "that should count as 'heavy' engagement.")


def compute_t9_rates(frame, entry_meta, downstream_outcome):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    is_heavy = classify_trauma_engagement(entry_meta).fillna(False)
    exposed_mask = first_trauma.notna() & eligible_reachable_mask(first_trauma)
    got = converted_within_window(first_trauma, down)

    rows = []
    for side, side_mask in [("Pre", first_trauma < OCT7), ("Post", first_trauma >= OCT7)]:
        for engagement, eng_mask in [("Heavy", is_heavy), ("Light", ~is_heavy)]:
            mask = exposed_mask & side_mask & eng_mask
            n = int(mask.sum())
            rate = got[mask].mean() * 1000 if n > 0 else np.nan
            rows.append({"side": side, "engagement": engagement, "n": n, "rate": rate})
    return pd.DataFrame(rows)


# --- T-Zone, STEP 1: Trauma -> downstream rate, split by zone (direct vs indirect exposure)
# WITHIN each of pre/post-Oct7 -- not a ported Trauma_specificity_probe_Step_5.py test, new
# here. Uses the same "zone" column (Near Gaza / Northern Border / Rest of Israel) already
# built in build_frame()/assign_zone(), windowed + censored the same way as everything else
# in this section. The question: does trauma diagnosed by someone with direct exposure (Near
# Gaza / Northern Border) carry a different downstream risk than trauma diagnosed by someone
# in the rest of the country (indirect/media exposure, "worried well")?
def compute_zone_rates(frame, downstream_outcome):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    eligible = first_trauma.notna() & eligible_reachable_mask(first_trauma)
    got = converted_within_window(first_trauma, down)

    rows = []
    for side, side_mask in [("Pre", first_trauma < OCT7), ("Post", first_trauma >= OCT7)]:
        for zone in ZONE_ORDER:
            mask = eligible & side_mask & (frame["zone"] == zone)
            n = int(mask.sum())
            rate = got[mask].mean() * 1000 if n > 0 else np.nan
            rows.append({"side": side, "zone": zone, "n": n, "rate": rate})
    return pd.DataFrame(rows)


# --- T-Yearly, STEP 1: Continuous year-of-entry RR trend (not bucketed) -----------------------
# Not a ported Step 5 test. TEST 8 there stratifies calendar time into 3 multi-year buckets
# (2017-2019 / 2020-Oct2023 / Oct2023-2026); this instead plots RR by SINGLE CALENDAR YEAR of
# trauma-entry as a continuous line, so a gradual pre-existing decline (background trend) looks
# visually different from a sharp break exactly at Oct 7. Same windowed+censored design as
# everything else in this section; each period's IRR is measured against that SAME period's
# general-population rate (compute_general_lexis_split() with the same periods), so a
# background trend in the general population doesn't show up as a change in the IRR.
# 2023 is SPLIT into two sub-periods (Jan 1-Oct 6, and Oct 7-Dec 31) instead of one calendar-
# year point -- a single 2023 point would average ~9 months of pre-war data together with ~3
# months of war-period data, which would blur exactly the break this chart exists to detect.
# CAVEAT (person-time era): periods close to STUDY_END are NO LONGER dropped the way the old
# eligible_reachable_mask() design dropped them -- a patient entered in 2025/2026 now
# contributes however many partial person-years they were actually observed for. Their point
# on this chart is still noisier than earlier years (less accumulated person-time, so a single
# early/late event swings the rate more) -- read the n at each point as a reliability signal,
# but don't expect it to read n=0/NaN near STUDY_END anymore.
YEARLY_TREND_PERIODS = []
for _year in range(2017, 2027):
    if _year == 2023:
        YEARLY_TREND_PERIODS.append(("2023\n(pre)",  pd.Timestamp("2023-01-01"), OCT7))
        YEARLY_TREND_PERIODS.append(("2023\n(post)", OCT7, pd.Timestamp("2024-01-01")))
    else:
        YEARLY_TREND_PERIODS.append((str(_year), pd.Timestamp(f"{_year}-01-01"),
                                    pd.Timestamp(f"{_year + 1}-01-01")))

def compute_yearly_rr_trend(frame, downstream_outcome):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]

    # PER-PERIOD general-population reference, using the SAME periods as the exposed side
    # (including the 2023 pre/post split) -- via continuous Lexis tracking, not an anchor.
    n_gen = int(first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES).isna().sum())
    gen_split = compute_general_lexis_split(frame, core.TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome,
                                            YEARLY_TREND_PERIODS)

    rows = []
    for label, start, end in YEARLY_TREND_PERIODS:
        period_entry = first_trauma.where((first_trauma >= start) & (first_trauma < end))
        n = int(period_entry.notna().sum())
        rate, _, _ = person_time_rate(period_entry, down, core.FIXED_WINDOW_DAYS)
        gen_rate = gen_split.loc[label, "rate"]
        rr = rate / gen_rate if (pd.notna(gen_rate) and gen_rate > 0 and pd.notna(rate)) else np.nan
        rows.append({"label": label, "n": n, "rr": rr})
    return pd.DataFrame(rows), n_gen


# --- T-BySecondDiagnosisYear: same "yearly trend" idea, but the YEAR on the x-axis is when the
# SECOND diagnosis (psychosis/bipolar) happened, not when the trauma was entered. Two lines per
# panel distinguish trauma diagnosed pre-war vs post-war, so a converter's year is attributed to
# whichever trauma-timing group they came from. Restricted to the ACTIVE window (FIXED_WINDOW_DAYS)
# per your choice -- a conversion only counts if it happened within that many months of the
# trauma entry, same "hit" definition as everywhere else in this script.
#
# This requires a proper Lexis-style split: each patient's restricted risk interval
# [entry, boundary) is cut at every PERIOD boundary it crosses (not just whole calendar years --
# see SECOND_DIAGNOSIS_PERIODS below, which splits 2023 itself at the exact Oct7 date), so
# person-time is attributed to the period it actually occurred in, and the event (if any) is
# attributed to the period the EXACT event date falls in -- an exact date-level split, not a
# whole-year approximation.
SECOND_DIAGNOSIS_YEAR_RANGE = list(range(2017, 2027))   # for x-tick labels only

SECOND_DIAGNOSIS_PERIODS = []
for _y in range(2017, 2027):
    if _y == 2023:
        SECOND_DIAGNOSIS_PERIODS.append(("2023\n(pre)",  pd.Timestamp("2023-01-01"), OCT7))
        SECOND_DIAGNOSIS_PERIODS.append(("2023\n(post)", OCT7, pd.Timestamp("2024-01-01")))
    else:
        SECOND_DIAGNOSIS_PERIODS.append((str(_y), pd.Timestamp(f"{_y}-01-01"),
                                        pd.Timestamp(f"{_y + 1}-01-01")))

def split_person_time_by_period(entry_dates, event_dates, window_days, periods):
    """Returns (events_by_label, person_years_by_label) dicts, by splitting each patient's
    restricted-window risk interval across the given (label, start, end) period boundaries --
    the event (if counted) is attributed to whichever period its EXACT date falls in."""
    valid = entry_dates.notna()
    entry = entry_dates[valid]
    event = event_dates.reindex(entry.index)

    prevalent = event.notna() & (event <= entry)   # same rule as person_time_events_and_years()
    entry, event = entry[~prevalent], event[~prevalent]

    window_end = entry + pd.to_timedelta(window_days, unit="D")
    boundary = window_end.where(window_end <= STUDY_END, STUDY_END)
    fu_end = followup_end_for(entry.index)
    boundary = boundary.where(boundary <= fu_end, fu_end)
    is_event = event.notna() & (event > entry) & (event <= boundary)
    end_time = boundary.where(~is_event, event)

    events_by_label = {label: 0 for label, _, _ in periods}
    py_by_label = {label: 0.0 for label, _, _ in periods}

    for label, p_start, p_end in periods:
        seg_start = entry.where(entry > p_start, p_start)
        seg_end = end_time.where(end_time < p_end, p_end)
        py_by_label[label] += member_days(seg_start, seg_end).clip(lower=0).sum() / 365.25

    for pid in entry.index:
        e_start, e_end = entry.loc[pid], end_time.loc[pid]
        if pd.isna(e_start) or pd.isna(e_end) or e_end <= e_start:
            continue
        if is_event.loc[pid]:
            ev_date = event.loc[pid]
            for label, p_start, p_end in periods:
                if p_start <= ev_date < p_end:
                    events_by_label[label] += 1
                    break

    return events_by_label, py_by_label


def compute_by_second_diagnosis_year(frame, downstream_outcome, side, exposure_outcomes=None):
    """side: 'pre' or 'post' -- trauma diagnosed before/on-after Oct 7. Returns a DataFrame
    (label, n_events, person_years, irr) using the currently active window. 2023 is split
    exactly at Oct7 (SECOND_DIAGNOSIS_PERIODS), same as the entry-side Yearly trend chart --
    not approximated by whole calendar year. exposure_outcomes = the first diagnosis (default:
    the active trauma exposure)."""
    exposure_outcomes = core.TRAUMA_EXPOSURE_OUTCOMES if exposure_outcomes is None else exposure_outcomes
    first_trauma = first_of(frame, exposure_outcomes)
    entry = first_trauma.where((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    down = frame[f"first_{downstream_outcome}"]

    events_by_label, py_by_label = split_person_time_by_period(
        entry, down, core.FIXED_WINDOW_DAYS, SECOND_DIAGNOSIS_PERIODS)

    # PER-PERIOD general reference via continuous Lexis tracking -- replaces the old single
    # anchor-averaged constant, so each period's IRR is measured against that SAME period's
    # general-population rate, not one rate repeated across every point on the chart.
    gen_split = compute_general_lexis_split(frame, exposure_outcomes, downstream_outcome,
                                            SECOND_DIAGNOSIS_PERIODS)

    rows = []
    for label, start, end in SECOND_DIAGNOSIS_PERIODS:
        py, n = py_by_label[label], events_by_label[label]
        rate = (n / py * 1000) if py > 0 else np.nan
        gen_rate = gen_split.loc[label, "rate"]
        irr = rate / gen_rate if (pd.notna(rate) and pd.notna(gen_rate) and gen_rate > 0) else np.nan
        rows.append({"label": label, "start": start, "end": end,
                    "n_events": n, "person_years": py, "irr": irr})
    return pd.DataFrame(rows)


MIN_N_THRESHOLDS = [10, 30]

# --- Boxplot comparison, redesigned around the EXACT date of the second diagnosis (not a
# whole-year approximation) -- every patient has a real date, so "before Oct7" / "on-after
# Oct7" is a clean partition with nothing dropped:
#   Group 1 (gray)  = trauma BEFORE Oct7  AND second diagnosis BEFORE Oct7
#   Group 2 (red)   = trauma ON/AFTER Oct7 (its own second diagnosis is automatically also
#                     on/after Oct7, since event > entry >= Oct7 by construction)
#   One-tail t-test: group 1 (gray) > group 2 (red)
#   Group 3 (gray, NOT tested) = trauma BEFORE Oct7 AND second diagnosis ON/AFTER Oct7
# Group 1 and Group 3 TOGETHER exactly partition the "pre" line's full dataset -- nothing
# dropped, nothing double-counted, no whole-year approximation for the split 2023 period.
# IMPORTANT CAVEAT, stated directly on the chart: each point is ONE PERIOD's IRR, not one
# patient -- small-sample test (typically <10 points per group), not a patient-level
# comparison. Treat p-values as suggestive, not definitive.
def compute_second_diagnosis_boxplot_values(frame, downstream_outcome, trauma_side,
                                            event_side="all", min_n=0, exposure_outcomes=None):
    """trauma_side: 'pre' or 'post' (which line to pull from). event_side: 'pre' (period ends
    by Oct7), 'post' (period starts on/after Oct7), or 'all' (no restriction)."""
    trend = compute_by_second_diagnosis_year(frame, downstream_outcome, trauma_side, exposure_outcomes)
    if event_side == "pre":
        trend = trend[trend["end"] <= OCT7]
    elif event_side == "post":
        trend = trend[trend["start"] >= OCT7]
    sub = trend[trend["n_events"] >= min_n]
    return sub["irr"].dropna().values


# --- PATIENT-LEVEL counterpart of the boxplot above: each point = ONE PATIENT, not one year ---
# An IRR has no per-patient value, so the per-patient measure is TIME TO SECOND DIAGNOSIS: days
# from the trauma entry to the downstream diagnosis, for patients who converted inside their
# follow-up (same rules as person_time_events_and_years(): prevalent cases excluded, event must
# fall after entry, within FIXED_WINDOW_DAYS, by fu_end/STUDY_END, and not within LAG_DAYS).
# Same 3 groups as the year-level boxplot, split on the exact dates of both diagnoses:
#   box 1 (gray) = trauma before Oct7, 2nd dx before Oct7
#   box 2 (red)  = trauma on/after Oct7 (2nd dx automatically also on/after Oct7)
#   box 3 (gray, NOT tested) = trauma before Oct7, 2nd dx on/after Oct7
# One-tail Mann-Whitney U, box 1 > box 2 (patient-level, skewed data -> rank test, not t-test).
# CAVEAT: this answers "how FAST did converters convert", not "how MANY converted" -- and box 1 /
# box 2 are both truncated by their own boundary (Oct7 / STUDY_END), which caps latencies for
# late entrants, so differences can partly reflect available follow-up, not biology.
def compute_time_to_second_dx(frame, downstream_outcome, trauma_side, event_side):
    """Days from trauma entry to 2nd diagnosis, one value per converting patient.
    trauma_side: 'pre'/'post' Oct7. event_side: 'pre'/'post' Oct7 (2nd dx date) or 'all'."""
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    entry = first_trauma.where((first_trauma < OCT7) if trauma_side == "pre" else (first_trauma >= OCT7))
    entry = entry.dropna()
    event = frame[f"first_{downstream_outcome}"].reindex(entry.index)

    window_end = entry + pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D")
    boundary = window_end.where(window_end <= STUDY_END, STUDY_END)
    fu_end = followup_end_for(entry.index)
    boundary = boundary.where(boundary <= fu_end, fu_end)

    is_event = event.notna() & (event > entry) & (event <= boundary)
    if LAG_DAYS > 0:
        is_event &= (event - entry).dt.days >= LAG_DAYS
    if event_side == "pre":
        is_event &= event < OCT7
    elif event_side == "post":
        is_event &= event >= OCT7
    return (event[is_event] - entry[is_event]).dt.days.values


def km_intervals(frame, downstream_outcome, group):
    """Per-patient (t_enter, t_exit, event) in days since trauma entry, for KM group 1/2/3."""
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    entry = first_trauma.where((first_trauma >= OCT7) if group == 2 else (first_trauma < OCT7)).dropna()
    event = frame[f"first_{downstream_outcome}"].reindex(entry.index)
    prevalent = event.notna() & (event <= entry)
    entry, event = entry[~prevalent], event[~prevalent]

    window_end = entry + pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D")
    boundary = window_end.where(window_end <= STUDY_END, STUDY_END)
    fu_end = followup_end_for(entry.index)
    boundary = boundary.where(boundary <= fu_end, fu_end)
    is_event = event.notna() & (event > entry) & (event <= boundary)
    if LAG_DAYS > 0:
        is_event &= (event - entry).dt.days >= LAG_DAYS
    exit_date = boundary.where(~is_event, event)

    enter_date = entry.copy()
    if group == 1:                       # censor at Oct7
        is_event &= exit_date < OCT7
        exit_date = exit_date.where(exit_date < OCT7, OCT7)
    elif group == 3:                     # delayed entry at Oct7, only if still at risk then
        still_at_risk = exit_date > OCT7
        enter_date, exit_date, is_event = (pd.Series(OCT7, index=entry.index)[still_at_risk],
                                           exit_date[still_at_risk], is_event[still_at_risk])
        entry = entry[still_at_risk]

    # one row per membership piece (counting-process form): a patient leaves the risk set
    # during a membership gap and rejoins after it; the event sits on the last piece
    rows, piece_start, piece_end, is_last = membership_pieces(enter_date, exit_date)
    piece_entry = entry.reindex(rows).values
    t_enter = ((piece_start.values - piece_entry) / np.timedelta64(1, "D")).astype(float)
    t_exit = ((piece_end.values - piece_entry) / np.timedelta64(1, "D")).astype(float)
    piece_event = is_event.reindex(rows).values & is_last
    keep = t_exit > t_enter
    return t_enter[keep], t_exit[keep], piece_event[keep]


def risk_set_sizes(t_enter, t_exit, times):
    """n at risk at each time t: entered before t and not yet exited (t_enter < t <= t_exit)."""
    return (np.searchsorted(np.sort(t_enter), times, side="left")
            - np.searchsorted(np.sort(t_exit), times, side="left"))


def km_cumulative_incidence(t_enter, t_exit, is_event):
    """Returns (times, cum_inc, ci_low, ci_high) -- 1-KM with delayed entry, Greenwood variance,
    log(-log) CI. times starts at 0 with cum_inc 0."""
    ev_times = np.unique(t_exit[is_event])
    d = np.array([np.sum((t_exit == t) & is_event) for t in ev_times]) if len(ev_times) else np.array([])
    n = risk_set_sizes(t_enter, t_exit, ev_times)
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(1 - d / n)
        green = np.cumsum(d / (n * (n - d)))
        log_s = np.log(surv)
        se_loglog = np.sqrt(green) / np.abs(log_s)
        lo_s = np.exp(-np.exp(np.log(-log_s) + 1.96 * se_loglog))
        hi_s = np.exp(-np.exp(np.log(-log_s) - 1.96 * se_loglog))
    times = np.concatenate([[0], ev_times])
    cum = np.concatenate([[0], 1 - surv])
    lo = np.concatenate([[0], np.nan_to_num(1 - hi_s, nan=0)])
    hi = np.concatenate([[0], np.nan_to_num(1 - lo_s, nan=1)])
    return times, cum, lo, hi


def logrank_test(a, b):
    """Two-sided log-rank p-value between two (t_enter, t_exit, is_event) groups."""
    times = np.unique(np.concatenate([a[1][a[2]], b[1][b[2]]]))
    if len(times) == 0:
        return np.nan
    d1 = np.array([np.sum((a[1] == t) & a[2]) for t in times])
    d2 = np.array([np.sum((b[1] == t) & b[2]) for t in times])
    n1, n2 = risk_set_sizes(a[0], a[1], times), risk_set_sizes(b[0], b[1], times)
    n, dd = n1 + n2, d1 + d2
    ok = n > 1
    e1 = dd[ok] * n1[ok] / n[ok]
    v = dd[ok] * (n1[ok] / n[ok]) * (1 - n1[ok] / n[ok]) * (n[ok] - dd[ok]) / (n[ok] - 1)
    if v.sum() <= 0:
        return np.nan
    chi2 = (d1[ok].sum() - e1.sum()) ** 2 / v.sum()
    return stats.chi2.sf(chi2, df=1)


# --- T-Yearly-Forecast: quantitative version of "gradual decline vs sharp break" ---------------
# Fits an OLS line on the PRE-WAR points only (2017 .. "2023 (pre)"), extrapolates it forward,
# and computes a 95% prediction interval around that extrapolation. If the actual post-war
# point(s) fall OUTSIDE the interval, that's a real break beyond continuation of whatever
# pre-existing trend the series already had -- not just eyeballing whether post-war is "the
# lowest point", which a noisy-but-already-declining series could produce on its own.
def fit_trend_and_predict(x_fit, y_fit, x_predict, alpha=0.05):
    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    valid = ~np.isnan(y_fit)
    x_fit, y_fit = x_fit[valid], y_fit[valid]
    n = len(x_fit)
    x_predict = np.asarray(x_predict, dtype=float)
    if n < 3:
        nan_arr = np.full_like(x_predict, np.nan)
        return {"slope": np.nan, "intercept": np.nan, "y_pred": nan_arr,
               "lower": nan_arr, "upper": nan_arr, "n_fit": n}

    xbar = x_fit.mean()
    Sxx = np.sum((x_fit - xbar) ** 2)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    sse = np.sum((y_fit - (slope * x_fit + intercept)) ** 2)
    dof = n - 2
    s = np.sqrt(sse / dof) if dof > 0 else np.nan
    t_crit = stats.t.ppf(1 - alpha / 2, dof) if dof > 0 else np.nan

    y_pred = slope * x_predict + intercept
    if dof > 0 and Sxx > 0:
        se_pred = s * np.sqrt(1 + 1/n + (x_predict - xbar) ** 2 / Sxx)
        lower, upper = y_pred - t_crit * se_pred, y_pred + t_crit * se_pred
    else:
        lower = upper = np.full_like(x_predict, np.nan)
    return {"slope": slope, "intercept": intercept, "y_pred": y_pred,
           "lower": lower, "upper": upper, "n_fit": n}


# --- T-Quarterly: same design as T-Yearly, at quarterly (3-month) resolution instead of yearly.
# Finer-grained version of the same question -- a real gradual trend should look smooth quarter
# to quarter; a data artifact or small-n noise will look jagged. Q_INDEX matches Statement 3's
# quarterly range (2017Q1-2026Q2) so the two charts' time axes line up if compared side by side.
Q_INDEX = pd.period_range("2017Q1", "2026Q2", freq="Q")
QUARTERLY_TREND_PERIODS = [(str(q), q.start_time, (q + 1).start_time) for q in Q_INDEX]
_OCT7_QUARTER_POS = next(i for i, q in enumerate(Q_INDEX) if q == pd.Period("2023Q4", freq="Q"))

def compute_quarterly_rr_trend(frame, downstream_outcome):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]

    n_gen = int(first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES).isna().sum())
    gen_split = compute_general_lexis_split(frame, core.TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome,
                                            QUARTERLY_TREND_PERIODS)

    rows = []
    for label, start, end in QUARTERLY_TREND_PERIODS:
        period_entry = first_trauma.where((first_trauma >= start) & (first_trauma < end))
        n = int(period_entry.notna().sum())
        rate, _, _ = person_time_rate(period_entry, down, core.FIXED_WINDOW_DAYS)
        gen_rate = gen_split.loc[label, "rate"]
        rr = rate / gen_rate if (pd.notna(gen_rate) and gen_rate > 0 and pd.notna(rate)) else np.nan
        rows.append({"label": label, "n": n, "rr": rr})
    return pd.DataFrame(rows), n_gen


# =============================================================================
# PART 2 -- STATEMENT 2: ALTERNATE ENTRY-DIAGNOSIS RR ("B1", extended to 8)
# =============================================================================

# --- S2, STEP 1: RR for one downstream outcome, across N exposures (windowed, censored) ---
# Excludes the downstream outcome itself from the exposure set (no self-pairs).
def compute_b1_raw_rr(frame, downstream_outcome):
    labels, rrs, ns, n_gens = [], [], [], []
    for exp_label, outcome_list in current_alt_exposures().items():
        if downstream_outcome in outcome_list:
            continue   # exposure == downstream outcome; skip self-pair
        n_exp, n_gen, comp = compute_exposed_vs_general(frame, outcome_list, [downstream_outcome])
        row = comp.iloc[0]
        labels.append(exp_label); rrs.append(row["rr"]); ns.append(n_exp); n_gens.append(n_gen)
    return labels, rrs, ns, n_gens


# --- S2, STEP 3: Pre/post-war RR heatmap across all 8 exposures x 2 focus outcomes ---
def compute_b1_prepost_matrix(frame, entry_side):
    """entry_side: 'pre' or 'post' (Oct 7 diagnosis date of the ENTRY exposure)."""
    rows = []
    for exp_label, outcome_list in current_alt_exposures().items():
        downstream = [o for o in FOCUS_OUTCOMES if o not in outcome_list]
        if not downstream:
            continue
        (n_pre, rr_pre), (n_post, rr_post), n_gen = compute_prepost_window_rr(
            frame, outcome_list, downstream, core.FIXED_WINDOW_DAYS)
        n_exp, rr_dict = (n_pre, rr_pre) if entry_side == "pre" else (n_post, rr_post)
        row = {"exposure": exp_label, "n_exp": n_exp, "n_gen": n_gen}
        for o in FOCUS_OUTCOMES:
            row[o] = rr_dict.get(o, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("exposure")


# --- S2, STEP 3b: How much did each cell's RR change from pre- to post-Oct7 entry? ---
# % change = (post_rr - pre_rr) / pre_rr * 100 -- green if it went UP (post_rr > pre_rr),
# red if it went DOWN, intensity by magnitude. Self-pairs and cells missing either side stay
# NaN (blank), same as the two heatmaps this one is derived from.
def compute_b1_prepost_change_matrix(pre_matrix, post_matrix):
    rows = []
    for exposure in pre_matrix.index:
        row = {"exposure": exposure,
              "n_pre": pre_matrix.loc[exposure, "n_exp"],
              "n_post": post_matrix.loc[exposure, "n_exp"] if exposure in post_matrix.index else np.nan}
        for o in FOCUS_OUTCOMES:
            pre_rr  = pre_matrix.loc[exposure, o] if exposure in pre_matrix.index else np.nan
            post_rr = post_matrix.loc[exposure, o] if exposure in post_matrix.index else np.nan
            if pd.notna(pre_rr) and pd.notna(post_rr) and pre_rr > 0:
                row[o] = (post_rr - pre_rr) / pre_rr * 100
            else:
                row[o] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("exposure")


# --- S2, STEP 4: Write Statement 2 pages (4 pages) for one age band ---
# --- S2 EXTENDED: every entry diagnosis x every second diagnosis (7x7), plus the bar charts for
# the second diagnoses slides 22/24 don't cover. Written at the END of the PDF (slides 43+) so
# slides 1-42 keep their numbers. Same windowed person-time design as the rest of Statement 2.
# The merged trauma diagnosis is also a SECOND diagnosis here: its first date as an outcome is
# the earliest of its components (column first_trauma_dx, added on the fly). Diagonal = self-pair
# (entry diagnosis == second diagnosis) and stays empty.
TRAUMA_DX_KEY = "trauma_dx"


def s2_targets(frame):
    """(label, downstream key, component outcomes) for the 8 second diagnoses, trauma first."""
    if f"first_{TRAUMA_DX_KEY}" not in frame.columns:
        frame[f"first_{TRAUMA_DX_KEY}"] = first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    return [(label, TRAUMA_DX_KEY if outcomes == core.TRAUMA_EXPOSURE_OUTCOMES else outcomes[0], outcomes)
            for label, outcomes in current_alt_exposures().items()]


def compute_s2_full_matrices(frame):
    """Returns (overall, pre, post): DataFrames indexed by entry-diagnosis label, columns n_exp,
    n_gen + one IRR column per downstream key. NaN on the diagonal (self-pair)."""
    targets = s2_targets(frame)
    overall, pre, post = [], [], []
    for exp_label, exp_outcomes in current_alt_exposures().items():
        downs = [key for _, key, comps in targets if comps != exp_outcomes]
        n_exp, n_gen, comp = compute_exposed_vs_general(frame, exp_outcomes, downs)
        row = {"exposure": exp_label, "n_exp": n_exp, "n_gen": n_gen}
        row.update(dict(zip(comp["outcome"], comp["rr"])))
        overall.append(row)
        (n_pre, rr_pre), (n_post, rr_post), n_gen_pp = compute_prepost_window_rr(
            frame, exp_outcomes, downs, core.FIXED_WINDOW_DAYS)
        pre.append({"exposure": exp_label, "n_exp": n_pre, "n_gen": n_gen_pp, **rr_pre})
        post.append({"exposure": exp_label, "n_exp": n_post, "n_gen": n_gen_pp, **rr_post})
    keys = [key for _, key, _ in targets]
    as_df = lambda rows: pd.DataFrame(rows).set_index("exposure").reindex(columns=["n_exp", "n_gen"] + keys)
    return as_df(overall), as_df(pre), as_df(post)


# =============================================================================
# COMORBIDITY -- risk of psychosis / bipolar when the patient has BOTH the trauma diagnosis and one
# other diagnosis X, against: no trauma and no X / trauma only / X only. Appended at the end of
# the presentation PDF (after Statement 2 extended) so earlier slide numbers don't move.
#
# Time-varying status, one comparison per (outcome, X). Each patient moves through statuses and
# every stretch of person-time is counted in the status the patient was in at that time:
#   none     : from fu_start (later of 2017 / age 15) until trauma, X, or fu_end -- no window
#   trauma   : from the trauma date until the earliest of trauma + FIXED_WINDOW_DAYS, X, fu_end
#   X only   : from the X date until the earliest of X + FIXED_WINDOW_DAYS, trauma, fu_end
#   both     : from the later of the two dates (any order) until + FIXED_WINDOW_DAYS / fu_end
# The outcome ends the stretch it falls in; an outcome on/before a stretch's start = prior case
# (no person-time). A patient whose trauma or X falls before age 15 (or, in a diagnoser mode, was
# not given by the matching diagnoser) is left out of that comparison, as in the rest of the file.
# Before/after Oct 7 = by the date the patient ENTERED that status; "none" is always the whole
# period (as the general population is on slides 2-3).
# =============================================================================
COMORB_STATES = ["none", "trauma", "x", "both"]


def comorb_state_labels(x_label):
    return {"none": f"no {core.TRAUMA_EXPOSURE_LABEL}, no {x_label}", "trauma": f"{core.TRAUMA_EXPOSURE_LABEL} only",
            "x": f"{x_label} only", "both": f"{core.TRAUMA_EXPOSURE_LABEL} + {x_label}"}


def comorb_x_list(outcome):
    """The other diagnoses X paired with trauma for this outcome: (label, outcome key)."""
    return [(label, outs[0]) for label, outs in current_alt_exposures().items()
            if outs != core.TRAUMA_EXPOSURE_OUTCOMES and outs[0] != outcome]


def comorbidity_segments(frame, outcome, x):
    """One row per (patient, status) stretch with person-time: state, start, py, event."""
    T, Tr = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES), first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    X, Xr = first_of(frame, [x]), first_of_raw(frame, [x])
    keep = ~(Tr.notna() & T.isna()) & ~(Xr.notna() & X.isna())
    T, X = T[keep], X[keep]
    O, fu_start, cap = frame.loc[keep, f"first_{outcome}"], frame.loc[keep, "fu_start"], frame.loc[keep, "fu_end"]
    win = pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D")
    earliest = lambda *cols: pd.concat(cols, axis=1).min(axis=1)   # skips NaT

    starts = {"none": fu_start,
              "trauma": T.where(T.notna() & (X.isna() | (X > T))),
              "x": X.where(X.notna() & (T.isna() | (T > X))),
              "both": pd.concat([T, X], axis=1).max(axis=1).where(T.notna() & X.notna())}
    ends = {"none": earliest(T, X, cap),
            "trauma": earliest(starts["trauma"] + win, X.where(X > T), cap),
            "x": earliest(starts["x"] + win, T.where(T > X), cap),
            "both": earliest(starts["both"] + win, cap)}
    parts = []
    for state in COMORB_STATES:
        start, end = starts[state], ends[state]
        ok = start.notna() & (end > start) & ~(O.notna() & (O <= start))
        start, end, o = start[ok], end[ok], O[ok]
        event = o.notna() & (o > start) & (o <= end)
        if LAG_DAYS > 0:
            event &= (o - start).dt.days >= LAG_DAYS
        exit_ = end.where(~event, o)
        parts.append(pd.DataFrame({"state": state, "start": start,
                                   "py": member_days(start, exit_) / 365.25, "event": event}))
    return pd.concat(parts, ignore_index=True)


def comorbidity_rates(seg):
    """Per state: events, person-years, rate /1,000 PY -- all years, and pre/post by entry date."""
    out = {}
    for side, mask in [("all", pd.Series(True, index=seg.index)), ("pre", seg["start"] < OCT7),
                       ("post", seg["start"] >= OCT7)]:
        for state in COMORB_STATES:
            sub = seg[(seg["state"] == state) & (mask if state != "none" else True)]
            ev, py = int(sub["event"].sum()), sub["py"].sum()
            out[(side, state)] = {"events": ev, "py": py, "patients": len(sub),
                                  "rate": ev / py * 1000 if py > 0 else np.nan}
    return out


def compute_comorbidity_tables(frame):
    """{outcome: DataFrame indexed by X label} with rates per state and the 3 IRRs (both vs
    none / trauma / x) for all years, pre, post, plus % change pre->post of each IRR."""
    tables, details = {}, {}
    for outcome in FOCUS_OUTCOMES:
        rows = {}
        for x_label, x in comorb_x_list(outcome):
            r = comorbidity_rates(comorbidity_segments(frame, outcome, x))
            details[(outcome, x_label)] = r
            row = {}
            for side in ["all", "pre", "post"]:
                for ref in ["none", "trauma", "x"]:
                    ref_rate = r[("all" if ref == "none" else side, ref)]["rate"]
                    both_rate = r[(side, "both")]["rate"]
                    row[f"{side}|{ref}"] = both_rate / ref_rate if ref_rate and ref_rate > 0 else np.nan
            for ref in ["none", "trauma", "x"]:
                a, b = row[f"pre|{ref}"], row[f"post|{ref}"]
                row[f"change|{ref}"] = (b - a) / a * 100 if pd.notna(a) and a > 0 and pd.notna(b) else np.nan
            row["n_both"] = r[("all", "both")]["patients"]
            for side in ["all", "pre", "post"]:
                row[f"{side}|ev"] = r[(side, "both")]["events"]   # comorbid events behind each IRR
            rows[x_label] = row
        tables[outcome] = pd.DataFrame(rows).T
    return tables, details


COMORB_REF_LABELS = {"none": "vs no trauma, no X", "trauma": "vs trauma only", "x": "vs X only"}


# =============================================================================
# PART 3 -- STATEMENT 3: ROSE vs DID-NOT-RISE (with Northern Border zone)
# =============================================================================

# --- S3, STEP 1: Zone populations = patient count per zone in this frame ---
def zone_populations(frame):
    return frame["zone"].value_counts().reindex(ZONE_ORDER).fillna(0)


# --- S3, STEP 2: Quarterly first-ever counts per zone for one item ('total' = all 9) ---
def quarterly_counts(frame, item, group_col="zone", groups=ZONE_ORDER):
    q_index = pd.period_range("2017Q1", "2026Q2", freq="Q")
    outcomes = ALL_OUTCOMES if item == "total" else [item]
    ev = pd.concat(
        [frame[[group_col, "fu_start", "fu_end", f"first_{o}"]].dropna(subset=[f"first_{o}"])
              .rename(columns={f"first_{o}": "date"})
         for o in outcomes], ignore_index=True)
    # only diagnoses made while the patient was under follow-up (same window as the denominator)
    ev = ev[(ev["date"] >= ev["fu_start"]) & (ev["date"] <= ev["fu_end"])]
    ev["quarter"] = ev["date"].dt.to_period("Q")
    ev = ev[ev["quarter"].isin(q_index)]
    return (ev.groupby(["quarter", group_col]).size()
           .unstack(group_col).reindex(index=q_index, columns=groups).fillna(0))


# --- S3, STEP 2b: Person-years under follow-up per quarter per zone (the rate denominator) ---
# Replaces a fixed zone headcount: a patient only counts toward a quarter for the days they
# were actually members (and 15+, and in Israel) during it -- so a zone whose population grew
# over the years no longer shows inflated early-year rates.
def quarterly_person_years(frame, q_index, group_col="zone", groups=ZONE_ORDER):
    rows = {}
    for q in q_index:
        seg_start = frame["fu_start"].clip(lower=q.start_time)
        seg_end   = frame["fu_end"].clip(upper=(q + 1).start_time)
        py = member_days(seg_start, seg_end).clip(lower=0) / 365.25
        rows[q] = py.groupby(frame[group_col]).sum()
    return pd.DataFrame(rows).T.reindex(index=q_index, columns=groups).fillna(0)


def to_rate_per_1000_py(counts, person_years):
    return counts / person_years.replace(0, np.nan) * 1000


# --- S3, STEP 3: Draw one quarterly per-1,000-by-zone panel ---
# Zones drawn in Statement 3. Northern Border is left out on request -- its residents are simply
# not shown (NOT merged into Rest of Israel), so the two remaining lines are unchanged.
STATEMENT3_ZONES = ["Near Gaza", "Rest of Israel"]


def prepost_ratio_text(rate_df):
    pre  = [q for q in rate_df.index if q <= pd.Period("2023Q3", freq="Q")]
    post = [q for q in rate_df.index if q >= pd.Period("2023Q4", freq="Q")]
    ratio = rate_df.loc[post].mean() / rate_df.loc[pre].mean().replace(0, np.nan)
    return "post/pre rate ratio -> " + "  |  ".join(
        f"{z}: x{ratio[z]:.2f}" for z in STATEMENT3_ZONES if z in ratio.index)


# --- S3, STEP 5: Same rates, one line per DISTANCE-FROM-GAZA band instead of per zone --------
# Northern Border residents are left out (not merged into a band), like the zone pages above.
# Patients with no distance value are left out too. In a population file that is already cut
# by distance (e.g. <=20 km), only the bands inside that range have lines.
DIST_BANDS = [(0, 7, "0-7 km"), (7, 20, "7-20 km"), (20, 40, "20-40 km"),
              (40, 80, "40-80 km"), (80, np.inf, "80+ km")]
DIST_BAND_LABELS = [label for _, _, label in DIST_BANDS]


def distance_band(frame):
    """Distance band per patient (lower bound exclusive, so exactly 20 km falls in 7-20 km,
    matching the <=20 km population); NaN for Northern Border residents / no distance."""
    band = pd.Series(np.nan, index=frame.index, dtype=object)
    for lo, hi, label in DIST_BANDS:
        band[(frame["distance_km"] > lo) & (frame["distance_km"] <= hi)] = label
    band[frame["distance_km"] == 0] = DIST_BANDS[0][2]
    band[frame["is_north_border"]] = np.nan
    return band


# =============================================================================
# PART 4 -- STATEMENT 5: WHO DIAGNOSES WHAT (profession by specialty, pre/post)
#
# Not from the docx or from Trauma_specificity_probe_Step_5.py -- added on request. Tests
# whether psychosis/bipolar are diagnosed mainly by a psychiatrist while every other outcome
# (including PTSD/ASR) is diagnosed at meaningfully higher rates by other professions, and
# whether that split holds BOTH before and after Oct 7 (checked separately, not pooled) --
# i.e. that the specialist-vs-generalist pattern itself is stable across the war, even though
# the diagnosis RATES around it (Statement 3) are not.
# =============================================================================

# --- S5, STEP 1: % of one outcome's first-ever diagnoses given by a psychiatrist, one side ---
# code 999 ("Unknown/Unassigned/Admin") is MISSING data, not a "non-psychiatrist" profession --
# excluded from the denominator entirely so it can't dilute the percentage either way.
def compute_pct_psychiatrist(outcome_meta, outcome, side, treat_missing_as_nonpsychiatrist):
    dates = outcome_meta[outcome]["date"]
    codes = normalize_profession_code(outcome_meta[outcome]["profession"])
    mask = dates.notna() & ((dates < OCT7) if side == "pre" else (dates >= OCT7))
    codes_side = codes[mask]
    if treat_missing_as_nonpsychiatrist:
        # code 999 counts in the denominator as a non-psychiatrist case (conservative: treats
        # "unknown who diagnosed" the same as "diagnosed by someone else").
        valid = codes_side.notna()
    else:
        # code 999 excluded from both numerator and denominator (it's missing data, not a
        # real profession) -- the treatment used everywhere else in this script.
        valid = codes_side.notna() & ~codes_side.isin(MISSING_PROFESSION_CODES)
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    is_psychiatrist = codes_side[valid].isin(PSYCHIATRIST_PROFESSION_CODES)
    return is_psychiatrist.mean() * 100, n


# --- S5, STEP 1b: Print real profession codes (decoded) + warn if the psychiatrist codes don't
# appear at all. Runs once per script execution (not once per age band / trauma mode) so the
# console log stays readable.
_profession_diagnostic_printed = False

def print_profession_diagnostic(outcome_meta):
    global _profession_diagnostic_printed
    if _profession_diagnostic_printed:
        return
    _profession_diagnostic_printed = True
    print("\n  [Statement 5 diagnostic] Profession codes found in the data (decoded):")
    any_match = False
    for outcome in ["psychosis", "bipolar"]:
        codes = normalize_profession_code(outcome_meta[outcome]["profession"]).dropna()
        vc = codes.value_counts()
        decoded = {profession_code_label(k): v for k, v in vc.head(8).items()}
        print(f"    {outcome} (top values): {decoded}")
        if vc.index.isin(PSYCHIATRIST_PROFESSION_CODES).any():
            any_match = True
    if not any_match:
        print(f"    *** WARNING: none of the psychiatrist codes {PSYCHIATRIST_PROFESSION_CODES} "
             f"appear in the data above. Statement 5 will read ~0% for every bar -- check "
             f"PSYCHIATRIST_PROFESSION_CODES against the real coding scheme (it may differ by "
             f"outcome table, or use different codes than the decoder given for this field). ***")
    else:
        print(f"    OK: psychiatrist codes {PSYCHIATRIST_PROFESSION_CODES} found in the data.")
    print("    NOTE: Statement 5 now renders TWO pages -- one excluding code 999 (Unknown/Admin)"
         " from both numerator and denominator, one counting it as non-psychiatrist -- see how"
         " much the two treatments differ before picking one for the final presentation.")


# =============================================================================
# PART 6 -- STATEMENT 6: LATENCY -- HOW LONG FROM TRAUMA TO THE SECOND DIAGNOSIS
#
# Not from the docx -- added on request, as the empirical basis for deciding whether LAG_DAYS
# (protopathic-bias exclusion, currently 0/off) is worth setting to a specific value: if these
# histograms show a suspicious early spike (many "conversions" in the first days/weeks), that's
# evidence for setting a lag; if the distribution is smooth, LAG_DAYS=0 is fine as-is.
# UNRESTRICTED by window_days/WINDOW_MODES on purpose -- this only exists to show the empirical
# shape of the latency distribution, so capping it at the window would hide exactly the
# information (long delays, or how much mass sits outside any given window) it exists to reveal.
# One page per NON-trauma outcome (7 pages: psychosis, bipolar, anxiety, depression, substance,
# personality, OCD), each with 2 panels (Pre-Oct7 entry / Post-Oct7 entry).
# =============================================================================

LATENCY_BINS_MONTHS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 36), (36, np.inf)]
LATENCY_BIN_LABELS  = ["0-1", "1-3", "3-6", "6-12", "12-24", "24-36", "36+"]

# --- S6, STEP 1: Latency (months) from trauma entry to this outcome, among converters, one side ---
def compute_latency_distribution(frame, downstream_outcome, side):
    first_trauma = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    side_mask = first_trauma.notna() & ((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    converted_mask = side_mask & down.notna() & (down > first_trauma)
    latency_months = (down[converted_mask] - first_trauma[converted_mask]).dt.days / 30.44

    counts = [int(((latency_months >= lo) & (latency_months < hi)).sum())
             for lo, hi in LATENCY_BINS_MONTHS]
    return counts, int(converted_mask.sum())


# =============================================================================
# PART 7 -- STATEMENT 7: DEMOGRAPHIC COMPARISON OF TRAUMA-DIAGNOSED PATIENTS, PRE VS POST OCT7
#
# Not from the docx -- added on request, since the research question is specifically about
# whether the CHARACTERISTICS of who gets a PTSD/ASR diagnosis changed across the war, not just
# the downstream psychosis/bipolar risk. Population here is EVERY patient with a trauma
# diagnosis (current TRAUMA_EXPOSURE_OUTCOMES/diagnoser mode), split into "before Oct7" vs
# "on/after Oct7" by their OWN trauma date -- no downstream outcome, no window, no person-time
# involved at all; this is purely "who are these two groups of patients."
#
# ONE separate PDF per run (not multiplied across the 12 trauma/window/diagnoser combinations),
# saved directly to output_dir alongside the 12 statement decks and the sanity-check report.
#
# Statistics: non-parametric throughout, matching this project's convention --
#   continuous/ordinal variables (age, Charlson score): Mann-Whitney U
#   categorical variables (gender, sector, district, zone, ...): Chi-square test of independence
# =============================================================================

def split_trauma_pre_post(frame):
    """Every patient with a trauma diagnosis (current TRAUMA_EXPOSURE_OUTCOMES/diagnoser mode),
    split into (pre, post) DataFrames by whether that diagnosis was before or on/after Oct7."""
    trauma_date = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    pre  = frame[trauma_date.notna() & (trauma_date < OCT7)]
    post = frame[trauma_date.notna() & (trauma_date >= OCT7)]
    return pre, post


def chi_square_pvalue(pre_series, post_series):
    pre_vals  = pre_series.dropna().astype(str)
    post_vals = post_series.dropna().astype(str)
    if len(pre_vals) == 0 or len(post_vals) == 0:
        return "ns"
    combined = pd.DataFrame({
        "value":  pd.concat([pre_vals, post_vals], ignore_index=True),
        "period": ["pre"] * len(pre_vals) + ["post"] * len(post_vals),
    })
    ct = pd.crosstab(combined["value"], combined["period"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return "ns"
    try:
        _, p, _, _ = stats.chi2_contingency(ct)
        return f"{p:.2e}" if p < 0.05 else "ns"
    except ValueError:
        return "ns"


# --- S7, STEP 2b: PTSD vs ASR composition -- did the MIX shift, not just the count? ---
def classify_trauma_composition(sub):
    has_ptsd = sub["first_ptsd"].notna()
    has_asr  = sub["first_asr"].notna()
    out = pd.Series("Both", index=sub.index)
    out[has_ptsd & ~has_asr] = "PTSD only"
    out[~has_ptsd & has_asr] = "ASR only"
    return out


# --- S7, STEP 2c: Prior psychiatric history BEFORE the trauma diagnosis -- reverse-causality /
# diagnostic-purity check (this was an open "still todo" item -- built here for the first
# time). If post-war patients have LESS prior history than pre-war ones, that supports "war
# trauma captures a broader, less pre-selected population" rather than "the same people who
# were already engaged with mental health services." ---
PRIOR_HISTORY_OUTCOMES = ["anxiety", "depression", "substance", "personality", "OCD", "psychosis", "bipolar"]

def has_any_prior_history(sub):
    entry = first_of(sub, core.TRAUMA_EXPOSURE_OUTCOMES)
    has_prior = pd.Series(False, index=sub.index)
    for o in PRIOR_HISTORY_OUTCOMES:
        has_prior |= (sub[f"first_{o}"].notna() & (sub[f"first_{o}"] < entry))
    return has_prior


# =============================================================================
# PART 7c -- IRR BY YEAR SINCE TRAUMA DIAGNOSIS (relative time, capped at Oct 7 2023)
#
# Different axis from everything else in this script: not calendar year, not year-of-second-
# diagnosis -- YEARS ELAPSED SINCE EACH PATIENT'S OWN TRAUMA DIAGNOSIS, pooled across patients
# by that relative index (year 1 since diagnosis, year 2, year 3, ...), with every patient's
# clock stopped at Oct 7 2023 regardless of when they were actually diagnosed. Only pre-war
# trauma patients can contribute at all -- a patient diagnosed on/after Oct7 has zero elapsed
# time before Oct7 by construction, so this naturally excludes them without needing an explicit
# filter (see compute_relative_year_split()'s loop condition).
#
# Two variants, each its own page (2 panels: Psychosis | Bipolar), in a SEPARATE PDF file:
#   Variant A: the final (possibly partial) year before Oct7 IS included -- e.g. entry
#     2020-03-01 gives full years 1/2/3, then a ~7-month PARTIAL contribution to "year 4"
#     (2023-03-01 to Oct7-2023).
#   Variant B: only COMPLETE relative years are counted -- the same patient contributes only
#     years 1/2/3; the partial ~7 months is dropped entirely (not folded into year 3, not
#     counted as year 4).
# IRR uses a single overall general-population reference rate (compute_general_rate_overall()),
# not a per-relative-year one -- there's no natural "year N of being unexposed" for a
# population that was never exposed to begin with, so this mirrors how H2 itself uses one
# constant reference rather than inventing a parallel structure for the general side.
# =============================================================================

def compute_relative_year_split(entry_dates, event_dates, max_years=10, full_years_only=False):
    """Lexis-splits person-time by YEARS SINCE ENTRY (not calendar year), every patient's own
    diagnosis date as their own day-zero, capped at OCT7 (not STUDY_END, not a window).
    full_years_only=False: a patient's final, possibly-partial year before Oct7 still
      contributes its partial person-time to that relative-year bucket.
    full_years_only=True: only buckets a patient reaches a FULL year for (entry + k years <=
      OCT7) are counted; the loop stops entirely at the first partial bucket for that patient.
    Returns a DataFrame (rel_year, n_events, person_years, rate)."""
    valid = entry_dates.notna()
    entry = entry_dates[valid]
    event = event_dates.reindex(entry.index)

    prevalent = event.notna() & (event <= entry)   # same rule as person_time_events_and_years()
    entry, event = entry[~prevalent], event[~prevalent]

    # per-patient cap: Oct 7, or earlier if they left Clalit / died before it
    fu_end = followup_end_for(entry.index)
    cap = fu_end.where(fu_end < OCT7, OCT7)

    is_event = event.notna() & (event > entry) & (event <= cap)
    end_time = cap.where(~is_event, event)

    events_by_year, py_by_year = {}, {}

    for pid in entry.index:
        e_start, e_end, p_cap = entry.loc[pid], end_time.loc[pid], cap.loc[pid]
        if pd.isna(e_start) or pd.isna(e_end) or e_end <= e_start:
            continue
        ev_date = event.loc[pid] if is_event.loc[pid] else None

        k = 1
        while True:
            bucket_start = e_start + pd.DateOffset(years=k - 1)
            bucket_full_end = e_start + pd.DateOffset(years=k)
            if bucket_start >= p_cap or bucket_start >= e_end or k > max_years:
                break
            is_full_year = bucket_full_end <= p_cap
            if full_years_only and not is_full_year:
                break   # stop here -- partial year not counted, nothing after it either

            bucket_end_capped = min(bucket_full_end, p_cap, e_end)
            contributed_days = member_days_one(pid, bucket_start, bucket_end_capped)
            if contributed_days > 0:
                py_by_year[k] = py_by_year.get(k, 0.0) + contributed_days / 365.25
                if ev_date is not None and bucket_start < ev_date <= bucket_end_capped:
                    events_by_year[k] = events_by_year.get(k, 0) + 1
            k += 1

    max_k = max(list(py_by_year.keys()) + [0])
    rows = []
    for k in range(1, max_k + 1):
        py, n = py_by_year.get(k, 0.0), events_by_year.get(k, 0)
        rate = (n / py * 1000) if py > 0 else np.nan
        rows.append({"rel_year": k, "n_events": n, "person_years": py, "rate": rate})
    return pd.DataFrame(rows)


def compute_relative_year_irr(frame, downstream_outcome, full_years_only=False):
    """Restricts to PRE-WAR trauma patients (post-war ones naturally contribute nothing --
    see module docstring above), then computes IRR per relative year against a single overall
    general-population reference rate."""
    trauma_date = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    entry = trauma_date.where(trauma_date < OCT7)   # pre-war only; NaT elsewhere is fine
    down = frame[f"first_{downstream_outcome}"]

    trend = compute_relative_year_split(entry, down, full_years_only=full_years_only)
    gen_rate, _, _ = compute_general_rate_overall(frame, core.TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome)
    trend["irr"] = trend["rate"] / gen_rate if (pd.notna(gen_rate) and gen_rate > 0) else np.nan
    return trend, gen_rate


# =============================================================================
# DESCRIPTIVE COMPANION PDF -- written next to every presentation_summary_*.pdf as
# presentation_summary_*_descriptive.pdf. No statistical tests: counts, person-years, follow-up,
# distributions -- the raw material behind each slide, so every slide's numbers can be read in
# context. Every page names the slide(s) it explains (numbering = the slide guide document).
# =============================================================================
FOLLOWUP_END_REASONS = ["2nd diagnosis (event)", "window ended", "left Clalit (membership end)",
                        "study end (30.6.2026)"]


def exposed_followup_detail(entry_dates, event_dates, window_days=None):
    """Per at-risk exposed patient (prevalent cases removed): entry, end, months followed, whether
    the follow-up ended in an event, and why it ended. Same rules as person_time_events_and_years().
    Returns (DataFrame, n_prevalent)."""
    window_days = core.FIXED_WINDOW_DAYS if window_days is None else window_days
    entry = entry_dates.dropna()
    event = event_dates.reindex(entry.index)
    prevalent = event.notna() & (event <= entry)
    entry, event = entry[~prevalent], event[~prevalent]

    window_end = entry + pd.to_timedelta(window_days, unit="D")
    fu_end = followup_end_for(entry.index)
    boundary = pd.concat([window_end, pd.Series(STUDY_END, index=entry.index), fu_end], axis=1).min(axis=1)
    is_event = event.notna() & (event > entry) & (event <= boundary)
    if LAG_DAYS > 0:
        is_event &= (event - entry).dt.days >= LAG_DAYS
    end = boundary.where(~is_event, event)

    reason = np.select(
        [is_event, boundary == window_end, (boundary == fu_end) & (fu_end < STUDY_END)],
        FOLLOWUP_END_REASONS[:3], default=FOLLOWUP_END_REASONS[3])
    detail = pd.DataFrame({"entry": entry, "end": end, "event": is_event, "reason": reason,
                           "months": member_days(entry, end) / 30.44})   # months as a member
    return detail, int(prevalent.sum())


def trauma_component_of_entry(frame, entry):
    """Which trauma diagnosis opened the entry (e.g. 'PTSD' / 'ASR' / 'both same day')."""
    comps = [o for o in core.TRAUMA_EXPOSURE_OUTCOMES if f"first_{o}" in frame.columns]
    hits = pd.DataFrame({o: frame.loc[entry.index, f"first_{o}"] == entry for o in comps})
    label = pd.Series("other", index=entry.index, dtype=object)
    for o in comps:
        label[hits[o]] = o.upper()
    if len(comps) > 1:
        label[hits.sum(axis=1) > 1] = "both same day"
    return label


DESC_AGE_BANDS = [(15, 25, "15-24"), (25, 35, "25-34"), (35, 45, "35-44"), (45, 55, "45-54"),
                  (55, 65, "55-64"), (65, 200, "65+")]


def second_dx_flow(frame, outcome):
    """Who is left after each rule of the 'By year of SECOND diagnosis' graph. Returns a list of
    (step, {"pre": n, "post": n}); the last count equals the sum of the n= labels on the graph."""
    tr = core.TRAUMA_EXPOSURE_LABEL
    raw = first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    entry_all = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    both = lambda n: {"pre": n, "post": n}
    steps = []
    if core._FULL_COHORT_N is not None:
        steps.append(("Patients in the input file", both(core._FULL_COHORT_N)))
    steps.append(("In this age band (and population)", both(len(frame))))
    steps.append((f"Ever diagnosed with {tr}", both(int(raw.notna().sum()))))
    steps.append((f"First {tr} at age 15 or older", both(int(raw.where(~(raw < frame['age15_date'])).notna().sum()))))
    if core.CURRENT_DIAGNOSER_MODE != "any_diagnoser":
        steps.append((f"First {tr} given by: {core.CURRENT_DIAGNOSER_LABEL}", both(int(entry_all.notna().sum()))))
    rules = {"pre": [], "post": []}
    window = pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D")
    for side in ("pre", "post"):
        entry = entry_all[(entry_all < OCT7) if side == "pre" else (entry_all >= OCT7)]
        down = frame[f"first_{outcome}"].reindex(entry.index)
        fu_end = followup_end_for(entry.index)
        keep = pd.Series(True, index=entry.index)
        rules[side].append(keep.sum())
        keep &= ~(down.notna() & (down <= entry))
        rules[side].append(keep.sum())
        keep &= down.notna() & (down > entry)
        rules[side].append(keep.sum())
        keep &= down <= entry + window
        rules[side].append(keep.sum())
        keep &= down <= STUDY_END
        rules[side].append(keep.sum())
        keep &= down <= fu_end
        rules[side].append(keep.sum())
        keep &= (down >= SECOND_DIAGNOSIS_PERIODS[0][1]) & (down < SECOND_DIAGNOSIS_PERIODS[-1][2])
        rules[side].append(keep.sum())
    names = [f"First {tr} before Oct 7 (pre) / on or after Oct 7 (post)",
             f"No second diagnosis (column) on or before the {tr} date (not a prior case)",
             f"Got the second diagnosis after the {tr} date (any time)",
             f"... within {core.FIXED_WINDOW_DAYS // 30} months of the {tr} date",
             f"... by the study end ({STUDY_END:%d.%m.%Y})",
             f"... before leaving Clalit (membership end)",
             f"... in {SECOND_DIAGNOSIS_YEAR_RANGE[0]}-{SECOND_DIAGNOSIS_YEAR_RANGE[-1]} (the graph's years)"]
    for i, name in enumerate(names):
        steps.append((name, {side: int(rules[side][i]) for side in ("pre", "post")}))
    return steps
