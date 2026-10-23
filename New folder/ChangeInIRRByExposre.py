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

# =============================================================================
# presentation_summary_step6.py
#
# Builds the curated presentation deck (now 5 "statements") described in
# PRE_POST_War_PTSD_misdiagnosis.docx plus additions made along the way, run
# once per TRAUMA MODE (merged PTSD/ASR, PTSD only, ASR only -- see
# TRAUMA_MODES) x 2 age bands = SIX standalone PDFs, one per (mode, age band)
# combination:
#   presentation_summary_age15_40_ptsd_asr.pdf   presentation_summary_age15_plus_ptsd_asr.pdf
#   presentation_summary_age15_40_ptsd_only.pdf  presentation_summary_age15_plus_ptsd_only.pdf
#   presentation_summary_age15_40_asr_only.pdf   presentation_summary_age15_plus_asr_only.pdf
# Splitting PTSD from ASR (instead of always merging them into one "trauma"
# exposure) tests whether the pre/post-war RR gap in Statement 1 is really a
# trauma-diagnosis-severity effect, or whether it's driven by a shift in the
# PTSD:ASR mix after Oct 7 (ASR is by definition acute/transient, PTSD
# requires sustained symptoms -- a war-period surge in ASR-heavy diagnosing
# could look like "weaker trauma signal" without being one).
#
# SECOND independent axis: DIAGNOSER MODE (any diagnoser / psychiatrist-only /
# non-psychiatrist-only -- see DIAGNOSER_MODES). Filters EVERY exposure in the
# script (trauma in Statement 1, and all 7 alternate exposures in Statement
# 2's B1 analysis) to only those whose earliest diagnosis was given by a
# psychiatrist / someone else / anyone. Tests the "diagnostic dilution"
# hypothesis directly: if the pre/post RR gap shrinks or vanishes once
# restricted to psychiatrist-diagnosed trauma only, that's evidence the gap
# is (at least partly) about WHO is diagnosing, not just WHEN. The __main__
# block runs this axis fully (3 diagnoser modes x 2 age bands = 6 files) for
# ONE trauma mode at a time (SELECTED_TRAUMA_MODE below) -- change that
# constant and re-run to get another trauma mode's 6 files, rather than
# generating all 3 trauma x 3 diagnoser x 2 age = 18 files in a single run.
#
# This script does NOT re-run every test in Step 3/4/5 -- it reproduces only
# the specific figures needed for the 3 statements, generalizing them where
# the docx asked for a broader version. Everything is self-contained (no
# cross-file imports) so it runs standalone, but every computation is a
# direct adaptation of logic already reviewed in:
#   - War_effect_directions_Step_4.py   (H2 exposed-vs-general, suppression)
#   - Trauma_specificity_probe_Step_5.py (TEST 2c pre/post-war heatmap, ALT_EXPOSURES)
#   - Alarms_Step5_1.py                  (3-zone scheme incl. Northern Border)
#
# METHODOLOGY NOTE -- EVERY "second illness after the first" calculation in this script
# (Statement 1 & 2, and TEST 2b/5/6/9's rate-based pieces) uses RESTRICTED PERSON-TIME, not
# binary cumulative risk in a fixed window:
#   1. Every patient contributes exactly however much follow-up time they were actually
#      observed for, up to window_days or STUDY_END, whichever comes first -- NOBODY IS
#      DROPPED for lacking full follow-up (the old design's eligible_reachable_mask() dropped
#      anyone without a full window_days of observed time before STUDY_END; person-time simply
#      lets them contribute a partial amount instead). See person_time_events_and_years().
#   2. The window itself is a WHOLE NUMBER OF YEARS ONLY (WINDOW_MODES: 12 or 24 months, never
#      18) -- this is what makes every entry-month see an equal mix of all 12 calendar months
#      inside its window, so no seasonal disorder pattern can bias a pre/post comparison by
#      cohort composition alone.
#   3. The general/reference population is measured via CONTINUOUS DYNAMIC-EXPOSURE LEXIS
#      TRACKING (compute_general_lexis_split()), not a fixed window from an anchor point: every
#      patient is tracked as "general" from fu_start (later of 2017 / age 15) until they get exposed
#      (raw exposure date, any diagnoser), get the downstream outcome, or fu_end --
#      whichever comes first -- and that continuous interval is split across whatever calendar
#      periods a given chart needs (a single whole-period total for H2/heatmaps/B1, or
#      per-year/per-quarter slices for the trend charts). This replaced an earlier design that
#      averaged 5 fixed pseudo-entry anchors (2017/2019/2021/2023/2025) -- that fixed secular
#      drift in the background rate, but a window-from-an-anchor never actually stayed inside
#      the calendar period it was meant to represent (e.g. a 2023 anchor + 24mo window bleeds
#      into 2024-2025), so a per-year general rate computed that way wasn't really "2023"'s
#      rate. Continuous tracking has no such window to bleed past a period boundary at all.
#   4. Rates are per 1,000 PERSON-YEARS (an incidence rate), and RR is more precisely an IRR
#      (incidence rate ratio) -- titles/axes say "IRR" for anything using this design.
#   5. Significance tests are the Poisson exact test (conditional-binomial form), not
#      Mann-Whitney/binomial -- those needed individual-level binary vectors that person-time
#      analysis doesn't produce the same way.
#   6. LAG_DAYS (default 0, off) is available to exclude a downstream diagnosis from counting
#      as a "hit" if it falls within LAG_DAYS of entry (protopathic-bias / same-episode
#      relabeling concern) -- not enabled by default; Statement 6's latency histograms are the
#      empirical basis for deciding whether a specific value is warranted.
#   7. Person-time starts at fu_start = later of 2017 / age 15, stops at fu_end = membership
#      end (left Clalit / death) or STUDY_END. Exposure before age 15 = excluded from both groups.
#      Exposed follow-up runs from the exposure date to the earliest of downstream diagnosis,
#      entry + FIXED_WINDOW_DAYS, fu_end. Prevalent cases (outcome already on/before the
#      at-risk start) are excluded. See FOLLOW-UP END below.
# This means Statement 1 page 1 and Statement 2 page 1 do NOT reproduce the original H2/B1
# images' raw numbers anymore (those used unrestricted, any-time-ever, patient-count-based
# follow-up) -- that divergence is intentional, per explicit instruction, not a bug.
#
# STATEMENT 1 (trauma -> psychosis/bipolar, largest excess of any exposure):
#   Page 1: H2 "exposed (PTSD/ASR) vs general population" bars, per outcome,
#           windowed + censored as above.
#           [generalized here to run on BOTH age bands -- the original only
#            ran age 15-40, and used unrestricted follow-up time]
#   Page 2: TEST-2c-style heatmap -- IRR within the window, split by whether
#           PTSD/ASR was diagnosed before or on/after Oct 7 2023.
#   Page 3: bar-chart view of page 2's exact same data -- one bar pair
#           (before/after) per downstream outcome, added on request as an
#           easier-to-read alternative to the heatmap.
#   Pages 4-9 (supporting): TEST 2b/5/6/9, reframed around PRE vs POST Oct7
#           PTSD/ASR entry specifically (the originals split by ALT_EXPOSURES,
#           not by time), all windowed + censored the same way -- each probes
#           a different candidate explanation for why the post-war RR is
#           lower than the pre-war RR:
#             2b: direct pre/post rate + RR, censored to full follow-up
#             5:  entry-diagnosis profession among converters (non-psychiatric
#                 diagnosers post-war would support "diagnostic dilution")
#             6:  downstream subtype at conversion (F23 vs spectrum; F30 vs
#                 F31) -- is post-war psychosis/bipolar diagnostically "weaker"
#             9:  inpatient vs outpatient trauma entry (exposure severity)
#             Zone: direct (Near Gaza/Northern Border) vs indirect (Rest of
#                 Israel) exposure -- not a ported Step 5 test, new here;
#                 reuses the same zone column as Statement 3.
#             Yearly trend: continuous (not bucketed) RR by single calendar
#                 year of entry -- not a ported Step 5 test either; TEST 8
#                 there uses 3 multi-year buckets, this uses one point per
#                 year (2023 itself split into pre/post-Oct7 sub-periods) so
#                 a gradual pre-existing decline looks different from a sharp
#                 break exactly at Oct 7.
#             By year of SECOND diagnosis: same idea, mirrored -- x-axis year
#                 is when the OUTCOME (psychosis/bipolar) happened, not when
#                 the trauma did, with 2 lines per panel (trauma pre-war /
#                 trauma post-war). Requires a proper Lexis split of each
#                 patient's restricted risk interval across PERIOD
#                 boundaries (split_person_time_by_period()) -- 2023 is
#                 split exactly at Oct7 (SECOND_DIAGNOSIS_PERIODS), an exact
#                 date-level split, not a whole-calendar-year approximation.
#                 Followed by 2 more pages showing only periods with
#                 n_events >= 10/30 (noisy low-n periods hidden as gaps),
#                 then 3 boxplot pages (no min-n, then 10/30) with 3 boxes
#                 each: [trauma+2nd-dx both pre-Oct7] vs [trauma+2nd-dx both
#                 on/after Oct7] (one-tailed Welch t-test between these two),
#                 plus [trauma pre-Oct7, 2nd-dx on/after Oct7] shown for
#                 reference only, not tested. The first two groups plus the
#                 reference group exactly partition the "pre" line's full
#                 dataset -- nothing dropped, nothing double-counted --
#                 see sanity check #18 for why this specific test is a
#                 small-sample, suggestive-only result.
#             Yearly trend forecast: quantitative follow-up to the above --
#                 fits an OLS line + 95% prediction interval on the PRE-WAR
#                 points only (2017.."2023 pre"), extrapolates forward, and
#                 flags (star marker) any post-war point that falls outside
#                 that interval. Answers "is post-war a real break, or just
#                 the continuation of whatever pre-existing trend/noise the
#                 series already had" quantitatively instead of by eye.
#             Quarterly trend: same design and same question as Yearly trend,
#                 at 3-month resolution instead of yearly -- a genuine gradual
#                 trend should still look smooth quarter to quarter; a small-n
#                 artifact will look jagged. Shares Statement 3's Q_INDEX
#                 range (2017Q1-2026Q2) so the two charts' time axes align.
#
# STATEMENT 2 ("B1" -- alternate entry-diagnoses can outrank trauma as a
#   predictor of later psychosis/bipolar):
#   Page 1: RR bar chart (windowed + censored, as above), one panel for
#           psychosis and one for bipolar, across ALL 7 possible entry
#           exposures (PTSD/ASR, Depression, Anxiety, Substance, Bipolar,
#           Psychosis, Personality -- excluding the outcome itself).
#           [the original B1 chart's source script was not available; this
#            reproduces its spirit as the multi-exposure counterpart of H2,
#            now on the same windowed footing as the rest of the script]
#   Page 2+3: pre/post-war heatmaps (whole-year window), generalized from
#           PTSD/ASR-only (TEST 2c) to all 7 alternate entry exposures, for
#           psychosis + bipolar as the two downstream columns.
#   Page 4: bar-chart view of pages 2+3's exact same data -- one panel per
#           downstream outcome, X-axis = the 7 exposures, 2 bars each
#           (before/after), added on request. NOTE: page 5 below (the %
#           change heatmap) is DERIVED from pages 2+3's same before/after
#           values, so it doesn't get its own separate bar-chart page --
#           that would just replot page 4's bars again.
#   Page 5: derived from pages 2+3 -- % change in IRR from pre- to post-Oct7
#           entry, per exposure/outcome cell (green = IRR went up, red = IRR
#           went down, diverging colormap centered at 0% change).
#   Run on both age bands.
#
# STATEMENT 3 (what rose vs. did not rise after Oct 7):
#   One page per item, quarterly new-dx-per-1,000-person-years by zone (person-years of
#   members under follow-up in that quarter, not a fixed zone headcount) -- same
#   metric as Direction 3's SUPPRESSION analysis, but with zones REDEFINED to
#   include a "Northern Border" category (per the docx's "ADD NORTH BORDER
#   HERE" note), using the exact 3-zone scheme from Alarms_Step5_1.py:
#     Near Gaza (distance_to_gaza_km <= 30) > Northern Border (residency
#     border == 'גבול הצפון') > Rest of Israel.
#   ROSE section:        total, ptsd, asr, anxiety, depression
#   DID-NOT-RISE section: psychosis, bipolar, substance, personality
#   Run on both age bands. NOT windowed the way Statements 1/2 are -- this is
#   a population-level incidence-rate analysis, not a "second illness after
#   the first" exposure→outcome design, so the fixed-window logic above
#   doesn't apply to it.
#
# STATEMENT 4 has no dedicated code section -- it's the claim that Statement
#   1 page 2 + TEST 2b already test directly (trauma diagnosed before Oct 7
#   carries a higher downstream RR than trauma diagnosed on/after Oct 7), so
#   there's a numbering gap here by design, not a missing page.
#
# STATEMENT 5 (who diagnoses what): TWO pages, 2 panels each (Before / On-after
#   Oct 7), % of each of the 8 outcomes' first-ever diagnoses given by a
#   psychiatrist -- one page excluding code 999 (Unknown/Admin) from both
#   numerator and denominator, one page counting it as non-psychiatrist (a
#   sensitivity comparison, since 999 is common in this field -- see
#   PROFESSION_CODE_LABELS). Not from the docx or Step 5 -- added on request,
#   to check (separately per side, not pooled) whether psychosis/bipolar are
#   diagnosed mainly by a psychiatrist while everything else is diagnosed at
#   higher rates by other professions, and whether that split is stable
#   across the war even though the RATES around it (Statement 3) are not.
#   NOT windowed either -- this is about who gave THIS diagnosis, not a
#   downstream-outcome design, so no follow-up time or censoring is involved.
#
# STATEMENT 6 (latency): SIX pages, one per non-trauma outcome (psychosis,
#   bipolar, anxiety, depression, substance, personality), each with 2 panels
#   (Pre-Oct7 / Post-Oct7 entry) -- a histogram of months from trauma entry to
#   that outcome's first diagnosis, among patients who got it. Not from the
#   docx -- added as the empirical basis for deciding whether LAG_DAYS is
#   worth setting to a nonzero value (a suspicious early spike would be
#   evidence for protopathic bias / same-episode relabeling). UNRESTRICTED by
#   window_days on purpose -- capping it would hide the very information
#   (long delays, mass outside any window) this exists to show.
#
# NOTE ON DATA ACCESS: this script was written and syntax/logic-tested
# against a SYNTHETIC file with the same columns (see
# make_synthetic_data.py) -- it has not been run against the real network
# extract. Before trusting the output, re-run the sanity checks at the
# bottom of this file against the real data.
# =============================================================================

OCT7 = pd.Timestamp("2023-10-07")

ALL_OUTCOMES = ["psychosis", "bipolar", "ptsd", "asr",
                "anxiety", "depression", "substance", "personality"]

OUTCOME_COLORS = {
    "psychosis":   "#e63946", "bipolar":     "#457b9d",
    "ptsd":        "#2a9d8f", "asr":         "#e9c46a",
    "anxiety":     "#f4a261", "depression":  "#264653",
    "substance":   "#6d6875", "personality": "#b5838d",
}

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
# Applies to EVERY exposure definition in the script (trauma in Statement 1, and all 7
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

# --- CHOOSE WHICH TRAUMA MODE TO RUN (edit this, then re-run) --------------------------------
# Unlike trauma mode (which auto-loops all 3 in export_all_trauma_modes()), diagnoser mode
# ALWAYS loops all 3 automatically for whichever single trauma mode is selected here -- so one
# run produces 2 age bands x 3 diagnoser modes = 6 files for this trauma mode. Change this and
# re-run to get the other trauma mode's 6 files instead of generating all 18 at once.
SELECTED_TRAUMA_MODE = "ptsd_asr"   # one of: "ptsd_asr", "ptsd_only", "asr_only"

# --- SLIM_RUN: True -> write ONLY presentation_summary_age15_40_<trauma>_24mo_any_diagnoser.pdf
# (one age band, one window, one diagnoser mode; no demographic / relative-year PDFs; sanity
# checks still print to the console but no report file). False -> the full 12-file run.
SLIM_RUN = True
SLIM_AGE_BAND, SLIM_WINDOW, SLIM_DIAGNOSER = "age15_40", "24mo", "any_diagnoser"

# --- Statement 1 & 2: exposure definitions -----------------------------------
# 7 possible entry exposures (every outcome group can be an "entry" diagnosis). The trauma
# entry is built dynamically from TRAUMA_EXPOSURE_OUTCOMES/LABEL so it follows the active
# trauma mode -- call current_alt_exposures_7() rather than a static module dict.
def current_alt_exposures_7():
    return {
        TRAUMA_EXPOSURE_LABEL: TRAUMA_EXPOSURE_OUTCOMES,
        "Depression":  ["depression"],
        "Anxiety":     ["anxiety"],
        "Substance":   ["substance"],
        "Bipolar":     ["bipolar"],
        "Psychosis":   ["psychosis"],
        "Personality": ["personality"],
    }
EXPOSURE_COLORS = {
    "PTSD/ASR": "#e63946", "PTSD": "#e63946", "ASR": "#e63946",   # same color, any trauma mode
    "Depression": "#264653", "Anxiety": "#f4a261",
    "Substance": "#6d6875", "Bipolar": "#457b9d", "Psychosis": "#e63946",
    "Personality": "#b5838d",
}
FOCUS_OUTCOMES = ["psychosis", "bipolar"]     # statement 1 & 2 downstream targets
S1_DOWNSTREAM  = ["psychosis", "bipolar", "anxiety",
                  "depression", "substance", "personality"]  # H2's original 6

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
FOLLOWUP_MIN_AGE     = 15
_followup_end = None   # per-patient fu_end Series (full-population index), set by set_followup()


def set_followup(frame):
    global _followup_end
    _followup_end = frame["fu_end"]


def followup_end_for(index):
    """fu_end for the given patients; STUDY_END for everyone if set_followup() wasn't called."""
    if _followup_end is None:
        return pd.Series(STUDY_END, index=index)
    return _followup_end.reindex(index)

# --- LAG_DAYS: protopathic-bias exclusion window, OFF by default (0) -----------------------
# If > 0, a downstream diagnosis occurring within LAG_DAYS of the entry diagnosis is NOT
# counted as a "hit" -- the concern is that a diagnosis given days-to-weeks after entry may
# reflect the SAME clinical episode being relabeled (e.g. an ASR diagnosis given in the chaos
# right after Oct 7 that gets corrected to a psychosis diagnosis three weeks later isn't really
# "trauma causing psychosis in 3 weeks" -- it's one episode, two labels), not a genuine delayed
# effect. Left at 0 (no exclusion) until the latency histograms (Statement 6) show whether
# there's actually a suspicious early spike that would justify a specific value -- kept as a
# named, easy-to-set constant for exactly that follow-up, not implemented blindly now.
LAG_DAYS = 0

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

def person_time_events_and_years(entry_dates, event_dates, window_days=None, lag_days=LAG_DAYS):
    """For a cohort with `entry_dates` (index-aligned Series, NaT = not in this cohort) and
    `event_dates` (first occurrence of the outcome, NaT = never), returns (n_events,
    total_person_years) using a window capped at `window_days` (default: current
    FIXED_WINDOW_DAYS) and STUDY_END, whichever is reached first. An event within `lag_days` of
    entry is NOT counted as a hit (protopathic-bias exclusion, off by default) -- that person's
    person-time then simply runs to the window/STUDY_END boundary as if no event had occurred
    (this script only has first-ever dates, not recurrences, so there is no later event to
    still count for them)."""
    entry, end_time, is_event, _ = exposed_intervals(entry_dates, event_dates, window_days, lag_days)
    days_at_risk = (end_time - entry).dt.days.clip(lower=0)
    person_years = days_at_risk.sum() / 365.25
    n_events = int(is_event.sum())
    return n_events, person_years


def exposed_intervals(entry_dates, event_dates, window_days=None, lag_days=LAG_DAYS):
    """Per at-risk exposed patient: (entry, end_time, is_event, event) -- the at-risk interval
    behind person_time_events_and_years(); also used by the adjusted Poisson model so both
    count exactly the same person-time."""
    window_days = FIXED_WINDOW_DAYS if window_days is None else window_days
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


def person_time_rate(entry_dates, event_dates, window_days=None, lag_days=LAG_DAYS):
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
        days = (overlap_end - seg_start).dt.days.clip(lower=0)
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
# "profession" metadata for all 8 outcomes, not just the 2 entry ones above -- and period DATES
# for the 4 outcomes (anxiety/depression/substance/personality) not already covered by
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
ZONE_COLORS = {"Near Gaza": "#e63946", "Northern Border": "#457b9d",
               "Rest of Israel": "#6c757d"}

STATEMENT3_ROSE         = ["total", "ptsd", "asr", "anxiety", "depression"]
STATEMENT3_DID_NOT_RISE = ["psychosis", "bipolar", "substance", "personality"]
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
             MEMBERSHIP_END_COL]
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
                    for o in ALL_OUTCOMES for p in PERIODS_RANGE]   # now covers all 8

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
    usecols = [c for c in NEEDED_COLS if c in available]
    missing = [c for c in NEEDED_COLS if c not in available]
    if missing:
        print(f"WARNING: {len(missing)} expected columns not found (will be skipped): {missing}")
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
        membership_end = pd.to_datetime(df[MEMBERSHIP_END_COL], errors="coerce")
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
    # Statement 5: profession at first-ever diagnosis, for ALL 8 outcomes (not just the trauma
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
# outcome_profession_meta (built for all 8 outcomes, not just the 2 trauma ones), so it applies
# to any of Statement 2's 7 alternate exposures too, not just trauma.
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
    window_days = FIXED_WINDOW_DAYS if window_days is None else window_days
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


# --- S1, STEP 2: Draw the exposed-vs-general grouped bar chart (H2 page 1 layout) ---
def draw_exposed_vs_general(ax, comp, n_exp, n_gen, exposure_label, title):
    x = np.arange(len(comp)); w = 0.38
    ax.bar(x - w/2, comp["exp_rate"], width=w,
          color=[OUTCOME_COLORS.get(o, "#888") for o in comp["outcome"]],
          alpha=0.9, label=f"After {exposure_label} (exposed)")
    ax.bar(x + w/2, comp["gen_rate"], width=w,
          color="#b0b0b0", alpha=0.9, label=f"General pop. (never {exposure_label})")
    for xi, row in zip(x, comp.itertuples()):
        top = max(row.exp_rate if pd.notna(row.exp_rate) else 0,
                 row.gen_rate if pd.notna(row.gen_rate) else 0)
        rr_txt = f"IRR {row.rr:.1f}" if pd.notna(row.rr) else "IRR n/a"
        exp_txt = f"{row.exp_rate:.1f}" if pd.notna(row.exp_rate) else "n/a"
        gen_txt = f"{row.gen_rate:.1f}" if pd.notna(row.gen_rate) else "n/a"
        ax.text(xi, top, f"{rr_txt}\nexp={exp_txt}, gen={gen_txt}\nn at risk={row.n_at_risk:,}",
               ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([o.upper() for o in comp["outcome"]], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Incidence per 1,000 person-years", fontsize=9)
    ax.set_xlabel(f"n exposed={n_exp:,} (before excluding prior cases per outcome) | n general={n_gen:,} | "
                 f"IRR = exposed rate / general rate (both per 1,000 person-years) | "
                 f"{FIXED_WINDOW_DAYS//30}mo window, restricted person-time", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")


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


# --- S1, STEP 4: Draw the pre/post-war RR heatmap pair (one exposure, N downstream cols) ---
def draw_prepost_heatmap(ax, rr_dict, n_exp, n_gen, downstream_outcomes, row_label):
    vals = np.array([[rr_dict.get(o, np.nan) for o in downstream_outcomes]])
    sns.heatmap(vals, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True,
               xticklabels=[o.upper() for o in downstream_outcomes],
               yticklabels=[f"{row_label}\n(n_exp={n_exp:,}, n_gen={n_gen:,})"], ax=ax,
               annot_kws={"fontsize": 9})
    ax.set_xlabel("Downstream outcome", fontsize=8)


# --- S1, STEP 4b: Bar-chart version of the same pre/post heatmap data -- one bar pair per
# downstream outcome (pre-war IRR, post-war IRR), same numbers as draw_prepost_heatmap() above,
# just as grouped bars instead of a color grid. ---
def draw_prepost_bar_chart(ax, rr_pre, rr_post, n_pre, n_post, downstream_outcomes, title):
    x = np.arange(len(downstream_outcomes)); w = 0.35
    pre_vals  = [rr_pre.get(o, np.nan) for o in downstream_outcomes]
    post_vals = [rr_post.get(o, np.nan) for o in downstream_outcomes]
    bars_pre  = ax.bar(x - w/2, [v if pd.notna(v) else 0 for v in pre_vals], width=w,
                       color="#95a5a6", alpha=0.85, label=f"Before Oct 7 (n_exp={n_pre:,})")
    bars_post = ax.bar(x + w/2, [v if pd.notna(v) else 0 for v in post_vals], width=w,
                       color="#e63946", alpha=0.85, label=f"On/after Oct 7 (n_exp={n_post:,})")
    for bar, v in zip(bars_pre, pre_vals):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars_post, post_vals):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([o.upper() for o in downstream_outcomes], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"IRR (exposed / general), within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


# --- S1, STEP 5: Write Statement 1 pages (3 pages) for one age band ---
def write_statement1(pdf, frame, age_label):
    n_exp, n_gen, comp = compute_exposed_vs_general(frame, TRAUMA_EXPOSURE_OUTCOMES, S1_DOWNSTREAM)
    fig, ax = plt.subplots(figsize=(13, 6))
    draw_exposed_vs_general(ax, comp, n_exp, n_gen, TRAUMA_EXPOSURE_LABEL,
                            f"STATEMENT 1 | H2 | After {TRAUMA_EXPOSURE_LABEL} vs general population | "
                            f"within {FIXED_WINDOW_DAYS//30}mo | {age_label}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    (n_pre, rr_pre), (n_post, rr_post), n_gen_pp = compute_prepost_window_rr(
        frame, TRAUMA_EXPOSURE_OUTCOMES, S1_DOWNSTREAM, FIXED_WINDOW_DAYS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle(f"STATEMENT 1 | {TRAUMA_EXPOSURE_LABEL} before vs on/after Oct 7 2023 | "
                f"IRR within {FIXED_WINDOW_DAYS//30}mo | {age_label}",
                fontsize=11, fontweight="bold")
    draw_prepost_heatmap(axes[0], rr_pre, n_pre, n_gen_pp, S1_DOWNSTREAM, "Before Oct 7")
    draw_prepost_heatmap(axes[1], rr_post, n_post, n_gen_pp, S1_DOWNSTREAM, "On/after Oct 7")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    draw_prepost_bar_chart(ax, rr_pre, rr_post, n_pre, n_post, S1_DOWNSTREAM,
                           f"STATEMENT 1 | {TRAUMA_EXPOSURE_LABEL} before vs on/after Oct 7 2023 | "
                           f"bar-chart view of the heatmap above | {age_label}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    pre_entry  = first_trauma.where(first_trauma < OCT7)
    post_entry = first_trauma.where(first_trauma >= OCT7)

    rate_pre,  a_pre,  py_pre  = person_time_rate(pre_entry,  down, FIXED_WINDOW_DAYS)
    rate_post, a_post, py_post = person_time_rate(post_entry, down, FIXED_WINDOW_DAYS)
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


def draw_t2b_panel(ax, frame, downstream_outcome, title):
    res = compute_t2b_rates(frame, downstream_outcome)
    rate_vals = [res["rate_pre"] if pd.notna(res["rate_pre"]) else 0,
                res["rate_post"] if pd.notna(res["rate_post"]) else 0]
    bars = ax.bar([0, 1], rate_vals, color=["#95a5a6", "#e63946"], alpha=0.85, width=0.5)
    for bar, n, rate in zip(bars, [res["n_pre"], res["n_post"]], rate_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f"{rate:.1f} / 1,000 PY\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Pre-Oct7 {TRAUMA_EXPOSURE_LABEL}\n(2017-2023)",
                       f"Post-Oct7 {TRAUMA_EXPOSURE_LABEL}\n(war period)"], fontsize=9)
    ax.set_ylabel(f"Rate within {FIXED_WINDOW_DAYS//30}mo of entry\nper 1,000 person-years", fontsize=8)
    rr_txt = f"{res['rr']:.2f}" if pd.notna(res["rr"]) else "n/a"
    ax.set_title(f"{title}\nIRR(post/pre)={rr_txt}  Poisson exact p={res['pval']}",
               fontsize=10, fontweight="bold")


# --- Shared helper: reachability (full follow-up) + within-window conversion, used by
# TEST 5/6/9 below so all three apply the exact same window+censoring rule as TEST 2b
# and compute_exposed_vs_general() above. ---
def eligible_reachable_mask(exposure_date, window_days=None):
    window_days = FIXED_WINDOW_DAYS if window_days is None else window_days
    return (exposure_date + pd.to_timedelta(window_days, unit="D")) <= STUDY_END


def converted_within_window(exposure_date, downstream_date, window_days=None):
    window_days = FIXED_WINDOW_DAYS if window_days is None else window_days
    return (downstream_date.notna() & (downstream_date > exposure_date) &
           (downstream_date <= exposure_date + pd.to_timedelta(window_days, unit="D")))


# --- T5, STEP 1: Entry-diagnosis profession breakdown among converters, pre vs post Oct7 ---
# Windowed + censored: only trauma-exposed patients with a full 18mo follow-up window count,
# and "converted" means the downstream dx fell within that same 18mo window.
def compute_t5_breakdown(frame, entry_meta, downstream_outcome, side):
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    eligible = first_trauma.notna() & eligible_reachable_mask(first_trauma)
    side_mask = eligible & ((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    converted_mask = side_mask & converted_within_window(first_trauma, down)

    profession = get_exposure_meta_field(entry_meta, TRAUMA_EXPOSURE_OUTCOMES, "profession")
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


def draw_t5_panel(ax, frame, entry_meta, downstream_outcome, title):
    labels, sides = [f"Pre-Oct7 {TRAUMA_EXPOSURE_LABEL}", f"Post-Oct7 {TRAUMA_EXPOSURE_LABEL}"], ["pre", "post"]
    breakdowns, all_categories = {}, set()
    for label, side in zip(labels, sides):
        pct, n = compute_t5_breakdown(frame, entry_meta, downstream_outcome, side)
        breakdowns[label] = (pct, n)
        all_categories |= set(pct.index)
    all_categories = sorted(all_categories, key=str)
    cmap = plt.cm.tab10
    bottom, x_pos = np.zeros(len(labels)), np.arange(len(labels))
    for i, cat in enumerate(all_categories):
        vals = [breakdowns[l][0].get(cat, 0) for l in labels]
        ax.bar(x_pos, vals, bottom=bottom, label=str(cat)[:20], color=cmap(i % 10), alpha=0.85)
        bottom += np.array(vals)
    for xi, l in zip(x_pos, labels):
        ax.text(xi, 102, f"n={breakdowns[l][1]:,}", ha="center", fontsize=7)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of converters (by entry-diagnosis profession)", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", ncol=2)


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
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
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


def draw_t6_panel(ax, frame, downstream_meta, downstream_outcome, classifier, title):
    labels, sides = [f"Pre-Oct7 {TRAUMA_EXPOSURE_LABEL}", f"Post-Oct7 {TRAUMA_EXPOSURE_LABEL}"], ["pre", "post"]
    breakdowns, all_categories = {}, set()
    for label, side in zip(labels, sides):
        pct, n = compute_t6_breakdown(frame, downstream_meta, downstream_outcome, classifier, side)
        breakdowns[label] = (pct, n)
        all_categories |= set(pct.index)
    all_categories = sorted(all_categories, key=str)
    cmap = plt.cm.Set1
    bottom, x_pos = np.zeros(len(labels)), np.arange(len(labels))
    for i, cat in enumerate(all_categories):
        vals = [breakdowns[l][0].get(cat, 0) for l in labels]
        ax.bar(x_pos, vals, bottom=bottom, label=cat, color=cmap(i % 9), alpha=0.85)
        bottom += np.array(vals)
    for xi, l in zip(x_pos, labels):
        ax.text(xi, 102, f"n={breakdowns[l][1]:,}", ha="center", fontsize=7)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of converters (by subtype)", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")


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
    facility_type = get_exposure_meta_field(entry_meta, TRAUMA_EXPOSURE_OUTCOMES, "facility type").astype(str)
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
    facility_type = get_exposure_meta_field(entry_meta, TRAUMA_EXPOSURE_OUTCOMES, "facility type")
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
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
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


def draw_t9_panel(ax, frame, entry_meta, downstream_outcome, title):
    df9 = compute_t9_rates(frame, entry_meta, downstream_outcome)
    sides = ["Pre", "Post"]
    x = np.arange(len(sides)); w = 0.35
    for i, (engagement, color, eng_label) in enumerate([
            ("Heavy", "#e63946", "Heavy (inpatient)"), ("Light", "#95a5a6", "Light (outpatient)")]):
        sub = df9[df9.engagement == engagement].set_index("side").reindex(sides)
        vals = [v if pd.notna(v) else 0 for v in sub["rate"].values]
        ns = [int(n) for n in sub["n"].values]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, width=w, alpha=0.85, label=eng_label, color=color)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"n={n:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}-Oct7 {TRAUMA_EXPOSURE_LABEL}" for s in sides], fontsize=9)
    ax.set_ylabel(f"{downstream_outcome.capitalize()} rate per 1,000", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)


# --- T-Zone, STEP 1: Trauma -> downstream rate, split by zone (direct vs indirect exposure)
# WITHIN each of pre/post-Oct7 -- not a ported Trauma_specificity_probe_Step_5.py test, new
# here. Uses the same "zone" column (Near Gaza / Northern Border / Rest of Israel) already
# built in build_frame()/assign_zone(), windowed + censored the same way as everything else
# in this section. The question: does trauma diagnosed by someone with direct exposure (Near
# Gaza / Northern Border) carry a different downstream risk than trauma diagnosed by someone
# in the rest of the country (indirect/media exposure, "worried well")?
def compute_zone_rates(frame, downstream_outcome):
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
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


# --- T-Zone, STEP 2: Draw panel -- grouped bars, 3 zones x (Pre, Post), one downstream outcome ---
def draw_zone_panel(ax, frame, downstream_outcome, title):
    df_zone = compute_zone_rates(frame, downstream_outcome)
    sides = ["Pre", "Post"]
    x = np.arange(len(sides)); w = 0.25
    for i, zone in enumerate(ZONE_ORDER):
        sub = df_zone[df_zone.zone == zone].set_index("side").reindex(sides)
        vals = [v if pd.notna(v) else 0 for v in sub["rate"].values]
        ns = [int(n) for n in sub["n"].values]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, width=w, alpha=0.85, label=ZONE_LABELS[zone], color=ZONE_COLORS[zone])
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"n={n:,}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}-Oct7 {TRAUMA_EXPOSURE_LABEL}" for s in sides], fontsize=9)
    ax.set_ylabel(f"{downstream_outcome.capitalize()} rate per 1,000", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)


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
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]

    # PER-PERIOD general-population reference, using the SAME periods as the exposed side
    # (including the 2023 pre/post split) -- via continuous Lexis tracking, not an anchor.
    n_gen = int(first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES).isna().sum())
    gen_split = compute_general_lexis_split(frame, TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome,
                                            YEARLY_TREND_PERIODS)

    rows = []
    for label, start, end in YEARLY_TREND_PERIODS:
        period_entry = first_trauma.where((first_trauma >= start) & (first_trauma < end))
        n = int(period_entry.notna().sum())
        rate, _, _ = person_time_rate(period_entry, down, FIXED_WINDOW_DAYS)
        gen_rate = gen_split.loc[label, "rate"]
        rr = rate / gen_rate if (pd.notna(gen_rate) and gen_rate > 0 and pd.notna(rate)) else np.nan
        rows.append({"label": label, "n": n, "rr": rr})
    return pd.DataFrame(rows), n_gen


# --- T-Yearly, STEP 2: Draw the continuous year-line panel, Oct 7 marked ---
def draw_yearly_trend_panel(ax, frame, downstream_outcome, title):
    trend, n_gen = compute_yearly_rr_trend(frame, downstream_outcome)
    x = np.arange(len(trend))
    ax.plot(x, trend["rr"], marker="o", markersize=5, linewidth=1.8,
           color=OUTCOME_COLORS.get(downstream_outcome, "#333333"))
    for xi, row in zip(x, trend.itertuples()):
        if pd.notna(row.rr):
            ax.text(xi, row.rr, f"n={int(row.n):,}", fontsize=6, ha="center", va="bottom")
    oct7_pos = next(i for i, l in enumerate(trend["label"]) if l == "2023\n(post)")
    ax.axvline(oct7_pos - 0.5, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(trend["label"], fontsize=8)
    ax.set_xlabel(f"Year of {TRAUMA_EXPOSURE_LABEL} entry (Oct 7 2023 marked, "
                 f"2023 split pre/post)", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(f"{title}  (n_gen={n_gen:,}, constant across periods)", fontsize=10, fontweight="bold")


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

    for pid in entry.index:
        e_start, e_end = entry.loc[pid], end_time.loc[pid]
        if pd.isna(e_start) or pd.isna(e_end) or e_end <= e_start:
            continue
        for label, p_start, p_end in periods:
            seg_start = max(e_start, p_start)
            seg_end = min(e_end, p_end)
            if seg_end > seg_start:
                py_by_label[label] += (seg_end - seg_start).days / 365.25
        if is_event.loc[pid]:
            ev_date = event.loc[pid]
            for label, p_start, p_end in periods:
                if p_start <= ev_date < p_end:
                    events_by_label[label] += 1
                    break

    return events_by_label, py_by_label


def compute_by_second_diagnosis_year(frame, downstream_outcome, side):
    """side: 'pre' or 'post' -- trauma diagnosed before/on-after Oct 7. Returns a DataFrame
    (label, n_events, person_years, irr) using the currently active window. 2023 is split
    exactly at Oct7 (SECOND_DIAGNOSIS_PERIODS), same as the entry-side Yearly trend chart --
    not approximated by whole calendar year."""
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    entry = first_trauma.where((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    down = frame[f"first_{downstream_outcome}"]

    events_by_label, py_by_label = split_person_time_by_period(
        entry, down, FIXED_WINDOW_DAYS, SECOND_DIAGNOSIS_PERIODS)

    # PER-PERIOD general reference via continuous Lexis tracking -- replaces the old single
    # anchor-averaged constant, so each period's IRR is measured against that SAME period's
    # general-population rate, not one rate repeated across every point on the chart.
    gen_split = compute_general_lexis_split(frame, TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome,
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


def draw_by_second_diagnosis_year_panel(ax, frame, downstream_outcome, title, min_n=0):
    oct7_frac = 2023 + (OCT7 - pd.Timestamp("2023-01-01")).days / 365

    def period_x(start, end):
        # continuous x-position = the period's own midpoint, in fractional-year units --
        # derived directly from its actual date range, not a hard-coded shift for one special
        # case. A whole-year period sits at ~year+0.5; the split 2023 sub-periods each sit at
        # their own true midpoint (~Q2 for pre, ~Nov 20 for post).
        mid = start + (end - start) / 2
        return mid.year + (mid - pd.Timestamp(f"{mid.year}-01-01")).days / 365

    for side, label_txt, color in [("pre", f"{TRAUMA_EXPOSURE_LABEL} before Oct 7", "#95a5a6"),
                                   ("post", f"{TRAUMA_EXPOSURE_LABEL} on/after Oct 7", "#e63946")]:
        trend = compute_by_second_diagnosis_year(frame, downstream_outcome, side)
        if min_n > 0:
            # Points below the threshold are masked to NaN (not dropped from the DataFrame) so
            # matplotlib draws a genuine GAP there instead of connecting neighboring reliable
            # points across the missing one, which would misleadingly imply a smooth trend.
            trend = trend.copy()
            trend.loc[trend["n_events"] < min_n, "irr"] = np.nan
        x_vals = [period_x(s, e) for s, e in zip(trend["start"], trend["end"])]
        ax.plot(x_vals, trend["irr"], marker="o", markersize=5, linewidth=1.8,
               color=color, label=label_txt)
        for xi, row in zip(x_vals, trend.itertuples()):
            if pd.notna(row.irr):
                ax.text(xi, row.irr, f"n={int(row.n_events)}", fontsize=6,
                       ha="center", va="bottom", color=color)
    ax.axvline(oct7_frac, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(SECOND_DIAGNOSIS_YEAR_RANGE)
    n_note = f" | points with n<{min_n} hidden" if min_n > 0 else ""
    ax.set_xlabel(f"Year of the SECOND diagnosis (Oct 7 2023 marked; 2023 itself is split "
                 f"exactly at Oct7, not approximated){n_note}", fontsize=7)
    ax.set_ylabel(f"IRR (exposed / general), within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")


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
                                            event_side="all", min_n=0):
    """trauma_side: 'pre' or 'post' (which line to pull from). event_side: 'pre' (period ends
    by Oct7), 'post' (period starts on/after Oct7), or 'all' (no restriction)."""
    trend = compute_by_second_diagnosis_year(frame, downstream_outcome, trauma_side)
    if event_side == "pre":
        trend = trend[trend["end"] <= OCT7]
    elif event_side == "post":
        trend = trend[trend["start"] >= OCT7]
    sub = trend[trend["n_events"] >= min_n]
    return sub["irr"].dropna().values


def draw_first_second_boxplot_combined(ax, frame, downstream_outcome, min_n=0):
    gray_pre  = compute_second_diagnosis_boxplot_values(frame, downstream_outcome, "pre",  "pre",  min_n)
    red_post  = compute_second_diagnosis_boxplot_values(frame, downstream_outcome, "post", "post", min_n)
    gray3     = compute_second_diagnosis_boxplot_values(frame, downstream_outcome, "pre",  "post", min_n)

    bp = ax.boxplot([gray_pre, red_post, gray3], positions=[1, 2, 3], widths=0.5,
                    labels=["trauma before Oct7\n(2nd dx: also before Oct7)",
                          "trauma on/after Oct7\n(2nd dx: also on/after Oct7)",
                          "trauma before Oct7\n(2nd dx: on/after Oct7)"],
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#95a5a6", "#e63946", "#95a5a6"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)

    rng = np.random.default_rng(42)   # fixed seed -- reproducible jitter, doesn't affect the test
    for pos, vals in zip([1, 2, 3], [gray_pre, red_post, gray3]):
        if len(vals) > 0:
            jitter = rng.normal(0, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, color="black", alpha=0.6, s=16, zorder=3)

    pval_str = "n/a (need >=2 each)"
    if len(gray_pre) >= 2 and len(red_post) >= 2:
        _, p = stats.ttest_ind(gray_pre, red_post, alternative="greater", equal_var=False)
        pval_str = f"{p:.3g}"

    ax.set_ylabel(f"IRR, within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(f"{downstream_outcome.upper()}\n"
               f"one-tail t-test, box 1 (gray) > box 2 (red): p={pval_str}  "
               f"(n: {len(gray_pre)}, {len(red_post)}, {len(gray3)})\n"
               f"box 3 (gray, right) is NOT part of the test",
               fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)


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
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    entry = first_trauma.where((first_trauma < OCT7) if trauma_side == "pre" else (first_trauma >= OCT7))
    entry = entry.dropna()
    event = frame[f"first_{downstream_outcome}"].reindex(entry.index)

    window_end = entry + pd.to_timedelta(FIXED_WINDOW_DAYS, unit="D")
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


def draw_time_to_second_dx_boxplot(ax, frame, downstream_outcome):
    box1 = compute_time_to_second_dx(frame, downstream_outcome, "pre",  "pre")
    box2 = compute_time_to_second_dx(frame, downstream_outcome, "post", "post")
    box3 = compute_time_to_second_dx(frame, downstream_outcome, "pre",  "post")
    groups = [box1, box2, box3]

    names = ["trauma before Oct7\n(2nd dx: also before Oct7)",
             "trauma on/after Oct7\n(2nd dx: also on/after Oct7)",
             "trauma before Oct7\n(2nd dx: on/after Oct7)"]
    labels = [f"{nm}\nn={len(v):,}" + (f", median={np.median(v):.0f}d" if len(v) else "")
              for nm, v in zip(names, groups)]
    bp = ax.boxplot(groups, positions=[1, 2, 3], widths=0.5, labels=labels,
                    patch_artist=True, showmeans=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#95a5a6", "#e63946", "#95a5a6"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)

    rng = np.random.default_rng(42)   # fixed seed -- reproducible jitter, doesn't affect the test
    for pos, vals in zip([1, 2, 3], groups):
        if len(vals) > 0:
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, color="black",
                      alpha=0.25 if len(vals) > 200 else 0.5, s=6, zorder=3, linewidths=0)

    pval_str = "n/a (need >=2 each)"
    if len(box1) >= 2 and len(box2) >= 2:
        _, p = stats.mannwhitneyu(box1, box2, alternative="greater")
        pval_str = f"{p:.3g}"

    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"Days from {TRAUMA_EXPOSURE_LABEL} to 2nd diagnosis\n"
                 f"(converters within {FIXED_WINDOW_DAYS//30}mo)", fontsize=8)
    ax.set_title(f"{downstream_outcome.upper()}\n"
               f"one-tail Mann-Whitney, box 1 (gray) > box 2 (red): p={pval_str}\n"
               f"box 3 (gray, right) is NOT part of the test",
               fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)


# --- PATIENT-LEVEL cumulative incidence (1 - Kaplan-Meier), ALL exposed patients ---------------
# Unlike the time-to-2nd-dx boxplot (converters only), every exposed patient is in the curve:
# time 0 = trauma entry, an event = the 2nd diagnosis, and a patient whose follow-up ends
# without one (window end, fu_end = left Clalit, STUDY_END) is CENSORED -- they leave the risk
# set instead of being counted as "never converted". At-risk rules are the same as
# person_time_events_and_years() (prevalent cases excluded, LAG_DAYS honored).
# Three curves, mirroring the boxplots:
#   1 (gray)        = trauma before Oct7, followed only UNTIL Oct7 (censored at Oct7)
#   2 (red)         = trauma on/after Oct7
#   3 (blue dashed) = trauma before Oct7, followed only FROM Oct7 -- delayed entry: each patient
#                     joins the risk set at their own time-since-trauma on Oct7 (left truncation)
# One page per pair in KM_PAGES (1 vs 2, 2 vs 3, 1 vs 3), each with a two-sided log-rank test.
KM_GROUP_STYLES = {1: ("#7f8c8d", "-",  "trauma before Oct7 (followed until Oct7)"),
                   2: ("#e63946", "-",  "trauma on/after Oct7"),
                   3: ("#457b9d", "--", "trauma before Oct7 (followed from Oct7)")}


def km_intervals(frame, downstream_outcome, group):
    """Per-patient (t_enter, t_exit, event) in days since trauma entry, for KM group 1/2/3."""
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    entry = first_trauma.where((first_trauma >= OCT7) if group == 2 else (first_trauma < OCT7)).dropna()
    event = frame[f"first_{downstream_outcome}"].reindex(entry.index)
    prevalent = event.notna() & (event <= entry)
    entry, event = entry[~prevalent], event[~prevalent]

    window_end = entry + pd.to_timedelta(FIXED_WINDOW_DAYS, unit="D")
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

    t_enter = (enter_date - entry).dt.days.values.astype(float)
    t_exit = (exit_date - entry).dt.days.values.astype(float)
    keep = t_exit > t_enter
    return t_enter[keep], t_exit[keep], is_event.values[keep]


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


def draw_km_cumulative_incidence(ax, frame, downstream_outcome, curves=(1, 2)):
    """Draws the two KM groups in `curves` (keys of KM_GROUP_STYLES) and log-rank tests them."""
    groups = {g: km_intervals(frame, downstream_outcome, g) for g in curves}
    horizon = FIXED_WINDOW_DAYS
    for g, (t_enter, t_exit, is_event) in groups.items():
        color, ls, label = KM_GROUP_STYLES[g]
        if len(t_exit) == 0:
            continue
        times, cum, lo, hi = km_cumulative_incidence(t_enter, t_exit, is_event)
        months = times / 30.44
        ax.step(months, cum * 100, where="post", color=color, linestyle=ls, linewidth=1.8,
               label=f"({g}) {label}: {cum[-1]*100:.2f}% [{lo[-1]*100:.2f}-{hi[-1]*100:.2f}] "
                     f"(n={len(t_exit):,}, events={int(is_event.sum()):,})")
        ax.fill_between(months, lo * 100, hi * 100, step="post", color=color, alpha=0.12)

    a, b = curves
    p = logrank_test(groups[a], groups[b])
    p_str = f"{p:.3g}" if pd.notna(p) else "n/a"

    # number-at-risk row under the x-axis, every 6 months
    horizon_mo = round(horizon / 30.44)
    ticks = np.arange(0, horizon_mo + 1, 6)
    ax.set_xticks(ticks)
    ax.set_xlim(0, horizon_mo)
    tick_days = np.clip(ticks * 30.44, 0.5, horizon)   # 0.5: just after entry; cap at window end
    for row, g in enumerate(curves):
        t_enter, t_exit, _ = groups[g]
        at_risk = risk_set_sizes(t_enter, t_exit, tick_days) if len(t_exit) else np.zeros(len(ticks))
        color, ls, _ = KM_GROUP_STYLES[g]
        y = -0.10 - row * 0.035
        for x, nr in zip(ticks, at_risk):
            ax.text(x, y, f"{int(nr):,}", transform=ax.get_xaxis_transform(),
                   ha="center", va="top", fontsize=6.5, color=color,
                   fontstyle="italic" if ls == "--" else "normal")
        ax.text(-0.04, y, f"at risk ({g}):", transform=ax.transAxes, ha="right", va="top",
               fontsize=6.5, color=color)

    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Months since {TRAUMA_EXPOSURE_LABEL} diagnosis", fontsize=8)
    ax.set_ylabel(f"Cumulative incidence of 2nd diagnosis (%)", fontsize=8)
    ax.set_title(f"{downstream_outcome.upper()}\n"
               f"log-rank (two-sided), curve {a} vs curve {b}: p={p_str}",
               fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left", title=f"% at {horizon//30}mo [95% CI]",
             title_fontsize=6.5)
    ax.grid(axis="y", alpha=0.3)


# One KM page per pair of curves (see KM_GROUP_STYLES)
KM_PAGES = [((1, 2), "PRE-war (until Oct7) vs POST-war trauma"),
            ((2, 3), "POST-war trauma vs PRE-war trauma followed from Oct7"),
            ((1, 3), "PRE-war trauma: follow-up before vs from Oct7 -- CAUTION: overlapping "
                     "patients, log-rank assumes independent groups, p is approximate")]


# --- Write the 2 min-n-filtered by-second-diagnosis-year pages, then the 3 boxplot pages
# (no minimum, n>=10, n>=30) -- each boxplot page has 2 panels (Psychosis | Bipolar), and EACH
# panel is ONE combined graph holding all 3 comparison groups (6 boxes), not 3 separate
# subplots. ---
def write_statement1_min_n_and_boxplot_pages(pdf, frame, age_label):
    for min_n in MIN_N_THRESHOLDS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"STATEMENT 1 (supporting) | By year of SECOND diagnosis, only n>={min_n} | "
                    f"within {FIXED_WINDOW_DAYS//30}mo of {TRAUMA_EXPOSURE_LABEL} entry | {age_label}\n"
                    f"same chart as before, years with fewer than {min_n} events hidden",
                    fontsize=11, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_by_second_diagnosis_year_panel(ax, frame, outcome, outcome.upper(), min_n=min_n)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    for min_n in [0] + MIN_N_THRESHOLDS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        n_note = f"only years with n>={min_n}" if min_n > 0 else "no minimum n"
        fig.suptitle(f"STATEMENT 1 (supporting) | 1st (trauma) vs 2nd (outcome) diagnosis-year "
                    f"IRR distribution ({n_note}) | {age_label}\n"
                    f"PRE-war & POST-war groups are t-tested (1st > 2nd); cross-reference group "
                    f"is NOT tested\nCAUTION: each point = one YEAR's IRR, not one patient -- "
                    f"small-sample test",
                    fontsize=10, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_first_second_boxplot_combined(ax, frame, outcome, min_n)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # patient-level page: each point = one patient (time to 2nd diagnosis), not one year's IRR
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"STATEMENT 1 (supporting) | Time from 1st (trauma) to 2nd (outcome) diagnosis, "
                f"PATIENT-LEVEL | {age_label}\n"
                f"each point = one patient who converted within {FIXED_WINDOW_DAYS//30}mo | "
                f"PRE-war vs POST-war tested (one-tail Mann-Whitney); cross-reference group NOT tested\n"
                f"CAUTION: measures speed of conversion among converters, not how many converted; "
                f"late entrants have less follow-up to convert in",
                fontsize=10, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_time_to_second_dx_boxplot(ax, frame, outcome)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # patient-level KM pages: cumulative incidence over ALL exposed patients (censoring-aware),
    # one page per pair of curves in KM_PAGES
    for curves, pair_label in KM_PAGES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
        delayed_note = ("curve 3 = pre-Oct7 trauma patients still at risk on Oct7, followed from "
                        "Oct7 (delayed entry); " if 3 in curves else "")
        fig.suptitle(f"STATEMENT 1 (supporting) | Cumulative incidence of 2nd diagnosis after "
                    f"{TRAUMA_EXPOSURE_LABEL}, PATIENT-LEVEL (1 - Kaplan-Meier) | {age_label}\n"
                    f"{pair_label} | all exposed patients; follow-up ending without a 2nd dx "
                    f"(window / left Clalit / study end) is censored\n"
                    f"{delayed_note}shaded = 95% CI",
                    fontsize=10, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_km_cumulative_incidence(ax, frame, outcome, curves)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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


def draw_yearly_trend_forecast_panel(ax, frame, downstream_outcome, title):
    trend, n_gen = compute_yearly_rr_trend(frame, downstream_outcome)
    x = np.arange(len(trend))
    fit_end = next(i for i, l in enumerate(trend["label"]) if l == "2023\n(pre)")   # inclusive
    fit_mask = x <= fit_end

    result = fit_trend_and_predict(x[fit_mask], trend["rr"].values[fit_mask], x)

    ax.fill_between(x, result["lower"], result["upper"], color="gray", alpha=0.18,
                    label="95% prediction interval\n(from pre-war trend)")
    ax.plot(x, result["y_pred"], linestyle="--", linewidth=1.3, color="gray",
           label="pre-war OLS trend")
    ax.plot(x, trend["rr"], marker="o", markersize=5, linewidth=1.8,
           color=OUTCOME_COLORS.get(downstream_outcome, "#333333"), label="actual RR")

    for xi, row in zip(x, trend.itertuples()):
        if xi > fit_end and pd.notna(row.rr):
            lo, hi = result["lower"][xi], result["upper"][xi]
            outside = pd.notna(lo) and pd.notna(hi) and not (lo <= row.rr <= hi)
            marker_color = "black" if outside else OUTCOME_COLORS.get(downstream_outcome, "#333333")
            ax.plot(xi, row.rr, marker="*" if outside else "o",
                   markersize=14 if outside else 5, color=marker_color, zorder=5)
            if outside:
                ax.text(xi, row.rr, "  outside PI", fontsize=6, ha="left", va="center")

    ax.axvline(fit_end + 0.5, color="black", linestyle=":", linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(trend["label"], fontsize=8)
    ax.set_xlabel(f"Year of {TRAUMA_EXPOSURE_LABEL} entry (fit uses 2017 .. 2023-pre only, "
                 f"n_fit={result['n_fit']})", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")


# --- T-Quarterly: same design as T-Yearly, at quarterly (3-month) resolution instead of yearly.
# Finer-grained version of the same question -- a real gradual trend should look smooth quarter
# to quarter; a data artifact or small-n noise will look jagged. Q_INDEX matches Statement 3's
# quarterly range (2017Q1-2026Q2) so the two charts' time axes line up if compared side by side.
Q_INDEX = pd.period_range("2017Q1", "2026Q2", freq="Q")
QUARTERLY_TREND_PERIODS = [(str(q), q.start_time, (q + 1).start_time) for q in Q_INDEX]
_OCT7_QUARTER_POS = next(i for i, q in enumerate(Q_INDEX) if q == pd.Period("2023Q4", freq="Q"))

def compute_quarterly_rr_trend(frame, downstream_outcome):
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]

    n_gen = int(first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES).isna().sum())
    gen_split = compute_general_lexis_split(frame, TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome,
                                            QUARTERLY_TREND_PERIODS)

    rows = []
    for label, start, end in QUARTERLY_TREND_PERIODS:
        period_entry = first_trauma.where((first_trauma >= start) & (first_trauma < end))
        n = int(period_entry.notna().sum())
        rate, _, _ = person_time_rate(period_entry, down, FIXED_WINDOW_DAYS)
        gen_rate = gen_split.loc[label, "rate"]
        rr = rate / gen_rate if (pd.notna(gen_rate) and gen_rate > 0 and pd.notna(rate)) else np.nan
        rows.append({"label": label, "n": n, "rr": rr})
    return pd.DataFrame(rows), n_gen


# Per-point n labels are dropped here (unlike the yearly version) -- ~38 quarterly points would
# make them illegible; hover/inspect the underlying DataFrame directly if a specific quarter's
# n is needed.
def draw_quarterly_trend_panel(ax, frame, downstream_outcome, title):
    trend, n_gen = compute_quarterly_rr_trend(frame, downstream_outcome)
    x = np.arange(len(trend))
    ax.plot(x, trend["rr"], marker="o", markersize=3, linewidth=1.3,
           color=OUTCOME_COLORS.get(downstream_outcome, "#333333"))
    ax.axvline(_OCT7_QUARTER_POS - 0.5, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    year_ticks = [i for i, q in enumerate(Q_INDEX) if q.quarter == 1]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(Q_INDEX[i].year) for i in year_ticks], fontsize=8)
    ax.set_xlabel(f"Quarter of {TRAUMA_EXPOSURE_LABEL} entry (Oct 7 2023 marked)", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(f"{title}  (n_gen={n_gen:,}, constant across quarters)", fontsize=10, fontweight="bold")


# --- PART 1B, STEP 5: Write the 4 supporting-test pages for one age band ---
def write_statement1_supporting(pdf, frame, entry_meta, downstream_meta, age_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 2b | {TRAUMA_EXPOSURE_LABEL} pre vs post Oct 7, "
                f"restricted person-time within {FIXED_WINDOW_DAYS//30}mo | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_t2b_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 5 | Entry-diagnosis profession among "
                f"converters within {FIXED_WINDOW_DAYS//30}mo, pre vs post Oct 7 | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_t5_panel(ax, frame, entry_meta, outcome, f"-> {outcome.upper()}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 6 | Downstream subtype at conversion "
                f"within {FIXED_WINDOW_DAYS//30}mo, pre vs post Oct 7 trauma entry | {age_label}",
                fontsize=11, fontweight="bold")
    draw_t6_panel(axes[0], frame, downstream_meta, "psychosis", classify_psychosis_subtype, "PSYCHOSIS subtype")
    draw_t6_panel(axes[1], frame, downstream_meta, "bipolar", classify_bipolar_subtype, "BIPOLAR subtype")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 9 | Trauma engagement severity "
                f"(inpatient vs outpatient), within {FIXED_WINDOW_DAYS//30}mo, "
                f"pre vs post Oct 7 | {age_label}",
                fontsize=11, fontweight="bold")
    print_facility_type_diagnostic(entry_meta)
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_t9_panel(ax, frame, entry_meta, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST Zone | direct (Near Gaza / Northern "
                f"Border) vs indirect exposure, within {FIXED_WINDOW_DAYS//30}mo, "
                f"pre vs post Oct 7 | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_zone_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | Yearly trend | continuous RR by year of "
                f"{TRAUMA_EXPOSURE_LABEL} entry, within {FIXED_WINDOW_DAYS//30}mo | {age_label}\n"
                f"gradual pre-existing decline vs a sharp break at Oct 7?",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_yearly_trend_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | By year of SECOND diagnosis | within "
                f"{FIXED_WINDOW_DAYS//30}mo of {TRAUMA_EXPOSURE_LABEL} entry | {age_label}\n"
                f"same idea as Yearly trend, but the x-axis year is when the second "
                f"diagnosis happened, not when the trauma did",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_by_second_diagnosis_year_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    write_statement1_min_n_and_boxplot_pages(pdf, frame, age_label)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | Yearly trend, pre-war forecast | "
                f"{age_label}\ndoes post-Oct7 fall outside where the PRE-WAR trend alone "
                f"would have predicted it?",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_yearly_trend_forecast_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | Quarterly trend | continuous RR by quarter of "
                f"{TRAUMA_EXPOSURE_LABEL} entry, within {FIXED_WINDOW_DAYS//30}mo | {age_label}\n"
                f"same question as Yearly trend, at 3-month resolution",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_quarterly_trend_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# PART 2 -- STATEMENT 2: ALTERNATE ENTRY-DIAGNOSIS RR ("B1", extended to 7)
# =============================================================================

# --- S2, STEP 1: RR for one downstream outcome, across N exposures (windowed, censored) ---
# Excludes the downstream outcome itself from the exposure set (no self-pairs).
def compute_b1_raw_rr(frame, downstream_outcome):
    labels, rrs, ns, n_gens = [], [], [], []
    for exp_label, outcome_list in current_alt_exposures_7().items():
        if downstream_outcome in outcome_list:
            continue   # exposure == downstream outcome; skip self-pair
        n_exp, n_gen, comp = compute_exposed_vs_general(frame, outcome_list, [downstream_outcome])
        row = comp.iloc[0]
        labels.append(exp_label); rrs.append(row["rr"]); ns.append(n_exp); n_gens.append(n_gen)
    return labels, rrs, ns, n_gens


# --- S2, STEP 2: Draw one B1 panel (bar chart) for one downstream outcome ---
def draw_b1_panel(ax, frame, downstream_outcome, title):
    labels, rrs, ns, n_gens = compute_b1_raw_rr(frame, downstream_outcome)
    x = np.arange(len(labels))
    bars = ax.bar(x, [r if pd.notna(r) else 0 for r in rrs],
                 color=[EXPOSURE_COLORS.get(l, "#888") for l in labels], alpha=0.85, width=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    for bar, rr, n, n_gen in zip(bars, rrs, ns, n_gens):
        txt = f"RR {rr:.1f}" if pd.notna(rr) else "RR n/a"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f"{txt}\nn_exp={n:,}\nn_gen={n_gen:,}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("IRR (exposed / general), downstream after exposure", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


# --- S2, STEP 3: Pre/post-war RR heatmap across all 7 exposures x 2 focus outcomes ---
def compute_b1_prepost_matrix(frame, entry_side):
    """entry_side: 'pre' or 'post' (Oct 7 diagnosis date of the ENTRY exposure)."""
    rows = []
    for exp_label, outcome_list in current_alt_exposures_7().items():
        downstream = [o for o in FOCUS_OUTCOMES if o not in outcome_list]
        if not downstream:
            continue
        (n_pre, rr_pre), (n_post, rr_post), n_gen = compute_prepost_window_rr(
            frame, outcome_list, downstream, FIXED_WINDOW_DAYS)
        n_exp, rr_dict = (n_pre, rr_pre) if entry_side == "pre" else (n_post, rr_post)
        row = {"exposure": exp_label, "n_exp": n_exp, "n_gen": n_gen}
        for o in FOCUS_OUTCOMES:
            row[o] = rr_dict.get(o, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("exposure")


def draw_b1_prepost_heatmap(ax, matrix, title):
    display_cols = [o for o in FOCUS_OUTCOMES]
    vals = matrix[display_cols].values.astype(float)
    row_labels = [f"{idx}\n(n_exp={int(n):,}, n_gen={int(g):,})"
                 for idx, n, g in zip(matrix.index, matrix["n_exp"], matrix["n_gen"])]
    sns.heatmap(vals, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True,
               xticklabels=[o.upper() for o in display_cols],
               yticklabels=row_labels, ax=ax, annot_kws={"fontsize": 8})
    ax.set_title(title, fontsize=10, fontweight="bold")


# --- S2, STEP 3a-bar: Bar-chart version of the pre/post heatmap above -- one panel per
# downstream outcome (matching the existing FOCUS_OUTCOMES 2-panel convention used elsewhere in
# Statement 2), X-axis = the 7 alternate exposures, 2 bars each (pre-war IRR, post-war IRR). ---
def draw_b1_prepost_bar_panel(ax, pre_matrix, post_matrix, downstream_outcome, title):
    exposures = pre_matrix.index.tolist()
    post_aligned = post_matrix.reindex(exposures)
    x = np.arange(len(exposures)); w = 0.35
    pre_vals  = pre_matrix[downstream_outcome].values
    post_vals = post_aligned[downstream_outcome].values
    bars_pre  = ax.bar(x - w/2, [v if pd.notna(v) else 0 for v in pre_vals], width=w,
                       color="#95a5a6", alpha=0.85, label="Before Oct 7")
    bars_post = ax.bar(x + w/2, [v if pd.notna(v) else 0 for v in post_vals], width=w,
                       color="#e63946", alpha=0.85, label="On/after Oct 7")
    for bar, v, n in zip(bars_pre, pre_vals, pre_matrix["n_exp"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{v:.1f}\nn={int(n):,}", ha="center", va="bottom", fontsize=6)
    for bar, v, n in zip(bars_post, post_vals, post_aligned["n_exp"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{v:.1f}\nn={int(n):,}" if pd.notna(n) else f"{v:.1f}",
                   ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(exposures, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)


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


# Fixed, tight color scale so moderate (real-world-typical) changes already reach strong
# color saturation -- letting sns.heatmap auto-scale vmin/vmax from the data (via center=0
# alone) stretches the range to whatever the single most extreme cell is, which washes out
# every more-typical cell to a pale shade. Values beyond the cap are CLIPPED to full color
# (indicated by the colorbar's extended triangle ends), not hidden -- the annotated % text
# always shows the true value regardless of clipping.
CHANGE_HEATMAP_COLOR_CAP = 50   # +-50% change already reaches full red/green saturation

def draw_b1_prepost_change_heatmap(ax, change_matrix, title):
    display_cols = FOCUS_OUTCOMES
    vals = change_matrix[display_cols].values.astype(float)
    row_labels = [f"{idx}\n(n_pre={int(np_):,}, n_post={int(npo):,})"
                 for idx, np_, npo in zip(change_matrix.index, change_matrix["n_pre"], change_matrix["n_post"])]
    annot = np.full(vals.shape, "", dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if pd.notna(vals[i, j]):
                annot[i, j] = f"{vals[i, j]:+.0f}%"
    cap = CHANGE_HEATMAP_COLOR_CAP
    sns.heatmap(vals, annot=annot, fmt="", cmap="RdYlGn", vmin=-cap, vmax=cap, cbar=True,
               xticklabels=[o.upper() for o in display_cols],
               yticklabels=row_labels, ax=ax, annot_kws={"fontsize": 8},
               mask=np.isnan(vals), linewidths=0.5, linecolor="#dddddd",
               cbar_kws={"extend": "both", "label": f"clipped at +-{cap}%"})
    ax.set_title(title, fontsize=10, fontweight="bold")


# --- S2, STEP 4: Write Statement 2 pages (4 pages) for one age band ---
# --- S2 EXTENDED: every entry diagnosis x every second diagnosis (7x7), plus the bar charts for
# the second diagnoses slides 22/24 don't cover. Written at the END of the PDF (slides 43+) so
# slides 1-42 keep their numbers. Same windowed person-time design as the rest of Statement 2.
# The merged trauma diagnosis is also a SECOND diagnosis here: its first date as an outcome is
# the earliest of its components (column first_trauma_dx, added on the fly). Diagonal = self-pair
# (entry diagnosis == second diagnosis) and stays empty.
TRAUMA_DX_KEY = "trauma_dx"


def s2_targets_7(frame):
    """(label, downstream key, component outcomes) for the 7 second diagnoses, trauma first."""
    if f"first_{TRAUMA_DX_KEY}" not in frame.columns:
        frame[f"first_{TRAUMA_DX_KEY}"] = first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES)
    return [(label, TRAUMA_DX_KEY if outcomes == TRAUMA_EXPOSURE_OUTCOMES else outcomes[0], outcomes)
            for label, outcomes in current_alt_exposures_7().items()]


def compute_s2_full_matrices(frame):
    """Returns (overall, pre, post): DataFrames indexed by entry-diagnosis label, columns n_exp,
    n_gen + one IRR column per downstream key. NaN on the diagonal (self-pair)."""
    targets = s2_targets_7(frame)
    overall, pre, post = [], [], []
    for exp_label, exp_outcomes in current_alt_exposures_7().items():
        downs = [key for _, key, comps in targets if comps != exp_outcomes]
        n_exp, n_gen, comp = compute_exposed_vs_general(frame, exp_outcomes, downs)
        row = {"exposure": exp_label, "n_exp": n_exp, "n_gen": n_gen}
        row.update(dict(zip(comp["outcome"], comp["rr"])))
        overall.append(row)
        (n_pre, rr_pre), (n_post, rr_post), n_gen_pp = compute_prepost_window_rr(
            frame, exp_outcomes, downs, FIXED_WINDOW_DAYS)
        pre.append({"exposure": exp_label, "n_exp": n_pre, "n_gen": n_gen_pp, **rr_pre})
        post.append({"exposure": exp_label, "n_exp": n_post, "n_gen": n_gen_pp, **rr_post})
    keys = [key for _, key, _ in targets]
    as_df = lambda rows: pd.DataFrame(rows).set_index("exposure").reindex(columns=["n_exp", "n_gen"] + keys)
    return as_df(overall), as_df(pre), as_df(post)


def draw_s2_full_heatmap(ax, matrix, targets, title, n_col="n_exp", change=False, vmax=None):
    keys = [key for _, key, _ in targets]
    vals = matrix[keys].values.astype(float)
    annot = np.full(vals.shape, "", dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if pd.notna(vals[i, j]):
                annot[i, j] = f"{vals[i, j]:+.0f}%" if change else f"{vals[i, j]:.1f}"
    if change:
        row_labels = [f"{idx}\n(n_pre={int(a):,}, n_post={int(b):,})"
                      for idx, a, b in zip(matrix.index, matrix["n_pre"], matrix["n_post"])]
        cap = CHANGE_HEATMAP_COLOR_CAP
        kw = dict(cmap="RdYlGn", vmin=-cap, vmax=cap, cbar_kws={"extend": "both", "label": f"clipped at +-{cap}%"})
    else:
        row_labels = [f"{idx}\n(n_exp={int(n):,})" for idx, n in zip(matrix.index, matrix[n_col])]
        kw = dict(cmap="YlOrRd", vmin=0, vmax=vmax, cbar_kws={"label": f"IRR within {FIXED_WINDOW_DAYS//30}mo"})
    sns.heatmap(vals, annot=annot, fmt="", mask=np.isnan(vals), linewidths=0.5, linecolor="#dddddd",
                xticklabels=[label for label, _, _ in targets], yticklabels=row_labels, ax=ax,
                annot_kws={"fontsize": 8}, **kw)
    ax.set_xlabel("Second diagnosis (downstream)", fontsize=8)
    ax.set_ylabel("Entry diagnosis", fontsize=8)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.set_title(title, fontsize=10, fontweight="bold")


def draw_s2_overall_bar_panel(ax, overall, downstream_key, title):
    """Slide-22 style: one bar per entry diagnosis, IRR for this second diagnosis."""
    sub = overall[overall[downstream_key].notna()]
    x = np.arange(len(sub))
    bars = ax.bar(x, sub[downstream_key].values, width=0.6, alpha=0.85,
                  color=[EXPOSURE_COLORS.get(l, "#888") for l in sub.index])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    for bar, (label, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"RR {row[downstream_key]:.1f}\nn_exp={int(row['n_exp']):,}\nn_gen={int(row['n_gen']):,}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("IRR (exposed / general), downstream after exposure", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


def write_statement2_extended(pdf, frame, age_label):
    targets = s2_targets_7(frame)
    overall, pre, post = compute_s2_full_matrices(frame)
    win = FIXED_WINDOW_DAYS // 30

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.suptitle(f"STATEMENT 2 (extended) | IRR, every entry diagnosis x every second diagnosis | "
                 f"within {win}mo, all years | {age_label}\nrow = entry diagnosis, column = second "
                 f"diagnosis; each row vs. its own general population (never had that entry dx)",
                 fontsize=11, fontweight="bold")
    draw_s2_full_heatmap(ax, overall, targets, "")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    fig.suptitle(f"STATEMENT 2 (extended) | IRR by entry diagnosis date, before vs on/after Oct 7 | "
                 f"within {win}mo | {age_label}", fontsize=11, fontweight="bold")
    keys = [key for _, key, _ in targets]
    shared_vmax = np.nanmax(np.concatenate([pre[keys].values.ravel(), post[keys].values.ravel()]).astype(float))
    draw_s2_full_heatmap(axes[0], pre, targets, "Entry diagnosed BEFORE Oct 7 2023 (same color scale)",
                         vmax=shared_vmax)
    draw_s2_full_heatmap(axes[1], post, targets, "Entry diagnosed ON/AFTER Oct 7 2023 (same color scale)",
                         vmax=shared_vmax)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    keys = [key for _, key, _ in targets]
    change = pd.DataFrame(index=pre.index)
    change["n_pre"], change["n_post"] = pre["n_exp"], post["n_exp"]
    for k in keys:
        change[k] = np.where(pre[k] > 0, (post[k] - pre[k]) / pre[k] * 100, np.nan)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.suptitle(f"STATEMENT 2 (extended) | How much did the IRR change, pre- vs post-Oct7 entry | "
                 f"{age_label}\ngreen = IRR increased after Oct 7, red = decreased (% change "
                 f"relative to the pre-Oct7 IRR)", fontsize=11, fontweight="bold")
    draw_s2_full_heatmap(ax, change, targets, "", change=True)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # bar charts for the second diagnoses slides 22/24 don't show -- 2 per page
    labels = {key: label for label, key, _ in targets}
    remaining = [key for key in ["anxiety", "depression", "substance", "personality", TRAUMA_DX_KEY]
                 if key in labels]
    pages = [remaining[i:i + 2] for i in range(0, len(remaining), 2)]
    for keys_on_page in pages:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"STATEMENT 2 (extended) | Alternate entry-diagnosis IRR | within {win}mo, "
                     f"restricted person-time | {age_label}", fontsize=11, fontweight="bold")
        for ax, key in zip(axes, keys_on_page):
            draw_s2_overall_bar_panel(ax, overall, key, labels[key].upper())
        for ax in axes[len(keys_on_page):]:
            ax.axis("off")
        plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    for keys_on_page in pages:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"STATEMENT 2 (extended) | Entry diagnosis before vs on/after Oct 7 | "
                     f"IRR within {win}mo | {age_label}", fontsize=11, fontweight="bold")
        for ax, key in zip(axes, keys_on_page):
            draw_b1_prepost_bar_panel(ax, pre.drop(index=pre.index[pre[key].isna() & post[key].isna()]),
                                      post, key, labels[key].upper())
        for ax in axes[len(keys_on_page):]:
            ax.axis("off")
        plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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
COMORB_STATE_COLORS = {"none": "#b0b0b0", "trauma": "#e63946", "x": "#457b9d", "both": "#2b2d42"}


def comorb_state_labels(x_label):
    return {"none": f"no {TRAUMA_EXPOSURE_LABEL}, no {x_label}", "trauma": f"{TRAUMA_EXPOSURE_LABEL} only",
            "x": f"{x_label} only", "both": f"{TRAUMA_EXPOSURE_LABEL} + {x_label}"}


def comorb_x_list(outcome):
    """The other diagnoses X paired with trauma for this outcome: (label, outcome key)."""
    return [(label, outs[0]) for label, outs in current_alt_exposures_7().items()
            if outs != TRAUMA_EXPOSURE_OUTCOMES and outs[0] != outcome]


def comorbidity_segments(frame, outcome, x):
    """One row per (patient, status) stretch with person-time: state, start, py, event."""
    T, Tr = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES), first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES)
    X, Xr = first_of(frame, [x]), first_of_raw(frame, [x])
    keep = ~(Tr.notna() & T.isna()) & ~(Xr.notna() & X.isna())
    T, X = T[keep], X[keep]
    O, fu_start, cap = frame.loc[keep, f"first_{outcome}"], frame.loc[keep, "fu_start"], frame.loc[keep, "fu_end"]
    win = pd.to_timedelta(FIXED_WINDOW_DAYS, unit="D")
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
                                   "py": (exit_ - start).dt.days / 365.25, "event": event}))
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


def draw_comorb_heatmap(ax, table, side, title, vmax=None):
    refs = ["none", "trauma", "x"]
    vals = table[[f"{side}|{r}" for r in refs]].values.astype(float)
    change = side == "change"
    annot = np.full(vals.shape, "", dtype=object)
    for i in range(vals.shape[0]):
        ev = None if change else int(table[f"{side}|ev"].iloc[i])
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                # comorbid events under each IRR: few events = unstable IRR
                annot[i, j] = f"{vals[i, j]:+.0f}%" if change else f"{vals[i, j]:.1f}\n({ev} events)"
    if change:
        cap = CHANGE_HEATMAP_COLOR_CAP
        kw = dict(cmap="RdYlGn", vmin=-cap, vmax=cap, center=0,
                  cbar_kws={"extend": "both", "label": f"% change, clipped at +-{cap}%"})
    else:
        kw = dict(cmap="YlOrRd", vmin=0, vmax=vmax, cbar_kws={"label": "IRR (comorbid / reference)"})
    sns.heatmap(vals, annot=annot, fmt="", mask=np.isnan(vals), linewidths=0.5, linecolor="#dddddd",
                xticklabels=[COMORB_REF_LABELS[r] for r in refs],
                yticklabels=[f"{TRAUMA_EXPOSURE_LABEL} + {x}\n(n={int(n):,})" for x, n in zip(table.index, table["n_both"])],
                ax=ax, annot_kws={"fontsize": 9}, **kw)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=7.5, rotation=0)
    ax.set_title(title, fontsize=10, fontweight="bold")


def draw_comorb_bar_panel(ax, r, outcome, x_label):
    labels = comorb_state_labels(x_label)
    vals = [r[("all", s)] for s in COMORB_STATES]
    x = np.arange(len(COMORB_STATES))
    bars = ax.bar(x, [v["rate"] if pd.notna(v["rate"]) else 0 for v in vals], width=0.6,
                  color=[COMORB_STATE_COLORS[s] for s in COMORB_STATES], alpha=0.9)
    for bar, v in zip(bars, vals):
        rate_txt = f"{v['rate']:.1f}" if pd.notna(v["rate"]) else "n/a"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{rate_txt}\nevents={v['events']:,}\nPY={v['py']:,.0f}", ha="center", va="bottom", fontsize=7)
    both = vals[3]["rate"]
    irr = lambda ref: both / ref if pd.notna(both) and pd.notna(ref) and ref > 0 else np.nan
    irr_txt = "  |  ".join(f"{name} {irr(vals[i]['rate']):.1f}" for i, name in
                           [(0, "IRR vs none"), (1, "vs trauma only"), (2, f"vs {x_label} only")])
    ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in COMORB_STATES], fontsize=7.5, rotation=12)
    ax.set_ylabel(f"{outcome.capitalize()} rate per 1,000 person-years", fontsize=8)
    ax.set_ylim(0, max([v["rate"] for v in vals if pd.notna(v["rate"])] + [0]) * 1.3 or 1)
    ax.set_title(f"{outcome.upper()} | {TRAUMA_EXPOSURE_LABEL} + {x_label}\n{irr_txt}",
                 fontsize=9, fontweight="bold")


def write_comorbidity_statement(pdf, frame, age_label):
    tables, details = compute_comorbidity_tables(frame)
    win = FIXED_WINDOW_DAYS // 30
    head = f"COMORBIDITY | {TRAUMA_EXPOSURE_LABEL} + another diagnosis X"
    note = ("each cell = rate in the comorbid status / rate in the reference status; statuses are "
            f"time-varying (a patient counts in each status only while in it), {win}mo window from entry")

    vmax = np.nanmax([tables[o][[f"all|{r}" for r in ["none", "trauma", "x"]]].values for o in FOCUS_OUTCOMES])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(f"{head} | IRR, all years | {age_label}\n{note}", fontsize=11, fontweight="bold")
    for ax, o in zip(axes, FOCUS_OUTCOMES):
        draw_comorb_heatmap(ax, tables[o], "all", f"-> {o.upper()}", vmax=vmax)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    vmax = np.nanmax([tables[o][[f"{s}|{r}" for s in ["pre", "post"] for r in ["none", "trauma", "x"]]].values
                      for o in FOCUS_OUTCOMES])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"{head} | IRR by date of entering the status, before vs on/after Oct 7 | {age_label}\n"
                 f"{note}; 'no trauma, no X' = whole period (same reference both sides); same color scale",
                 fontsize=11, fontweight="bold")
    for row, o in enumerate(FOCUS_OUTCOMES):
        draw_comorb_heatmap(axes[row, 0], tables[o], "pre", f"-> {o.upper()} | BEFORE Oct 7", vmax=vmax)
        draw_comorb_heatmap(axes[row, 1], tables[o], "post", f"-> {o.upper()} | ON/AFTER Oct 7", vmax=vmax)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(f"{head} | How much did each IRR change, pre- vs post-Oct7 | {age_label}\n"
                 f"green = IRR increased after Oct 7, red = decreased (% change relative to the pre-Oct7 IRR)",
                 fontsize=11, fontweight="bold")
    for ax, o in zip(axes, FOCUS_OUTCOMES):
        draw_comorb_heatmap(ax, tables[o], "change", f"-> {o.upper()}")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    for o in FOCUS_OUTCOMES:
        xs = list(tables[o].index)
        for i in range(0, len(xs), 2):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
            fig.suptitle(f"{head} | rates of the 4 statuses, all years, within {win}mo | {age_label}",
                         fontsize=11, fontweight="bold")
            for ax, x_label in zip(axes, xs[i:i + 2]):
                draw_comorb_bar_panel(ax, details[(o, x_label)], o, x_label)
            for ax in axes[len(xs[i:i + 2]):]:
                ax.axis("off")
            plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_statement2(pdf, frame, age_label):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"STATEMENT 2 | Alternate entry-diagnosis IRR | within "
                f"{FIXED_WINDOW_DAYS//30}mo, restricted person-time | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_b1_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    pre_matrix  = compute_b1_prepost_matrix(frame, "pre")
    post_matrix = compute_b1_prepost_matrix(frame, "post")
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    fig.suptitle(f"STATEMENT 2 | Entry diagnosis before vs on/after Oct 7 | "
                f"IRR within {FIXED_WINDOW_DAYS//30}mo | {age_label}",
                fontsize=11, fontweight="bold")
    draw_b1_prepost_heatmap(axes[0], pre_matrix,  "Entry diagnosed BEFORE Oct 7 2023")
    draw_b1_prepost_heatmap(axes[1], post_matrix, "Entry diagnosed ON/AFTER Oct 7 2023")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"STATEMENT 2 | Entry diagnosis before vs on/after Oct 7 | "
                f"bar-chart view of the heatmap above | {age_label}",
                fontsize=11, fontweight="bold")
    draw_b1_prepost_bar_panel(axes[0], pre_matrix, post_matrix, "psychosis", "PSYCHOSIS")
    draw_b1_prepost_bar_panel(axes[1], pre_matrix, post_matrix, "bipolar", "BIPOLAR")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    change_matrix = compute_b1_prepost_change_matrix(pre_matrix, post_matrix)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"STATEMENT 2 | How much did the RR change, pre- vs post-Oct7 entry | "
                f"{age_label}\ngreen = RR increased after Oct 7, red = RR decreased "
                f"(% change relative to the pre-Oct7 RR)",
                fontsize=11, fontweight="bold")
    draw_b1_prepost_change_heatmap(ax, change_matrix, "")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# PART 3 -- STATEMENT 3: ROSE vs DID-NOT-RISE (with Northern Border zone)
# =============================================================================

# --- S3, STEP 1: Zone populations = patient count per zone in this frame ---
def zone_populations(frame):
    return frame["zone"].value_counts().reindex(ZONE_ORDER).fillna(0)


# --- S3, STEP 2: Quarterly first-ever counts per zone for one item ('total' = all 8) ---
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
        py = (seg_end - seg_start).dt.days.clip(lower=0) / 365.25
        rows[q] = py.groupby(frame[group_col]).sum()
    return pd.DataFrame(rows).T.reindex(index=q_index, columns=groups).fillna(0)


def to_rate_per_1000_py(counts, person_years):
    return counts / person_years.replace(0, np.nan) * 1000


# --- S3, STEP 3: Draw one quarterly per-1,000-by-zone panel ---
# Zones drawn in Statement 3. Northern Border is left out on request -- its residents are simply
# not shown (NOT merged into Rest of Israel), so the two remaining lines are unchanged.
STATEMENT3_ZONES = ["Near Gaza", "Rest of Israel"]


def draw_rate_panel(ax, rate_df, oct7_pos, title, subtitle=None):
    x = np.arange(len(rate_df.index))
    for z in STATEMENT3_ZONES:
        if z in rate_df.columns:
            ax.plot(x, rate_df[z].values, marker="o", markersize=3.5,
                   linewidth=1.8, color=ZONE_COLORS[z], label=ZONE_LABELS[z])
    ax.axvline(oct7_pos - 0.5, color="black", linewidth=2)
    ax.text(oct7_pos - 0.45, ax.get_ylim()[1] * 0.97, "Oct 7", fontsize=8, va="top")
    year_ticks = [i for i, q in enumerate(rate_df.index) if q.quarter == 1]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(rate_df.index[i].year) for i in year_ticks], fontsize=8)
    ax.set_ylabel("New diagnoses per 1,000 person-years\n(members under follow-up)", fontsize=9)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=3)


def prepost_ratio_text(rate_df):
    pre  = [q for q in rate_df.index if q <= pd.Period("2023Q3", freq="Q")]
    post = [q for q in rate_df.index if q >= pd.Period("2023Q4", freq="Q")]
    ratio = rate_df.loc[post].mean() / rate_df.loc[pre].mean().replace(0, np.nan)
    return "post/pre rate ratio -> " + "  |  ".join(
        f"{z}: x{ratio[z]:.2f}" for z in STATEMENT3_ZONES if z in ratio.index)


# --- S3, STEP 4: Write Statement 3 pages (ROSE section, then DID-NOT-RISE section) ---
def write_statement3(pdf, frame, age_label):
    q_index = pd.period_range("2017Q1", "2026Q2", freq="Q")
    person_years = quarterly_person_years(frame, q_index)
    oct7_pos = next(i for i, q in enumerate(q_index) if q == pd.Period("2023Q4", freq="Q"))

    for section_label, items in [("ROSE AFTER OCT 7", STATEMENT3_ROSE),
                                 ("DID NOT RISE", STATEMENT3_DID_NOT_RISE)]:
        for item in items:
            label = "ALL 8 OUTCOMES" if item == "total" else item.upper()
            rate = to_rate_per_1000_py(quarterly_counts(frame, item), person_years)
            fig, ax = plt.subplots(figsize=(13, 5.5))
            draw_rate_panel(ax, rate, oct7_pos,
                            f"STATEMENT 3 | {section_label} | {label} | "
                            f"New dx per 1,000 person-years | Quarterly by zone | {age_label}",
                            subtitle=prepost_ratio_text(rate))
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    write_statement3_by_distance(pdf, frame, age_label)


# --- S3, STEP 5: Same rates, one line per DISTANCE-FROM-GAZA band instead of per zone --------
# Northern Border residents are left out (not merged into a band), like the zone pages above.
# Patients with no distance value are left out too. In a population file that is already cut
# by distance (e.g. <=20 km), only the bands inside that range have lines.
DIST_BANDS = [(0, 7, "0-7 km"), (7, 20, "7-20 km"), (20, 40, "20-40 km"),
              (40, 80, "40-80 km"), (80, np.inf, "80+ km")]
DIST_BAND_LABELS = [label for _, _, label in DIST_BANDS]
DIST_BAND_COLORS = dict(zip(DIST_BAND_LABELS, ["#9d0208", "#e85d04", "#faa307", "#2a9d8f", "#264653"]))


def distance_band(frame):
    """Distance band per patient (lower bound exclusive, so exactly 20 km falls in 7-20 km,
    matching the <=20 km population); NaN for Northern Border residents / no distance."""
    band = pd.Series(np.nan, index=frame.index, dtype=object)
    for lo, hi, label in DIST_BANDS:
        band[(frame["distance_km"] > lo) & (frame["distance_km"] <= hi)] = label
    band[frame["distance_km"] == 0] = DIST_BANDS[0][2]
    band[frame["is_north_border"]] = np.nan
    return band


def write_statement3_by_distance(pdf, frame, age_label):
    q_index = pd.period_range("2017Q1", "2026Q2", freq="Q")
    oct7_pos = next(i for i, q in enumerate(q_index) if q == pd.Period("2023Q4", freq="Q"))
    f = frame.assign(dist_band=distance_band(frame))
    f = f[f["dist_band"].notna()]
    person_years = quarterly_person_years(f, q_index, "dist_band", DIST_BAND_LABELS)
    shown = [b for b in DIST_BAND_LABELS if person_years[b].sum() > 0]
    year_ticks = [i for i, q in enumerate(q_index) if q.quarter == 1]

    for section_label, items in [("ROSE AFTER OCT 7", STATEMENT3_ROSE),
                                 ("DID NOT RISE", STATEMENT3_DID_NOT_RISE)]:
        fig, axes = plt.subplots(3, 2, figsize=(15, 13))
        fig.suptitle(f"STATEMENT 3 | {section_label} | by distance from Gaza (Northern Border "
                     f"excluded) | New dx per 1,000 person-years | Quarterly | {age_label}",
                     fontsize=11, fontweight="bold")
        for ax, item in zip(axes.flat, items):
            rate = to_rate_per_1000_py(
                quarterly_counts(f, item, "dist_band", DIST_BAND_LABELS), person_years)
            x = np.arange(len(q_index))
            for b in shown:
                ax.plot(x, rate[b].values, marker="o", markersize=2.5, linewidth=1.4,
                        color=DIST_BAND_COLORS[b], label=b)
            ax.axvline(oct7_pos - 0.5, color="black", linewidth=1.5)
            ax.set_xticks(year_ticks)
            ax.set_xticklabels([str(q_index[i].year) for i in year_ticks], fontsize=7)
            pre, post = rate.iloc[:oct7_pos].mean(), rate.iloc[oct7_pos:].mean()
            ratio = post / pre.replace(0, np.nan)
            ax.set_xlabel("post/pre -> " + " | ".join(f"{b}: x{ratio[b]:.2f}" for b in shown
                                                      if pd.notna(ratio[b])), fontsize=7)
            ax.set_ylabel("per 1,000 person-years", fontsize=8)
            ax.set_title("ALL 8 OUTCOMES" if item == "total" else item.upper(),
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left", ncol=3)
        for ax in list(axes.flat)[len(items):]:
            ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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


# --- S5, STEP 2: Draw one side's panel -- all 8 outcomes, % diagnosed by a psychiatrist ---
def draw_profession_specialist_panel(ax, outcome_meta, side, title, treat_missing_as_nonpsychiatrist):
    pcts, ns = [], []
    for outcome in ALL_OUTCOMES:
        pct, n = compute_pct_psychiatrist(outcome_meta, outcome, side, treat_missing_as_nonpsychiatrist)
        pcts.append(pct if pd.notna(pct) else 0)
        ns.append(n)
    x = np.arange(len(ALL_OUTCOMES))
    bars = ax.bar(x, pcts, color=[OUTCOME_COLORS.get(o, "#888") for o in ALL_OUTCOMES], alpha=0.85)
    for bar, n, pct in zip(bars, ns, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{pct:.0f}%\nn={n:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([o.upper() for o in ALL_OUTCOMES], rotation=30, ha="right", fontsize=8)
    denom_note = ("code 999/Unknown counted as non-psychiatrist" if treat_missing_as_nonpsychiatrist
                 else "code 999/Unknown excluded from n")
    ax.set_ylabel(f"% of first-ever diagnoses given by a psychiatrist\n({denom_note})", fontsize=7.5)
    ax.set_ylim(0, 112)
    ax.set_title(title, fontsize=10, fontweight="bold")


# --- S5, STEP 3: Write BOTH Statement 5 pages (Pre/Post x 2 missing-data treatments) ---
def write_statement5(pdf, outcome_meta, age_label):
    print_profession_diagnostic(outcome_meta)
    for treat_missing, page_label in [
            (False, "code 999 (Unknown/Admin) EXCLUDED from numerator and denominator"),
            (True,  "code 999 (Unknown/Admin) COUNTED as non-psychiatrist")]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"STATEMENT 5 | Who diagnoses what -- % given by a psychiatrist, "
                    f"checked separately pre vs post Oct 7 | {age_label}\n{page_label}",
                    fontsize=11, fontweight="bold")
        draw_profession_specialist_panel(axes[0], outcome_meta, "pre", "Before Oct 7 2023", treat_missing)
        draw_profession_specialist_panel(axes[1], outcome_meta, "post", "On/after Oct 7 2023", treat_missing)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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
# One page per NON-trauma outcome (6 pages: psychosis, bipolar, anxiety, depression, substance,
# personality), each with 2 panels (Pre-Oct7 entry / Post-Oct7 entry).
# =============================================================================

LATENCY_BINS_MONTHS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 36), (36, np.inf)]
LATENCY_BIN_LABELS  = ["0-1", "1-3", "3-6", "6-12", "12-24", "24-36", "36+"]

# --- S6, STEP 1: Latency (months) from trauma entry to this outcome, among converters, one side ---
def compute_latency_distribution(frame, downstream_outcome, side):
    first_trauma = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    down = frame[f"first_{downstream_outcome}"]
    side_mask = first_trauma.notna() & ((first_trauma < OCT7) if side == "pre" else (first_trauma >= OCT7))
    converted_mask = side_mask & down.notna() & (down > first_trauma)
    latency_months = (down[converted_mask] - first_trauma[converted_mask]).dt.days / 30.44

    counts = [int(((latency_months >= lo) & (latency_months < hi)).sum())
             for lo, hi in LATENCY_BINS_MONTHS]
    return counts, int(converted_mask.sum())


# --- S6, STEP 2: Draw one side's latency histogram for one outcome ---
def draw_latency_panel(ax, frame, downstream_outcome, side, title):
    counts, n_converted = compute_latency_distribution(frame, downstream_outcome, side)
    x = np.arange(len(LATENCY_BIN_LABELS))
    bars = ax.bar(x, counts, color=OUTCOME_COLORS.get(downstream_outcome, "#888888"), alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f"{c}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(LATENCY_BIN_LABELS, fontsize=8)
    ax.set_xlabel("Months from trauma entry to this diagnosis", fontsize=8)
    ax.set_ylabel(f"Number of converters (n={n_converted:,})", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


# --- S6, STEP 3: Write one page per non-trauma outcome (6 pages), Pre/Post panels each ---
def write_statement6(pdf, frame, age_label):
    for outcome in S1_DOWNSTREAM:   # psychosis, bipolar, anxiety, depression, substance, personality
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f"STATEMENT 6 | Latency: months from {TRAUMA_EXPOSURE_LABEL} entry to "
                    f"first {outcome.upper()} | {age_label}\n"
                    f"unrestricted by window -- shows full empirical distribution",
                    fontsize=11, fontweight="bold")
        draw_latency_panel(axes[0], frame, outcome, "pre",  "Pre-Oct7 entry")
        draw_latency_panel(axes[1], frame, outcome, "post", "Post-Oct7 entry")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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
    trauma_date = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
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


# --- S7, STEP 1: One panel for a CONTINUOUS/ORDINAL variable -- boxplot (+ Mann-Whitney U if
# show_stats) ---
def draw_continuous_demo_panel(ax, pre, post, col, title, ylabel, show_stats=True):
    pre_vals  = pd.to_numeric(pre[col], errors="coerce").dropna().values
    post_vals = pd.to_numeric(post[col], errors="coerce").dropna().values

    bp = ax.boxplot([pre_vals, post_vals], labels=["Before Oct7", "On/after Oct7"],
                    patch_artist=True, widths=0.5, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#95a5a6", "#e63946"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)

    median_txt = ""
    if len(pre_vals) > 0 and len(post_vals) > 0:
        median_txt = f"  |  median: {np.median(pre_vals):.1f} vs {np.median(post_vals):.1f}"
    ax.set_ylabel(ylabel, fontsize=9)
    if show_stats:
        pval_str = "ns"
        if len(pre_vals) > 0 and len(post_vals) > 0:
            _, p = stats.mannwhitneyu(pre_vals, post_vals, alternative="two-sided")
            pval_str = f"{p:.2e}" if p < 0.05 else "ns"
        ax.set_title(f"{title}\nMann-Whitney U p={pval_str}  (n: {len(pre_vals):,}, {len(post_vals):,})"
                   f"{median_txt}", fontsize=10, fontweight="bold")
    else:
        ax.set_title(f"{title}\n(n: {len(pre_vals):,}, {len(post_vals):,}){median_txt}",
                   fontsize=10, fontweight="bold")


# --- S7, STEP 2: One panel for a CATEGORICAL variable -- grouped % bars (+ Chi-square if
# show_stats) ---
def draw_categorical_demo_panel(ax, pre, post, col, title, top_n=8, show_stats=True):
    pre_vc  = pre[col].dropna().astype(str).value_counts(normalize=True) * 100
    post_vc = post[col].dropna().astype(str).value_counts(normalize=True) * 100
    top_cats = list(dict.fromkeys(list(pre_vc.head(top_n).index) + list(post_vc.head(top_n).index)))

    x = np.arange(len(top_cats)); w = 0.38
    pre_heights  = [pre_vc.get(c, 0) for c in top_cats]
    post_heights = [post_vc.get(c, 0) for c in top_cats]
    ax.bar(x - w/2, pre_heights,  width=w, color="#95a5a6", alpha=0.85, label="Before Oct7")
    ax.bar(x + w/2, post_heights, width=w, color="#e63946", alpha=0.85, label="On/after Oct7")
    ax.set_xticks(x); ax.set_xticklabels(top_cats, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("% within period", fontsize=9)

    n_pre, n_post = pre[col].notna().sum(), post[col].notna().sum()
    top_note = "" if len(top_cats) == pre[col].dropna().nunique() == post[col].dropna().nunique() \
              else f" (top {top_n} shown)"
    if show_stats:
        pval_str = chi_square_pvalue(pre[col], post[col])
        ax.set_title(f"{title}\nChi-square p={pval_str}  (n: {n_pre:,}, {n_post:,}){top_note}",
                   fontsize=10, fontweight="bold")
    else:
        ax.set_title(f"{title}\n(n: {n_pre:,}, {n_post:,}){top_note}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


# --- S7, STEP 2b: PTSD vs ASR composition -- did the MIX shift, not just the count? ---
def classify_trauma_composition(sub):
    has_ptsd = sub["first_ptsd"].notna()
    has_asr  = sub["first_asr"].notna()
    out = pd.Series("Both", index=sub.index)
    out[has_ptsd & ~has_asr] = "PTSD only"
    out[~has_ptsd & has_asr] = "ASR only"
    return out


def draw_trauma_composition_panel(ax, pre, post, show_stats=True):
    pre_class, post_class = classify_trauma_composition(pre), classify_trauma_composition(post)
    cats = ["PTSD only", "ASR only", "Both"]
    pre_vc, post_vc = pre_class.value_counts(normalize=True) * 100, post_class.value_counts(normalize=True) * 100

    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w/2, [pre_vc.get(c, 0) for c in cats],  width=w, color="#95a5a6", alpha=0.85, label="Before Oct7")
    ax.bar(x + w/2, [post_vc.get(c, 0) for c in cats], width=w, color="#e63946", alpha=0.85, label="On/after Oct7")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("% within period", fontsize=9)

    if show_stats:
        pval_str = chi_square_pvalue(pre_class, post_class)
        ax.set_title(f"PTSD vs ASR COMPOSITION -- did the MIX shift?\n"
                   f"Chi-square p={pval_str}  (n: {len(pre_class):,}, {len(post_class):,})",
                   fontsize=10, fontweight="bold")
    else:
        ax.set_title(f"PTSD vs ASR COMPOSITION -- did the MIX shift?\n"
                   f"(n: {len(pre_class):,}, {len(post_class):,})", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


# --- S7, STEP 2c: Prior psychiatric history BEFORE the trauma diagnosis -- reverse-causality /
# diagnostic-purity check (this was an open "still todo" item -- built here for the first
# time). If post-war patients have LESS prior history than pre-war ones, that supports "war
# trauma captures a broader, less pre-selected population" rather than "the same people who
# were already engaged with mental health services." ---
PRIOR_HISTORY_OUTCOMES = ["anxiety", "depression", "substance", "personality", "psychosis", "bipolar"]

def has_any_prior_history(sub):
    entry = first_of(sub, TRAUMA_EXPOSURE_OUTCOMES)
    has_prior = pd.Series(False, index=sub.index)
    for o in PRIOR_HISTORY_OUTCOMES:
        has_prior |= (sub[f"first_{o}"].notna() & (sub[f"first_{o}"] < entry))
    return has_prior


def draw_prior_history_panel(ax, pre, post, show_stats=True):
    pre_prior, post_prior = has_any_prior_history(pre), has_any_prior_history(post)
    pre_pct  = pre_prior.mean() * 100 if len(pre_prior) else np.nan
    post_pct = post_prior.mean() * 100 if len(post_prior) else np.nan

    ax.bar([0, 1], [pre_pct, post_pct], color=["#95a5a6", "#e63946"], alpha=0.85, width=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Before Oct7", "On/after Oct7"], fontsize=9)
    ax.set_ylabel("% with ANY prior psychiatric dx\nbefore the trauma diagnosis", fontsize=8.5)
    for xi, v, n in zip([0, 1], [pre_pct, post_pct], [len(pre_prior), len(post_prior)]):
        if pd.notna(v):
            ax.text(xi, v + 1, f"{v:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max([v for v in [pre_pct, post_pct] if pd.notna(v)] + [10]) * 1.3)

    if show_stats:
        pval_str = chi_square_pvalue(pre_prior.map({True: "yes", False: "no"}),
                                     post_prior.map({True: "yes", False: "no"}))
        ax.set_title(f"PRIOR PSYCHIATRIC HISTORY (any, before trauma dx)\n"
                   f"Chi-square p={pval_str}  -- reverse-causality check", fontsize=9.5, fontweight="bold")
    else:
        ax.set_title("PRIOR PSYCHIATRIC HISTORY (any, before trauma dx)\n"
                   "-- reverse-causality check", fontsize=9.5, fontweight="bold")


def draw_prior_history_breakdown_panel(ax, pre, post):
    def pct_by_type(sub):
        entry = first_of(sub, TRAUMA_EXPOSURE_OUTCOMES)
        n = len(sub)
        out = {}
        for o in PRIOR_HISTORY_OUTCOMES:
            had_prior = sub[f"first_{o}"].notna() & (sub[f"first_{o}"] < entry)
            out[o] = had_prior.sum() / n * 100 if n > 0 else np.nan
        return out
    pre_pct, post_pct = pct_by_type(pre), pct_by_type(post)
    cats = PRIOR_HISTORY_OUTCOMES
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w/2, [pre_pct[c] for c in cats],  width=w, color="#95a5a6", alpha=0.85, label="Before Oct7")
    ax.bar(x + w/2, [post_pct[c] for c in cats], width=w, color="#e63946", alpha=0.85, label="On/after Oct7")
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in cats], rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("% with THIS prior dx before trauma", fontsize=9)
    ax.set_title("PRIOR HISTORY BY TYPE (before trauma dx)\n(descriptive breakdown of the panel to the left)",
               fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8)


# --- S7, STEP 2d: Who diagnosed the trauma + what facility, for ALL trauma patients (not just
# the small subgroup who later converted, unlike TEST 5/9 in Statement 1). Answers a more basic
# question: did general trauma-diagnosis PRACTICE itself shift after Oct7? ---
def draw_trauma_all_patients_profession_panel(ax, pre, post, entry_meta, show_stats=True):
    profession = get_exposure_meta_field(entry_meta, TRAUMA_EXPOSURE_OUTCOMES, "profession")
    def labels_for(idx):
        codes = normalize_profession_code(profession.reindex(idx))
        codes = codes[codes.notna() & ~codes.isin(MISSING_PROFESSION_CODES)]
        return codes.apply(profession_code_label)
    pre_labels, post_labels = labels_for(pre.index), labels_for(post.index)
    pre_vc, post_vc = pre_labels.value_counts(normalize=True) * 100, post_labels.value_counts(normalize=True) * 100
    top_cats = list(dict.fromkeys(list(pre_vc.head(8).index) + list(post_vc.head(8).index)))

    x = np.arange(len(top_cats)); w = 0.38
    ax.bar(x - w/2, [pre_vc.get(c, 0) for c in top_cats],  width=w, color="#95a5a6", alpha=0.85, label="Before Oct7")
    ax.bar(x + w/2, [post_vc.get(c, 0) for c in top_cats], width=w, color="#e63946", alpha=0.85, label="On/after Oct7")
    ax.set_xticks(x); ax.set_xticklabels(top_cats, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("% within period (code 999 excluded)", fontsize=8.5)

    if show_stats:
        pval_str = chi_square_pvalue(pre_labels, post_labels)
        ax.set_title(f"WHO DIAGNOSED THE TRAUMA (ALL trauma patients, not just converters)\n"
                   f"Chi-square p={pval_str}  (n: {len(pre_labels):,}, {len(post_labels):,})",
                   fontsize=9.5, fontweight="bold")
    else:
        ax.set_title(f"WHO DIAGNOSED THE TRAUMA (ALL trauma patients, not just converters)\n"
                   f"(n: {len(pre_labels):,}, {len(post_labels):,})", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7)


def draw_trauma_all_patients_facility_panel(ax, pre, post, entry_meta, show_stats=True):
    facility = get_exposure_meta_field(entry_meta, TRAUMA_EXPOSURE_OUTCOMES, "facility type")
    pre_fac  = facility.reindex(pre.index).dropna().astype(str)
    post_fac = facility.reindex(post.index).dropna().astype(str)
    pre_vc, post_vc = pre_fac.value_counts(normalize=True) * 100, post_fac.value_counts(normalize=True) * 100
    top_cats = list(dict.fromkeys(list(pre_vc.head(8).index) + list(post_vc.head(8).index)))

    x = np.arange(len(top_cats)); w = 0.38
    ax.bar(x - w/2, [pre_vc.get(c, 0) for c in top_cats],  width=w, color="#95a5a6", alpha=0.85, label="Before Oct7")
    ax.bar(x + w/2, [post_vc.get(c, 0) for c in top_cats], width=w, color="#e63946", alpha=0.85, label="On/after Oct7")
    ax.set_xticks(x); ax.set_xticklabels(top_cats, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("% within period", fontsize=9)

    if show_stats:
        pval_str = chi_square_pvalue(pre_fac, post_fac)
        ax.set_title(f"TRAUMA DIAGNOSIS FACILITY TYPE (ALL trauma patients)\n"
                   f"Chi-square p={pval_str}  (n: {len(pre_fac):,}, {len(post_fac):,})",
                   fontsize=9.5, fontweight="bold")
    else:
        ax.set_title(f"TRAUMA DIAGNOSIS FACILITY TYPE (ALL trauma patients)\n"
                   f"(n: {len(pre_fac):,}, {len(post_fac):,})", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7)


# --- S7, STEP 2e: Timing of post-war trauma diagnoses -- immediate surge or gradual trickle? --
def draw_days_since_oct7_panel(ax, post):
    entry = first_of(post, TRAUMA_EXPOSURE_OUTCOMES)
    days = (entry - OCT7).dt.days.dropna()
    days = days[days >= 0]
    if len(days) == 0:
        ax.text(0.5, 0.5, "No post-war trauma patients", transform=ax.transAxes,
               ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("TIMING OF POST-WAR TRAUMA DIAGNOSES", fontsize=10, fontweight="bold")
        return
    bins = np.arange(0, int(days.max()) + 15, 14)   # 2-week bins
    ax.hist(days, bins=bins, color="#e63946", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Days from Oct 7 to trauma diagnosis (2-week bins)", fontsize=9)
    ax.set_ylabel("n patients", fontsize=9)
    ax.set_title(f"TIMING OF POST-WAR TRAUMA DIAGNOSES\n(n={len(days):,}, median={days.median():.0f}d)",
               fontsize=10, fontweight="bold")


# --- S7, STEP 3: Write the whole demographic-comparison PDF (its own file, not per-mode) ---
def write_demographic_comparison_pdf(frame_full, entry_meta_full, output_dir, trauma_mode=None):
    trauma_mode = trauma_mode or SELECTED_TRAUMA_MODE
    set_trauma_mode(trauma_mode)
    pre, post = split_trauma_pre_post(frame_full)
    print(f"\n=== Demographic comparison | trauma={TRAUMA_EXPOSURE_LABEL} | "
         f"n_pre={len(pre):,} | n_post={len(post):,} ===")

    output_pdf = os.path.join(output_dir, f"demographic_comparison_{trauma_mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic comparison | {TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | age, health burden", fontsize=11, fontweight="bold")
        draw_continuous_demo_panel(axes[0], pre, post, "age_at_oct7", "AGE (at Oct 7)", "Age (years)")
        draw_continuous_demo_panel(axes[1], pre, post, "charlson_score",
                                   "CHARLSON COMORBIDITY INDEX", "Charlson score")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic comparison | {TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | gender, geographic zone", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "gender", "GENDER")
        draw_categorical_demo_panel(axes[1], pre, post, "zone", "GEOGRAPHIC ZONE (distance to Gaza)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.suptitle(f"Demographic comparison | {TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | sector, district", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "sector", "SECTOR")
        draw_categorical_demo_panel(axes[1], pre, post, "district", "DISTRICT")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic comparison | {TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | socioeconomic status, immigration", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "ses_5level", "SOCIOECONOMIC SCORE (5-level)")
        draw_categorical_demo_panel(axes[1], pre, post, "is_immigrant", "IS IMMIGRANT (has immigration date)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.suptitle(f"Demographic comparison | {TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | country of birth, mailing language", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "country_of_birth", "COUNTRY OF BIRTH")
        draw_categorical_demo_panel(axes[1], pre, post, "mailing_language", "MAILING LANGUAGE")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        fig.suptitle(f"Demographic comparison | {TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | insurance type", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(ax, pre, post, "insurance_type", "INSURANCE TYPE")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"  Saved: {output_pdf}")
    return output_pdf


# =============================================================================
# PART 7b -- DEMOGRAPHIC INSIGHTS (descriptive only, no statistical tests)
#
# The Tier-1/Tier-2 items proposed on request: same underlying comparisons (trauma
# composition, prior psychiatric history, who/where diagnosed for ALL trauma patients --
# not just converters, continuous geography, 3-level SES, timing of post-war diagnoses), but
# in a SEPARATE file and with every p-value/statistical-test annotation removed -- just counts,
# percentages, and medians. Reuses the exact same draw functions as the main demographic-
# comparison PDF via show_stats=False, so the underlying numbers are identical between the two
# files; only the presence of the test annotation differs.
# =============================================================================

def write_demographic_insights_pdf(frame_full, entry_meta_full, output_dir, trauma_mode=None):
    trauma_mode = trauma_mode or SELECTED_TRAUMA_MODE
    set_trauma_mode(trauma_mode)
    pre, post = split_trauma_pre_post(frame_full)
    print(f"\n=== Demographic insights (descriptive only) | trauma={TRAUMA_EXPOSURE_LABEL} | "
         f"n_pre={len(pre):,} | n_post={len(post):,} ===")

    output_pdf = os.path.join(output_dir, f"demographic_insights_{trauma_mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | trauma composition, prior history",
                    fontsize=11, fontweight="bold")
        draw_trauma_composition_panel(axes[0], pre, post, show_stats=False)
        draw_prior_history_panel(axes[1], pre, post, show_stats=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | prior history by type", fontsize=11, fontweight="bold")
        draw_prior_history_breakdown_panel(ax, pre, post)   # already descriptive-only
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.suptitle(f"Demographic insights (descriptive) | {TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | trauma diagnosis practice, ALL patients "
                    f"(not just converters)", fontsize=11, fontweight="bold")
        draw_trauma_all_patients_profession_panel(axes[0], pre, post, entry_meta_full, show_stats=False)
        draw_trauma_all_patients_facility_panel(axes[1], pre, post, entry_meta_full, show_stats=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | geography (continuous), SES (3-level)",
                    fontsize=11, fontweight="bold")
        draw_continuous_demo_panel(axes[0], pre, post, "distance_km",
                                   "DISTANCE TO GAZA (continuous)", "Distance (km)", show_stats=False)
        draw_categorical_demo_panel(axes[1], pre, post, "ses_3level", "SOCIOECONOMIC SCORE (3-level)",
                                    show_stats=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {TRAUMA_EXPOSURE_LABEL} | "
                    f"timing of post-war diagnoses", fontsize=11, fontweight="bold")
        draw_days_since_oct7_panel(ax, post)   # already descriptive-only
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"  Saved: {output_pdf}")
    return output_pdf


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
            contributed_days = (bucket_end_capped - bucket_start).days
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
    trauma_date = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    entry = trauma_date.where(trauma_date < OCT7)   # pre-war only; NaT elsewhere is fine
    down = frame[f"first_{downstream_outcome}"]

    trend = compute_relative_year_split(entry, down, full_years_only=full_years_only)
    gen_rate, _, _ = compute_general_rate_overall(frame, TRAUMA_EXPOSURE_OUTCOMES, downstream_outcome)
    trend["irr"] = trend["rate"] / gen_rate if (pd.notna(gen_rate) and gen_rate > 0) else np.nan
    return trend, gen_rate


def draw_relative_year_panel(ax, frame, downstream_outcome, title, full_years_only=False):
    trend, gen_rate = compute_relative_year_irr(frame, downstream_outcome, full_years_only)
    if len(trend) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center",
               fontsize=10, color="gray")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    ax.plot(trend["rel_year"], trend["irr"], marker="o", markersize=6, linewidth=1.8,
           color=OUTCOME_COLORS.get(downstream_outcome, "#333333"))
    for _, row in trend.iterrows():
        if pd.notna(row["irr"]):
            ax.text(row["rel_year"], row["irr"], f"n={int(row['n_events'])}", fontsize=7,
                   ha="center", va="bottom")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(trend["rel_year"].astype(int))
    year_note = "final partial year included" if not full_years_only else "only COMPLETE years counted"
    ax.set_xlabel(f"Year since trauma diagnosis ({year_note})", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general, gen_rate={gen_rate:.2f})" if pd.notna(gen_rate)
                 else "IRR (exposed / general)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


def write_relative_year_pdf(frame_full, output_dir, trauma_mode=None):
    trauma_mode = trauma_mode or SELECTED_TRAUMA_MODE
    set_trauma_mode(trauma_mode)
    print(f"\n=== IRR by year since trauma diagnosis | trauma={TRAUMA_EXPOSURE_LABEL} ===")

    output_pdf = os.path.join(output_dir, f"irr_by_year_since_diagnosis_{trauma_mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"IRR by year since {TRAUMA_EXPOSURE_LABEL} diagnosis | capped at Oct 7 2023 | "
                    f"final PARTIAL year included\n"
                    f"pre-war trauma patients only (post-war have no elapsed time before Oct7)",
                    fontsize=11, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_relative_year_panel(ax, frame_full, outcome, outcome.upper(), full_years_only=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"IRR by year since {TRAUMA_EXPOSURE_LABEL} diagnosis | capped at Oct 7 2023 | "
                    f"only COMPLETE years counted\n"
                    f"pre-war trauma patients only (post-war have no elapsed time before Oct7)",
                    fontsize=11, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_relative_year_panel(ax, frame_full, outcome, outcome.upper(), full_years_only=True)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"  Saved: {output_pdf}")
    return output_pdf


# =============================================================================
# MASTER EXPORT
# =============================================================================

# --- Load once, reused across all 3 trauma-mode runs (avoids re-reading the large CSV 3x) ---
def load_and_build(input_file):
    print("Loading input file...")
    df = load_frame(input_file)
    print(f"Loaded {len(df):,} rows.")
    built = build_frame(df)
    set_followup(built[0])
    return built


# =============================================================================
# PART 8 -- ADJUSTED IRR: Poisson regression with log(person-years) as offset
#
# Every IRR above is CRUDE: exposed rate / general rate, nothing else taken into account. The two
# groups also differ in age, sex, sector, SES and comorbidity, and if any of those affects the
# outcome on its own, the crude IRR mixes the trauma effect with the group differences
# (confounding). Here the SAME person-time as the crude IRR (exposed_intervals() /
# general_intervals() -- same windows, membership end, age-15 rule, prevalent-case exclusion) is
# fed into a Poisson regression:
#   log(events) = log(person-years) + exposure + attained age + sex + sector + SES + Charlson
# and exp(coefficient of exposure) = the ADJUSTED IRR: how much the exposure raises the rate
# among patients who are otherwise similar on those factors.
#   - Attained age: each stretch of person-time is split by the age the patient was at that time
#     (5-year bands), so time spent at 22 counts as age 20-24 even if the patient is 40 at Oct 7.
#   - Sex, sector, SES (5-level), Charlson (0/1/2/3+) are per-patient categories; a missing value
#     is its own category. Each covariate's reference is its largest category (by person-years).
#   - Region / zone is deliberately NOT in the model (on request); no calendar-year term either.
#   - Fitted on AGGREGATED cells (exposure x age band x sex x sector x SES x Charlson: summed
#     events and person-years) -- for a Poisson model this gives exactly the same estimates as
#     the patient-level fit, at a fraction of the size. Fitted by IRLS in numpy (no statsmodels
#     dependency); 95% CI = Wald, from the inverse Fisher information.
#   - Crude IRR in these plots = the same model with the exposure term only -- it equals the
#     crude IRR on the earlier slides exactly (sanity check 11b).
#   - Patients with no birth date can't be given an attained age and are left out of the model.
#   - CAVEAT: the Charlson score is "last until 2023" -- for some patients it was measured after
#     their entry, so it may partly reflect consequences rather than only prior health.
# =============================================================================
ADJ_AGE_BANDS = [(a, a + 5, f"{a}-{a + 4}") for a in range(15, 65, 5)] + [(65, 200, "65+")]
ADJ_COVARIATES = ["Sex", "Sector", "SES (5-level)", "Charlson"]
ADJ_COLORS = {"crude": "#95a5a6", "adjusted": "#e63946"}


def _as_category(series):
    """Clean per-patient category: numbers like '3.0' -> '3', blanks/NaN -> 'missing'."""
    s = series.astype(object)
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).str.strip()
    is_int = num.notna() & (num == num.round())
    out[is_int] = num[is_int].astype(int).astype(str)
    out[s.isna() | out.isin(["", "nan", "None", "NaT"])] = "missing"
    return out


def adjustment_covariates(frame):
    charlson = pd.cut(frame["charlson_score"], [-np.inf, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    return pd.DataFrame({
        "Sex": _as_category(frame["gender"]),
        "Sector": _as_category(frame["sector"]),
        "SES (5-level)": _as_category(frame["ses_5level"]),
        "Charlson": charlson.astype(object).where(charlson.notna(), "missing"),
    }, index=frame.index)


def _age_band_bounds(birth):
    """[(label, band_start, band_end)] dates per patient (5-year bands; 365.25-day years, cut at
    midnight -- at most a day off a true birthday, irrelevant at this scale, and whole days so the
    bands add up to exactly the unsplit person-time). The first band starts at birth (nobody is at
    risk before 15 anyway) and the last runs to age 150, so no person-time falls outside."""
    def at_age(years):
        return (birth + pd.to_timedelta(years * 365.25, unit="D")).dt.normalize()
    bounds = []
    for i, (lo, hi, label) in enumerate(ADJ_AGE_BANDS):
        b_start = birth if i == 0 else at_age(lo)
        bounds.append((label, b_start, at_age(min(hi, 150))))
    return bounds


def _aggregate_intervals(start, end, is_event, event, cell, bounds, exposure_label):
    """Splits each [start, end) interval by attained-age band and sums events and person-years
    per (covariate cell, age band). Returns a small DataFrame."""
    idx = start.index
    parts = []
    for label, b_start, b_end in bounds:
        b_start, b_end = b_start.reindex(idx), b_end.reindex(idx)
        s = start.where(start > b_start, b_start)
        e = end.where(end < b_end, b_end)
        py = (e - s).dt.days.clip(lower=0).fillna(0) / 365.25
        ev = (is_event & (event >= b_start) & (event < b_end)).astype(int)
        keep = (py > 0) | (ev > 0)
        if keep.any():
            agg = pd.DataFrame({"cell": cell.reindex(idx)[keep].values, "py": py[keep].values,
                                "events": ev[keep].values}).groupby("cell").sum()
            agg["age"] = label
            parts.append(agg.reset_index())
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cell", "py", "events", "age"])
    out["exposure"] = exposure_label
    return out


def build_poisson_table(frame, exposure_outcomes, downstream_outcome):
    """Aggregated person-time table for one exposure definition and one outcome. Exposed time is
    labeled 'before Oct 7' / 'on/after Oct 7' by the date of entry (as in Statement 1); general
    time is 'general'. Returns (table, n_no_birth)."""
    has_birth = frame["birth"].notna()
    f = frame[has_birth]
    covs = adjustment_covariates(f)
    cell = covs.groupby(ADJ_COVARIATES, sort=False).ngroup()
    cell_values = covs.assign(cell=cell.values).drop_duplicates("cell").set_index("cell")
    bounds = _age_band_bounds(f["birth"])
    down = f[f"first_{downstream_outcome}"]

    entry_all = first_of(f, exposure_outcomes)
    parts = []
    for label, entry in [("before Oct 7", entry_all.where(entry_all < OCT7)),
                         ("on/after Oct 7", entry_all.where(entry_all >= OCT7))]:
        e_start, e_end, e_is_event, e_event = exposed_intervals(entry, down)
        parts.append(_aggregate_intervals(e_start, e_end, e_is_event, e_event, cell, bounds, label))
    g_start, g_end, g_is_event, g_event = general_intervals(f, exposure_outcomes, downstream_outcome)
    parts.append(_aggregate_intervals(g_start, g_end, g_is_event, g_event, cell, bounds, "general"))

    table = pd.concat(parts, ignore_index=True)
    table = table.groupby(["exposure", "age", "cell"], as_index=False)[["py", "events"]].sum()
    table = table.join(cell_values, on="cell")
    return table[table["py"] > 0].reset_index(drop=True), int((~has_birth).sum())


def fit_poisson(table, exposure_col, covariates, exposure_ref="general"):
    """Poisson GLM, log link, offset log(py), fitted by IRLS. `covariates` are categorical
    columns of `table`, each coded against its largest category by person-years. Returns a
    DataFrame of terms (term, group, level, irr, lo, hi, p) -- intercept excluded."""
    cols, names = [np.ones(len(table))], [("Intercept", "", "")]
    for col in [exposure_col] + covariates:
        if col == exposure_col:
            ref = exposure_ref
        else:
            ref = table.groupby(col)["py"].sum().idxmax()
        for level in sorted(table[col].astype(str).unique()):
            if level == str(ref):
                continue
            cols.append((table[col].astype(str) == level).astype(float).values)
            names.append((f"{col}: {level}", col, level))
    X = np.column_stack(cols)
    y = table["events"].values.astype(float)
    offset = np.log(table["py"].values)

    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum(), 0.5) / table["py"].sum())
    for _ in range(100):
        eta = np.clip(X @ beta + offset, -50, 50)
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / mu
        XtW = X.T * mu
        beta_new = np.linalg.pinv(XtW @ X) @ (XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    mu = np.exp(np.clip(X @ beta + offset, -50, 50))
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv((X.T * mu) @ X)), 0, None))

    rows = []
    for (term, group, level), b, s in zip(names[1:], beta[1:], se[1:]):
        level_events = table.loc[table[group].astype(str) == level, "events"].sum()
        if level_events == 0:   # no events in this level -> estimate runs to 0, CI meaningless
            rows.append({"term": term, "group": group, "level": level, "irr": np.nan,
                         "lo": np.nan, "hi": np.nan, "p": np.nan, "events": 0})
            continue
        p = 2 * stats.norm.sf(abs(b / s)) if s > 0 else np.nan
        rows.append({"term": term, "group": group, "level": level, "irr": np.exp(b),
                     "lo": np.exp(b - 1.96 * s), "hi": np.exp(b + 1.96 * s), "p": p,
                     "events": int(level_events)})
    return pd.DataFrame(rows)


def adjusted_irr_results(frame, exposure_outcomes, downstream_outcome):
    """Crude and adjusted IRR for the exposure overall and split before / on-after Oct 7, plus
    the full adjusted (overall) model's terms."""
    table, n_no_birth = build_poisson_table(frame, exposure_outcomes, downstream_outcome)
    pooled = table.assign(exposure=table["exposure"].where(table["exposure"] == "general", "exposed"))
    out = {"n_no_birth": n_no_birth, "table": table}
    for key, tbl in [("overall", pooled), ("prepost", table)]:
        for kind, covs in [("crude", []), ("adjusted", ["age"] + ADJ_COVARIATES)]:
            out[(key, kind)] = fit_poisson(tbl, "exposure", covs)
    return out


def _exposure_row(res, level):
    r = res[res["level"] == level]
    return r.iloc[0] if len(r) else None


def rtl(text):
    """Hebrew text for matplotlib (which draws right-to-left text reversed): python-bidi if it's
    installed, otherwise a plain reversal (fine for the single-word category labels used here)."""
    text = str(text)
    if not any("֐" <= c <= "׿" for c in text):
        return text
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except ImportError:
        return text[::-1]


FOREST_TICKS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]


def draw_forest(ax, rows, title, xlabel="IRR (log scale), 95% CI"):
    """rows: list of (label, irr, lo, hi, color) top to bottom; label-only rows (irr None) are
    bold group headers. Value + CI printed in a column to the right of the plot."""
    n = len(rows)
    finite = [v for _, irr, lo, hi, _ in rows if irr is not None for v in (irr, lo, hi)
              if v is not None and np.isfinite(v) and v > 0]
    lo_lim = max(min(finite + [0.5]) / 1.2, 0.05)
    hi_lim = min(max(finite + [2.0]) * 1.2, 50)
    for i, (label, irr, lo, hi, color) in enumerate(rows):
        yy = n - 1 - i
        if irr is None:
            continue
        if pd.isna(irr):
            ax.text(1.02, yy, "n/a (no events)", transform=ax.get_yaxis_transform(),
                    fontsize=7, va="center", ha="left", color="gray")
        else:
            ax.plot([max(lo, lo_lim), min(hi, hi_lim)], [yy, yy], color=color, linewidth=1.6)
            ax.plot(irr, yy, marker="s", color=color, markersize=6)
            ax.text(1.02, yy, f"{irr:.2f} [{lo:.2f}-{hi:.2f}]", transform=ax.get_yaxis_transform(),
                    fontsize=7, va="center", ha="left")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(lo_lim, hi_lim)
    ticks = [t for t in FOREST_TICKS if lo_lim <= t <= hi_lim]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    for tick, r in zip(ax.get_yticklabels(), rows[::-1]):
        if r[1] is None:
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


def write_adjusted_forest_pages(pdf, frame, age_label):
    adj_note = (f"Adjusted for attained age (5-yr bands), sex, sector, SES (5-level), Charlson "
                f"(0/1/2/3+) -- Poisson regression, offset log(person-years), 95% Wald CI. "
                f"Same person-time as the crude IRR: within {FIXED_WINDOW_DAYS // 30}mo of entry, "
                f"membership end, age 15+, prevalent cases excluded.")
    results = {o: adjusted_irr_results(frame, TRAUMA_EXPOSURE_OUTCOMES, o) for o in FOCUS_OUTCOMES}

    # --- page 1: trauma, crude vs adjusted, overall and before / after Oct 7 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"ADJUSTED IRR | {TRAUMA_EXPOSURE_LABEL} vs general population | crude vs adjusted, "
                 f"overall and by entry before / on-after Oct 7 | {age_label}\n{adj_note}",
                 fontsize=9.5, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        res = results[outcome]
        rows = []
        for header, key, level in [(f"{TRAUMA_EXPOSURE_LABEL} (any time)", "overall", "exposed"),
                                   (f"{TRAUMA_EXPOSURE_LABEL} before Oct 7", "prepost", "before Oct 7"),
                                   (f"{TRAUMA_EXPOSURE_LABEL} on/after Oct 7", "prepost", "on/after Oct 7")]:
            rows.append((header, None, None, None, None))
            for kind in ["crude", "adjusted"]:
                r = _exposure_row(res[(key, kind)], level)
                rows.append((f"   {kind}",) + ((r["irr"], r["lo"], r["hi"]) if r is not None
                                                else (np.nan, np.nan, np.nan)) + (ADJ_COLORS[kind],))
        n_note = f" ({res['n_no_birth']:,} without birth date left out)" if res["n_no_birth"] else ""
        draw_forest(ax, rows, f"-> {outcome.upper()}{n_note}")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(wspace=0.55)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # --- page 2: every term of the adjusted overall model ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 13))
    fig.suptitle(f"ADJUSTED IRR | full model, all covariates | {TRAUMA_EXPOSURE_LABEL} (any time) "
                 f"vs general | {age_label}\nEach row = IRR of that level vs its reference "
                 f"(the largest category), with all other terms held fixed.\n{adj_note}",
                 fontsize=9.5, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        terms = results[outcome][("overall", "adjusted")]
        table = results[outcome]["table"]
        rows = []
        for group in ["exposure", "age"] + ADJ_COVARIATES:
            g = terms[terms["group"] == group]
            if group == "exposure":
                ref = "general"
            else:
                ref = table.groupby(group)["py"].sum().idxmax()
            rows.append((f"{'Exposure' if group == 'exposure' else 'Attained age' if group == 'age' else group}"
                         f"  (ref: {rtl(ref)})", None, None, None, None))
            for r in g.itertuples():
                rows.append((f"   {rtl(r.level)}", r.irr, r.lo, r.hi, ADJ_COLORS["adjusted"]))
        draw_forest(ax, rows, f"-> {outcome.upper()}")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(wspace=0.55)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # --- page 3: all 7 entry exposures (Statement 2), crude vs adjusted, one model each ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f"ADJUSTED IRR | all entry diagnoses (Statement 2), each vs its own general "
                 f"population | crude vs adjusted | {age_label}\n{adj_note}",
                 fontsize=9.5, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        rows = []
        for exp_label, outcome_list in current_alt_exposures_7().items():
            if outcome in outcome_list:
                continue   # exposure == outcome; skip self-pair (as in B1)
            res = adjusted_irr_results(frame, outcome_list, outcome) if outcome_list != TRAUMA_EXPOSURE_OUTCOMES \
                else results[outcome]
            rows.append((exp_label, None, None, None, None))
            for kind in ["crude", "adjusted"]:
                r = _exposure_row(res[("overall", kind)], "exposed")
                rows.append((f"   {kind}",) + ((r["irr"], r["lo"], r["hi"]) if r is not None
                                                else (np.nan, np.nan, np.nan)) + (ADJ_COLORS[kind],))
        draw_forest(ax, rows, f"-> {outcome.upper()}")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(wspace=0.55)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# DESCRIPTIVE COMPANION PDF -- written next to every presentation_summary_*.pdf as
# presentation_summary_*_descriptive.pdf. No statistical tests: counts, person-years, follow-up,
# distributions -- the raw material behind each slide, so every slide's numbers can be read in
# context. Every page names the slide(s) it explains (numbering = the slide guide document).
# =============================================================================
FOLLOWUP_END_REASONS = ["2nd diagnosis (event)", "window ended", "left Clalit (membership end)",
                        "study end (30.6.2026)"]
FOLLOWUP_END_COLORS = ["#e63946", "#95a5a6", "#457b9d", "#f4a261"]


def fmt_cell(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.1f}"


def draw_table_page(pdf, title, df, note=None, font_size=8):
    """One PDF page holding `df` as a table (index becomes the first column)."""
    table_df = df.reset_index()
    n_rows = len(table_df)
    fig, ax = plt.subplots(figsize=(14, 0.6 + 0.3 * (n_rows + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=[[fmt_cell(v) for v in row] for row in table_df.itertuples(index=False)],
                   colLabels=[str(c) for c in table_df.columns], loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.auto_set_column_width(col=list(range(len(table_df.columns))))
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e9ecef"); cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left", fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    if note:
        ax.text(0, -0.04, note, transform=ax.transAxes, fontsize=7.5, ha="left", va="top", wrap=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def exposed_followup_detail(entry_dates, event_dates, window_days=None):
    """Per at-risk exposed patient (prevalent cases removed): entry, end, months followed, whether
    the follow-up ended in an event, and why it ended. Same rules as person_time_events_and_years().
    Returns (DataFrame, n_prevalent)."""
    window_days = FIXED_WINDOW_DAYS if window_days is None else window_days
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
                           "months": (end - entry).dt.days / 30.44})
    return detail, int(prevalent.sum())


def trauma_component_of_entry(frame, entry):
    """Which trauma diagnosis opened the entry (e.g. 'PTSD' / 'ASR' / 'both same day')."""
    comps = [o for o in TRAUMA_EXPOSURE_OUTCOMES if f"first_{o}" in frame.columns]
    hits = pd.DataFrame({o: frame.loc[entry.index, f"first_{o}"] == entry for o in comps})
    label = pd.Series("other", index=entry.index, dtype=object)
    for o in comps:
        label[hits[o]] = o.upper()
    if len(comps) > 1:
        label[hits.sum(axis=1) > 1] = "both same day"
    return label


def desc_overview_page(pdf, frame, age_label):
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    exposed = entry.notna()
    general = first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES).isna()
    pre, post = exposed & (entry < OCT7), exposed & (entry >= OCT7)
    left_early = frame["fu_end"] < STUDY_END
    female = frame["gender"].astype(str).str.lower().str.startswith(("f", "נ"))
    rows = {}
    for name, mask in [("All patients in age band", pd.Series(True, index=frame.index)),
                       (f"Exposed: {TRAUMA_EXPOSURE_LABEL} (any time)", exposed),
                       (f"  {TRAUMA_EXPOSURE_LABEL} before Oct 7", pre),
                       (f"  {TRAUMA_EXPOSURE_LABEL} on/after Oct 7", post),
                       (f"General (never {TRAUMA_EXPOSURE_LABEL})", general),
                       (f"Excluded ({TRAUMA_EXPOSURE_LABEL} before age 15"
                        + ("" if CURRENT_DIAGNOSER_MODE == "any_diagnoser" else " / other diagnoser") + ")",
                        ~exposed & ~general)]:
        sub = frame[mask]
        rows[name] = {"n patients": int(mask.sum()),
                      "% of age band": mask.mean() * 100,
                      "median age at Oct 7": sub["age_at_oct7"].median(),
                      "% female": female[mask].mean() * 100 if mask.any() else np.nan,
                      "% left Clalit before 30.6.2026": left_early[mask].mean() * 100 if mask.any() else np.nan}
    df = pd.DataFrame(rows).T
    df.index.name = "Group"
    draw_table_page(pdf, f"DESCRIPTIVE | Overview (all slides) | who is in this file | {age_label}", df,
                    note=f"Age = age on Oct 7 2023. Exposure = earliest {TRAUMA_EXPOSURE_LABEL} diagnosis "
                         f"({CURRENT_DIAGNOSER_LABEL}). Data from {REGISTRY_START:%d.%m.%Y} to "
                         f"{STUDY_END:%d.%m.%Y}. '% female' assumes the gender column starts with F/נ for "
                         f"women -- check against the real coding.")

    # which trauma diagnosis opened the entry, pre vs post
    comp = trauma_component_of_entry(frame, entry[exposed])
    side = np.where(entry[exposed] < OCT7, "before Oct 7", "on/after Oct 7")
    mix = pd.crosstab(comp, side)
    mix_pct = mix / mix.sum() * 100
    mix_df = pd.concat({"n": mix, "% of column": mix_pct}, axis=1)
    mix_df.columns = [f"{a}: {b}" for a, b in mix_df.columns]
    mix_df.index.name = "First trauma diagnosis"
    draw_table_page(pdf, f"DESCRIPTIVE | Overview | which trauma diagnosis opened the entry, "
                         f"before vs after Oct 7 | {age_label}", mix_df,
                    note="A shift toward ASR after Oct 7 means the post-war exposed group is made of "
                         "more acute, often transient diagnoses -- relevant to every pre/post comparison.")


DESC_AGE_BANDS = [(15, 25, "15-24"), (25, 35, "25-34"), (35, 45, "35-44"), (45, 55, "45-54"),
                  (55, 65, "55-64"), (65, 200, "65+")]


def desc_population_table_page(pdf, frame, age_label):
    """'Table 1': the population behind the adjusted model, by group -- every covariate the model
    adjusts for, n (% of column). Plus person-years / events actually entering the model."""
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    general = first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES).isna()
    groups = {"General population": general,
              f"{TRAUMA_EXPOSURE_LABEL} before Oct 7": entry.notna() & (entry < OCT7),
              f"{TRAUMA_EXPOSURE_LABEL} on/after Oct 7": entry.notna() & (entry >= OCT7)}
    covs = adjustment_covariates(frame)
    age_band = pd.Series("missing", index=frame.index, dtype=object)
    for lo, hi, label in DESC_AGE_BANDS:
        age_band[(frame["age_at_oct7"] >= lo) & (frame["age_at_oct7"] < hi)] = label
    covs.insert(0, "Age at Oct 7", age_band)

    rows = {}
    rows[("", "n patients")] = {g: f"{int(m.sum()):,}" for g, m in groups.items()}
    rows[("", "age at Oct 7, median [IQR]")] = {
        g: (f"{frame.loc[m, 'age_at_oct7'].median():.1f} [{frame.loc[m, 'age_at_oct7'].quantile(.25):.1f}-"
            f"{frame.loc[m, 'age_at_oct7'].quantile(.75):.1f}]" if m.any() else "-")
        for g, m in groups.items()}
    for var in covs.columns:
        levels = covs[var].value_counts().index.tolist()
        levels = [l for l in levels if l != "missing"] + (["missing"] if "missing" in levels else [])
        if var == "Age at Oct 7":
            levels = [l for _, _, l in DESC_AGE_BANDS if l in levels] + [l for l in levels if l == "missing"]
        elif var in ("SES (5-level)", "Charlson"):   # ordinal -- in order, not by size
            levels = sorted(l for l in levels if l != "missing") + [l for l in levels if l == "missing"]
        for level in levels:
            rows[(var, level)] = {g: (f"{int((covs.loc[m, var] == level).sum()):,} "
                                      f"({(covs.loc[m, var] == level).mean() * 100:.1f}%)" if m.any() else "-")
                                  for g, m in groups.items()}
    df = pd.DataFrame(rows).T
    df.index = [f"{a}: {rtl(b)}" if a else b for a, b in df.index]
    df.index.name = "Characteristic"
    draw_table_page(pdf, f"DESCRIPTIVE | Adjusted-IRR slides | the population used, by group | {age_label}",
                    df, font_size=7,
                    note=f"Patients counted by group as in Statement 1 (exposure = earliest {TRAUMA_EXPOSURE_LABEL}, "
                         f"{CURRENT_DIAGNOSER_LABEL}). Patients diagnosed before Oct 7 and their pre-trauma time "
                         f"also contribute to the general population's person-time in the model (dynamic "
                         f"exposure) -- these columns count each patient once, by their own group. Sex, sector, "
                         f"SES and Charlson are the model's covariates as coded there; age in the model is "
                         f"ATTAINED age (age during each stretch of person-time), shown here as age at Oct 7.")

    # person-years and events that actually enter the model, per exposure group and outcome
    rows = {}
    for outcome in FOCUS_OUTCOMES:
        table, n_no_birth = build_poisson_table(frame, TRAUMA_EXPOSURE_OUTCOMES, outcome)
        for grp in ["general", "before Oct 7", "on/after Oct 7"]:
            t = table[table["exposure"] == grp]
            py, ev = t["py"].sum(), t["events"].sum()
            rows[f"{outcome.upper()} | {grp}"] = {
                "person-years": py, "events": int(ev),
                "crude rate / 1,000 PY": ev / py * 1000 if py > 0 else np.nan,
                "patients left out (no birth date)": n_no_birth}
    df = pd.DataFrame(rows).T
    df.index.name = "Outcome | group"
    draw_table_page(pdf, f"DESCRIPTIVE | Adjusted-IRR slides | person-years and events in the Poisson model "
                         f"| {age_label}", df,
                    note="Same person-time as the crude IRR slides (window, membership end, age 15+, prevalent "
                         "cases excluded); 'general' includes the pre-exposure time of patients who are later "
                         "exposed.")


def desc_statement1_pages(pdf, frame, age_label):
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    win = FIXED_WINDOW_DAYS // 30
    rows, prepost_rows = {}, {}
    for o in S1_DOWNSTREAM:
        down = frame[f"first_{o}"]
        det, n_prev = exposed_followup_detail(entry, down)
        g_rate, g_ev, g_py = compute_general_rate_overall(frame, TRAUMA_EXPOSURE_OUTCOMES, o)
        py = det["months"].sum() * 30.44 / 365.25
        rows[o.upper()] = {"exposed (any)": int(entry.notna().sum()), "prior cases excluded": n_prev,
                           "n at risk": len(det), "events": int(det["event"].sum()),
                           "person-years": py, "rate /1,000 PY": det["event"].sum() / py * 1000 if py else np.nan,
                           "median months followed": det["months"].median(),
                           "general events": int(g_ev), "general PY": g_py, "general rate": g_rate}
        r = {}
        for side_name, side_mask in [("pre", det["entry"] < OCT7), ("post", det["entry"] >= OCT7)]:
            sub = det[side_mask]
            spy = sub["months"].sum() * 30.44 / 365.25
            r[f"{side_name}: at risk"] = len(sub)
            r[f"{side_name}: events"] = int(sub["event"].sum())
            r[f"{side_name}: PY"] = spy
            r[f"{side_name}: rate"] = sub["event"].sum() / spy * 1000 if spy else np.nan
            r[f"{side_name}: median months"] = sub["months"].median()
        prepost_rows[o.upper()] = r
    df = pd.DataFrame(rows).T; df.index.name = "Outcome"
    draw_table_page(pdf, f"DESCRIPTIVE | Slide 1 (H2) | numbers behind each bar, within {win}mo | {age_label}",
                    df, font_size=7.5,
                    note="rate = events / person-years x 1,000; IRR on slide 1 = rate / general rate. "
                         "'prior cases excluded' = already had that outcome on/before the trauma date. "
                         "General PY = person-years of never-exposed time since 2017 (no window).")
    df2 = pd.DataFrame(prepost_rows).T; df2.index.name = "Outcome"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 2-3 (+4) | before vs after Oct 7, numbers behind each "
                         f"cell | {age_label}", df2, font_size=7.5,
                    note="Both sides are divided by the SAME general rate on slides 2-3 (see slide 1 table). "
                         "Slide 4 (TEST 2b) = post rate / pre rate from these two columns directly.")

    # why did follow-up end -- pre vs post, per focus outcome
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"DESCRIPTIVE | Slides 1-4, 17-19 | why each exposed patient's follow-up ended "
                 f"(within {win}mo) | {age_label}", fontsize=11, fontweight="bold")
    for ax, o in zip(axes, FOCUS_OUTCOMES):
        det, _ = exposed_followup_detail(entry, frame[f"first_{o}"])
        sides = [("before Oct 7", det[det["entry"] < OCT7]), ("on/after Oct 7", det[det["entry"] >= OCT7])]
        bottom = np.zeros(len(sides))
        for reason, color in zip(FOLLOWUP_END_REASONS, FOLLOWUP_END_COLORS):
            vals = np.array([(s["reason"] == reason).mean() * 100 if len(s) else 0 for _, s in sides])
            ax.bar(range(len(sides)), vals, bottom=bottom, color=color, label=reason, width=0.55)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 3:
                    ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
            bottom += vals
        ax.set_xticks(range(len(sides)))
        ax.set_xticklabels([f"{n}\n(n at risk={len(s):,})" for n, s in sides], fontsize=8)
        ax.set_ylabel("% of at-risk exposed patients", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(o.upper(), fontsize=10, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # how long were they followed -- pre vs post
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"DESCRIPTIVE | Slides 1-4, 16-19 | months of follow-up per exposed patient "
                 f"(max {win}) | {age_label}", fontsize=11, fontweight="bold")
    bins = np.arange(0, win + 1.01, 1)
    for ax, o in zip(axes, FOCUS_OUTCOMES):
        det, _ = exposed_followup_detail(entry, frame[f"first_{o}"])
        for name, sub, color in [("before Oct 7", det[det["entry"] < OCT7], "#95a5a6"),
                                 ("on/after Oct 7", det[det["entry"] >= OCT7], "#e63946")]:
            if len(sub):
                ax.hist(sub["months"], bins=bins, alpha=0.55, color=color,
                        weights=np.full(len(sub), 100 / len(sub)),
                        label=f"{name} (n={len(sub):,}, median {sub['months'].median():.1f}mo)")
        ax.set_xlabel("Months followed", fontsize=8)
        ax.set_ylabel("% of group", fontsize=8)
        ax.set_title(o.upper(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # monthly new trauma entries, by which diagnosis opened them
    exposed_entry = entry.dropna()
    comp = trauma_component_of_entry(frame, exposed_entry)
    monthly = pd.crosstab(exposed_entry.dt.to_period("M"), comp)
    monthly = monthly.reindex(pd.period_range("2017-01", STUDY_END.strftime("%Y-%m"), freq="M"), fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(monthly))
    bottom = np.zeros(len(monthly))
    for i, col in enumerate(monthly.columns):
        ax.bar(x, monthly[col].values, bottom=bottom, width=1.0, label=col,
               color=["#e63946", "#457b9d", "#f4a261", "#95a5a6"][i % 4])
        bottom += monthly[col].values
    oct7_x = list(monthly.index).index(pd.Period("2023-10", freq="M"))
    ax.axvline(oct7_x, color="black", linestyle="--", linewidth=1.2)
    year_ticks = [i for i, p in enumerate(monthly.index) if p.month == 1]
    ax.set_xticks(year_ticks); ax.set_xticklabels([str(monthly.index[i].year) for i in year_ticks], fontsize=8)
    ax.set_ylabel("New exposed patients per month", fontsize=8)
    ax.set_title(f"DESCRIPTIVE | Slides 2-3, 9, 20-21 | new {TRAUMA_EXPOSURE_LABEL} entries per month "
                 f"(first-ever diagnosis) | {age_label}", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def desc_old_method_tests_page(pdf, frame, entry_meta, age_label):
    """Slides 5-8 use the older full-window design -- show who that design keeps and the base mix
    (all eligible exposed, not just converters) of profession / setting / zone, pre vs post."""
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    eligible = entry.notna() & eligible_reachable_mask(entry)
    last_entry = STUDY_END - pd.to_timedelta(FIXED_WINDOW_DAYS, unit="D")
    rows = {}
    for side_name, side_mask in [("before Oct 7", entry < OCT7), ("on/after Oct 7", entry >= OCT7)]:
        r = {"exposed": int((entry.notna() & side_mask).sum()),
             "kept (full window by 30.6.2026)": int((eligible & side_mask).sum())}
        for o in FOCUS_OUTCOMES:
            conv = eligible & side_mask & converted_within_window(entry, frame[f"first_{o}"])
            r[f"converters: {o}"] = int(conv.sum())
        rows[side_name] = r
    df = pd.DataFrame(rows).T; df.index.name = "Trauma diagnosed"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 5-8 | sample sizes under the full-window design | {age_label}",
                    df, note=f"Slides 5-8 keep only patients diagnosed by {last_entry:%d.%m.%Y} (full "
                             f"{FIXED_WINDOW_DAYS//30}-month window before study end), count a share of "
                             f"patients (not person-years), and do not stop at membership end.")

    profession = normalize_profession_code(
        get_exposure_meta_field(entry_meta, TRAUMA_EXPOSURE_OUTCOMES, "profession"))
    prof_label = profession.map(lambda c: "missing / 999" if pd.isna(c) or c in MISSING_PROFESSION_CODES
                                else profession_code_label(c))
    heavy = classify_trauma_engagement(entry_meta).fillna(False).map(
        {True: "hospital setting", False: "other setting"})
    side = pd.Series(np.where(entry < OCT7, "before Oct 7", "on/after Oct 7"), index=frame.index)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    fig.suptitle(f"DESCRIPTIVE | Slides 5, 7, 8 | base mix of ALL kept exposed patients (not only "
                 f"converters), before vs after Oct 7 | {age_label}", fontsize=11, fontweight="bold")
    for ax, (title, series) in zip(axes, [("Profession of trauma diagnoser (slide 5)", prof_label),
                                          ("Setting of trauma diagnosis (slide 7)", heavy),
                                          ("Zone of residence (slide 8)", frame["zone"])]):
        tab = pd.crosstab(series[eligible], side[eligible])
        if title.startswith("Profession"):
            top = tab.sum(axis=1).sort_values(ascending=False).index[:6]
            other = tab.drop(index=top).sum()
            tab = tab.loc[top]
            if other.sum() > 0:
                tab.loc["other"] = other
        pct = tab / tab.sum() * 100
        bottom = np.zeros(pct.shape[1])
        for i, cat in enumerate(pct.index):
            ax.bar(range(pct.shape[1]), pct.loc[cat].values, bottom=bottom, label=str(cat)[:28],
                   color=plt.cm.tab10(i % 10), width=0.55)
            bottom += pct.loc[cat].values
        ax.set_xticks(range(pct.shape[1]))
        ax.set_xticklabels([f"{c}\n(n={int(tab[c].sum()):,})" for c in pct.columns], fontsize=8)
        ax.set_ylim(0, 100); ax.set_ylabel("% of kept exposed patients", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def desc_yearly_pages(pdf, frame, age_label):
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    for o in FOCUS_OUTCOMES:
        down = frame[f"first_{o}"]
        gen = compute_general_lexis_split(frame, TRAUMA_EXPOSURE_OUTCOMES, o, YEARLY_TREND_PERIODS)
        rows = {}
        for label, start, end in YEARLY_TREND_PERIODS:
            period_entry = entry.where((entry >= start) & (entry < end))
            det, n_prev = exposed_followup_detail(period_entry, down)
            py = det["months"].sum() * 30.44 / 365.25
            rows[label.replace("\n", " ")] = {
                "new exposed": int(period_entry.notna().sum()), "prior cases excl.": n_prev,
                "at risk": len(det), "events": int(det["event"].sum()), "PY": py,
                "rate": det["event"].sum() / py * 1000 if py else np.nan,
                "general events": int(gen.loc[label, "n_events"]), "general PY": gen.loc[label, "person_years"],
                "general rate": gen.loc[label, "rate"]}
        df = pd.DataFrame(rows).T; df.index.name = "Year of trauma dx"
        draw_table_page(pdf, f"DESCRIPTIVE | Slides 9, 20, 21 | {o.upper()} by year of {TRAUMA_EXPOSURE_LABEL} "
                             f"diagnosis | {age_label}", df, font_size=7.5,
                        note="Each point on slides 9/20 = rate / general rate of that row. Rows with few "
                             "events (or little PY, e.g. 2026) give unstable IRRs.")


def desc_second_dx_year_page(pdf, frame, age_label):
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    rows = {}
    for o in FOCUS_OUTCOMES:
        down = frame[f"first_{o}"]
        for side_name, side_entry in [("trauma pre", entry.where(entry < OCT7)),
                                      ("trauma post", entry.where(entry >= OCT7))]:
            ev, py = split_person_time_by_period(side_entry, down, FIXED_WINDOW_DAYS, SECOND_DIAGNOSIS_PERIODS)
            rows[f"{o.upper()} | {side_name} | events"] = {k.replace("\n", " "): v for k, v in ev.items()}
            rows[f"{o.upper()} | {side_name} | PY"] = {k.replace("\n", " "): v for k, v in py.items()}
    df = pd.DataFrame(rows).T; df.index.name = "Outcome | group | measure"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 10-15 | events and person-years by calendar period of the "
                         f"2nd diagnosis | {age_label}", df, font_size=7,
                    note="Slides 11-12 hide periods with fewer than 10 / 30 events; slides 13-15 box each "
                         "period's IRR. Box 1 = 'trauma pre' periods before Oct 7, box 2 = 'trauma post', "
                         "box 3 = 'trauma pre' periods from Oct 7.")


def desc_patient_level_page(pdf, frame, age_label):
    rows = {}
    box_names = {1: "box 1: trauma & 2nd dx before Oct 7", 2: "box 2: both on/after Oct 7",
                 3: "box 3: trauma before, 2nd dx after"}
    for o in FOCUS_OUTCOMES:
        for g, (side_t, side_e) in {1: ("pre", "pre"), 2: ("post", "post"), 3: ("pre", "post")}.items():
            days = compute_time_to_second_dx(frame, o, side_t, side_e)
            rows[f"{o.upper()} | {box_names[g]}"] = {
                "converters": len(days),
                "median days": np.median(days) if len(days) else np.nan,
                "Q1 days": np.percentile(days, 25) if len(days) else np.nan,
                "Q3 days": np.percentile(days, 75) if len(days) else np.nan}
        for g in (1, 2, 3):
            t_enter, t_exit, is_event = km_intervals(frame, o, g)
            rows[f"{o.upper()} | KM curve {g}"] = {
                "KM patients": len(t_exit), "KM events": int(is_event.sum()),
                "KM censored": int((~is_event).sum()),
                "median days at risk": np.median(t_exit - t_enter) if len(t_exit) else np.nan}
    df = pd.DataFrame(rows).T; df.index.name = "Outcome | group"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 16-19 | patient-level groups: converters (box plot) and "
                         f"Kaplan-Meier curves | {age_label}", df, font_size=7.5,
                    note="KM curve 1 = trauma before Oct 7 followed until Oct 7; curve 2 = trauma on/after "
                         "Oct 7; curve 3 = trauma before Oct 7 followed from Oct 7 (delayed entry).")


def desc_statement2_page(pdf, frame, age_label):
    rows = {}
    for exp_label, outcome_list in current_alt_exposures_7().items():
        entry = first_of(frame, outcome_list)
        r = {"exposed": int(entry.notna().sum()),
             "before Oct 7": int((entry < OCT7).sum()), "on/after Oct 7": int((entry >= OCT7).sum())}
        for o in FOCUS_OUTCOMES:
            if o in outcome_list:
                r[f"{o}: at risk"] = r[f"{o}: events"] = "self"
                continue
            det, _ = exposed_followup_detail(entry, frame[f"first_{o}"])
            r[f"{o}: at risk"] = len(det)
            r[f"{o}: events"] = int(det["event"].sum())
            r[f"{o}: events pre / post"] = (f"{int(det.loc[det['entry'] < OCT7, 'event'].sum()):,} / "
                                            f"{int(det.loc[det['entry'] >= OCT7, 'event'].sum()):,}")
        rows[exp_label] = r
    df = pd.DataFrame(rows).T; df.index.name = "Entry diagnosis"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 22-25 | each entry diagnosis: group sizes and events "
                         f"within {FIXED_WINDOW_DAYS//30}mo | {age_label}", df, font_size=7.5,
                    note="Each row's general population = never diagnosed with THAT entry diagnosis, so it "
                         "differs per row. 'self' = the entry diagnosis is the outcome itself (not shown).")


def desc_statement3_pages(pdf, frame, age_label):
    q_index = pd.period_range("2017Q1", "2026Q2", freq="Q")
    py = quarterly_person_years(frame, q_index)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(q_index))
    for z in ZONE_ORDER:
        ax.plot(x, py[z].values, marker="o", markersize=3, label=ZONE_LABELS[z], color=ZONE_COLORS[z])
    oct7_pos = list(q_index).index(pd.Period("2023Q4", freq="Q"))
    ax.axvline(oct7_pos - 0.5, color="black", linestyle="--", linewidth=1.2)
    year_ticks = [i for i, q in enumerate(q_index) if q.quarter == 1]
    ax.set_xticks(year_ticks); ax.set_xticklabels([str(q_index[i].year) for i in year_ticks], fontsize=8)
    ax.set_ylabel("Person-years of members per quarter", fontsize=8)
    ax.set_title(f"DESCRIPTIVE | Slides 26-34 | the denominator: person-years under follow-up per quarter, "
                 f"by zone | {age_label}", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    rows = {}
    for z in ZONE_ORDER:
        m = frame["zone"] == z
        r = {"patients": int(m.sum()), "% of age band": m.mean() * 100,
             f"% ever {TRAUMA_EXPOSURE_LABEL}": entry[m].notna().mean() * 100 if m.any() else np.nan}
        for item in ["ptsd", "asr", "psychosis", "bipolar"]:
            if f"first_{item}" not in frame.columns:
                continue
            d = frame.loc[m, f"first_{item}"]
            # same rule as Statement 3: only diagnoses made during follow-up (from age 15 / 2017)
            in_fu = (d >= frame.loc[m, "fu_start"]) & (d <= frame.loc[m, "fu_end"])
            r[f"{item}: new dx pre"] = int((in_fu & (d < OCT7)).sum())
            r[f"{item}: new dx post"] = int((in_fu & (d >= OCT7)).sum())
        rows[ZONE_LABELS[z]] = r
    df = pd.DataFrame(rows).T; df.index.name = "Zone"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 26-34 | zone sizes and raw counts of new diagnoses | {age_label}",
                    df, font_size=7.5,
                    note=f"pre = {REGISTRY_START:%d.%m.%Y} to 6.10.2023 (~6.8 years); post = 7.10.2023 to "
                         f"{STUDY_END:%d.%m.%Y} (~2.7 years) -- compare the slides' rates, not these raw counts.")


def desc_statement5_page(pdf, outcome_meta, age_label):
    rows = {}
    for o in ALL_OUTCOMES:
        dates = outcome_meta[o]["date"]
        codes = normalize_profession_code(outcome_meta[o]["profession"])
        r = {}
        for side_name, side_mask in [("pre", dates < OCT7), ("post", dates >= OCT7)]:
            m = dates.notna() & side_mask
            n = int(m.sum())
            r[f"{side_name}: first dx"] = n
            r[f"{side_name}: % code 999"] = codes[m].isin(MISSING_PROFESSION_CODES).mean() * 100 if n else np.nan
            r[f"{side_name}: % no profession"] = codes[m].isna().mean() * 100 if n else np.nan
        rows[o.upper()] = r
    df = pd.DataFrame(rows).T; df.index.name = "Diagnosis"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 35-36 | first-ever diagnoses and missing diagnoser "
                         f"profession | {age_label}", df,
                    note="Slide 35 drops code 999 from the denominator, slide 36 counts it as "
                         "non-psychiatrist -- the larger '% code 999', the more the two slides differ.")


def desc_statement6_page(pdf, frame, age_label):
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES)
    rows = {}
    for o in S1_DOWNSTREAM:
        down = frame[f"first_{o}"]
        for side_name, side_mask in [("pre", entry < OCT7), ("post", entry >= OCT7)]:
            conv = entry.notna() & side_mask & down.notna() & (down > entry)
            lat = (down[conv] - entry[conv]).dt.days / 30.44
            n = len(lat)
            rows[f"{o.upper()} | trauma {side_name}"] = {
                "exposed": int((entry.notna() & side_mask).sum()), "converters (any time)": n,
                "median months": lat.median() if n else np.nan,
                "% within 1 month": (lat < 1).mean() * 100 if n else np.nan,
                f"% within {FIXED_WINDOW_DAYS//30} months": (lat < FIXED_WINDOW_DAYS / 30.44).mean() * 100 if n else np.nan,
                "max months possible": (STUDY_END - entry[entry.notna() & side_mask].min()).days / 30.44
                                       if (entry.notna() & side_mask).any() else np.nan}
    df = pd.DataFrame(rows).T; df.index.name = "Outcome | trauma timing"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 37-42 | latency summary (no window, no membership cap) | "
                         f"{age_label}", df, font_size=7.5,
                    note="Post-Oct7 trauma patients can be followed at most ~33 months, so their latency "
                         "distribution is cut short by design -- compare shapes up to that point only.")


def write_descriptive_pdf(output_pdf, frame, entry_meta, outcome_profession_meta, age_label):
    with PdfPages(output_pdf) as pdf:
        desc_overview_page(pdf, frame, age_label)
        desc_population_table_page(pdf, frame, age_label)
        desc_statement1_pages(pdf, frame, age_label)
        desc_old_method_tests_page(pdf, frame, entry_meta, age_label)
        desc_yearly_pages(pdf, frame, age_label)
        desc_second_dx_year_page(pdf, frame, age_label)
        desc_patient_level_page(pdf, frame, age_label)
        desc_statement2_page(pdf, frame, age_label)
        desc_statement3_pages(pdf, frame, age_label)
        desc_statement5_page(pdf, outcome_profession_meta, age_label)
        desc_statement6_page(pdf, frame, age_label)
    print(f"  Saved: {output_pdf}")


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
}
EXTRA_POPULATIONS = ["gaza20km", "over20km_nonorth"]   # run for the SLIM combination only


def apply_population(frame, entry_meta, downstream_meta, outcome_profession_meta, population_key):
    mask_fn = POPULATIONS[population_key][2]
    if mask_fn is None:
        return frame, entry_meta, downstream_meta, outcome_profession_meta
    mask = mask_fn(frame).fillna(False).astype(bool)
    return (frame[mask].copy(),
            {o: d.loc[mask].copy() for o, d in entry_meta.items()},
            {o: d.loc[mask].copy() for o, d in downstream_meta.items()},
            {o: d.loc[mask].copy() for o, d in outcome_profession_meta.items()})


def export_all(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full,
               output_dir, file_suffix="", age_keys=None, population_key="all"):
    os.makedirs(output_dir, exist_ok=True)
    pop_label, pop_suffix, _ = POPULATIONS[population_key]
    for age_key, (age_label, lo, hi) in AGE_BANDS.items():
        if age_keys is not None and age_key not in age_keys:
            continue
        frame, entry_meta, downstream_meta, outcome_profession_meta = apply_age(
            frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full, lo, hi)
        frame, entry_meta, downstream_meta, outcome_profession_meta = apply_population(
            frame, entry_meta, downstream_meta, outcome_profession_meta, population_key)
        label = (f"{age_label} | {CURRENT_DIAGNOSER_LABEL}"
                if CURRENT_DIAGNOSER_MODE != "any_diagnoser" else age_label)
        if pop_label:
            label = f"{label} | {pop_label}"
        print(f"\n=== {age_label} | trauma={TRAUMA_EXPOSURE_LABEL} | window={FIXED_WINDOW_DAYS}d | "
             f"diagnoser={CURRENT_DIAGNOSER_LABEL} | population={pop_label or 'all'} | n={len(frame):,} ===")
        output_pdf = os.path.join(output_dir, f"presentation_summary_{age_key}{file_suffix}{pop_suffix}.pdf")
        with PdfPages(output_pdf) as pdf:
            print("  Statement 1...")
            write_statement1(pdf, frame, label)
            print("  Statement 1 (supporting tests)...")
            write_statement1_supporting(pdf, frame, entry_meta, downstream_meta, label)
            print("  Statement 2...")
            write_statement2(pdf, frame, label)
            print("  Statement 3...")
            write_statement3(pdf, frame, label)
            print("  Statement 5...")
            write_statement5(pdf, outcome_profession_meta, label)
            print("  Statement 6...")
            write_statement6(pdf, frame, label)
            print("  Statement 2 (extended, appended at the end)...")
            write_statement2_extended(pdf, frame, label)
            print("  Comorbidity (appended at the end)...")
            write_comorbidity_statement(pdf, frame, label)
            print("  Adjusted IRR forest plots (appended at the end)...")
            write_adjusted_forest_pages(pdf, frame, label)
        print(f"  Saved: {output_pdf}")
        print("  Descriptive companion...")
        write_descriptive_pdf(output_pdf.replace(".pdf", "_descriptive.pdf"),
                              frame, entry_meta, outcome_profession_meta, label)


def set_trauma_mode(mode_key):
    global TRAUMA_EXPOSURE_OUTCOMES, TRAUMA_EXPOSURE_LABEL
    TRAUMA_EXPOSURE_OUTCOMES, TRAUMA_EXPOSURE_LABEL = TRAUMA_MODES[mode_key]


# --- Run the full pipeline 3 times -- once per TRAUMA_MODES entry (merged PTSD/ASR, PTSD
# only, ASR only) -- writing 6 separate PDFs total (3 modes x 2 age bands), each name-tagged
# with its mode so they never overwrite each other:
#   presentation_summary_age15_40_ptsd_asr.pdf   / presentation_summary_age15_plus_ptsd_asr.pdf
#   presentation_summary_age15_40_ptsd_only.pdf  / presentation_summary_age15_plus_ptsd_only.pdf
#   presentation_summary_age15_40_asr_only.pdf   / presentation_summary_age15_plus_asr_only.pdf
# The CSV is read only ONCE (data loading doesn't depend on trauma mode); only the analysis
# functions re-run per mode, reading TRAUMA_EXPOSURE_OUTCOMES/LABEL each time.
def export_all_trauma_modes(input_file, output_dir):
    frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full = \
        load_and_build(input_file)
    for mode_key in TRAUMA_MODES:
        set_trauma_mode(mode_key)
        print(f"\n{'='*70}\nTRAUMA MODE: {mode_key} ({TRAUMA_EXPOSURE_LABEL})\n{'='*70}")
        export_all(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full,
                  output_dir, file_suffix=f"_{mode_key}")


# --- Run the full pipeline 3 times -- once per DIAGNOSER_MODES entry (any diagnoser,
# psychiatrist only, non-psychiatrist only) -- for the SINGLE trauma mode named by
# SELECTED_TRAUMA_MODE above. Writes 6 separate PDFs (2 age bands x 3 diagnoser modes):
#   presentation_summary_age15_40_<trauma>_any_diagnoser.pdf
#   presentation_summary_age15_40_<trauma>_psychiatrist_only.pdf
#   presentation_summary_age15_40_<trauma>_non_psychiatrist_only.pdf
#   (and the same 3 for age15_plus)
# To get a different trauma mode's 6 files: change SELECTED_TRAUMA_MODE above and re-run --
# this does NOT auto-loop all 3 trauma modes the way export_all_trauma_modes() does, by
# explicit request (3 trauma x 3 diagnoser x 2 age = 18 files at once was considered too many
# to generate in one run; one trauma mode's 6 files at a time is the intended workflow).
# The filter applies to EVERY exposure in the script -- Statement 1's trauma exposure AND all
# 7 of Statement 2's alternate exposures -- via first_of()/diagnoser_excluded_mask(), not just
# TEST 5/9's existing profession breakdowns.
# =============================================================================
# AUTOMATED SANITY CHECKS -- runs automatically at the end of export_all_diagnoser_modes(),
# using the already-loaded data (no reload). Prints PASS/FAIL/INFO for each check and a final
# summary; does not raise on failure (some checks are informational, not hard invariants), so
# a failed check won't stop the PDFs you already generated from being usable -- but a FAIL here
# means something in this run's numbers should not be trusted until it's investigated.
# Also writes everything printed during the run (including the nested profession/facility-type
# diagnostic printouts) to sanity_checks_report.txt in output_dir, so it's reviewable without
# having to keep the console log around.
# =============================================================================

class _TeeStdout:
    """Writes to both the real stdout and an in-memory buffer, so console output during
    run_sanity_checks() is unaffected but everything printed also ends up captured for the
    report file -- including print() calls made by functions run_sanity_checks() calls into
    (print_profession_diagnostic(), print_facility_type_diagnostic()), not just its own."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def run_sanity_checks(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full,
                      output_dir=None):
    buffer = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _TeeStdout(real_stdout, buffer)
    try:
        _run_sanity_checks_body(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full)
    finally:
        sys.stdout = real_stdout

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "sanity_checks_report.txt")
        header = (f"Sanity checks report\nGenerated: {pd.Timestamp.now().isoformat(timespec='seconds')}\n"
                 f"Trauma mode: {TRAUMA_EXPOSURE_LABEL} | Window: {FIXED_WINDOW_DAYS} days | "
                 f"Diagnoser: {CURRENT_DIAGNOSER_LABEL}\n" + "=" * 70 + "\n")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header + buffer.getvalue())
        print(f"\nSanity check report saved to: {report_path}")


def _run_sanity_checks_body(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full):
    print("\n" + "=" * 70)
    print("AUTOMATED SANITY CHECKS")
    print("=" * 70)
    results = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        results.append((name, bool(condition)))

    def info(name, detail):
        print(f"  [INFO] {name}: {detail}")

    set_trauma_mode("ptsd_asr")
    set_diagnoser_mode("any_diagnoser", outcome_profession_meta_full)
    set_window_mode("12mo")
    frame = frame_full   # full (un-age-banded) population for these checks

    # --- 1. Required derived columns exist on the loaded frame ---
    expected_cols = [f"first_{o}" for o in ALL_OUTCOMES] + ["zone", "age_at_oct7", "patient id",
                                                           "fu_start", "fu_end"]
    missing = [c for c in expected_cols if c not in frame.columns]
    check("all expected derived columns present on the loaded frame", not missing, f"missing={missing}")

    # --- 1b. Follow-up end (membership end) ---
    info("fu_end before STUDY_END (left Clalit or died)",
         f"{int((frame['fu_end'] < STUDY_END).sum()):,}")
    check("fu_end never later than STUDY_END", bool((frame["fu_end"] <= STUDY_END).all()))
    entry = first_of(frame, TRAUMA_EXPOSURE_OUTCOMES).dropna()
    info(f"{TRAUMA_EXPOSURE_LABEL} patients diagnosed AFTER membership end (contribute 0 days)",
         f"{int((entry > frame.loc[entry.index, 'fu_end']).sum()):,} of {len(entry):,}")
    raw = first_of_raw(frame, TRAUMA_EXPOSURE_OUTCOMES)
    info(f"{TRAUMA_EXPOSURE_LABEL} patients excluded (first diagnosis before age 15)",
         f"{int((raw < frame['age15_date']).sum()):,} of {int(raw.notna().sum()):,}")
    check("no exposed entry is before its patient's 15th birthday",
          not bool((entry < frame.loc[entry.index, "age15_date"]).any()))
    info("patients whose follow-up starts after 2017 (turned 15 after 1.1.2017)",
         f"{int((frame['fu_start'] > REGISTRY_START).sum()):,}")

    # --- 2. zone_populations sums to len(frame), for every age band (exhaustive/mutually
    # exclusive zone assignment -- nobody should be silently dropped) ---
    for age_key, (age_label, lo, hi) in AGE_BANDS.items():
        f_age, _, _, _ = apply_age(frame_full, entry_meta_full, downstream_meta_full,
                                   outcome_profession_meta_full, lo, hi)
        pops = zone_populations(f_age)
        check(f"zone_populations sums to len(frame) [{age_key}]",
             int(pops.sum()) == len(f_age), f"{int(pops.sum())} vs {len(f_age)}")

    # --- 3. Statement 2 B1's TRAUMA_EXPOSURE_LABEL bar matches Statement 1 H2's bar exactly --
    # both call compute_exposed_vs_general() the same way, just plotted in different charts. ---
    for outcome in FOCUS_OUTCOMES:
        _, _, comp = compute_exposed_vs_general(frame, TRAUMA_EXPOSURE_OUTCOMES, [outcome])
        rr_h2 = comp.iloc[0]["rr"]
        labels, rrs, _, _ = compute_b1_raw_rr(frame, outcome)
        if TRAUMA_EXPOSURE_LABEL in labels:
            rr_b1 = rrs[labels.index(TRAUMA_EXPOSURE_LABEL)]
            match = (pd.isna(rr_h2) and pd.isna(rr_b1)) or \
                   (pd.notna(rr_h2) and pd.notna(rr_b1) and abs(rr_h2 - rr_b1) < 1e-9)
            check(f"Statement 2 B1's {TRAUMA_EXPOSURE_LABEL} bar == Statement 1 H2 bar [{outcome}]",
                 match, f"H2={rr_h2}, B1={rr_b1}")

    # --- 4. n_gen is invariant across diagnoser modes -- the general population is defined by
    # RAW exposure status only, diagnoser mode should never change who counts as "general". ---
    n_gens = {}
    for diag_key in DIAGNOSER_MODES:
        set_diagnoser_mode(diag_key, outcome_profession_meta_full)
        n_exp, n_gen, _ = compute_exposed_vs_general(frame, TRAUMA_EXPOSURE_OUTCOMES, ["psychosis"])
        n_gens[diag_key] = n_gen
    check("n_gen is invariant across diagnoser modes", len(set(n_gens.values())) == 1, str(n_gens))
    set_diagnoser_mode("any_diagnoser", outcome_profession_meta_full)

    # --- 5. Trauma-mode headcount relationship: ptsd_only + asr_only n_exp should be >= merged
    # n_exp (merged uses min-date pooling, so anyone with EITHER counts once there but could
    # count in both single-outcome runs if they have both diagnoses -- union effect, never a
    # deficit). If merged is LARGER than the sum, TRAUMA_EXPOSURE_OUTCOMES is broken. ---
    n_exps = {}
    for trauma_key in TRAUMA_MODES:
        set_trauma_mode(trauma_key)
        n_exps[trauma_key] = int(first_of(frame, TRAUMA_EXPOSURE_OUTCOMES).notna().sum())
    set_trauma_mode("ptsd_asr")
    check("ptsd_only + asr_only n_exp >= merged (ptsd_asr) n_exp",
         n_exps["ptsd_only"] + n_exps["asr_only"] >= n_exps["ptsd_asr"], str(n_exps))

    # --- 6. Pre+post n_exp sums to exactly the pooled H2 n_exp (same exposure, same eligibility
    # rule, just split by OCT7 -- must partition exactly, no overlap or gap). ---
    n_exp_pooled, _, _ = compute_exposed_vs_general(frame, TRAUMA_EXPOSURE_OUTCOMES, ["psychosis"])
    (n_pre, _), (n_post, _), _ = compute_prepost_window_rr(
        frame, TRAUMA_EXPOSURE_OUTCOMES, ["psychosis"], FIXED_WINDOW_DAYS)
    check("pre_n_exp + post_n_exp == pooled H2 n_exp",
         (n_pre + n_post) == n_exp_pooled, f"{n_pre}+{n_post} vs {n_exp_pooled}")

    # --- 7/8. Profession & facility-type coding diagnostics (already-existing diagnostic
    # printers, now folded into the pass/fail summary instead of being a silent side effect). --
    global _profession_diagnostic_printed, _facility_type_diagnostic_printed
    _profession_diagnostic_printed = False
    any_psychiatrist_code = any(
        normalize_profession_code(outcome_profession_meta_full[o]["profession"]).dropna()
        .isin(PSYCHIATRIST_PROFESSION_CODES).any()
        for o in ["psychosis", "bipolar"])
    print_profession_diagnostic(outcome_profession_meta_full)
    check("psychiatrist profession codes (71/73) appear somewhere in the data", any_psychiatrist_code)

    _facility_type_diagnostic_printed = False
    facility_vals = get_exposure_meta_field(entry_meta_full, TRAUMA_EXPOSURE_OUTCOMES, "facility type")
    any_hospital_match = facility_vals.dropna().astype(str).apply(
        lambda v: any(p in v for p in HOSPITAL_FACILITY_PATTERNS)).any()
    print_facility_type_diagnostic(entry_meta_full)
    check("a hospital-type pattern matches at least one real facility-type value (TEST 9)",
         any_hospital_match)

    # --- 9. Regression guard for the default-parameter-capture bug (window_days=FIXED_WINDOW_DAYS
    # in a def line silently freezes at definition time) -- confirms set_window_mode() actually
    # changes eligible_reachable_mask()'s behavior at call time, not just the module constant. --
    test_date = pd.Series([pd.Timestamp("2025-06-01")])
    set_window_mode("12mo")
    result_12 = bool(eligible_reachable_mask(test_date).iloc[0])
    set_window_mode("24mo")
    result_24 = bool(eligible_reachable_mask(test_date).iloc[0])
    check("set_window_mode() propagates to eligible_reachable_mask() (no stale-default bug)",
         result_12 is True and result_24 is False, f"12mo->{result_12}, 24mo->{result_24}")
    set_window_mode("12mo")

    # --- 10. SECOND_DIAGNOSIS_PERIODS' split-2023 boundaries are exactly correct: the "pre"
    # sub-period must end exactly at Oct7, the "post" sub-period must start exactly at Oct7 --
    # this is what makes Group 1/Group 3 of the boxplot (and the line chart's period_x()
    # placement) an exact, gap-free, non-overlapping partition with nothing approximated. ---
    period_map = {label: (start, end) for label, start, end in SECOND_DIAGNOSIS_PERIODS}
    pre_end   = period_map["2023\n(pre)"][1]
    post_start = period_map["2023\n(post)"][0]
    check("SECOND_DIAGNOSIS_PERIODS' 2023 split is exact at Oct7 (pre ends == post starts == Oct7)",
         pre_end == OCT7 and post_start == OCT7, f"pre_end={pre_end}, post_start={post_start}, OCT7={OCT7}")

    # --- 11. Boxplot Group 1 (2nd dx pre-Oct7) + Group 3 (2nd dx on/after Oct7), both drawn
    # from the SAME "pre" trauma line, must exactly partition its full period set -- every
    # period is either pre-Oct7 or on/after-Oct7, none dropped, none in both. ---
    trend_pre_full = compute_by_second_diagnosis_year(frame, "psychosis", "pre")
    n_pre_labels  = int((trend_pre_full["end"] <= OCT7).sum())
    n_post_labels = int((trend_pre_full["start"] >= OCT7).sum())
    check("boxplot Group1 + Group3 periods exactly partition the 'pre' line (no gap/overlap)",
         n_pre_labels + n_post_labels == len(trend_pre_full),
         f"{n_pre_labels}+{n_post_labels} vs {len(trend_pre_full)} total periods")

    # --- 11b. Adjusted-IRR model: its CRUDE IRR (exposure term only) must equal the crude IRR of
    # Statement 1 exactly, on the same patients (those with a birth date) -- proves the model
    # sees exactly the same events and person-years as every crude slide. ---
    f_birth = frame[frame["birth"].notna()]
    for outcome in FOCUS_OUTCOMES:
        _, _, comp = compute_exposed_vs_general(f_birth, TRAUMA_EXPOSURE_OUTCOMES, [outcome])
        res = adjusted_irr_results(f_birth, TRAUMA_EXPOSURE_OUTCOMES, outcome)
        r = _exposure_row(res[("overall", "crude")], "exposed")
        irr_model = r["irr"] if r is not None else np.nan
        irr_crude = comp.iloc[0]["rr"]
        # 0.1% tolerance: the slides' general rate drops an outcome dated exactly on STUDY_END
        check(f"Poisson crude IRR == Statement 1 crude IRR (within 0.1%) [{outcome}]",
              pd.notna(irr_model) and pd.notna(irr_crude) and abs(irr_model / irr_crude - 1) < 1e-3,
              f"model={irr_model}, crude={irr_crude}")

    # --- 11c. The two geographic population variants never share a patient ---
    in_a = POPULATIONS["gaza20km"][2](frame).fillna(False)
    in_b = POPULATIONS["over20km_nonorth"][2](frame).fillna(False)
    check("<=20 km and >20 km-no-North populations are disjoint", not bool((in_a & in_b).any()),
          f"<=20km={int(in_a.sum()):,}, >20km no North={int(in_b.sum()):,}, "
          f"in neither={int((~in_a & ~in_b).sum()):,}")

    # --- 12. Informational only (data hypotheses, not code-correctness -- printed, not asserted) --
    for item in STATEMENT3_ROSE + STATEMENT3_DID_NOT_RISE:
        pass   # Statement 3's own charts already show post/pre ratios directly; nothing extra
              # to compute here without re-deriving the whole quarterly pipeline redundantly.
    info("TEST 2b vs H2 rate_pre", "expected to be in the same ballpark (H2 pools pre+post; "
        "pre-Oct7 usually dominates the pooled cohort) -- read both charts' printed rate "
        "values directly, not asserted here since 'ballpark' isn't a hard equality.")

    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print("-" * 70)
    print(f"SANITY CHECKS: {n_pass}/{n_total} passed" +
         (" -- ALL CLEAR" if n_pass == n_total else " -- SEE FAILURES ABOVE BEFORE TRUSTING THIS RUN"))
    print("=" * 70)
    return results


def export_all_diagnoser_modes(input_file, output_dir, trauma_mode=None):
    trauma_mode = trauma_mode or SELECTED_TRAUMA_MODE
    frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full = \
        load_and_build(input_file)
    set_trauma_mode(trauma_mode)
    print(f"\n{'='*70}\nTRAUMA MODE: {trauma_mode} ({TRAUMA_EXPOSURE_LABEL})\n{'='*70}")
    window_keys    = [SLIM_WINDOW] if SLIM_RUN else list(WINDOW_MODES)
    diagnoser_keys = [SLIM_DIAGNOSER] if SLIM_RUN else list(DIAGNOSER_MODES)
    age_keys       = [SLIM_AGE_BAND] if SLIM_RUN else None
    for window_key in window_keys:
        set_window_mode(window_key)
        print(f"\n{'#'*70}\nWINDOW MODE: {window_key} ({FIXED_WINDOW_DAYS} days)\n{'#'*70}")
        for diagnoser_key in diagnoser_keys:
            set_diagnoser_mode(diagnoser_key, outcome_profession_meta_full)
            print(f"\n{'-'*70}\nDIAGNOSER MODE: {diagnoser_key} ({CURRENT_DIAGNOSER_LABEL})\n{'-'*70}")
            export_all(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full,
                      output_dir, file_suffix=f"_{trauma_mode}_{window_key}_{diagnoser_key}",
                      age_keys=age_keys)

    # geographic sub-population files, for the SLIM combination (age15_40 / 24mo / any diagnoser):
    #   presentation_summary_age15_40_<trauma>_24mo_any_diagnoser_gaza20km.pdf
    #   presentation_summary_age15_40_<trauma>_24mo_any_diagnoser_over20km_nonorth.pdf
    # (+ a _descriptive.pdf next to each)
    set_window_mode(SLIM_WINDOW)
    set_diagnoser_mode(SLIM_DIAGNOSER, outcome_profession_meta_full)
    for population_key in EXTRA_POPULATIONS:
        print(f"\n{'-'*70}\nPOPULATION: {POPULATIONS[population_key][0]}\n{'-'*70}")
        export_all(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full,
                  output_dir, file_suffix=f"_{trauma_mode}_{SLIM_WINDOW}_{SLIM_DIAGNOSER}",
                  age_keys=[SLIM_AGE_BAND], population_key=population_key)

    if SLIM_RUN:
        run_sanity_checks(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full)
        set_diagnoser_mode("any_diagnoser", outcome_profession_meta_full)
        set_window_mode("12mo")
        return

    # ONE demographic-comparison PDF, not multiplied across the 12 combinations above -- uses
    # the "any_diagnoser" / trauma_mode setting (demographics of the trauma diagnosis itself,
    # not filtered by who diagnosed it).
    set_diagnoser_mode("any_diagnoser", outcome_profession_meta_full)
    write_demographic_comparison_pdf(frame_full, entry_meta_full, output_dir, trauma_mode=trauma_mode)
    write_demographic_insights_pdf(frame_full, entry_meta_full, output_dir, trauma_mode=trauma_mode)
    write_relative_year_pdf(frame_full, output_dir, trauma_mode=trauma_mode)

    run_sanity_checks(frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full,
                      output_dir=output_dir)
    # restore the module to the "selected" state in case anything runs after this interactively
    set_trauma_mode(trauma_mode)
    set_diagnoser_mode("any_diagnoser", outcome_profession_meta_full)
    set_window_mode("12mo")


# =============================================================================
# PATHS -- adjust to the actual final matched-cohort file before running
# =============================================================================

INPUT_FILE = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar\matched_cohort_FirstEverOutcomes.csv"
OUTPUT_DIR = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PDF_PostWarPsychosis\PresentationSummary"

if __name__ == "__main__":
    # SLIM_RUN=True (default): 1 file only -- see SLIM_RUN above. SLIM_RUN=False: 12 files,
    # 2 age bands x 3 diagnoser modes x 2 window modes, for SELECTED_TRAUMA_MODE above.
    # To run all 3 trauma modes without diagnoser/window variation instead (the older,
    # simpler 6-file workflow from before these features), call
    # export_all_trauma_modes(INPUT_FILE, OUTPUT_DIR) instead.
    export_all_diagnoser_modes(INPUT_FILE, OUTPUT_DIR)

# =============================================================================
# HISTORICAL NOTES -- context for past changes that aren't re-testable as automated checks
# (either because they compare against a script/design that no longer exists, or because
# they're pure narrative). The actual sanity checks now run automatically -- see
# run_sanity_checks(), called at the end of export_all_diagnoser_modes().
#
# - Every RR/IRR number in this file has changed at least twice since the original H2/B1
#   images (docx screenshots): first from unrestricted lifetime follow-up to a rate-vs-general
#   design, then from binary risk-in-a-window to restricted person-time, then from
#   anchor-averaged to continuous Lexis-tracked general rates. None of these are regressions --
#   each was an explicit, requested methodology change. Don't expect this file's numbers to
#   match any earlier run or any of the original docx images.
# - TEST 2b's rate_pre is computed differently from H2's exp_rate (2b anchors to each
#   exposure's own entry date and never touches a general-population reference at all), so
#   they won't be numerically identical -- but should land in the same ballpark, since H2
#   pools pre+post and pre-Oct7 usually dominates the pooled cohort.
# - Statement 5 and Statement 3 are IDENTICAL across all 3 trauma-mode files (neither reads
#   TRAUMA_EXPOSURE_OUTCOMES/LABEL) -- don't go looking for a difference between the
#   ptsd_asr/ptsd_only/asr_only versions of those two specifically.
# - The boxplot pages' one-tailed Welch t-test compares SETS OF YEARLY IRR VALUES (typically
#   <10 per group), not patient-level data -- likely underpowered, especially post-war (capped
#   at ~3 calendar years). Treat the p-value as suggestive, not equivalent in strength to the
#   Poisson exact tests used elsewhere in this script.
# =============================================================================
