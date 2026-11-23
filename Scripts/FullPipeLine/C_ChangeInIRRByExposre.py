# C_ChangeInIRRByExposre.py -- MAIN: run this file. The pipeline is split into 9 files that must
# sit in the same directory:
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
# The description below is the original one for the whole pipeline.
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
    AGE_BANDS, DIAGNOSER_MODES, EXTRA_POPULATIONS, POPULATIONS, SELECTED_TRAUMA_MODE,
    SLIM_AGE_BAND, SLIM_DIAGNOSER, SLIM_RUN, SLIM_WINDOW, TRAUMA_MODES, WINDOW_MODES, apply_age,
    apply_population, load_and_build, set_diagnoser_mode, set_trauma_mode, set_window_mode,
)
from C_IRR_Models import (  # noqa: F401
    _era_group_label,
)
from C_IRR_Checks import (  # noqa: F401
    run_sanity_checks,
)
from C_IRR_Tables import (  # noqa: F401
    desc_overview_page, desc_patient_level_page, desc_population_table_page,
    desc_second_dx_flow_page, desc_second_dx_year_page, desc_statement2_page, desc_statement5_page,
    desc_statement6_page, desc_yearly_pages,
)
from C_IRR_Graphs_Statement1 import (  # noqa: F401
    write_statement1, write_statement1_supporting,
)
from C_IRR_Graphs_Statements2to6 import (  # noqa: F401
    write_comorbidity_statement, write_statement2, write_statement2_extended, write_statement3,
    write_statement5, write_statement6,
)
from C_IRR_Graphs_Demog_Models import (  # noqa: F401
    desc_old_method_tests_page, desc_statement1_pages, desc_statement3_pages, section_divider,
    write_adjusted_forest_pages, write_demographic_comparison_pdf, write_demographic_insights_pdf,
    write_era_pages, write_relative_year_pdf,
)

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
# script (trauma in Statement 1, and all 8 alternate exposures in Statement
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
#           psychosis and one for bipolar, across ALL 8 possible entry
#           exposures (PTSD/ASR, Depression, Anxiety, Substance, Bipolar,
#           Psychosis, Personality, OCD -- excluding the outcome itself).
#           [the original B1 chart's source script was not available; this
#            reproduces its spirit as the multi-exposure counterpart of H2,
#            now on the same windowed footing as the rest of the script]
#   Page 2+3: pre/post-war heatmaps (whole-year window), generalized from
#           PTSD/ASR-only (TEST 2c) to all 8 alternate entry exposures, for
#           psychosis + bipolar as the two downstream columns.
#   Page 4: bar-chart view of pages 2+3's exact same data -- one panel per
#           downstream outcome, X-axis = the 8 exposures, 2 bars each
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
#   NOT-YET-CLASSIFIED section: OCD
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
#   Oct 7), % of each of the 9 outcomes' first-ever diagnoses given by a
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
# STATEMENT 6 (latency): SEVEN pages, one per non-trauma outcome (psychosis,
#   bipolar, anxiety, depression, substance, personality, OCD), each with 2 panels
#   (Pre-Oct7 / Post-Oct7 entry) -- a histogram of months from trauma entry to
#   that outcome's first diagnosis, among patients who got it. Not from the
#   docx -- added as the empirical basis for deciding whether LAG_DAYS is
#   worth setting to a nonzero value (a suspicious early spike would be
#   evidence for protopathic bias / same-episode relabeling). UNRESTRICTED by
#   window_days on purpose -- capping it would hide the very information
#   (long delays, mass outside any window) this exists to show.
# =============================================================================


def write_descriptive_pdf(output_pdf, frame, entry_meta, outcome_profession_meta, age_label):
    with PdfPages(output_pdf) as pdf:
        desc_overview_page(pdf, frame, age_label)
        desc_population_table_page(pdf, frame, age_label)
        desc_statement1_pages(pdf, frame, age_label)
        desc_old_method_tests_page(pdf, frame, entry_meta, age_label)
        desc_yearly_pages(pdf, frame, age_label)
        desc_second_dx_year_page(pdf, frame, age_label)
        desc_second_dx_flow_page(pdf, frame, age_label)
        desc_patient_level_page(pdf, frame, age_label)
        desc_statement2_page(pdf, frame, age_label)
        desc_statement3_pages(pdf, frame, age_label)
        desc_statement5_page(pdf, outcome_profession_meta, age_label)
        desc_statement6_page(pdf, frame, age_label)
    print(f"  Saved: {output_pdf}")


# =============================================================================
# MASTER EXPORT
# =============================================================================


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
        label = (f"{age_label} | {core.CURRENT_DIAGNOSER_LABEL}"
                if core.CURRENT_DIAGNOSER_MODE != "any_diagnoser" else age_label)
        if pop_label:
            label = f"{label} | {pop_label}"
        print(f"\n=== {age_label} | trauma={core.TRAUMA_EXPOSURE_LABEL} | window={core.FIXED_WINDOW_DAYS}d | "
             f"diagnoser={core.CURRENT_DIAGNOSER_LABEL} | population={pop_label or 'all'} | n={len(frame):,} ===")
        output_pdf = os.path.join(output_dir, f"presentation_summary_{age_key}{file_suffix}{pop_suffix}.pdf")
        with PdfPages(output_pdf) as pdf:
            print("  Statement 1...")
            write_statement1(pdf, frame, label)
            section_divider(pdf, "STATEMENT 1 -- SUPPORTING TESTS", label)
            print("  Statement 1 (supporting tests)...")
            write_statement1_supporting(pdf, frame, entry_meta, downstream_meta, label)
            section_divider(pdf, "STATEMENT 2", label)
            print("  Statement 2...")
            write_statement2(pdf, frame, label)
            section_divider(pdf, "STATEMENT 3", label)
            print("  Statement 3...")
            write_statement3(pdf, frame, label)
            section_divider(pdf, "STATEMENT 5", label)
            print("  Statement 5...")
            write_statement5(pdf, outcome_profession_meta, label)
            section_divider(pdf, "STATEMENT 6", label)
            print("  Statement 6...")
            write_statement6(pdf, frame, label)
            section_divider(pdf, "STATEMENT 2 -- EXTENDED", label)
            print("  Statement 2 (extended, appended at the end)...")
            write_statement2_extended(pdf, frame, label)
            section_divider(pdf, "COMORBIDITY", label)
            print("  Comorbidity (appended at the end)...")
            write_comorbidity_statement(pdf, frame, label)
            section_divider(pdf, "ADJUSTED IRR -- vs GENERAL POPULATION", label)
            print("  Adjusted IRR forest plots (appended at the end)...")
            write_adjusted_forest_pages(pdf, frame, label)
            section_divider(pdf, "ERA COMPARISON 1",
                            f"{core.TRAUMA_EXPOSURE_LABEL}: {_era_group_label('post')}  vs  {_era_group_label('pre')}")
            print("  Era comparisons (pre / post / crossing Oct 7, appended at the end)...")
            write_era_pages(pdf, frame, label)
        print(f"  Saved: {output_pdf}")
        print("  Descriptive companion...")
        write_descriptive_pdf(output_pdf.replace(".pdf", "_descriptive.pdf"),
                              frame, entry_meta, outcome_profession_meta, label)


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
        print(f"\n{'='*70}\nTRAUMA MODE: {mode_key} ({core.TRAUMA_EXPOSURE_LABEL})\n{'='*70}")
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


def export_all_diagnoser_modes(input_file, output_dir, trauma_mode=None):
    trauma_mode = trauma_mode or SELECTED_TRAUMA_MODE
    frame_full, entry_meta_full, downstream_meta_full, outcome_profession_meta_full = \
        load_and_build(input_file)
    set_trauma_mode(trauma_mode)
    print(f"\n{'='*70}\nTRAUMA MODE: {trauma_mode} ({core.TRAUMA_EXPOSURE_LABEL})\n{'='*70}")
    window_keys    = [SLIM_WINDOW] if SLIM_RUN else list(WINDOW_MODES)
    diagnoser_keys = [SLIM_DIAGNOSER] if SLIM_RUN else list(DIAGNOSER_MODES)
    age_keys       = [SLIM_AGE_BAND] if SLIM_RUN else None
    for window_key in window_keys:
        set_window_mode(window_key)
        print(f"\n{'#'*70}\nWINDOW MODE: {window_key} ({core.FIXED_WINDOW_DAYS} days)\n{'#'*70}")
        for diagnoser_key in diagnoser_keys:
            set_diagnoser_mode(diagnoser_key, outcome_profession_meta_full)
            print(f"\n{'-'*70}\nDIAGNOSER MODE: {diagnoser_key} ({core.CURRENT_DIAGNOSER_LABEL})\n{'-'*70}")
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

INPUT_FILE = r"\\10.100.117.220\Projects$\R12-MHRC\Psychosis_Oct7_6Year_Amir\sagifolder\OutPutWorkingFiles\PsychosisPostWar\matched_cohort_PeriodOutComes_Membership.csv"   # step B2 output
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
