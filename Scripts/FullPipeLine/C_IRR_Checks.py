# C_IRR_Checks.py -- LOGIC 4/4: the automated sanity checks, run at the end of
# export_all_diagnoser_modes().
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
import C_IRR_Analyses as analyses   # run-time state -- read as analyses.NAME
from C_IRR_Core import (  # noqa: F401
    AGE_BANDS, ALL_OUTCOMES, DIAGNOSER_MODES, FOCUS_OUTCOMES, MEMBERSHIP_PERIODS_COL, OCT7,
    POPULATIONS, PSYCHIATRIST_PROFESSION_CODES, REGISTRY_START, STUDY_END, TRAUMA_MODES, apply_age,
    compute_general_lexis_split, compute_general_rate_overall, first_of, first_of_raw,
    general_intervals, get_exposure_meta_field, member_days, membership_pieces,
    normalize_profession_code, parse_membership_gaps, periods_from_years,
    person_time_events_and_years, set_diagnoser_mode, set_trauma_mode, set_window_mode,
)
from C_IRR_Analyses import (  # noqa: F401
    HOSPITAL_FACILITY_PATTERNS, SECOND_DIAGNOSIS_PERIODS, compute_b1_raw_rr,
    compute_by_second_diagnosis_year, compute_exposed_vs_general, compute_prepost_window_rr,
    eligible_reachable_mask, print_facility_type_diagnostic, print_profession_diagnostic,
    zone_populations,
)
from C_IRR_Models import (  # noqa: F401
    ERA_COMPARISONS, _exposure_row, adjusted_irr_results, build_poisson_table, era_cohort,
    era_results, group_person_time,
)


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
                 f"Trauma mode: {core.TRAUMA_EXPOSURE_LABEL} | Window: {core.FIXED_WINDOW_DAYS} days | "
                 f"Diagnoser: {core.CURRENT_DIAGNOSER_LABEL}\n" + "=" * 70 + "\n")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header + buffer.getvalue())
        print(f"\nSanity check report saved to: {report_path}")


def _check_membership_gap_logic(frame, gaps, all_days, kept_days, check, sample_size=5000):
    """1d. Re-checks the membership-gap person-time with a separate, simpler calculation."""
    t = pd.Timestamp
    # (a) the worked example: member 1.1.2017-1.1.2018 and 1.1.2019-1.1.2020 -> 2 years
    example = parse_membership_gaps(pd.Series(["01/01/2017-01/01/2018:01/01/2019-01/01/2020"], index=["ex"]))
    ex_days = int(member_days(pd.Series(t("2017-01-01"), index=["ex"]), pd.Series(t("2020-01-01"), index=["ex"]),
                              gaps=example).iloc[0])
    check("example: member 1.1.2017-1.1.2018 + 1.1.2019-1.1.2020 counts 730 days (2 years), not 1,095",
          ex_days == 730, f"got {ex_days}")

    # (b) one period, or none: person-time is exactly as before the change
    has_gap = frame.index.isin(gaps.index)
    check("patients without a gap: member days = calendar days",
          bool((kept_days[~has_gap] == all_days[~has_gap]).all()),
          f"{int((kept_days[~has_gap] != all_days[~has_gap]).sum()):,} differ")

    periods_text = frame[MEMBERSHIP_PERIODS_COL].astype(str) if MEMBERSHIP_PERIODS_COL in frame.columns \
        else pd.Series("", index=frame.index)
    several = periods_text.str.contains(":")
    check("every patient with several membership periods has a gap", bool(has_gap[several.values].all()),
          f"{int((~has_gap[several.values]).sum()):,} without")
    if not several.any():
        return

    # (c) fu_end = the last period's end (capped at STUDY_END) -- B2 output read with the right date order
    last_end = pd.to_datetime(periods_text[several].str.split(":").str[-1].str.split("-").str[-1],
                              format="%d/%m/%Y", errors="coerce").clip(upper=STUDY_END)
    differ = frame.loc[several, "fu_end"] != last_end
    check("several periods: fu_end = last membership end (capped at STUDY_END)", not bool(differ.any()),
          f"{int(differ.sum()):,} of {int(several.sum()):,} differ")

    # (d) separate calculation on a sample: days inside [fu_start, fu_end] covered by
    # (before the first period) + each period + (after the last period)
    rows = frame.index[several.values][:sample_size]   # chosen from the text, not from the parsed gaps

    def covered_days(text, start, end):
        periods = sorted(tuple(pd.to_datetime(p.split("-"), format="%d/%m/%Y")) for p in text.split(":"))
        covered = [(t.min, periods[0][0])] + periods + [(periods[-1][1], t.max)]
        return sum(max(0, (min(end, e) - max(start, s)).days) for s, e in covered)

    expected = pd.Series([covered_days(periods_text[r], frame.at[r, "fu_start"], frame.at[r, "fu_end"])
                          for r in rows], index=rows)
    differ = expected != kept_days[rows]
    check(f"person-time matches a separate period-by-period calculation ({len(rows):,} patients with several periods)",
          not bool(differ.any()), f"{int(differ.sum()):,} differ")

    # (e) the KM pieces add up to the same member days
    piece_rows, piece_start, piece_end, _ = membership_pieces(frame.loc[rows, "fu_start"], frame.loc[rows, "fu_end"])
    piece_days = pd.Series((piece_end - piece_start).dt.days.clip(lower=0).values, index=piece_rows)
    piece_days = piece_days.groupby(level=0).sum().reindex(rows, fill_value=0)
    differ = piece_days != kept_days[rows]
    check("KM curve pieces add up to the member days", not bool(differ.any()), f"{int(differ.sum()):,} differ")


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
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES).dropna()
    info(f"{core.TRAUMA_EXPOSURE_LABEL} patients diagnosed AFTER membership end (contribute 0 days)",
         f"{int((entry > frame.loc[entry.index, 'fu_end']).sum()):,} of {len(entry):,}")
    raw = first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    info(f"{core.TRAUMA_EXPOSURE_LABEL} patients excluded (first diagnosis before age 15)",
         f"{int((raw < frame['age15_date']).sum()):,} of {int(raw.notna().sum()):,}")
    check("no exposed entry is before its patient's 15th birthday",
          not bool((entry < frame.loc[entry.index, "age15_date"]).any()))
    info("patients whose follow-up starts after 2017 (turned 15 after 1.1.2017)",
         f"{int((frame['fu_start'] > REGISTRY_START).sum()):,}")

    # --- 1c. Membership gaps (left Clalit and came back) ---
    gaps = core._membership_gaps
    info("patients with a membership gap (gap days are not person-time)",
         f"{gaps.index.nunique():,}")
    all_days = (frame["fu_end"] - frame["fu_start"]).dt.days.clip(lower=0)
    kept_days = member_days(frame["fu_start"], frame["fu_end"]).clip(lower=0)
    info("follow-up person-years removed as membership gaps",
         f"{(all_days.sum() - kept_days.sum()) / 365.25:,.0f} of {all_days.sum() / 365.25:,.0f}")
    check("member person-time never exceeds calendar follow-up", bool((kept_days <= all_days).all()))
    if len(gaps):
        dates = pd.concat({o: frame[f"first_{o}"] for o in ALL_OUTCOMES}, axis=1).reindex(gaps.index)
        in_gap = dates.gt(gaps["gap_start"], axis=0) & dates.lt(gaps["gap_end"], axis=0)
        info("first-ever diagnoses dated inside a membership gap (still counted as events)",
             f"{int(in_gap.values.sum()):,}")
    _check_membership_gap_logic(frame, gaps, all_days, kept_days, check)

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
        _, _, comp = compute_exposed_vs_general(frame, core.TRAUMA_EXPOSURE_OUTCOMES, [outcome])
        rr_h2 = comp.iloc[0]["rr"]
        labels, rrs, _, _ = compute_b1_raw_rr(frame, outcome)
        if core.TRAUMA_EXPOSURE_LABEL in labels:
            rr_b1 = rrs[labels.index(core.TRAUMA_EXPOSURE_LABEL)]
            match = (pd.isna(rr_h2) and pd.isna(rr_b1)) or \
                   (pd.notna(rr_h2) and pd.notna(rr_b1) and abs(rr_h2 - rr_b1) < 1e-9)
            check(f"Statement 2 B1's {core.TRAUMA_EXPOSURE_LABEL} bar == Statement 1 H2 bar [{outcome}]",
                 match, f"H2={rr_h2}, B1={rr_b1}")

    # --- 4. n_gen is invariant across diagnoser modes -- the general population is defined by
    # RAW exposure status only, diagnoser mode should never change who counts as "general". ---
    n_gens = {}
    for diag_key in DIAGNOSER_MODES:
        set_diagnoser_mode(diag_key, outcome_profession_meta_full)
        n_exp, n_gen, _ = compute_exposed_vs_general(frame, core.TRAUMA_EXPOSURE_OUTCOMES, ["psychosis"])
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
        n_exps[trauma_key] = int(first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES).notna().sum())
    set_trauma_mode("ptsd_asr")
    check("ptsd_only + asr_only n_exp >= merged (ptsd_asr) n_exp",
         n_exps["ptsd_only"] + n_exps["asr_only"] >= n_exps["ptsd_asr"], str(n_exps))

    # --- 6. Pre+post n_exp sums to exactly the pooled H2 n_exp (same exposure, same eligibility
    # rule, just split by OCT7 -- must partition exactly, no overlap or gap). ---
    n_exp_pooled, _, _ = compute_exposed_vs_general(frame, core.TRAUMA_EXPOSURE_OUTCOMES, ["psychosis"])
    (n_pre, _), (n_post, _), _ = compute_prepost_window_rr(
        frame, core.TRAUMA_EXPOSURE_OUTCOMES, ["psychosis"], core.FIXED_WINDOW_DAYS)
    check("pre_n_exp + post_n_exp == pooled H2 n_exp",
         (n_pre + n_post) == n_exp_pooled, f"{n_pre}+{n_post} vs {n_exp_pooled}")

    # --- 7/8. Profession & facility-type coding diagnostics (already-existing diagnostic
    # printers, now folded into the pass/fail summary instead of being a silent side effect). --
    analyses._profession_diagnostic_printed = False
    any_psychiatrist_code = any(
        normalize_profession_code(outcome_profession_meta_full[o]["profession"]).dropna()
        .isin(PSYCHIATRIST_PROFESSION_CODES).any()
        for o in ["psychosis", "bipolar"])
    print_profession_diagnostic(outcome_profession_meta_full)
    check("psychiatrist profession codes (71/73) appear somewhere in the data", any_psychiatrist_code)

    analyses._facility_type_diagnostic_printed = False
    facility_vals = get_exposure_meta_field(entry_meta_full, core.TRAUMA_EXPOSURE_OUTCOMES, "facility type")
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
        _, _, comp = compute_exposed_vs_general(f_birth, core.TRAUMA_EXPOSURE_OUTCOMES, [outcome])
        res = adjusted_irr_results(f_birth, core.TRAUMA_EXPOSURE_OUTCOMES, outcome)
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
    in_c = POPULATIONS["no7km_nonorth"][2](frame).fillna(False).astype(bool)
    no_dist = frame["distance_km"].isna() & ~frame["is_north_border"].astype(bool)
    check("no-0-7km-no-North population: nobody <=7 km or North, everyone with missing distance kept",
          not bool((in_c & ((frame["distance_km"] <= 7) | frame["is_north_border"].astype(bool))).any())
          and bool(in_c[no_dist].all()),
          f"kept={int(in_c.sum()):,}, of them missing distance={int((in_c & no_dist).sum()):,}")

    # --- 11d. Era comparison: the model's crude IRR == (events/PY post) / (events/PY pre), the
    # pre era has no person-time or events on/after Oct 7, and nobody is at risk past the cap ---
    cohort = era_cohort(frame)
    res = era_results(frame, cohort, "psychosis")
    if res is not None:
        c, pt = res["crude"], res["pt"]
        ratio = (c["events_cmp"] / c["py_cmp"]) / (c["events_ref"] / c["py_ref"]) \
            if c["events_ref"] and c["py_cmp"] else np.nan
        check("Era model crude IRR == post/pre rate ratio [psychosis]",
              pd.notna(c["irr"]) and abs(c["irr"] / ratio - 1) < 1e-6, f"model={c['irr']}, ratio={ratio}")
        pre_entry = cohort.loc[pt.index, "entry"][pt["era"] == "pre"]
        pre_py = pt.loc[pt["era"] == "pre", "py"]
        check("Era pre-era person-time ends by Oct 7 and within the cap",
              bool(((pre_entry + pd.to_timedelta(pre_py * 365.25, unit="D")) <= OCT7 + pd.Timedelta(hours=1)).all())
              and bool((pt["py"] <= core.FIXED_WINDOW_DAYS / 365.25 + 1e-9).all()),
              f"max py={pt['py'].max():.3f}")
    # crossing group: time only from Oct 7, never past X + cap, and nobody with Y by Oct 7
    cr = group_person_time(cohort, frame, "psychosis", "crossing")
    if len(cr):
        cr_entry = cohort.loc[cr.index, "entry"]
        cap_left = (cr_entry + pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D") - OCT7).dt.days / 365.25
        y = frame.loc[cr.index, "first_psychosis"]
        check("Era crossing group: from Oct 7, within X + cap, nobody with Y by Oct 7",
              bool((cr_entry < OCT7).all()) and bool((cr["py"] <= cap_left + 1e-9).all())
              and not bool((y <= OCT7).any()), f"n={len(cr):,}, events={int(cr['events'].sum())}")

    # --- 11e. General-population Lexis split adds up: the same follow-up split into calendar years
    # gives the same person-years as the one whole-study period, and the same events -- except an
    # outcome dated exactly on STUDY_END, which the whole-study period (end-exclusive) leaves out. ---
    years = periods_from_years(range(REGISTRY_START.year, STUDY_END.year + 1))
    for outcome in FOCUS_OUTCOMES:
        yearly = compute_general_lexis_split(frame, core.TRAUMA_EXPOSURE_OUTCOMES, outcome, years)
        _, overall_events, overall_py = compute_general_rate_overall(frame, core.TRAUMA_EXPOSURE_OUTCOMES, outcome)
        _, _, is_event, down = general_intervals(frame, core.TRAUMA_EXPOSURE_OUTCOMES, outcome)
        on_study_end = int((is_event & (down == STUDY_END)).sum())
        yearly_py, yearly_events = yearly["person_years"].sum(), int(yearly["n_events"].sum())
        check(f"general person-time split by year adds up to the whole-study total [{outcome}]",
              abs(yearly_py - overall_py) <= 1e-6 * max(overall_py, 1)
              and yearly_events - int(overall_events) == on_study_end,
              f"PY {yearly_py:,.2f} vs {overall_py:,.2f}; events {yearly_events:,} vs "
              f"{int(overall_events):,} + {on_study_end} on STUDY_END")

    # --- 11f. The adjusted model's table holds exactly the crude exposed person-time: its exposed
    # rows (before + on/after Oct 7) have the same events and person-years as Statement 1's
    # exposed side (11b only compares the IRR, which a common scaling error would not change). ---
    for outcome in FOCUS_OUTCOMES:
        table, _ = build_poisson_table(f_birth, core.TRAUMA_EXPOSURE_OUTCOMES, outcome)
        exposed_rows = table[table["exposure"] != "general"]
        n_events, person_years = person_time_events_and_years(
            first_of(f_birth, core.TRAUMA_EXPOSURE_OUTCOMES), f_birth[f"first_{outcome}"])
        model_events, model_py = int(exposed_rows["events"].sum()), exposed_rows["py"].sum()
        check(f"Poisson table exposed events / person-years == Statement 1 exposed side [{outcome}]",
              model_events == n_events and abs(model_py - person_years) <= 1e-6 * max(person_years, 1),
              f"events {model_events:,} vs {n_events:,}; PY {model_py:,.2f} vs {person_years:,.2f}")

    # --- 11g. Every era comparison (not only post vs pre): crude IRR == the plain rate ratio, each
    # subgroup's level rows add up to the overall row (events and person-years, both groups), and
    # every interaction p-value is a probability. ---
    for cmp_group, ref_group in ERA_COMPARISONS:
        res = era_results(frame, cohort, "psychosis", cmp=cmp_group, ref=ref_group)
        name = f"Era {cmp_group} vs {ref_group} [psychosis]"
        if res is None:
            info(name, "nobody at risk in one of the groups -- not checked")
            continue
        c = res["crude"]
        ratio = ((c["events_cmp"] / c["py_cmp"]) / (c["events_ref"] / c["py_ref"])
                 if c["events_ref"] and c["events_cmp"] else np.nan)
        irr_ok = (pd.isna(c["irr"]) and pd.isna(ratio)) or (pd.notna(c["irr"]) and abs(c["irr"] / ratio - 1) < 1e-6)
        bad = []
        for sg, (sg_rows, p_int) in res["subgroups"].items():
            for key in ("events_ref", "events_cmp"):
                if sum(r[key] for r in sg_rows) != c[key]:
                    bad.append(f"{sg} {key}")
            for key in ("py_ref", "py_cmp"):
                if abs(sum(r[key] for r in sg_rows) - c[key]) > 1e-6 * max(c[key], 1):
                    bad.append(f"{sg} {key}")
            if pd.notna(p_int) and not 0 <= p_int <= 1:
                bad.append(f"{sg} interaction p={p_int}")
        check(f"{name}: crude IRR == rate ratio, subgroup rows add up to the overall row",
              irr_ok and not bad, f"IRR {c['irr']} vs ratio {ratio}" + (f"; mismatches: {bad}" if bad else ""))

    # --- 11h. Diagnoser modes split the exposed patients: psychiatrist-only and non-psychiatrist-
    # only never share a patient and both sit inside any-diagnoser; the rest (profession 999 /
    # missing) is exposed in any-diagnoser only. ---
    exposed = {}
    for mode in DIAGNOSER_MODES:
        set_diagnoser_mode(mode, outcome_profession_meta_full)
        exposed[mode] = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES).notna()
    set_diagnoser_mode("any_diagnoser", outcome_profession_meta_full)
    psych, non_psych, any_dx = (exposed["psychiatrist_only"], exposed["non_psychiatrist_only"],
                                exposed["any_diagnoser"])
    check("psychiatrist-only and non-psychiatrist-only exposed never overlap, both within any-diagnoser",
          not bool((psych & non_psych).any()) and not bool(((psych | non_psych) & ~any_dx).any()),
          f"psychiatrist={int(psych.sum()):,}, non-psychiatrist={int(non_psych.sum()):,}, "
          f"any={int(any_dx.sum()):,}")
    info("exposed with an unknown diagnoser (code 999 / missing) -- counted in any-diagnoser only",
         f"{int((any_dx & ~psych & ~non_psych).sum()):,}")

    # --- 12. Informational only (data hypotheses, not code-correctness -- printed, not asserted) --
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
