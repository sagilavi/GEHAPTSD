# C_IRR_Tables.py -- TABLE FORMATTING: cell/number formatting, Hebrew right-to-left text, the
# generic table page, and the table-only pages of the descriptive companion PDF.
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
from C_IRR_Core import (  # noqa: F401
    ALL_OUTCOMES, FOCUS_OUTCOMES, MISSING_PROFESSION_CODES, OCT7, REGISTRY_START, S1_DOWNSTREAM,
    STUDY_END, compute_general_lexis_split, current_alt_exposures, first_of, first_of_raw,
    normalize_profession_code,
)
from C_IRR_Analyses import (  # noqa: F401
    DESC_AGE_BANDS, SECOND_DIAGNOSIS_PERIODS, YEARLY_TREND_PERIODS, compute_time_to_second_dx,
    exposed_followup_detail, km_intervals, second_dx_flow, split_person_time_by_period,
    trauma_component_of_entry,
)
from C_IRR_Models import (  # noqa: F401
    adjustment_covariates, build_poisson_table,
)


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


def _fmt_p(p):
    return "n/a" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_irr(r):
    if pd.isna(r["irr"]):
        return "n/a (an era has no events)"
    return f"{r['irr']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}], p = {_fmt_p(r['p'])}"


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


def desc_overview_page(pdf, frame, age_label):
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    exposed = entry.notna()
    general = first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES).isna()
    pre, post = exposed & (entry < OCT7), exposed & (entry >= OCT7)
    left_early = frame["fu_end"] < STUDY_END
    female = frame["gender"].astype(str).str.lower().str.startswith(("f", "נ"))
    rows = {}
    for name, mask in [("All patients in age band", pd.Series(True, index=frame.index)),
                       (f"Exposed: {core.TRAUMA_EXPOSURE_LABEL} (any time)", exposed),
                       (f"  {core.TRAUMA_EXPOSURE_LABEL} before Oct 7", pre),
                       (f"  {core.TRAUMA_EXPOSURE_LABEL} on/after Oct 7", post),
                       (f"General (never {core.TRAUMA_EXPOSURE_LABEL})", general),
                       (f"Excluded ({core.TRAUMA_EXPOSURE_LABEL} before age 15"
                        + ("" if core.CURRENT_DIAGNOSER_MODE == "any_diagnoser" else " / other diagnoser") + ")",
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
                    note=f"Age = age on Oct 7 2023. Exposure = earliest {core.TRAUMA_EXPOSURE_LABEL} diagnosis "
                         f"({core.CURRENT_DIAGNOSER_LABEL}). Data from {REGISTRY_START:%d.%m.%Y} to "
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


def desc_population_table_page(pdf, frame, age_label):
    """'Table 1': the population behind the adjusted model, by group -- every covariate the model
    adjusts for, n (% of column). Plus person-years / events actually entering the model."""
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    general = first_of_raw(frame, core.TRAUMA_EXPOSURE_OUTCOMES).isna()
    groups = {"General population": general,
              f"{core.TRAUMA_EXPOSURE_LABEL} before Oct 7": entry.notna() & (entry < OCT7),
              f"{core.TRAUMA_EXPOSURE_LABEL} on/after Oct 7": entry.notna() & (entry >= OCT7)}
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
        elif var in ("SES (3-level)", "Charlson"):   # ordinal -- in order, not by size
            order = {"low": 0, "medium": 1, "high": 2}
            levels = (sorted((l for l in levels if l != "missing"), key=lambda l: (order.get(l.strip().lower(), 9), l))
                      + [l for l in levels if l == "missing"])
        for level in levels:
            rows[(var, level)] = {g: (f"{int((covs.loc[m, var] == level).sum()):,} "
                                      f"({(covs.loc[m, var] == level).mean() * 100:.1f}%)" if m.any() else "-")
                                  for g, m in groups.items()}
    df = pd.DataFrame(rows).T
    df.index = [f"{a}: {rtl(b)}" if a else b for a, b in df.index]
    df.index.name = "Characteristic"
    draw_table_page(pdf, f"DESCRIPTIVE | Adjusted-IRR slides | the population used, by group | {age_label}",
                    df, font_size=7,
                    note=f"Patients counted by group as in Statement 1 (exposure = earliest {core.TRAUMA_EXPOSURE_LABEL}, "
                         f"{core.CURRENT_DIAGNOSER_LABEL}). Patients diagnosed before Oct 7 and their pre-trauma time "
                         f"also contribute to the general population's person-time in the model (dynamic "
                         f"exposure) -- these columns count each patient once, by their own group. Sex, sector, "
                         f"SES and Charlson are the model's covariates as coded there; age in the model is "
                         f"ATTAINED age (age during each stretch of person-time), shown here as age at Oct 7.")

    # person-years and events that actually enter the model, per exposure group and outcome
    rows = {}
    for outcome in FOCUS_OUTCOMES:
        table, n_no_birth = build_poisson_table(frame, core.TRAUMA_EXPOSURE_OUTCOMES, outcome)
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


def desc_yearly_pages(pdf, frame, age_label):
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    for o in FOCUS_OUTCOMES:
        down = frame[f"first_{o}"]
        gen = compute_general_lexis_split(frame, core.TRAUMA_EXPOSURE_OUTCOMES, o, YEARLY_TREND_PERIODS)
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
        draw_table_page(pdf, f"DESCRIPTIVE | Slides 9, 20, 21 | {o.upper()} by year of {core.TRAUMA_EXPOSURE_LABEL} "
                             f"diagnosis | {age_label}", df, font_size=7.5,
                        note="Each point on slides 9/20 = rate / general rate of that row. Rows with few "
                             "events (or little PY, e.g. 2026) give unstable IRRs.")


def desc_second_dx_year_page(pdf, frame, age_label):
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    rows = {}
    for o in FOCUS_OUTCOMES:
        down = frame[f"first_{o}"]
        for side_name, side_entry in [("trauma pre", entry.where(entry < OCT7)),
                                      ("trauma post", entry.where(entry >= OCT7))]:
            ev, py = split_person_time_by_period(side_entry, down, core.FIXED_WINDOW_DAYS, SECOND_DIAGNOSIS_PERIODS)
            rows[f"{o.upper()} | {side_name} | events"] = {k.replace("\n", " "): v for k, v in ev.items()}
            rows[f"{o.upper()} | {side_name} | PY"] = {k.replace("\n", " "): v for k, v in py.items()}
    df = pd.DataFrame(rows).T; df.index.name = "Outcome | group | measure"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 10-15 | events and person-years by calendar period of the "
                         f"2nd diagnosis | {age_label}", df, font_size=7,
                    note="Slides 11-12 hide periods with fewer than 10 / 30 events; slides 13-15 box each "
                         "period's IRR. Box 1 = 'trauma pre' periods before Oct 7, box 2 = 'trauma post', "
                         "box 3 = 'trauma pre' periods from Oct 7.")


def desc_second_dx_flow_page(pdf, frame, age_label):
    """Patient flow behind the 'By year of SECOND diagnosis' graph: how many are left after each
    rule, and how much each rule removed."""
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    columns, checks = {}, []
    for o in FOCUS_OUTCOMES:
        steps = second_dx_flow(frame, o)
        for side, side_name in [("pre", "before Oct 7"), ("post", "on/after Oct 7")]:
            col, previous = {}, None
            for name, counts in steps:
                n = counts[side]
                change = "" if previous is None or previous == 0 else f" ({(n - previous) / previous * 100:+.0f}%)"
                col[name] = f"{n:,}{change}"
                previous = n
            on_graph = sum(split_person_time_by_period(
                entry.where((entry < OCT7) if side == "pre" else (entry >= OCT7)), frame[f"first_{o}"],
                core.FIXED_WINDOW_DAYS, SECOND_DIAGNOSIS_PERIODS)[0].values())
            mismatch = "" if on_graph == previous else "  MISMATCH"
            col["Sum of the n= labels on the graph (check)"] = f"{on_graph:,}{mismatch}"
            checks.append(on_graph == previous)
            columns[f"{o.upper()}\n{core.TRAUMA_EXPOSURE_LABEL} {side_name}"] = col
    df = pd.DataFrame(columns); df.index.name = "Step (each row keeps only who passed it)"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 10-15 | patient flow behind 'By year of SECOND diagnosis' | "
                         f"{age_label}", df, font_size=7,
                    note="Each number = patients still in after that step; in brackets = change from the row "
                         "above. The last step is the number of second diagnoses on the graph (its n= labels "
                         "added up). " + ("Check OK: the flow and the graph agree." if all(checks)
                                          else "CHECK FAILED: the flow and the graph do not agree."))


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
    for exp_label, outcome_list in current_alt_exposures().items():
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
                         f"within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}", df, font_size=7.5,
                    note="Each row's general population = never diagnosed with THAT entry diagnosis, so it "
                         "differs per row. 'self' = the entry diagnosis is the outcome itself (not shown).")


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
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
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
                f"% within {core.FIXED_WINDOW_DAYS//30} months": (lat < core.FIXED_WINDOW_DAYS / 30.44).mean() * 100 if n else np.nan,
                "max months possible": (STUDY_END - entry[entry.notna() & side_mask].min()).days / 30.44
                                       if (entry.notna() & side_mask).any() else np.nan}
    df = pd.DataFrame(rows).T; df.index.name = "Outcome | trauma timing"
    draw_table_page(pdf, f"DESCRIPTIVE | Slides 37-42 | latency summary (no window, no membership cap) | "
                         f"{age_label}", df, font_size=7.5,
                    note="Post-Oct7 trauma patients can be followed at most ~33 months, so their latency "
                         "distribution is cut short by design -- compare shapes up to that point only.")
