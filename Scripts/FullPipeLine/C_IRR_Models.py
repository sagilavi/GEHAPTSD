# C_IRR_Models.py -- LOGIC 3/4: adjusted IRR (Poisson regression, log person-years offset) and the
# era comparisons (pre / post / crossing Oct 7).
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
    ALL_OUTCOMES, OCT7, REGISTRY_START, STUDY_END, exposed_intervals, first_of,
    followup_end_for, general_intervals, member_days,
)


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
#   - Sex, sector ("marketing sector" column), SES ("socioeconomic score three level scale"), Charlson
#     (0/1/2/3+) are per-patient categories; a missing value is its own category. SES's reference is
#     Medium (ADJ_REFERENCE); every other covariate's reference is its largest category (by person-years).
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
ADJ_COVARIATES = ["Sex", "Sector", "SES (3-level)", "Charlson"]
# Fixed reference level per covariate; any covariate not listed uses its largest category.
ADJ_REFERENCE = {"SES (3-level)": "Medium"}
_reported_missing_refs = set()


def covariate_reference(table, col):
    """Reference level of a covariate: ADJ_REFERENCE (matched ignoring case) if the data has it,
    otherwise the largest category by person-years (with a one-time warning)."""
    largest = table.groupby(col)["py"].sum().idxmax()
    wanted = ADJ_REFERENCE.get(col)
    if wanted is None:
        return largest
    levels = [str(l) for l in table[col].unique()]
    matches = [l for l in levels if l.strip().lower() == wanted.lower()]
    if matches:
        return matches[0]
    if col not in _reported_missing_refs:
        _reported_missing_refs.add(col)
        print(f"WARNING: reference '{wanted}' not found for {col} (levels: {sorted(levels)}) -- "
              f"using the largest category '{largest}' instead")
    return largest


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
        "Sector": _as_category(frame["marketing_sector"]),
        "SES (3-level)": _as_category(frame["ses_3level"]),
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
        py = member_days(s, e).clip(lower=0).fillna(0) / 365.25
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


def _poisson_design(table, exposure_col, covariates, exposure_ref):
    """Intercept + dummy columns: the exposure against `exposure_ref`, each covariate against its
    ADJ_REFERENCE level (else its largest category by person-years). Returns (columns, [(term, group, level)])."""
    cols, names = [np.ones(len(table))], [("Intercept", "", "")]
    for col in [exposure_col] + covariates:
        if col == exposure_col:
            ref = exposure_ref
        else:
            ref = covariate_reference(table, col)
        for level in sorted(table[col].astype(str).unique()):
            if level == str(ref):
                continue
            cols.append((table[col].astype(str) == level).astype(float).values)
            names.append((f"{col}: {level}", col, level))
    return cols, names


def _irls(X, y, offset):
    """Poisson GLM (log link) by IRLS. Returns (beta, covariance, log-likelihood without the
    constant log(y!) term -- fine for likelihood-ratio tests between nested models)."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum(), 0.5) / np.exp(offset).sum())
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
    cov = np.linalg.pinv((X.T * mu) @ X)
    loglik = float(np.sum(y * np.log(np.clip(mu, 1e-300, None)) - mu))
    return beta, cov, loglik


def fit_poisson(table, exposure_col, covariates, exposure_ref="general"):
    """Poisson GLM, log link, offset log(py), fitted by IRLS. `covariates` are categorical
    columns of `table`, each coded against its largest category by person-years. Returns a
    DataFrame of terms (term, group, level, irr, lo, hi, p) -- intercept excluded."""
    cols, names = _poisson_design(table, exposure_col, covariates, exposure_ref)
    X = np.column_stack(cols)
    y = table["events"].values.astype(float)
    beta, cov, _ = _irls(X, y, np.log(table["py"].values))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

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


# =============================================================================
# PART 9 -- ERA COMPARISON: X diagnosed BEFORE vs ON/AFTER Oct 7 (reviewer's design)
#   - X = the trauma exposure (TRAUMA_EXPOSURE_OUTCOMES), Y = every other outcome.
#   - Cohort: patients whose first-ever X is dated from REGISTRY_START (2017) on, at age 15+
#     (first_of() -- the diagnoser mode applies). Pre era = X before Oct 7, post = on/after.
#   - Person-time from X until the earliest of: Y, X + FIXED_WINDOW_DAYS (the lag cap, same in
#     both eras), leaving Clalit, STUDY_END -- and for the pre era also the end of the era: pre
#     time and events stop at Oct 7 (censored at Oct 6, 2023). Prevalent Y (on/before X) excluded.
#   - Model: Poisson, offset log(person-years): era (ref = pre) + age AT X (not at Oct 7 -- the
#     pre group was diagnosed years earlier, so age at Oct 7 is shifted by design) + sex + sector
#     + SES + Charlson. Every covariate is in the model regardless of its SMD.
#   - Subgroups: an era x subgroup interaction is added to the adjusted model (the subgroup's
#     main effect too, if it isn't a confounder already). IRR in each level = exp(b_era +
#     b_era x level); interaction p = likelihood-ratio test against the same model without it.
#   - Table 1 / love plot: SMD pre vs post per covariate level (|SMD| > 0.1 = imbalance worth
#     noting). The SMD describes the imbalance -- it does not choose the confounders.
# =============================================================================
ERA_AGE_BANDS = [(15, 25, "15-24"), (25, 35, "25-34"), (35, 45, "35-44"), (45, 65, "45-64"),
                 (65, 200, "65+")]
ERA_COVARIATES = ["Age at X"] + ADJ_COVARIATES
# 3-category distance: a confounder in the "adjusted + distance" model only (the main adjusted
# model is unchanged). Missing distance goes with 20+ -- no distance = not near a border.
ERA_DIST3 = "Distance (3 cat)"
ERA_DIST3_LEVELS = ["0-7 km", "7-20 km", "20+ km / missing"]
ERA_SUBGROUPS = ["Sex", "Sector", "Age at X", "Distance from Gaza", ERA_DIST3]
SMD_THRESHOLD = 0.1


def _level_order(level):
    """Sort key: the 3 distance categories in distance order, everything else alphabetically."""
    return (ERA_DIST3_LEVELS.index(level), "") if level in ERA_DIST3_LEVELS else (99, str(level))


def era_outcomes():
    """Y = every outcome that isn't part of the current trauma exposure."""
    return [o for o in ALL_OUTCOMES if o not in core.TRAUMA_EXPOSURE_OUTCOMES]


def era_cohort(frame):
    """One row per patient with an incident X from 2017 on: entry, era, age at X (years and band),
    the confounders and the distance subgroup."""
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    entry = entry[entry >= REGISTRY_START]
    f = frame.loc[entry.index]
    age_x = (entry - f["birth"]).dt.days / 365.25
    age_band = pd.Series("missing", index=entry.index, dtype=object)
    for lo, hi, label in ERA_AGE_BANDS:
        age_band[(age_x >= lo) & (age_x < hi)] = label
    distance = pd.Series("missing", index=entry.index, dtype=object)
    distance[f["distance_km"] <= 20] = "<=20 km"
    distance[f["distance_km"] > 20] = ">20 km (not North)"
    distance[f["is_north_border"].astype(bool)] = "Northern Border"
    dist3 = pd.Series(ERA_DIST3_LEVELS[2], index=entry.index, dtype=object)
    dist3[f["distance_km"] <= 20] = ERA_DIST3_LEVELS[1]
    dist3[f["distance_km"] <= 7] = ERA_DIST3_LEVELS[0]
    cohort = adjustment_covariates(f)
    cohort.insert(0, "Age at X", age_band)
    cohort["Distance from Gaza"] = distance
    cohort[ERA_DIST3] = dist3
    cohort["entry"] = entry
    cohort["era"] = np.where(entry < OCT7, "pre", "post")
    cohort["age_x"] = age_x
    return cohort


def era_person_time(cohort, frame, outcome):
    """Per at-risk cohort patient for Y = `outcome` (prevalent Y removed): py, events, covariates.
    Same interval as the exposed side everywhere else (exposed_intervals), plus the pre-era cut."""
    start, end, is_event, event = exposed_intervals(cohort["entry"], frame[f"first_{outcome}"])
    pre = cohort.loc[start.index, "era"] == "pre"
    is_event = is_event & ~(pre & (event >= OCT7))
    end = end.where(~(pre & (end > OCT7)), OCT7)
    py = member_days(start, end).clip(lower=0) / 365.25
    return cohort.drop(columns=["entry"]).loc[start.index].assign(py=py, events=is_event.astype(int))


# --- Third group, "crossing": X before Oct 7, followed FROM Oct 7 -----------------------------
# The pre-diagnosed patients' second diagnosis after Oct 7. A group can't be defined by Y itself
# (everyone in it would be a case and the rate would mean nothing), so it is defined by its
# exposure time: pre-era patients still at risk on Oct 7 -- no Y by Oct 7, the cap (counted from
# X, as for everyone) not yet reached, still a Clalit member -- followed from Oct 7 until the
# earliest of Y, X + cap, leaving Clalit, STUDY_END. The same patient can be in "pre" (time up to
# Oct 6) and "crossing" (time from Oct 7) -- the two never share a day of person-time.
ERA_GROUP_LABELS = {"pre": "X before Oct 7", "post": "X on/after Oct 7",
                    "crossing": "X before Oct 7, Y-free on Oct 7, followed from Oct 7"}
ERA_COMPARISONS = [("post", "pre"), ("crossing", "pre"), ("crossing", "post")]   # (comparison, reference)


def _crossing_boundary(cohort):
    """Pre-era patients' latest possible follow-up end: X + cap, leaving Clalit, STUDY_END."""
    pre = cohort[cohort["era"] == "pre"]
    window_end = pre["entry"] + pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D")
    return pd.concat([window_end, pd.Series(STUDY_END, index=pre.index),
                      followup_end_for(pre.index)], axis=1).min(axis=1)


def group_cohort(cohort, group):
    """Patients of one comparison group (Table 1 -- before the Y-specific prevalent exclusion)."""
    if group == "crossing":
        boundary = _crossing_boundary(cohort)
        return cohort.loc[boundary[boundary > OCT7].index].assign(era="crossing")
    return cohort[cohort["era"] == group]


def group_person_time(cohort, frame, outcome, group):
    """era_person_time() rows of one group; for "crossing", time counted from Oct 7."""
    if group != "crossing":
        pt = era_person_time(cohort, frame, outcome)
        return pt[pt["era"] == group]
    members = group_cohort(cohort, "crossing")
    boundary = _crossing_boundary(cohort).reindex(members.index)
    event = frame[f"first_{outcome}"].reindex(members.index)
    at_risk = ~(event.notna() & (event <= OCT7))   # Y on/before Oct 7: not at risk of a first Y
    members, boundary, event = members[at_risk], boundary[at_risk], event[at_risk]
    is_event = event.notna() & (event <= boundary)
    if core.LAG_DAYS > 0:
        is_event &= (event - members["entry"]).dt.days >= core.LAG_DAYS
    end = boundary.where(~is_event, event)
    py = member_days(pd.Series(OCT7, index=end.index), end).clip(lower=0) / 365.25
    return members.drop(columns=["entry"]).assign(py=py, events=is_event.astype(int))


def _era_fit(cells, covariates, subgroup=None, cmp="post", ref="pre"):
    """Poisson fit of group (cmp vs ref) + covariates [+ group x subgroup]. Returns (rows, loglik,
    rank): rows = one IRR (cmp vs ref) overall, or one per subgroup level."""
    covariates = list(covariates)
    if subgroup and subgroup not in covariates:
        covariates.append(subgroup)
    cols, names = _poisson_design(cells, "era", covariates, ref)
    post = (cells["era"] == cmp).values.astype(float)
    levels = [None]
    if subgroup:
        sg_ref = str(covariate_reference(cells, subgroup))
        levels = [sg_ref] + [l for l in sorted(cells[subgroup].astype(str).unique()) if l != sg_ref]
        cols += [post * (cells[subgroup].astype(str) == l).values for l in levels[1:]]
    X = np.column_stack(cols)
    beta, cov, loglik = _irls(X, cells["events"].values.astype(float), np.log(cells["py"].values))

    rows = []
    for k, level in enumerate(levels):
        sub = cells if level is None else cells[cells[subgroup].astype(str) == level]
        ev = sub.groupby("era")["events"].sum().reindex([ref, cmp], fill_value=0)
        py = sub.groupby("era")["py"].sum().reindex([ref, cmp], fill_value=0)
        row = {"level": level, "events_ref": int(ev[ref]), "events_cmp": int(ev[cmp]),
               "py_ref": py[ref], "py_cmp": py[cmp],
               "irr": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
        if ev[ref] > 0 and ev[cmp] > 0:   # a group with no events -> IRR runs to 0 / inf
            if k == 0:
                b, v = beta[1], cov[1, 1]   # column 1 = "era: <cmp>"
            else:
                m = len(names) + k - 1
                b, v = beta[1] + beta[m], cov[1, 1] + cov[m, m] + 2 * cov[1, m]
            se = np.sqrt(max(v, 0))
            row.update(irr=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se),
                       p=2 * stats.norm.sf(abs(b / se)) if se > 0 else np.nan)
        rows.append(row)
    return rows, loglik, np.linalg.matrix_rank(X)


def era_results(frame, cohort, outcome, cmp="post", ref="pre"):
    """Crude + adjusted IRR (cmp vs ref) and the subgroup interaction fits for one X -> Y pair.
    None if either group has nobody at risk."""
    pt = pd.concat([group_person_time(cohort, frame, outcome, ref),
                    group_person_time(cohort, frame, outcome, cmp)])
    cells = pt.groupby(["era"] + ERA_COVARIATES + ["Distance from Gaza", ERA_DIST3],
                       as_index=False)[["py", "events"]].sum()
    cells = cells[cells["py"] > 0].reset_index(drop=True)
    if cells.empty or cells["era"].nunique() < 2:
        return None
    fit = lambda covs, sg=None: _era_fit(cells, covs, sg, cmp=cmp, ref=ref)
    out = {"pt": pt, "crude": fit([])[0][0], "adjusted": fit(ERA_COVARIATES)[0][0],
           "adjusted + distance": fit(ERA_COVARIATES + [ERA_DIST3])[0][0], "subgroups": {}}
    for sg in ERA_SUBGROUPS:
        covs = ERA_COVARIATES if sg in ERA_COVARIATES else ERA_COVARIATES + [sg]
        _, ll0, rank0 = fit(covs)
        rows, ll1, rank1 = fit(covs, sg)
        p_int = stats.chi2.sf(max(2 * (ll1 - ll0), 0), rank1 - rank0) if rank1 > rank0 else np.nan
        out["subgroups"][sg] = (rows, p_int)
    return out


def smd_rows(cohort, cmp="post", ref="pre"):
    """Table-1 rows ref vs cmp: age at X as a mean, then every level of each covariate as
    n (%) with its SMD (cmp - ref). Both distance variables are listed: the 4-level subgroup and
    the 3-category one used in the "adjusted + distance" model."""
    pre, post = cohort[cohort["era"] == ref], cohort[cohort["era"] == cmp]
    rows = []
    a, b = pre["age_x"].dropna(), post["age_x"].dropna()
    sd = np.sqrt((a.var() + b.var()) / 2) if len(a) > 1 and len(b) > 1 else np.nan
    rows.append(("Age at X (years)", "mean (SD)", f"{a.mean():.1f} ({a.std():.1f})",
                 f"{b.mean():.1f} ({b.std():.1f})", (b.mean() - a.mean()) / sd if sd > 0 else np.nan))
    for cov in ERA_COVARIATES + ["Distance from Gaza", ERA_DIST3]:
        for level in sorted(cohort[cov].astype(str).unique(), key=_level_order):
            n1, n2 = int((pre[cov] == level).sum()), int((post[cov] == level).sum())
            p1, p2 = n1 / max(len(pre), 1), n2 / max(len(post), 1)
            v = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
            rows.append((cov, level, f"{n1:,} ({p1:.1%})", f"{n2:,} ({p2:.1%})",
                         (p2 - p1) / np.sqrt(v) if v > 0 else np.nan))
    return pd.DataFrame(rows, columns=["Covariate", "Level", "pre", "post", "SMD"])


def _era_group_label(group):
    return ERA_GROUP_LABELS[group].replace("X ", f"{core.TRAUMA_EXPOSURE_LABEL} ")
