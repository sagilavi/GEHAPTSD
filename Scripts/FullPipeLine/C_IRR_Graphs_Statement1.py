# C_IRR_Graphs_Statement1.py -- GRAPHS 1/3: colors, and the chart panels and pages of
# Statement 1 and its supporting tests (PART 1, 1B).
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
    FOCUS_OUTCOMES, OCT7, S1_DOWNSTREAM, ZONE_LABELS, ZONE_ORDER,
)
from C_IRR_Analyses import (  # noqa: F401
    MIN_N_THRESHOLDS, Q_INDEX, SECOND_DIAGNOSIS_YEAR_RANGE, _OCT7_QUARTER_POS,
    classify_bipolar_subtype, classify_psychosis_subtype, compute_by_second_diagnosis_year,
    compute_exposed_vs_general, compute_prepost_window_rr, compute_quarterly_rr_trend,
    compute_second_diagnosis_boxplot_values, compute_t2b_rates, compute_t5_breakdown,
    compute_t6_breakdown, compute_t9_rates, compute_time_to_second_dx, compute_yearly_rr_trend,
    compute_zone_rates, fit_trend_and_predict, km_cumulative_incidence, km_intervals, logrank_test,
    print_facility_type_diagnostic, risk_set_sizes,
)

OUTCOME_COLORS = {
    "psychosis":   "#e63946", "bipolar":     "#457b9d",
    "ptsd":        "#2a9d8f", "asr":         "#e9c46a",
    "anxiety":     "#f4a261", "depression":  "#264653",
    "substance":   "#6d6875", "personality": "#b5838d",
    "OCD":         "#8ab17d",
}
EXPOSURE_COLORS = {
    "PTSD/ASR": "#e63946", "PTSD": "#e63946", "ASR": "#e63946",   # same color, any trauma mode
    "Depression": "#264653", "Anxiety": "#f4a261",
    "Substance": "#6d6875", "Bipolar": "#457b9d", "Psychosis": "#e63946",
    "Personality": "#b5838d", "OCD": "#8ab17d",
}
ZONE_COLORS = {"Near Gaza": "#e63946", "Northern Border": "#457b9d",
               "Rest of Israel": "#6c757d"}


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
                 f"{core.FIXED_WINDOW_DAYS//30}mo window, restricted person-time", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")


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
    ax.set_ylabel(f"IRR (exposed / general), within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


# --- S1, STEP 5: Write Statement 1 pages (3 pages) for one age band ---
def write_statement1(pdf, frame, age_label):
    n_exp, n_gen, comp = compute_exposed_vs_general(frame, core.TRAUMA_EXPOSURE_OUTCOMES, S1_DOWNSTREAM)
    fig, ax = plt.subplots(figsize=(13, 6))
    draw_exposed_vs_general(ax, comp, n_exp, n_gen, core.TRAUMA_EXPOSURE_LABEL,
                            f"STATEMENT 1 | H2 | After {core.TRAUMA_EXPOSURE_LABEL} vs general population | "
                            f"within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    (n_pre, rr_pre), (n_post, rr_post), n_gen_pp = compute_prepost_window_rr(
        frame, core.TRAUMA_EXPOSURE_OUTCOMES, S1_DOWNSTREAM, core.FIXED_WINDOW_DAYS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle(f"STATEMENT 1 | {core.TRAUMA_EXPOSURE_LABEL} before vs on/after Oct 7 2023 | "
                f"IRR within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}",
                fontsize=11, fontweight="bold")
    draw_prepost_heatmap(axes[0], rr_pre, n_pre, n_gen_pp, S1_DOWNSTREAM, "Before Oct 7")
    draw_prepost_heatmap(axes[1], rr_post, n_post, n_gen_pp, S1_DOWNSTREAM, "On/after Oct 7")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    draw_prepost_bar_chart(ax, rr_pre, rr_post, n_pre, n_post, S1_DOWNSTREAM,
                           f"STATEMENT 1 | {core.TRAUMA_EXPOSURE_LABEL} before vs on/after Oct 7 2023 | "
                           f"bar-chart view of the heatmap above | {age_label}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def draw_t2b_panel(ax, frame, downstream_outcome, title):
    res = compute_t2b_rates(frame, downstream_outcome)
    rate_vals = [res["rate_pre"] if pd.notna(res["rate_pre"]) else 0,
                res["rate_post"] if pd.notna(res["rate_post"]) else 0]
    bars = ax.bar([0, 1], rate_vals, color=["#95a5a6", "#e63946"], alpha=0.85, width=0.5)
    for bar, n, rate in zip(bars, [res["n_pre"], res["n_post"]], rate_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f"{rate:.1f} / 1,000 PY\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Pre-Oct7 {core.TRAUMA_EXPOSURE_LABEL}\n(2017-2023)",
                       f"Post-Oct7 {core.TRAUMA_EXPOSURE_LABEL}\n(war period)"], fontsize=9)
    ax.set_ylabel(f"Rate within {core.FIXED_WINDOW_DAYS//30}mo of entry\nper 1,000 person-years", fontsize=8)
    rr_txt = f"{res['rr']:.2f}" if pd.notna(res["rr"]) else "n/a"
    ax.set_title(f"{title}\nIRR(post/pre)={rr_txt}  Poisson exact p={res['pval']}",
               fontsize=10, fontweight="bold")


def draw_t5_panel(ax, frame, entry_meta, downstream_outcome, title):
    labels, sides = [f"Pre-Oct7 {core.TRAUMA_EXPOSURE_LABEL}", f"Post-Oct7 {core.TRAUMA_EXPOSURE_LABEL}"], ["pre", "post"]
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


def draw_t6_panel(ax, frame, downstream_meta, downstream_outcome, classifier, title):
    labels, sides = [f"Pre-Oct7 {core.TRAUMA_EXPOSURE_LABEL}", f"Post-Oct7 {core.TRAUMA_EXPOSURE_LABEL}"], ["pre", "post"]
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
    ax.set_xticks(x); ax.set_xticklabels([f"{s}-Oct7 {core.TRAUMA_EXPOSURE_LABEL}" for s in sides], fontsize=9)
    ax.set_ylabel(f"{downstream_outcome.capitalize()} rate per 1,000", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)


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
    ax.set_xticks(x); ax.set_xticklabels([f"{s}-Oct7 {core.TRAUMA_EXPOSURE_LABEL}" for s in sides], fontsize=9)
    ax.set_ylabel(f"{downstream_outcome.capitalize()} rate per 1,000", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)


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
    ax.set_xlabel(f"Year of {core.TRAUMA_EXPOSURE_LABEL} entry (Oct 7 2023 marked, "
                 f"2023 split pre/post)", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(f"{title}  (n_gen={n_gen:,}, constant across periods)", fontsize=10, fontweight="bold")


def draw_by_second_diagnosis_year_panel(ax, frame, downstream_outcome, title, min_n=0):
    oct7_frac = 2023 + (OCT7 - pd.Timestamp("2023-01-01")).days / 365

    def period_x(start, end):
        # continuous x-position = the period's own midpoint, in fractional-year units --
        # derived directly from its actual date range, not a hard-coded shift for one special
        # case. A whole-year period sits at ~year+0.5; the split 2023 sub-periods each sit at
        # their own true midpoint (~Q2 for pre, ~Nov 20 for post).
        mid = start + (end - start) / 2
        return mid.year + (mid - pd.Timestamp(f"{mid.year}-01-01")).days / 365

    for side, label_txt, color in [("pre", f"{core.TRAUMA_EXPOSURE_LABEL} before Oct 7", "#95a5a6"),
                                   ("post", f"{core.TRAUMA_EXPOSURE_LABEL} on/after Oct 7", "#e63946")]:
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
    ax.set_ylabel(f"IRR (exposed / general), within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")


def draw_first_second_boxplot_combined(ax, frame, downstream_outcome, min_n=0,
                                       exposure_outcomes=None, first_label="trauma"):
    """exposure_outcomes / first_label: the first diagnosis (default: the active trauma exposure)."""
    values = [compute_second_diagnosis_boxplot_values(frame, downstream_outcome, side, event_side,
                                                      min_n, exposure_outcomes)
              for side, event_side in [("pre", "pre"), ("post", "post"), ("pre", "post")]]
    gray_pre, red_post, gray3 = values

    bp = ax.boxplot([gray_pre, red_post, gray3], positions=[1, 2, 3], widths=0.5,
                    labels=[f"{first_label} before Oct7\n(2nd dx: also before Oct7)",
                          f"{first_label} on/after Oct7\n(2nd dx: also on/after Oct7)",
                          f"{first_label} before Oct7\n(2nd dx: on/after Oct7)"],
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

    ax.set_ylabel(f"IRR, within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(f"{downstream_outcome.upper()}\n"
               f"one-tail t-test, box 1 (gray) > box 2 (red): p={pval_str}  "
               f"(n: {len(gray_pre)}, {len(red_post)}, {len(gray3)})\n"
               f"box 3 (gray, right) is NOT part of the test",
               fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)


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
    ax.set_ylabel(f"Days from {core.TRAUMA_EXPOSURE_LABEL} to 2nd diagnosis\n"
                 f"(converters within {core.FIXED_WINDOW_DAYS//30}mo)", fontsize=8)
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


def draw_km_cumulative_incidence(ax, frame, downstream_outcome, curves=(1, 2)):
    """Draws the two KM groups in `curves` (keys of KM_GROUP_STYLES) and log-rank tests them."""
    groups = {g: km_intervals(frame, downstream_outcome, g) for g in curves}
    horizon = core.FIXED_WINDOW_DAYS
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
    ax.set_xlabel(f"Months since {core.TRAUMA_EXPOSURE_LABEL} diagnosis", fontsize=8)
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
                    f"within {core.FIXED_WINDOW_DAYS//30}mo of {core.TRAUMA_EXPOSURE_LABEL} entry | {age_label}\n"
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
                f"each point = one patient who converted within {core.FIXED_WINDOW_DAYS//30}mo | "
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
                    f"{core.TRAUMA_EXPOSURE_LABEL}, PATIENT-LEVEL (1 - Kaplan-Meier) | {age_label}\n"
                    f"{pair_label} | all exposed patients; follow-up ending without a 2nd dx "
                    f"(window / left Clalit / study end) is censored\n"
                    f"{delayed_note}shaded = 95% CI",
                    fontsize=10, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_km_cumulative_incidence(ax, frame, outcome, curves)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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
    ax.set_xlabel(f"Year of {core.TRAUMA_EXPOSURE_LABEL} entry (fit uses 2017 .. 2023-pre only, "
                 f"n_fit={result['n_fit']})", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")


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
    ax.set_xlabel(f"Quarter of {core.TRAUMA_EXPOSURE_LABEL} entry (Oct 7 2023 marked)", fontsize=8)
    ax.set_ylabel(f"IRR (exposed / general), within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(f"{title}  (n_gen={n_gen:,}, constant across quarters)", fontsize=10, fontweight="bold")


# --- PART 1B, STEP 5: Write the 4 supporting-test pages for one age band ---
def write_statement1_supporting(pdf, frame, entry_meta, downstream_meta, age_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 2b | {core.TRAUMA_EXPOSURE_LABEL} pre vs post Oct 7, "
                f"restricted person-time within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_t2b_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 5 | Entry-diagnosis profession among "
                f"converters within {core.FIXED_WINDOW_DAYS//30}mo, pre vs post Oct 7 | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_t5_panel(ax, frame, entry_meta, outcome, f"-> {outcome.upper()}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 6 | Downstream subtype at conversion "
                f"within {core.FIXED_WINDOW_DAYS//30}mo, pre vs post Oct 7 trauma entry | {age_label}",
                fontsize=11, fontweight="bold")
    draw_t6_panel(axes[0], frame, downstream_meta, "psychosis", classify_psychosis_subtype, "PSYCHOSIS subtype")
    draw_t6_panel(axes[1], frame, downstream_meta, "bipolar", classify_bipolar_subtype, "BIPOLAR subtype")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST 9 | Trauma engagement severity "
                f"(inpatient vs outpatient), within {core.FIXED_WINDOW_DAYS//30}mo, "
                f"pre vs post Oct 7 | {age_label}",
                fontsize=11, fontweight="bold")
    print_facility_type_diagnostic(entry_meta)
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_t9_panel(ax, frame, entry_meta, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | TEST Zone | direct (Near Gaza / Northern "
                f"Border) vs indirect exposure, within {core.FIXED_WINDOW_DAYS//30}mo, "
                f"pre vs post Oct 7 | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_zone_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | Yearly trend | continuous RR by year of "
                f"{core.TRAUMA_EXPOSURE_LABEL} entry, within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}\n"
                f"gradual pre-existing decline vs a sharp break at Oct 7?",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_yearly_trend_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"STATEMENT 1 (supporting) | By year of SECOND diagnosis | within "
                f"{core.FIXED_WINDOW_DAYS//30}mo of {core.TRAUMA_EXPOSURE_LABEL} entry | {age_label}\n"
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
                f"{core.TRAUMA_EXPOSURE_LABEL} entry, within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}\n"
                f"same question as Yearly trend, at 3-month resolution",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_quarterly_trend_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
