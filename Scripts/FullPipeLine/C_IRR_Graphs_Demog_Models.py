# C_IRR_Graphs_Demog_Models.py -- GRAPHS 3/3: demographic comparison / insights and
# relative-year PDFs, adjusted IRR forest plots, era comparison love / forest pages, gray
# section dividers, and the descriptive companion pages that include charts.
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
    FOCUS_OUTCOMES, MISSING_PROFESSION_CODES, OCT7, REGISTRY_START, S1_DOWNSTREAM,
    STUDY_END, ZONE_LABELS, ZONE_ORDER, compute_general_rate_overall,
    current_alt_exposures, first_of, get_exposure_meta_field, normalize_profession_code,
    profession_code_label, set_trauma_mode,
)
from C_IRR_Analyses import (  # noqa: F401
    FOLLOWUP_END_REASONS, PRIOR_HISTORY_OUTCOMES, chi_square_pvalue, classify_trauma_composition,
    classify_trauma_engagement, compute_relative_year_irr, converted_within_window,
    eligible_reachable_mask, exposed_followup_detail, has_any_prior_history,
    quarterly_person_years, split_trauma_pre_post, trauma_component_of_entry,
)
from C_IRR_Models import (  # noqa: F401
    ADJ_COVARIATES, ERA_AGE_BANDS, ERA_COMPARISONS, ERA_DIST3, ERA_DIST3_LEVELS, SMD_THRESHOLD,
    _era_group_label, _exposure_row, _level_order, adjusted_irr_results, covariate_reference,
    era_cohort, era_outcomes, era_results, group_cohort, smd_rows,
)
from C_IRR_Tables import (  # noqa: F401
    _fmt_irr, _fmt_p, draw_table_page, rtl,
)
from C_IRR_Graphs_Statement1 import (  # noqa: F401
    OUTCOME_COLORS, ZONE_COLORS,
)


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
        entry = first_of(sub, core.TRAUMA_EXPOSURE_OUTCOMES)
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
    profession = get_exposure_meta_field(entry_meta, core.TRAUMA_EXPOSURE_OUTCOMES, "profession")
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
    facility = get_exposure_meta_field(entry_meta, core.TRAUMA_EXPOSURE_OUTCOMES, "facility type")
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
    entry = first_of(post, core.TRAUMA_EXPOSURE_OUTCOMES)
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
    trauma_mode = trauma_mode or core.SELECTED_TRAUMA_MODE
    set_trauma_mode(trauma_mode)
    pre, post = split_trauma_pre_post(frame_full)
    print(f"\n=== Demographic comparison | trauma={core.TRAUMA_EXPOSURE_LABEL} | "
         f"n_pre={len(pre):,} | n_post={len(post):,} ===")

    output_pdf = os.path.join(output_dir, f"demographic_comparison_{trauma_mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic comparison | {core.TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | age, health burden", fontsize=11, fontweight="bold")
        draw_continuous_demo_panel(axes[0], pre, post, "age_at_oct7", "AGE (at Oct 7)", "Age (years)")
        draw_continuous_demo_panel(axes[1], pre, post, "charlson_score",
                                   "CHARLSON COMORBIDITY INDEX", "Charlson score")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic comparison | {core.TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | gender, geographic zone", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "gender", "GENDER")
        draw_categorical_demo_panel(axes[1], pre, post, "zone", "GEOGRAPHIC ZONE (distance to Gaza)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.suptitle(f"Demographic comparison | {core.TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | sector, district", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "sector", "SECTOR")
        draw_categorical_demo_panel(axes[1], pre, post, "district", "DISTRICT")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic comparison | {core.TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | socioeconomic status, immigration", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "ses_5level", "SOCIOECONOMIC SCORE (5-level)")
        draw_categorical_demo_panel(axes[1], pre, post, "is_immigrant", "IS IMMIGRANT (has immigration date)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.suptitle(f"Demographic comparison | {core.TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
                    f"on/after Oct 7 | country of birth, mailing language", fontsize=11, fontweight="bold")
        draw_categorical_demo_panel(axes[0], pre, post, "country_of_birth", "COUNTRY OF BIRTH")
        draw_categorical_demo_panel(axes[1], pre, post, "mailing_language", "MAILING LANGUAGE")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        fig.suptitle(f"Demographic comparison | {core.TRAUMA_EXPOSURE_LABEL} diagnosed before vs "
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
    trauma_mode = trauma_mode or core.SELECTED_TRAUMA_MODE
    set_trauma_mode(trauma_mode)
    pre, post = split_trauma_pre_post(frame_full)
    print(f"\n=== Demographic insights (descriptive only) | trauma={core.TRAUMA_EXPOSURE_LABEL} | "
         f"n_pre={len(pre):,} | n_post={len(post):,} ===")

    output_pdf = os.path.join(output_dir, f"demographic_insights_{trauma_mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {core.TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | trauma composition, prior history",
                    fontsize=11, fontweight="bold")
        draw_trauma_composition_panel(axes[0], pre, post, show_stats=False)
        draw_prior_history_panel(axes[1], pre, post, show_stats=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {core.TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | prior history by type", fontsize=11, fontweight="bold")
        draw_prior_history_breakdown_panel(ax, pre, post)   # already descriptive-only
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.suptitle(f"Demographic insights (descriptive) | {core.TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | trauma diagnosis practice, ALL patients "
                    f"(not just converters)", fontsize=11, fontweight="bold")
        draw_trauma_all_patients_profession_panel(axes[0], pre, post, entry_meta_full, show_stats=False)
        draw_trauma_all_patients_facility_panel(axes[1], pre, post, entry_meta_full, show_stats=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {core.TRAUMA_EXPOSURE_LABEL} diagnosed "
                    f"before vs on/after Oct 7 | geography (continuous), SES (3-level)",
                    fontsize=11, fontweight="bold")
        draw_continuous_demo_panel(axes[0], pre, post, "distance_km",
                                   "DISTANCE TO GAZA (continuous)", "Distance (km)", show_stats=False)
        draw_categorical_demo_panel(axes[1], pre, post, "ses_3level", "SOCIOECONOMIC SCORE (3-level)",
                                    show_stats=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Demographic insights (descriptive) | {core.TRAUMA_EXPOSURE_LABEL} | "
                    f"timing of post-war diagnoses", fontsize=11, fontweight="bold")
        draw_days_since_oct7_panel(ax, post)   # already descriptive-only
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"  Saved: {output_pdf}")
    return output_pdf


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
    trauma_mode = trauma_mode or core.SELECTED_TRAUMA_MODE
    set_trauma_mode(trauma_mode)
    print(f"\n=== IRR by year since trauma diagnosis | trauma={core.TRAUMA_EXPOSURE_LABEL} ===")

    output_pdf = os.path.join(output_dir, f"irr_by_year_since_diagnosis_{trauma_mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"IRR by year since {core.TRAUMA_EXPOSURE_LABEL} diagnosis | capped at Oct 7 2023 | "
                    f"final PARTIAL year included\n"
                    f"pre-war trauma patients only (post-war have no elapsed time before Oct7)",
                    fontsize=11, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_relative_year_panel(ax, frame_full, outcome, outcome.upper(), full_years_only=False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"IRR by year since {core.TRAUMA_EXPOSURE_LABEL} diagnosis | capped at Oct 7 2023 | "
                    f"only COMPLETE years counted\n"
                    f"pre-war trauma patients only (post-war have no elapsed time before Oct7)",
                    fontsize=11, fontweight="bold")
        for ax, outcome in zip(axes, FOCUS_OUTCOMES):
            draw_relative_year_panel(ax, frame_full, outcome, outcome.upper(), full_years_only=True)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"  Saved: {output_pdf}")
    return output_pdf
ADJ_COLORS = {"crude": "#95a5a6", "adjusted": "#e63946"}


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
    adj_note = (f"Adjusted for attained age (5-yr bands), sex, marketing sector, SES (3-level, ref Medium), Charlson "
                f"(0/1/2/3+) -- Poisson regression, offset log(person-years), 95% Wald CI. "
                f"Same person-time as the crude IRR: within {core.FIXED_WINDOW_DAYS // 30}mo of entry, "
                f"membership end, age 15+, prevalent cases excluded.")
    results = {o: adjusted_irr_results(frame, core.TRAUMA_EXPOSURE_OUTCOMES, o) for o in FOCUS_OUTCOMES}

    # --- page 1: trauma, crude vs adjusted, overall and before / after Oct 7 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"ADJUSTED IRR | {core.TRAUMA_EXPOSURE_LABEL} vs general population | crude vs adjusted, "
                 f"overall and by entry before / on-after Oct 7 | {age_label}\n{adj_note}",
                 fontsize=9.5, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        res = results[outcome]
        rows = []
        for header, key, level in [(f"{core.TRAUMA_EXPOSURE_LABEL} (any time)", "overall", "exposed"),
                                   (f"{core.TRAUMA_EXPOSURE_LABEL} before Oct 7", "prepost", "before Oct 7"),
                                   (f"{core.TRAUMA_EXPOSURE_LABEL} on/after Oct 7", "prepost", "on/after Oct 7")]:
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
    fig.suptitle(f"ADJUSTED IRR | full model, all covariates | {core.TRAUMA_EXPOSURE_LABEL} (any time) "
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
                ref = covariate_reference(table, group)
            rows.append((f"{'Exposure' if group == 'exposure' else 'Attained age' if group == 'age' else group}"
                         f"  (ref: {rtl(ref)})", None, None, None, None))
            for r in g.itertuples():
                rows.append((f"   {rtl(r.level)}", r.irr, r.lo, r.hi, ADJ_COLORS["adjusted"]))
        draw_forest(ax, rows, f"-> {outcome.upper()}")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(wspace=0.55)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # --- page 3: all 8 entry exposures (Statement 2), crude vs adjusted, one model each ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f"ADJUSTED IRR | all entry diagnoses (Statement 2), each vs its own general "
                 f"population | crude vs adjusted | {age_label}\n{adj_note}",
                 fontsize=9.5, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        rows = []
        for exp_label, outcome_list in current_alt_exposures().items():
            if outcome in outcome_list:
                continue   # exposure == outcome; skip self-pair (as in B1)
            res = adjusted_irr_results(frame, outcome_list, outcome) if outcome_list != core.TRAUMA_EXPOSURE_OUTCOMES \
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
ERA_COLORS = {"crude": ADJ_COLORS["crude"], "adjusted": ADJ_COLORS["adjusted"],
              "adjusted + distance": "#9d0208", "subgroup": "#457b9d"}


def section_divider(pdf, title, subtitle=""):
    """Full gray page naming the next section -- separates the sections of the main PDF."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#d3d6da")
    fig.text(0.5, 0.55, title, ha="center", va="center", fontsize=30, fontweight="bold", color="#343a40")
    if subtitle:
        fig.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=14, color="#495057", wrap=True)
    pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)


def write_era_pages(pdf, frame, age_label):
    """One section per comparison in ERA_COMPARISONS, each preceded by a gray divider page."""
    cohort = era_cohort(frame)
    for k, (cmp, ref) in enumerate(ERA_COMPARISONS):
        if k > 0:
            section_divider(pdf, f"ERA COMPARISON {k + 1}",
                            f"{core.TRAUMA_EXPOSURE_LABEL}: {_era_group_label(cmp)}  vs  {_era_group_label(ref)}")
        write_era_comparison(pdf, frame, cohort, age_label, cmp, ref)


def write_era_comparison(pdf, frame, cohort, age_label, cmp="post", ref="pre"):
    x_label = core.TRAUMA_EXPOSURE_LABEL
    cap = f"{core.FIXED_WINDOW_DAYS // 30}mo"
    design_note = (f"Cohort: first-ever {x_label} from 2017 at age 15+. Person-time from {x_label} to the "
                   f"earliest of Y, {cap} after {x_label}, leaving Clalit, 30.6.2026; pre era also "
                   f"censored at 6.10.2023. Prevalent Y excluded.")
    if "crossing" in (cmp, ref):
        design_note += (f"\nCrossing group: {x_label} before Oct 7 and no Y by 7.10.2023; person-time from "
                        f"7.10.2023 to the earliest of Y, {cap} after {x_label}, leaving Clalit, 30.6.2026.")
    adj_note = ("Adjusted: age at X (" + "/".join(l for _, _, l in ERA_AGE_BANDS) + "), sex, marketing sector, "
                "SES (3-level, ref Medium), Charlson (0/1/2/3+). Poisson, offset log(person-years), 95% Wald CI.")
    vs = f"{cmp} vs {ref}"
    groups = pd.concat([group_cohort(cohort, ref), group_cohort(cohort, cmp)])
    n_ref, n_cmp = int((groups["era"] == ref).sum()), int((groups["era"] == cmp).sum())
    if n_ref == 0 or n_cmp == 0:
        print(f"  Era comparison {vs} skipped: {ref}={n_ref:,}, {cmp}={n_cmp:,} patients")
        return

    # --- Table 1: the two groups' whole X cohort ---
    t1 = smd_rows(groups, cmp=cmp, ref=ref)
    t1["Level"] = t1["Level"].map(rtl)
    t1["SMD"] = t1["SMD"].map(lambda s: "-" if pd.isna(s) else f"{s:+.3f}" + (" *" if abs(s) > SMD_THRESHOLD else ""))
    t1 = t1.rename(columns={"pre": f"{ref.capitalize()} (n={n_ref:,})", "post": f"{cmp.capitalize()} (n={n_cmp:,})"})
    draw_table_page(pdf, f"ERA COMPARISON | Table 1 | {_era_group_label(cmp)} vs {_era_group_label(ref)} | "
                         f"{age_label}", t1.set_index("Covariate"),
                    note=(f"SMD = standardized mean difference, {cmp} - {ref} (* = |SMD| > {SMD_THRESHOLD}: an "
                          f"imbalance worth noting).\nEvery confounder enters the adjusted model regardless of "
                          f"its SMD; 'Distance from Gaza' is a subgroup only; '{ERA_DIST3}' is a subgroup and "
                          f"also the extra confounder of the 'adjusted + distance' model.\n{design_note}"),
                    font_size=7)

    for outcome in era_outcomes():
        res = era_results(frame, cohort, outcome, cmp=cmp, ref=ref)
        pair = f"{x_label} -> {outcome.upper()}"
        if res is None:
            print(f"  Era comparison {vs} {pair}: nobody at risk, skipped")
            continue
        crude, adjusted = res["crude"], res["adjusted"]
        counts = (f"{ref}: {crude['events_ref']:,} events / {crude['py_ref']:,.0f} PY, "
                  f"{cmp}: {crude['events_cmp']:,} events / {crude['py_cmp']:,.0f} PY")

        # --- love plot: SMD per covariate level, in this pair's at-risk cohort ---
        smd = smd_rows(res["pt"], cmp=cmp, ref=ref)
        smd = smd[smd["Covariate"] != "Distance from Gaza"].iloc[::-1].reset_index(drop=True)
        labels = [f"{c}: {rtl(l)}" if l != "mean (SD)" else c for c, l in zip(smd["Covariate"], smd["Level"])]
        fig, ax = plt.subplots(figsize=(11, 0.28 * len(smd) + 2.2))
        abs_smd = smd["SMD"].abs()
        colors = np.where(abs_smd > SMD_THRESHOLD, "#e63946", "#2a9d8f")
        ax.hlines(range(len(smd)), 0, abs_smd.fillna(0), color="#dddddd", linewidth=1)
        ax.scatter(abs_smd, range(len(smd)), c=colors, s=30, zorder=3)
        ax.axvline(SMD_THRESHOLD, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(smd)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, max(abs_smd.max() * 1.15 if abs_smd.notna().any() else 0, 0.25))
        ax.set_xlabel(f"|SMD| {vs} (dashed line = 0.1)", fontsize=8)
        ax.set_title(f"LOVE PLOT | {pair} | {_era_group_label(cmp)} vs {_era_group_label(ref)} | {age_label}\n"
                     f"Crude IRR {vs}: {_fmt_irr(crude)}\n"
                     f"{counts}\n{design_note}", fontsize=8.5, fontweight="bold", loc="left")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- forest: adjusted IRR overall and per subgroup level, with the interaction p ---
        rows = [("Overall", None, None, None, None)]
        for kind in ["crude", "adjusted", "adjusted + distance"]:
            r = res[kind]
            rows.append((f"   {kind}", r["irr"], r["lo"], r["hi"], ERA_COLORS[kind]))
        for sg, (sg_rows, p_int) in res["subgroups"].items():
            rows.append((f"{sg}  (interaction p = {_fmt_p(p_int)})", None, None, None, None))
            for r in sorted(sg_rows, key=lambda r: _level_order(r["level"])):
                rows.append((f"   {rtl(r['level'])}  ({r['events_ref']:,} / {r['events_cmp']:,} events)",
                             r["irr"], r["lo"], r["hi"], ERA_COLORS["subgroup"]))
        fig, ax = plt.subplots(figsize=(11, 0.3 * len(rows) + 2.2))
        draw_forest(ax, rows, "", xlabel=f"Adjusted IRR {vs} (log scale), 95% CI")
        ax.set_title(f"FOREST | {pair} | adjusted IRR {_era_group_label(cmp)} vs {_era_group_label(ref)}, "
                     f"overall and per subgroup | "
                     f"{age_label}\nAdjusted IRR: {_fmt_irr(adjusted)}   |   Adjusted + distance: "
                     f"{_fmt_irr(res['adjusted + distance'])}\n{adj_note} 'Adjusted + distance' also adjusts "
                     f"for {ERA_DIST3} ({' / '.join(ERA_DIST3_LEVELS)}).\n"
                     f"Subgroup rows: group x subgroup interaction in the adjusted model; events = {ref} / {cmp}; "
                     f"interaction p = likelihood-ratio test.\n{design_note}",
                     fontsize=8.5, fontweight="bold", loc="left")
        plt.tight_layout()
        fig.subplots_adjust(right=0.8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
FOLLOWUP_END_COLORS = ["#e63946", "#95a5a6", "#457b9d", "#f4a261"]


def desc_statement1_pages(pdf, frame, age_label):
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    win = core.FIXED_WINDOW_DAYS // 30
    rows, prepost_rows = {}, {}
    for o in S1_DOWNSTREAM:
        down = frame[f"first_{o}"]
        det, n_prev = exposed_followup_detail(entry, down)
        g_rate, g_ev, g_py = compute_general_rate_overall(frame, core.TRAUMA_EXPOSURE_OUTCOMES, o)
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
    ax.set_title(f"DESCRIPTIVE | Slides 2-3, 9, 20-21 | new {core.TRAUMA_EXPOSURE_LABEL} entries per month "
                 f"(first-ever diagnosis) | {age_label}", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def desc_old_method_tests_page(pdf, frame, entry_meta, age_label):
    """Slides 5-8 use the older full-window design -- show who that design keeps and the base mix
    (all eligible exposed, not just converters) of profession / setting / zone, pre vs post."""
    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    eligible = entry.notna() & eligible_reachable_mask(entry)
    last_entry = STUDY_END - pd.to_timedelta(core.FIXED_WINDOW_DAYS, unit="D")
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
                             f"{core.FIXED_WINDOW_DAYS//30}-month window before study end), count a share of "
                             f"patients (not person-years), and do not stop at membership end.")

    profession = normalize_profession_code(
        get_exposure_meta_field(entry_meta, core.TRAUMA_EXPOSURE_OUTCOMES, "profession"))
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

    entry = first_of(frame, core.TRAUMA_EXPOSURE_OUTCOMES)
    rows = {}
    for z in ZONE_ORDER:
        m = frame["zone"] == z
        r = {"patients": int(m.sum()), "% of age band": m.mean() * 100,
             f"% ever {core.TRAUMA_EXPOSURE_LABEL}": entry[m].notna().mean() * 100 if m.any() else np.nan}
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
