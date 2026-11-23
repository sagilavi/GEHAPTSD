# C_IRR_Graphs_Statements2to6.py -- GRAPHS 2/3: chart panels and pages of Statement 2
# (+ extended), comorbidity, Statement 3 (zones, distance), Statement 5 (professions),
# Statement 6 (latency).
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
    ALL_OUTCOMES, FOCUS_OUTCOMES, S1_DOWNSTREAM, STATEMENT3_SECTIONS, ZONE_LABELS,
)
from C_IRR_Analyses import (  # noqa: F401
    COMORB_REF_LABELS, COMORB_STATES, DIST_BAND_LABELS, LATENCY_BIN_LABELS, MIN_N_THRESHOLDS,
    STATEMENT3_ZONES, TRAUMA_DX_KEY, comorb_state_labels, compute_b1_prepost_change_matrix,
    compute_b1_prepost_matrix, compute_b1_raw_rr, compute_comorbidity_tables,
    compute_latency_distribution, compute_pct_psychiatrist, compute_s2_full_matrices,
    distance_band, prepost_ratio_text, print_profession_diagnostic, quarterly_counts,
    quarterly_person_years, s2_targets, to_rate_per_1000_py,
)
from C_IRR_Graphs_Statement1 import (  # noqa: F401
    EXPOSURE_COLORS, OUTCOME_COLORS, ZONE_COLORS, draw_first_second_boxplot_combined,
)


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
# Statement 2), X-axis = the 8 alternate exposures, 2 bars each (pre-war IRR, post-war IRR). ---
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
    ax.set_ylabel(f"IRR (exposed / general), within {core.FIXED_WINDOW_DAYS//30}mo", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)


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
        kw = dict(cmap="YlOrRd", vmin=0, vmax=vmax, cbar_kws={"label": f"IRR within {core.FIXED_WINDOW_DAYS//30}mo"})
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
    targets = s2_targets(frame)
    overall, pre, post = compute_s2_full_matrices(frame)
    win = core.FIXED_WINDOW_DAYS // 30

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
    remaining = [key for key in ["anxiety", "depression", "substance", "personality", "OCD", TRAUMA_DX_KEY]
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
COMORB_STATE_COLORS = {"none": "#b0b0b0", "trauma": "#e63946", "x": "#457b9d", "both": "#2b2d42"}


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
                yticklabels=[f"{core.TRAUMA_EXPOSURE_LABEL} + {x}\n(n={int(n):,})" for x, n in zip(table.index, table["n_both"])],
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
    ax.set_title(f"{outcome.upper()} | {core.TRAUMA_EXPOSURE_LABEL} + {x_label}\n{irr_txt}",
                 fontsize=9, fontweight="bold")


def write_comorbidity_statement(pdf, frame, age_label):
    tables, details = compute_comorbidity_tables(frame)
    win = core.FIXED_WINDOW_DAYS // 30
    head = f"COMORBIDITY | {core.TRAUMA_EXPOSURE_LABEL} + another diagnosis X"
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
                f"{core.FIXED_WINDOW_DAYS//30}mo, restricted person-time | {age_label}",
                fontsize=11, fontweight="bold")
    for ax, outcome in zip(axes, FOCUS_OUTCOMES):
        draw_b1_panel(ax, frame, outcome, outcome.upper())
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    pre_matrix  = compute_b1_prepost_matrix(frame, "pre")
    post_matrix = compute_b1_prepost_matrix(frame, "post")
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    fig.suptitle(f"STATEMENT 2 | Entry diagnosis before vs on/after Oct 7 | "
                f"IRR within {core.FIXED_WINDOW_DAYS//30}mo | {age_label}",
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

    write_trauma_all_second_boxplot_pages(pdf, frame, age_label)


# --- S2, STEP 6: Year-IRR boxplots, trauma -> EVERY second diagnosis (pages right after the %
# change heatmap). The same 3-box chart as Statement 1's trauma boxplot (which covers psychosis
# and bipolar only), for all 7 non-trauma diagnoses, 2 per page, in every minimum-n version.
def write_trauma_all_second_boxplot_pages(pdf, frame, age_label):
    seconds = S1_DOWNSTREAM   # psychosis, bipolar, anxiety, depression, substance, personality, OCD
    for min_n in [0] + MIN_N_THRESHOLDS:
        n_note = f"only years with n>={min_n}" if min_n > 0 else "no minimum n"
        for i in range(0, len(seconds), 2):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig.suptitle(f"STATEMENT 2 | First diagnosis: {core.TRAUMA_EXPOSURE_LABEL} -> second: every other "
                         f"diagnosis | diagnosis-year IRR distribution ({n_note}) | {age_label}\n"
                         f"Boxes 1 and 2 are t-tested (1st > 2nd); box 3 is NOT tested\n"
                         f"CAUTION: each point = one YEAR's IRR, not one patient -- small-sample test",
                         fontsize=10, fontweight="bold")
            page = seconds[i:i + 2]
            for ax, outcome in zip(axes, page):
                draw_first_second_boxplot_combined(ax, frame, outcome, min_n)
            for ax in axes[len(page):]:
                ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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


# --- S3, STEP 4: Write Statement 3 pages (ROSE section, then DID-NOT-RISE section) ---
def write_statement3(pdf, frame, age_label):
    q_index = pd.period_range("2017Q1", "2026Q2", freq="Q")
    person_years = quarterly_person_years(frame, q_index)
    oct7_pos = next(i for i, q in enumerate(q_index) if q == pd.Period("2023Q4", freq="Q"))

    for section_label, items in STATEMENT3_SECTIONS:
        for item in items:
            label = "ALL 9 OUTCOMES" if item == "total" else item.upper()
            rate = to_rate_per_1000_py(quarterly_counts(frame, item), person_years)
            fig, ax = plt.subplots(figsize=(13, 5.5))
            draw_rate_panel(ax, rate, oct7_pos,
                            f"STATEMENT 3 | {section_label} | {label} | "
                            f"New dx per 1,000 person-years | Quarterly by zone | {age_label}",
                            subtitle=prepost_ratio_text(rate))
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    write_statement3_by_distance(pdf, frame, age_label)
DIST_BAND_COLORS = dict(zip(DIST_BAND_LABELS, ["#9d0208", "#e85d04", "#faa307", "#2a9d8f", "#264653"]))


def write_statement3_by_distance(pdf, frame, age_label):
    q_index = pd.period_range("2017Q1", "2026Q2", freq="Q")
    oct7_pos = next(i for i, q in enumerate(q_index) if q == pd.Period("2023Q4", freq="Q"))
    f = frame.assign(dist_band=distance_band(frame))
    f = f[f["dist_band"].notna()]
    person_years = quarterly_person_years(f, q_index, "dist_band", DIST_BAND_LABELS)
    shown = [b for b in DIST_BAND_LABELS if person_years[b].sum() > 0]
    year_ticks = [i for i, q in enumerate(q_index) if q.quarter == 1]

    for section_label, items in STATEMENT3_SECTIONS:
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
            ax.set_title("ALL 9 OUTCOMES" if item == "total" else item.upper(),
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left", ncol=3)
        for ax in list(axes.flat)[len(items):]:
            ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# --- S5, STEP 2: Draw one side's panel -- all 9 outcomes, % diagnosed by a psychiatrist ---
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


# --- S6, STEP 3: Write one page per non-trauma outcome (7 pages), Pre/Post panels each ---
def write_statement6(pdf, frame, age_label):
    for outcome in S1_DOWNSTREAM:   # psychosis, bipolar, anxiety, depression, substance, personality, OCD
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f"STATEMENT 6 | Latency: months from {core.TRAUMA_EXPOSURE_LABEL} entry to "
                    f"first {outcome.upper()} | {age_label}\n"
                    f"unrestricted by window -- shows full empirical distribution",
                    fontsize=11, fontweight="bold")
        draw_latency_panel(axes[0], frame, outcome, "pre",  "Pre-Oct7 entry")
        draw_latency_panel(axes[1], frame, outcome, "post", "Post-Oct7 entry")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
