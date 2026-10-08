# -*- coding: utf-8 -*-
"""
Intervention validation cohort: sensitivity of network metrics to Tai Chi.

Data come from a published 12-week randomised controlled trial of Tai Chi for
subthreshold depression (103 participants; Tai Chi vs waitlist control). The
workbook holds two sheets, pre- and post-intervention CES-D item scores.

Reported analyses
-----------------
    1  Baseline balance between the Tai Chi and control arms
       (the sex row reproduces that of Table S2)
    2  Pre-post change in CES-D total and in the SNCI, within and between arms
       (paired t within arms; independent-samples pooled t and Cohen's d between
       arms).  The CES-D total row is Fig. 5a.  The SNCI row here standardises
       against the pooled RCT network and therefore does NOT reproduce Fig. 5b:
       Fig. 5b, 5c and 5d all use the SNCI and energy projected onto the 3,447-
       participant FDSC reference network and are produced by
       ``src/rct/rct_snci_projection.py``.
    3  Symptom-level response rate (>= 50% improvement per item), with
       Benjamini-Hochberg correction across the 20 items (Table S10 / Fig. 5e)

Notes on conventions
--------------------
Between-group comparisons of response rates and of the baseline sex split use
the uncorrected Pearson chi-squared test, and Fisher's exact test is substituted
when the smallest *expected* frequency falls below 5 (Methods and Fig. 5 legend
of the manuscript).  This is the scipy default off: ``correction=False``.  Using
scipy's default Yates correction changes the raw P values (for example
Depressed: uncorrected P = 0.0023 versus corrected P = 0.0046) and therefore the
Benjamini-Hochberg conclusion (q = 0.045 versus 0.076).

Outputs (results/):
    rct_baseline_balance.csv
    rct_change_summary.csv
    rct_snci_cesd_correlation.csv
    rct_response_rates.csv

Usage:
    python src/rct/rct_intervention.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Network machinery is imported from this repository.  The routines are the
# same implementations the script was written against (nodewise LASSO + EBIC +
# AND symmetrisation; E = -s.h - 0.5 s.J.s; SNCI = clip(50 + 10 (E - mu)/sigma,
# 0, 100); percolation + Louvain; four-centrality voting), so the numbers are
# unchanged.  Only the import paths and the data/output locations differ from
# the original working copy.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DATA_DIR, OUT_DIR  # noqa: E402
from cross_cohort_logistic import compute_snci, fit_ising_network  # noqa: E402
from analysis_all import ITEM_NAMES, compute_energy  # noqa: E402

# CES-D cut-off for the high- vs low-symptom contrast (>= 16).
CESD_CUTOFF = 16

# RCT workbook, sheets 干预前 / 干预后.  Override with CESD_RCT_XLSX.
RCT_XLSX = Path(os.environ.get("CESD_RCT_XLSX",
                               DATA_DIR / "RCT_总103例_CESD.xlsx"))

# Column offsets in the RCT workbook: participant metadata occupy the first
# nine columns, the 20 CES-D items follow.
ITEM_START, ITEM_END = 9, 29
GROUP_MAP = {"学校随机干预组": 1, "学校随机对照组": 0}


def load_rct(path: Path | None = None):
    """Return (pre, post, item columns, group, covariates) from the workbook."""
    path = path or RCT_XLSX
    pre = pd.read_excel(path, sheet_name="干预前")
    post = pd.read_excel(path, sheet_name="干预后")
    items = list(pre.columns[ITEM_START:ITEM_END])
    if len(items) != 20:
        raise ValueError(f"expected 20 CES-D item columns, found {len(items)} in {path}")

    pre_x = pre[items].apply(pd.to_numeric, errors="coerce")
    post_x = post[items].apply(pd.to_numeric, errors="coerce")
    group = pre["批次"].map(GROUP_MAP).values
    cov = {
        "age": pd.to_numeric(pre["年龄"], errors="coerce").values,
        "male": (pre["性别"] == "男").astype(float).values,
        "education_years": pd.to_numeric(pre["受教育年限"], errors="coerce").values,
    }
    return pre_x, post_x, items, group, cov


def _paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test, guarding against all-zero differences."""
    d = np.asarray(b) - np.asarray(a)
    d = d[~np.isnan(d)]
    if np.allclose(d, 0):
        return float("nan"), 1.0
    return stats.wilcoxon(a, b)


def baseline_balance(pre_x, group, covariates) -> pd.DataFrame:
    """Compare the two arms at baseline on total score and covariates."""
    total = pre_x.sum(axis=1).values
    rows = []
    for name, values in [("CES-D total", total), *covariates.items()]:
        a, b = np.asarray(values)[group == 1], np.asarray(values)[group == 0]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"variable": name,
                     "tai_chi_mean_sd": f"{a.mean():.2f} ± {a.std(ddof=1):.2f}",
                     "control_mean_sd": f"{b.mean():.2f} ± {b.std(ddof=1):.2f}",
                     "statistic": round(float(t), 3), "p": round(float(p), 4)})
    n_tc, n_ct = int((group == 1).sum()), int((group == 0).sum())
    # Uncorrected Pearson chi-squared, matching the manuscript (Methods, Fig. 5
    # legend); scipy's Yates correction is switched off explicitly.
    chi2, p_sex = stats.chi2_contingency(
        pd.crosstab(group, covariates["male"]), correction=False)[:2]
    rows.append({"variable": "sex (male), n (%)",
                 "tai_chi_mean_sd": f"{int(covariates['male'][group == 1].sum())} "
                                    f"({100 * covariates['male'][group == 1].mean():.1f}%)",
                 "control_mean_sd": f"{int(covariates['male'][group == 0].sum())} "
                                    f"({100 * covariates['male'][group == 0].mean():.1f}%)",
                 "statistic": round(float(chi2), 3), "p": round(float(p_sex), 4)})
    print(f"[balance] Tai Chi n = {n_tc}, control n = {n_ct}")
    return pd.DataFrame(rows)


def _cohen_d_ind(a, b):
    """Cohen's d for a two-sample contrast (pooled SD)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def change_summary(pre_x, post_x, group, pre_snci, post_snci) -> pd.DataFrame:
    """Pre-post change in CES-D total and SNCI, within and between arms.

    Within an arm the paired t test and Cohen's d_z are reported alongside the
    Wilcoxon signed-rank test; between arms the pooled independent-samples t
    test and Cohen's d are reported on the change scores.  The CES-D total row
    is Fig. 5a (t = -2.91, d = -0.57, P = 0.004).  The SNCI row uses the pooled
    RCT network as its reference and is not Fig. 5b; see the module docstring.
    """
    total_pre = pre_x.sum(axis=1).values
    total_post = post_x.sum(axis=1).values
    d_total = total_post - total_pre
    d_snci = post_snci - pre_snci

    rows = []
    for name, pre, post, delta in [("CES-D total", total_pre, total_post, d_total),
                                   ("SNCI", pre_snci, post_snci, d_snci)]:
        for arm, mask in [("Tai Chi", group == 1), ("control", group == 0),
                          ("all", np.ones(len(group), bool))]:
            stat, p = _paired_wilcoxon(pre[mask], post[mask])
            t_paired, p_paired = stats.ttest_rel(pre[mask], post[mask])
            dz = float(delta[mask].mean() / delta[mask].std(ddof=1)) \
                if delta[mask].std(ddof=1) > 0 else float("nan")
            rows.append({"metric": name, "arm": arm, "n": int(mask.sum()),
                         "pre_mean": round(float(pre[mask].mean()), 2),
                         "post_mean": round(float(post[mask].mean()), 2),
                         "change_mean": round(float(delta[mask].mean()), 2),
                         "change_sd": round(float(delta[mask].std(ddof=1)), 2),
                         "t_paired": round(float(t_paired), 3),
                         "p_paired": round(float(p_paired), 4),
                         "cohen_dz": round(dz, 3),
                         "wilcoxon_stat": round(float(stat), 1) if np.isfinite(stat) else None,
                         "p_within": round(float(p), 4)})
        a, b = delta[group == 1], delta[group == 0]
        t, p_between = stats.ttest_ind(a, b, equal_var=True)
        rows.append({"metric": name, "arm": "Tai Chi vs control", "n": len(group),
                     "pre_mean": None, "post_mean": None,
                     "change_mean": round(float(a.mean() - b.mean()), 2),
                     "change_sd": None,
                     "t_paired": round(float(t), 3),
                     "p_paired": round(float(p_between), 4),
                     "cohen_dz": round(_cohen_d_ind(a, b), 3),
                     "wilcoxon_stat": None, "p_within": None})
    print(f"[change] pre-post CES-D change: Tai Chi {d_total[group == 1].mean():.2f}, "
          f"control {d_total[group == 0].mean():.2f}")
    print(f"[change] r(SNCI, CES-D total) at baseline = "
          f"{stats.pearsonr(pre_snci, total_pre)[0]:.3f}; "
          f"r(delta SNCI, delta total) = {stats.pearsonr(d_snci, d_total)[0]:.3f}")
    return pd.DataFrame(rows)


def snci_cesd_correlation(pre_x, post_x, group, pre_snci, post_snci) -> pd.DataFrame:
    """Pearson r between the changes in SNCI and in CES-D total.

    NOTE ON THE SNCI REFERENCE.  This script standardises the SNCI against the
    network fitted on the pooled RCT baseline data (see ``main``), so the
    correlations reported here (Tai Chi 0.76, control 0.75) are *not* the
    within-arm correlations in Fig. 5d (Tai Chi 0.71, control 0.66).  Fig. 5b
    (delta SNCI, t = -2.65) and Fig. 5d are obtained from the SNCI projected
    onto the 3,447-participant FDSC reference network, i.e. from
    ``src/rct/rct_snci_projection.py``
    (``SNCI_103_per_subject.csv``): regress and correlate against the CES-D
    total built from ``core_symptom_change_103_per_subject.csv``.
    Both values are worth reporting side by side, since they show that the
    SNCI-CES-D coupling in the trial does not depend on which reference network
    is used.
    """
    d_total = post_x.sum(axis=1).values - pre_x.sum(axis=1).values
    d_snci = post_snci - pre_snci
    rows = []
    for arm, mask in [("Tai Chi", group == 1), ("control", group == 0),
                      ("all", np.ones(len(group), bool))]:
        r, p = stats.pearsonr(d_snci[mask], d_total[mask])
        rows.append({"snci_reference": "pooled RCT baseline network", "arm": arm,
                     "n": int(mask.sum()),
                     "r_delta_snci_delta_cesd": round(float(r), 3),
                     "p": round(float(p), 6)})
    r0, _ = stats.pearsonr(pre_snci, pre_x.sum(axis=1).values)
    rows.append({"snci_reference": "pooled RCT baseline network",
                 "arm": "baseline (SNCI vs CES-D total)", "n": int(len(group)),
                 "r_delta_snci_delta_cesd": round(float(r0), 3), "p": None})
    print("[corr] SNCI standardised on the pooled RCT network (not the FDSC "
          "reference frame): r(delta SNCI, delta CES-D total) = "
          + "; ".join(f"{r['arm']} {r['r_delta_snci_delta_cesd']:.3f}"
                      for r in rows[:3]))
    return pd.DataFrame(rows)


def response_rates(pre_x, post_x, group) -> pd.DataFrame:
    """Per-item response rate (>= 50% improvement) with BH correction.

    Participants whose baseline item score is 0 are excluded from that item,
    since a relative improvement is undefined.

    Between-group testing follows the manuscript (Methods, Fig. 5 legend):
    uncorrected Pearson chi-squared, replaced by Fisher's exact test when the
    smallest expected frequency is below 5.  This reproduces Table S10.
    """
    items = list(pre_x.columns)
    p_pre = pre_x.values.astype(float)
    p_post = post_x.values.astype(float)
    rows = []
    for j, col in enumerate(items):
        base, follow = p_pre[:, j], p_post[:, j]
        eligible = base > 0
        resp = np.zeros(len(base), bool)
        resp[eligible] = (base[eligible] - follow[eligible]) / base[eligible] >= 0.5
        n_elig = int(eligible.sum())
        n_tc = int((resp & (group == 1)).sum())
        n_ct = int((resp & (group == 0)).sum())
        denom_tc = int((eligible & (group == 1)).sum())
        denom_ct = int((eligible & (group == 0)).sum())
        table = np.array([[n_tc, denom_tc - n_tc], [n_ct, denom_ct - n_ct]])
        expected = (table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True)
                    / table.sum())
        try:
            if expected.min() < 5:
                chi2_v, p = float("nan"), stats.fisher_exact(table)[1]
            else:
                chi2_v, p = stats.chi2_contingency(table, correction=False)[:2]
        except Exception:
            chi2_v, p = float("nan"), float("nan")
        rows.append({"item": ITEM_NAMES[j] if j < len(ITEM_NAMES) else col,
                     "n_eligible": n_elig,
                     "response_rate_tai_chi": round(n_tc / denom_tc, 4) if denom_tc else np.nan,
                     "response_rate_control": round(n_ct / denom_ct, 4) if denom_ct else np.nan,
                     "chi2": round(chi2_v, 2) if np.isfinite(chi2_v) else None,
                     "p_raw": p})
    df = pd.DataFrame(rows)
    order = df["p_raw"].fillna(1).argsort()
    q = np.empty(len(df))
    m = len(df)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, df["p_raw"].iloc[idx] * m / k)
        q[idx] = prev
    df["q_BH"] = np.round(q, 4)
    df["p_raw"] = df["p_raw"].round(4)
    sig = df.loc[df["q_BH"] < 0.05, "item"].tolist()
    print(f"[response] items significant after BH correction: {sig}")
    return df


def main() -> None:
    pre_x, post_x, items, group, cov = load_rct()

    # Networks are estimated on the pooled baseline data; the resulting h and J
    # are then applied to all participants at both time points, so the SNCI is
    # standardised against a single reference distribution.
    # This is the *pooled RCT* reference network, which is what the original
    # intervention-validation script used.  The manuscript's Fig. 5b-5d instead
    # project the trial onto the 3,447-participant FDSC reference network; that
    # path lives in src/rct/rct_snci_projection.py.
    binary_pre = (pre_x.values >= 2).astype(float)
    net = fit_ising_network(binary_pre)
    _, pre_snci, mu, sigma = compute_snci(binary_pre, net)
    # Post-intervention energy is evaluated with the same network parameters.
    energy_post = compute_energy((post_x.values >= 2).astype(float),
                                 net["intercept"], net["j_sym"])
    post_snci = np.clip(50 + 10 * (energy_post - mu) / sigma, 0, 100)

    print(f"[rct] N = {len(group)}, network edges = {net['n_edges']}, "
          f"mu = {mu:.3f}, sigma = {sigma:.3f}")

    balance = baseline_balance(pre_x, group, cov)
    changes = change_summary(pre_x, post_x, group, pre_snci, post_snci)
    corr = snci_cesd_correlation(pre_x, post_x, group, pre_snci, post_snci)
    response = response_rates(pre_x, post_x, group)

    for df, name in [(balance, "rct_baseline_balance.csv"),
                     (changes, "rct_change_summary.csv"),
                     (corr, "rct_snci_cesd_correlation.csv"),
                     (response, "rct_response_rates.csv")]:
        df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
        print(f"[export] {OUT_DIR / name}")


if __name__ == "__main__":
    main()
