#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table S2 - baseline characteristics of the Tai Chi randomised controlled trial.

The 103 participants (Tai Chi n = 49, waitlist control n = 54) are compared
between arms on sex, age, years of education, the CES-D total and each of the
20 CES-D items.  Continuous variables are summarised as median (IQR) and tested
with the Mann-Whitney U statistic; sex is tested with Pearson's chi-squared.

Conventions
-----------
The published table mixes two normal approximations for the U statistic, and
this script reproduces both:

* The 20 CES-D items use the **uncorrected** normal approximation,
  ``z = (U - n1 n2 / 2) / sqrt(n1 n2 (n + 1) / 12)``, with a two-sided normal
  P.  This reproduces every published item Z and P to two decimals (for example
  Sad z = 1.588 -> 1.59, Dislike z = -1.532 -> -1.53).
* The three continuous rows (age, years of education, CES-D total) use the
  **tie-corrected** normal approximation, i.e. scipy's asymptotic default.
  The published magnitudes and P values follow this (age |z| = 1.744, P = 0.081;
  CES-D total |z| = 0.589, P = 0.558), with the sign reversed relative to the
  Tai-Chi-minus-control direction used here.
* Sex uses the uncorrected Pearson chi-squared test, matching the convention
  adopted throughout the manuscript (chi2 = 0.087, P = 0.768).

Three cells of the published table are not reproducible from the deposited data
and are flagged by ``--check``; all three look like manual entry slips, since
the remaining 21 rows agree cell for cell (including the total-sample and Tai
Chi columns of the CES-D total, and every other item IQR under the same
quantile algorithm):

* the control-arm median (IQR) of the CES-D total, computed here as
  20.50 (16.00, 26.75) versus 22.00 (16.50, 27.50) in the manuscript -- no
  standard quantile algorithm (R types 1-9, SPSS, Excel) returns a median of
  22.00 for this column;
* the control-arm upper quartile of Appetite, computed here as 1.00 versus
  1.25 in the manuscript -- again not returned by any standard quantile
  algorithm;
* the years-of-education P value, computed here as 0.469 versus 0.479 in the
  manuscript, while its Z (-0.725) is reproduced exactly.

The published sign of the Mann-Whitney Z for age and for the CES-D total is
reversed relative to the Tai-Chi-minus-control direction used here; the sign of
a Mann-Whitney statistic is arbitrary (it depends on the order of the two
groups), so the script reports only a note for those two rows.

Inputs (data/, see data/README.md)
----------------------------------
    RCT_总103例_CESD.xlsx    sheets 干预前 / 干预后, read through
                             ``src/rct/rct_intervention.load_rct``

Outputs (results/)
------------------
    table_s2_rct_baseline.csv
    table_s2_rct_baseline.md
    表S2_RCT基线特征.docx

Usage
-----
    python tables/table_s2_rct_baseline.py           # write the table
    python tables/table_s2_rct_baseline.py --check   # also diff vs published
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "src" / "rct")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import OUT_DIR  # noqa: E402
from rct_intervention import ITEM_NAMES, load_rct  # noqa: E402

CSV = OUT_DIR / "table_s2_rct_baseline.csv"
MD = OUT_DIR / "table_s2_rct_baseline.md"
DOCX = OUT_DIR / "表S2_RCT基线特征.docx"

# --------------------------------------------------------------------------
# Published values (NMH_Supple20261006.docx, Table S2), used by --check only.
# Medians are (total sample, Tai Chi, control); Z and P are the published cells.
# --------------------------------------------------------------------------
# Each entry is (total median, Tai Chi median, control median, statistic string,
# P string), transcribed verbatim so that the comparison respects the precision
# actually printed in the supplement.
PUBLISHED = {
    "Sex (male)":      ("33 (32.04%)", "15 (30.61%)", "18 (33.33%)", "0.087", "0.768"),
    "age":             ("19.00 (18.00, 20.00)", "19.00 (19.00, 20.00)",
                        "19.00 (18.00, 20.00)", "-1.744", "0.081"),
    "education_years": ("14.00 (13.00, 14.00)", "13.00 (13.00, 14.00)",
                        "14.00 (13.00, 14.00)", "-0.725", "0.479"),
    "CES-D total":     ("20.00 (17.00, 27.00)", "20.00 (18.00, 27.00)",
                        "22.00 (16.50, 27.50)", "-0.589", "0.558"),
    "Bothered":        ("1.00 (1.00, 1.00)", "1.00 (0.00, 1.00)",
                        "1.00 (1.00, 1.00)", "-1.13", "0.260"),
    "Appetite":        ("1.00 (0.00, 1.00)", "1.00 (0.00, 1.00)",
                        "1.00 (0.00, 1.25)", "-0.72", "0.470"),
    "Blues":           ("1.00 (1.00, 1.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 1.00)", "0.36", "0.716"),
    "AsGood":          ("2.00 (1.00, 2.00)", "2.00 (1.00, 2.00)",
                        "2.00 (1.00, 2.00)", "0.90", "0.369"),
    "Mind":            ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 2.00)", "0.28", "0.779"),
    "Depressed":       ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 1.00)", "0.77", "0.444"),
    "Effort":          ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 1.75)", "0.96", "0.338"),
    "Hopeful":         ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.50 (1.00, 2.00)", "-0.79", "0.428"),
    "Failure":         ("1.00 (1.00, 1.00)", "1.00 (1.00, 1.00)",
                        "1.00 (1.00, 1.00)", "-0.30", "0.764"),
    "Fearful":         ("1.00 (0.00, 1.00)", "1.00 (1.00, 1.00)",
                        "1.00 (0.00, 2.00)", "0.38", "0.704"),
    "Sleep":           ("1.00 (0.00, 2.00)", "1.00 (0.00, 1.00)",
                        "1.00 (0.00, 2.00)", "-1.26", "0.208"),
    "Happy":           ("2.00 (1.00, 2.00)", "2.00 (1.00, 2.00)",
                        "2.00 (1.00, 2.00)", "0.91", "0.362"),
    "TalkedLess":      ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 2.00)", "0.43", "0.665"),
    "Lonely":          ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 2.00)", "-0.26", "0.797"),
    "Unfriendly":      ("1.00 (0.00, 1.00)", "1.00 (0.00, 1.00)",
                        "1.00 (0.00, 1.00)", "-0.44", "0.663"),
    "EnjoyedLife":     ("2.00 (1.00, 2.00)", "2.00 (1.00, 2.00)",
                        "2.00 (1.00, 2.00)", "0.54", "0.588"),
    "Crying":          ("1.00 (1.00, 2.00)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 2.00)", "0.12", "0.908"),
    "Sad":             ("1.00 (1.00, 1.50)", "1.00 (1.00, 2.00)",
                        "1.00 (1.00, 1.00)", "1.59", "0.112"),
    "Dislike":         ("1.00 (0.00, 1.00)", "1.00 (0.00, 1.00)",
                        "1.00 (0.00, 1.00)", "-1.53", "0.126"),
    "Going":           ("0.00 (0.00, 1.00)", "0.00 (0.00, 1.00)",
                        "0.00 (0.00, 0.00)", "0.61", "0.544"),
}

_VAR_KEY = {"Age (years)": "age", "Years of education": "education_years",
            "CES-D total score": "CES-D total", "Male, n (%)": "Sex (male)"}


def _key(variable: str) -> str | None:
    if variable in _VAR_KEY:
        return _VAR_KEY[variable]
    if variable.startswith("CES-D_"):
        return variable[len("CES-D_"):]
    return variable if variable in PUBLISHED else None


def _median_iqr(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.2f} ({q1:.2f}, {q3:.2f})"


def _manwhitney_z(a: np.ndarray, b: np.ndarray, tie_corrected: bool) -> float:
    """Standardised U statistic; ``a`` is the Tai Chi arm, ``b`` the control arm."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided",
                           method="asymptotic").statistic
    n1, n2 = len(a), len(b)
    n = n1 + n2
    mu = n1 * n2 / 2
    if tie_corrected:
        _, counts = np.unique(np.concatenate([a, b]), return_counts=True)
        ties = np.sum(counts ** 3 - counts)
        sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    else:
        sigma = np.sqrt(n1 * n2 * (n + 1) / 12)
    return float((u - mu) / sigma)


def _z_p(a: np.ndarray, b: np.ndarray, tie_corrected: bool):
    z = _manwhitney_z(a, b, tie_corrected)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def _fmt_z(z: float) -> str:
    return f"{z:.3f}" if abs(z) >= 0.0005 else "0.000"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _n_pct(k: int, n: int) -> str:
    return f"{k} ({100 * k / n:.2f}%)"


def build_table():
    pre_x, _post_x, _items, group, cov = load_rct()
    total = pre_x.sum(axis=1).values.astype(float)
    tc, ctl = group == 1, group == 0

    rows = []

    # ---- sex (uncorrected Pearson chi-squared) --------------------------
    male = cov["male"].astype(float)
    table = np.array([[int(((male == 1) & tc).sum()), int(((male == 0) & tc).sum())],
                      [int(((male == 1) & ctl).sum()), int(((male == 0) & ctl).sum())]])
    chi2, p_sex = stats.chi2_contingency(table, correction=False)[:2]
    n_tc, n_ctl = int(tc.sum()), int(ctl.sum())
    rows.append({"Variable": "Sex", "Total sample": "", "Tai Chi group": "",
                 "Control group": "", "χ²/Z": "", "P": ""})
    rows.append({"Variable": "Male, n (%)",
                 "Total sample": _n_pct(int(male.sum()), len(male)),
                 "Tai Chi group": _n_pct(int(male[tc].sum()), n_tc),
                 "Control group": _n_pct(int(male[ctl].sum()), n_ctl),
                 "χ²/Z": f"{chi2:.3f}", "P": _fmt_p(p_sex)})
    rows.append({"Variable": "Female, n (%)",
                 "Total sample": _n_pct(int((male == 0).sum()), len(male)),
                 "Tai Chi group": _n_pct(int((male == 0)[tc].sum()), n_tc),
                 "Control group": _n_pct(int((male == 0)[ctl].sum()), n_ctl),
                 "χ²/Z": "", "P": ""})

    # ---- continuous rows: tie-corrected Z -------------------------------
    continuous = [("Age (years)", cov["age"].astype(float)),
                  ("Years of education", cov["education_years"].astype(float)),
                  ("CES-D total score", total)]
    for label, v in continuous:
        a, b = v[tc], v[ctl]
        z, p = _z_p(a, b, tie_corrected=True)
        rows.append({"Variable": label,
                     "Total sample": _median_iqr(v),
                     "Tai Chi group": _median_iqr(a),
                     "Control group": _median_iqr(b),
                     "χ²/Z": _fmt_z(z), "P": _fmt_p(p)})

    # ---- the 20 CES-D items: uncorrected Z ------------------------------
    for j, name in enumerate(ITEM_NAMES):
        v = pre_x.values[:, j].astype(float)
        a, b = v[tc], v[ctl]
        z, p = _z_p(a, b, tie_corrected=False)
        rows.append({"Variable": f"CES-D_{name}",
                     "Total sample": _median_iqr(v),
                     "Tai Chi group": _median_iqr(a),
                     "Control group": _median_iqr(b),
                     "χ²/Z": _fmt_z(z), "P": _fmt_p(p)})

    df = pd.DataFrame(rows)
    print(f"[s2] N = {len(male)} (Tai Chi {n_tc} / control {n_ctl}); "
          f"sex chi2 = {chi2:.3f}, P = {p_sex:.3f}")
    return df


# --------------------------------------------------------------------------
# published-value regression check
# --------------------------------------------------------------------------
def _dec(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def check(df: pd.DataFrame):
    """Compare against the published table at the printed precision.

    Returns ``(substantive, notes)``: ``substantive`` are differences that could
    change the reading of the table (a median that differs, or a P that differs
    beyond the third decimal); ``notes`` are sign-only differences in Z, which
    are arbitrary because the sign of a Mann-Whitney statistic depends on the
    order of the two groups.
    """
    substantive, notes = [], []
    for _, r in df.iterrows():
        var = r["Variable"]
        key = _key(var)
        if key is None:
            continue
        med_t, med_tc, med_ctl, stat_s, p_s = PUBLISHED[key]
        for what, got, want in [("median total", r["Total sample"], med_t),
                                ("median Tai Chi", r["Tai Chi group"], med_tc),
                                ("median control", r["Control group"], med_ctl)]:
            if str(got) != want:
                substantive.append(f"{var:22s} {what:15s} pipeline={got:<21s} published={want}")

        got_s = str(r["χ²/Z"])
        if got_s not in ("", "nan"):
            g, w = float(got_s), float(stat_s)
            d = _dec(stat_s)
            if f"{abs(g):.{d}f}" != f"{abs(w):.{d}f}":
                substantive.append(f"{var:22s} {'statistic':15s} pipeline={got_s:<21s} published={stat_s}")
            elif np.sign(g) != np.sign(w):
                notes.append(f"{var:22s} Z sign only      pipeline={got_s:<21s} published={stat_s}")

        got_p = str(r["P"])
        if got_p not in ("", "nan"):
            if got_p == "<0.001":
                gp = 0.0
            else:
                gp = float(got_p)
            wp = float(p_s)
            if abs(gp - wp) > 0.005:
                substantive.append(f"{var:22s} {'P':15s} pipeline={got_p:<21s} published={p_s}")
    return substantive, notes


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_docx(df: pd.DataFrame, path: Path) -> None:
    try:
        from docx import Document
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.shared import Pt
    except ImportError:
        print("[skip] python-docx not installed; docx not written")
        return

    doc = Document()
    cap = doc.add_paragraph()
    run = cap.add_run("Table S2 Baseline characteristics of the Tai Chi "
                      "randomised controlled trial")
    run.bold = True
    run.font.size = Pt(10.5)
    rfonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rfonts.set(qn(attr), "Times New Roman" if attr != "w:eastAsia" else "宋体")

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(10)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, h in enumerate(df.columns):
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(row[h]))
            r.font.size = Pt(10)
    doc.save(path)
    print(f"[export] {path}")


def main() -> None:
    df = build_table()
    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    MD.write_text(to_markdown(df), encoding="utf-8")
    print(f"[export] {CSV}\n[export] {MD}")
    to_docx(df, DOCX)

    if "--check" in sys.argv:
        substantive, notes = check(df)
        print(f"\n[s2] substantive differences from the published Table S2: "
              f"{len(substantive)}")
        for d in substantive:
            print("  " + d)
        print(f"[s2] Z sign-only differences (sign of a Mann-Whitney Z is "
              f"arbitrary): {len(notes)}")
        for d in notes:
            print("  " + d)


if __name__ == "__main__":
    main()
