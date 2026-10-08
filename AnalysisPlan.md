# Analysis plan — CES-D symptom networks, the Ising model and SNCI

**Network-derived clinical indicator of symptom system stability and core symptoms in
depression: a cross-cultural validation study**

Version 1.0 · last revised 2026-10-07

This document describes, in the order in which the analyses are run, the statistical
protocol behind every number reported in the manuscript and its supplement. It is the
narrative companion to the code: each section names the script that implements it and the
output file it writes. The plan is written after the analyses were completed and documents
the protocol as implemented, not a pre-registration.

**Contents**

1. [Cohorts and data](#1-cohorts-and-data)
2. [Item scoring and binarisation](#2-item-scoring-and-binarisation)
3. [Network estimation: the Ising model](#3-network-estimation-the-ising-model)
4. [Network energy and SNCI](#4-network-energy-and-snci)
5. [Network topology: percolation, communities, bridge strength](#5-network-topology-percolation-communities-bridge-strength)
6. [Core-symptom identification](#6-core-symptom-identification)
7. [Cross-cohort validation](#7-cross-cohort-validation)
8. [Sensitivity analyses](#8-sensitivity-analyses)
9. [Tai Chi RCT: intervention validation](#9-tai-chi-rct-intervention-validation)
10. [Software, seeds and file preparation](#10-software-seeds-and-file-preparation)
11. [References](#11-references)

---

## 1. Cohorts and data

Four cohorts contribute to the study. The FDSC cohort is the discovery sample; CFPS 2012
and MIDUS M2 are independent validation samples; the Tai Chi RCT provides the
intervention sample. Two of the four files are restricted (see
[`data/README.md`](data/README.md)).

| Role | Cohort | N | File |
|---|---|---|---|
| Discovery | FDSC | 3,447 | `3447例-CESD数据已反向计分(3).xlsx` |
| Validation | CFPS 2012 | 31,033 (youth 15–30 subgroup: 7,281) | `CFPS_cesd20_scores.csv` |
| Validation (longitudinal) | CFPS 2012 / 2016 matched | 3,976 | `CFPS_2012_2016_两期完整20项匹配(5).xlsx` |
| Validation | MIDUS M2 | 1,255 | `MIDUS_CESD20.xlsx` |
| Intervention | Tai Chi RCT | 103 (Tai Chi 49 / control 54) | `RCT_总103例_CESD.xlsx` |

The depressed-versus-non-depressed contrasts use the conventional CES-D cut-off of a
**total score ≥ 16** to define the high-symptom group.

## 2. Item scoring and binarisation

The 20 CES-D items are scored 0–3 ([Radloff, 1977](#11-references)). Items **4, 8, 12 and
16** are positively worded and are reverse-scored before any analysis. The FDSC data file
is already reverse-scored on disk (`REVERSED_ALREADY = True`); the reversion is applied
programmatically only for the cohorts that need it.

Each item is binarised at a cut-off of **≥ 2** on the 0–3 scale — a symptom is counted as
present if the respondent reports it "most or all of the time" at least at the moderate
level. Node states are then coded **s = 2x − 1 ∈ {−1, +1}** for estimation.

## 3. Network estimation: the Ising model

The 20 CES-D items are modelled as an **Ising** network: a pairwise Markov random field on
binary variables whose joint distribution is

$$P(s) \propto \exp\Big(\sum_{i} h_i s_i + \sum_{i<j} J_{ij} s_i s_j\Big),$$

where $h_i$ are the external-field (threshold) parameters and $J_{ij}$ the pairwise
interaction (edge) parameters.

Estimation follows the regularised pseudo-likelihood procedure of
[van Borkulo et al. (2014)](#11-references), implemented in
[`src/analysis_all.py`](src/analysis_all.py) (`fit_ising_network`) and reused by every
downstream script:

- **Nodewise L1-penalised logistic regression.** For each node $i$, a logistic regression
  of $s_i$ on all other nodes is fitted with the `saga` solver (`l1_ratio = 1.0`,
  `max_iter = 3000`, `tol = 1e-4`, `random_state = 42`).
- **Tuning by EBIC.** The penalty is chosen over a grid of 12 values,
  $C \in \{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0\}$, minimising
  the extended Bayesian information criterion ([Chen & Chen, 2008](#11-references))

  $$\mathrm{EBIC} = 2n\,\overline{\ell\ell} + k\log n + 2\gamma\,k\log(p-1),$$

  with $\gamma = 0.25$, $k$ the number of non-zero coefficients and $p = 20$ nodes.
- **AND symmetrisation.** An edge $(i,j)$ is retained only if **both** directions are
  non-zero ($|\beta| > 10^{-4}$); the two coefficients are then averaged,
  $J = (\beta_{i \to j} + \beta_{j \to i})/2$, preserving sign. This AND rule is used
  instead of the OR rule throughout.
- **Rescaling to the Ising parameters.** Because the logistic regressions are fitted in
  the ±1 domain, the coefficients are divided by two: $h = \beta_0/2$ and $J = \beta/2$.

The estimated parameters are saved to `results/ising_params_3447_pm1.npz` and to the
sheets of `results/3447例_分析结果_pm1.xlsx`.

## 4. Network energy and SNCI

For each participant, the **network energy** is evaluated directly from the fitted
parameters in the same ±1 domain as the estimation:

$$E = -\sum_i h_i s_i - \tfrac{1}{2}\sum_{i<j} J_{ij} s_i s_j.$$

The **Symptom Network Clinical Indicator (SNCI)** standardises the energy onto a
0–100 clinical scale,

$$\mathrm{SNCI} = \mathrm{clip}\!\left(50 + 10\,\frac{E-\mu}{\sigma},\; 0,\; 100\right),$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the energy distribution —
**computed within each cohort independently**, so that each cohort's SNCI is centred on
its own energy scale.

For the RCT, SNCI and energy are obtained by **projecting the 103 participants onto the
FDSC 3,447-case reference network** (frozen $h$, $J$). This reference frame, not the
pooled RCT baseline network, determines the RCT energy and SNCI values reported in the
manuscript (see §9).

## 5. Network topology: percolation, communities, bridge strength

- **Percolation.** Sweeping 60 equally spaced thresholds on $|J|$, the critical threshold
  $p_c$ is the largest threshold at which the **largest connected component still contains
  ≥ 50 % of the nodes**. Reproduced for FDSC from `J_correct.xlsx`.
- **Communities (Louvain).** Modularity is optimised with the Louvain algorithm
  ([Blondel et al., 2008](#11-references)) at **resolution = 1.0** in the primary analysis
  (sensitivity at 1.5 and 2.0). Modularity $Q$ is the resolution-aware networkx value.
- **Bridge strength and node predictability.** Bridge strength identifies nodes connecting
  communities; node predictability is the variance in a node explained by its neighbours
  ([Haslbeck & Waldorp, 2018](#11-references)).

Implemented in [`src/analysis_all.py`](src/analysis_all.py); the whole topology block is
repeated for CFPS and MIDUS in [`src/cross_cohort_analysis.py`](src/cross_cohort_analysis.py).

## 6. Core-symptom identification

Core symptoms are defined by stability under case resampling, not by a single centrality
score.

- **Engine (point estimate).** Four centrality indices are computed — **strength**,
  **bridge strength**, **percolation centrality** and **external field** — and each
  contributes its top-6 nodes, ties included. A node entering the top-6 of **at least 3 of
  the 4** indices seeds the resampling engine. This vote defines the candidate set only; it
  is not itself the reported result.
- **Final criterion.** 5,000 bootstrap resamples of the cases are drawn; a symptom is
  **core** if it is selected in **≥ 50 %** of resamples. Random-number generation uses
  `default_rng(20240825 + iter_idx)`, so iteration $k$ draws the same resample across every
  scenario and the results can be compared column-wise.
- Implementation: `python src/analysis_all.py --core-boot --boot-iters 5000`; output
  `results/bootstrap_pm1_n5000.csv`.

## 7. Cross-cohort validation

- **Discriminant validity.** SNCI is used to separate high- from low-symptom respondents
  by ROC analysis; the AUC and its 95 % CI, the Youden-optimal cut-off, sensitivity and
  specificity are reported for FDSC (Fig. 1f) and for each validation cohort (Fig. 4c).
- **Convergent signal.** The SNCI–CES-D correlation is reported per cohort (Fig. 4b).
- **Structural comparison.** Regularised networks for CFPS 2012 and MIDUS M2 are compared
  with FDSC on edge weights, global strength and community structure, and with the
  Network Comparison Test (permutation seed `20240826`;
  [van Borkulo et al., 2023](#11-references)).
- **Consensus skeleton.** A consensus network across the three symptom cohorts, with its
  consistency indicators, is produced by [`src/consensus_network.py`](src/consensus_network.py)
  (Table S7, Fig. S1).
- Implementation: [`src/cross_cohort_analysis.py`](src/cross_cohort_analysis.py),
  [`src/compute_cross_cohort_core.py`](src/compute_cross_cohort_core.py),
  [`src/run_bootstrap_parallel.py`](src/run_bootstrap_parallel.py). Random-number generation
  for this block is `RANDOM_STATE × 1000003 + iter_idx` (`RANDOM_STATE = 42`).

## 8. Sensitivity analyses

Six dimensions are varied, each with 5,000 bootstraps, in
[`src/sensitivity_analysis.py`](src/sensitivity_analysis.py):

| Dimension | Settings |
|---|---|
| Node encoding | 0 / 1 versus ±1 |
| EBIC γ | 0, 0.25 (primary), 0.5 |
| Percolation operationalisation | alternative threshold sweeps |
| Louvain resolution | 1.5 and 2.0 |
| Decision threshold | alternative core-symptom cut-offs |
| $p_c$ reproduction | three readings of the critical threshold |

Run as `python src/sensitivity_analysis.py <point|scenarios|cutoff|threshold|topk|percolation>`.
The main and sensitivity analyses **share** the `20240825 + iter_idx` seed system; it is
deliberately kept separate from the cross-cohort/RCT system, and merging the two would
change already-published values.

## 9. Tai Chi RCT: intervention validation

The 103-participant RCT (Tai Chi 49 / wait-list control 54) is analysed in
[`src/rct/`](src/rct/README.md). Between-arm and within-arm changes are computed for
ΔCES-D, ΔSNCI and Δenergy; energy and SNCI come from projecting participants onto the FDSC
reference network (§4).

- **Between-arm tests.** Two-sample $t$-tests on the change scores, pooled-variance by
  default and Welch where Levene's test indicates unequal variances; Cohen's $d$ for the
  between-arm contrast and $d_z$ for paired contrasts.
- **Within-arm tests.** Paired $t$-tests on pre–post change.
- **Association.** Per-arm correlation between ΔSNCI and ΔCES-D (Fig. 5d).
- **Response rates.** A participant responds on an item if it improves by **≥ 50 %** from
  the pre-treatment score (denominator: participants with a baseline score > 0). Between-arm
  comparisons use the **uncorrected Pearson χ²**, falling back to Fisher's exact test when
  the smallest *expected* frequency is < 5, with Benjamini–Hochberg control across the 20
  items (Fig. 5e, Table S10).
- **Core-symptom priority.** The core symptom set is the cross-cohort core, Depressed and
  Sad. A linear mixed-effects model tests whether the core symptoms improve more than the
  non-core symptoms under Tai Chi, with a random intercept for participant and a cluster
  bootstrap ($B = 5{,}000$);
  [`src/rct_mixedlm_core_priority.py`](src/rct_mixedlm_core_priority.py) (Table S14).

Item-level change tables (Tables S11–S13) are produced as part of
[`src/rct/rct_snci_projection.py`](src/rct/rct_snci_projection.py).

## 10. Software, seeds and file preparation

**Software.** Python 3.13 on Windows. Dependencies and version ranges are in
[`requirements.txt`](requirements.txt): numpy ≥ 2.0, pandas ≥ 2.2, scipy ≥ 1.13,
scikit-learn ≥ 1.5, networkx ≥ 3.3, python-louvain ≥ 0.16, statsmodels ≥ 0.14 (mixed
models), matplotlib ≥ 3.9 (consensus-network and ROC panels), openpyxl ≥ 3.1 and
python-docx ≥ 1.1 (table output).

**Seeds.** All model fitting, Louvain community detection, ROC analysis and network layout
use `seed = 42`. Two resampling streams are used and must not be merged:

| Stream | Seed | Used by |
|---|---|---|
| Main / sensitivity | `20240825 + iter_idx` | FDSC core resampling, sensitivity scenarios |
| Cross-cohort / RCT | `42 × 1000003 + iter_idx` | CFPS, MIDUS, RCT analyses |
| NCT permutation | `20240826` | Network comparison test |

**File preparation.** Analyses read the cohort files from `data/` and write to
`results/`; both are excluded by `.gitignore`. All paths are centralised in
[`config.py`](config.py) and can be redirected with the environment variables
`CESD_DATA_DIR` and `CESD_OUT_DIR`. The scripts are **illustrative**: the restricted
cohorts are not distributed with the repository, and no machine-specific absolute paths
remain in the code. Every long-running resampling task writes a `done` column so it can
be resumed.

## 11. References

1. Radloff LS. The CES-D scale: a self-report depression scale for research in the general population. *Applied Psychological Measurement*. 1977;1(3):385–401.
2. van Borkulo CD, Borsboom D, Epskamp S, Tio P, Schoevers RA, Noorthoorn E, Waldorp LJ. A new method for constructing networks from binary data. *Scientific Reports*. 2014;4:5918.
3. Chen J, Chen Z. Extended Bayesian information criteria for model selection with large model spaces. *Biometrika*. 2008;95(3):759–771.
4. Haslbeck JMB, Waldorp LJ. How well do network models predict observations? On the importance of predictability in network models. *Behavior Research Methods*. 2018;50(2):853–861.
5. Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*. 2008;2008:P10008.
6. Epskamp S, Borsboom D, Fried EI. Estimating psychological networks and their accuracy: a tutorial paper. *Behavior Research Methods*. 2018;50(1):195–212.
7. van Borkulo CD, van Bork R, Boschloo L, Kossakowski JJ, Tio P, Schoevers RA, Borsboom D, Waldorp LJ. Comparing network structures on three aspects: a permutation test. *Psychological Methods*. 2023;28(6):1273–1285.
8. Seabold S, Perktold J. statsmodels: econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*. 2010:92–96.
