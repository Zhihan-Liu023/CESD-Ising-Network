# -*- coding: utf-8 -*-
"""
==============================================================================
Table S14 —— 太极拳对核心症状「优先改善」效应的混合效应模型
（cluster bootstrap, B = 5000）
==============================================================================

【模型】
  长表：n = 103 受试者 × 20 个 CES-D 条目 = 2060 观测
  因变量：Δ = 干预后 − 干预前（逐条目，负值 = 改善）
  固定效应：change ~ tai_bin × core_bin
  随机效应：受试者随机截距（1 | subject）
  估计：极大似然（ML, reml=False）

【核心症状】
  跨队列交叉验证得到的共同核心症状 {Depressed, Sad}。

【推断】
  受试者层面 cluster bootstrap，B = 5000（有放回重抽 103 名受试者），
  每轮重估模型；95% CI 取经验分布的 2.5 / 97.5 百分位，
  p 取 2 × min(P(β>0), P(β<0))。

【事后比较】由固定效应线性组合给出
  太极组内（核心 − 非核心）  = b_core + b_inter
  对照组内（核心 − 非核心）  = b_core
  组间·核心（太极 − 对照）   = b_tai + b_inter
  组间·非核心（太极 − 对照） = b_tai
  Cohen's d = |diff| / 合并标准差（两个比较单元内条目层面 Δ 的 pooled SD）

【输出】<OUT_DIR>/
  mixedlm_core_vs_noncore_bootstrap_FE.csv      （固定效应）
  mixedlm_core_vs_noncore_bootstrap_simple.csv  （事后比较）

【依赖】numpy pandas statsmodels openpyxl
【运行】
  python rct_mixedlm_core_priority.py --iters 5000 --jobs 8
  python rct_mixedlm_core_priority.py --iters 200 --jobs 4      # 快速自检
==============================================================================
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# --- 路径配置（config.py 统一提供，可用环境变量覆盖）----------------------
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent
_REPO = _SRC.parent
for _p in (str(_SRC), str(_REPO)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from config import DATA_DIR as CONFIG_DATA_DIR, OUT_DIR as CONFIG_OUT_DIR  # noqa: E402
# ---------------------------------------------------------------------------

# ============================== CONFIG（换机器只改这里） ==============================
RCT_XLSX = Path(CONFIG_DATA_DIR) / "RCT_总103例_CESD.xlsx"
OUT_DIR = Path(CONFIG_OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHEET_PRE = "干预前"
SHEET_POST = "干预后"
GROUP_MAP = {"学校随机干预组": 1, "学校随机对照组": 0}   # 1 = 太极拳组

# 条目顺序 = 表中第 10–29 列（0-based 9:29）
ITEM_NAMES = [
    "Bothered", "Appetite", "Blues", "AsGood", "Mind",
    "Depressed", "Effort", "Hopeful", "Failure", "Fearful",
    "Sleep", "Happy", "TalkedLess", "Lonely", "Unfriendly",
    "EnjoyedLife", "Crying", "Sad", "Dislike", "Going",
]

# 核心症状 = 跨队列交叉验证得到的共同核心症状
CORE_SYMPTOMS = ["Depressed", "Sad"]

RANDOM_STATE = 42          # 与主分析核心症状 bootstrap 同一惯例：
                           #   seeds = range(42, 42+B)，每轮 RandomState(seed).randint 受试者重抽样
N_BOOT = 5000
N_JOBS = 8
# ====================================================================================


# ------------------------------ 数据 ------------------------------
def load_wide():
    """返回 (change_matrix n×20, tai n)。Δ = 干预后 − 干预前。"""
    pre = pd.read_excel(RCT_XLSX, sheet_name=SHEET_PRE)
    post = pd.read_excel(RCT_XLSX, sheet_name=SHEET_POST)
    cols = list(pre.columns[9:29])
    if len(cols) != 20:
        raise ValueError(f"CES-D 条目列数不为 20（实得 {len(cols)}）")
    xp = pre[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    xq = post[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    keep = ~(np.isnan(xp).any(axis=1) | np.isnan(xq).any(axis=1))
    xp, xq = xp[keep], xq[keep]
    tai = pre.loc[keep, "批次"].map(GROUP_MAP).values.astype(float)
    return xq - xp, tai


def build_long(delta, tai, core_mask, subject_idx=None):
    """把 n×20 的 Δ 矩阵摊成长表；subject_idx 为 None 时用原始编号。"""
    n, p = delta.shape
    if subject_idx is None:
        subject_idx = np.arange(n)
    # cluster bootstrap：subject 取重抽样后的位置编号（0..n-1），使同一名受试者
    # 被多次抽中时各自成为独立 cluster，而不是被合并成一个。
    sub = np.repeat(np.arange(n), p)
    item = np.tile(np.arange(p), n)
    return pd.DataFrame({
        "subject": sub,
        "item": item,
        "change": delta[subject_idx].ravel(),
        "tai_bin": np.repeat(tai[subject_idx], p),
        "core_bin": np.tile(core_mask.astype(float), n),
    })


# ------------------------------ 模型 ------------------------------
def fit_ml(long_df):
    # 随机截距只按受试者设定，条目层面的差异归入残差。
    md = smf.mixedlm("change ~ tai_bin * core_bin", long_df,
                     groups=long_df["subject"])
    return md.fit(reml=False)


FE_TERMS = ["Intercept", "tai_bin", "core_bin", "tai_bin:core_bin"]


def contrasts_from_params(params):
    b0 = params.get("Intercept", np.nan)
    b_tai = params.get("tai_bin", np.nan)
    b_core = params.get("core_bin", np.nan)
    b_int = params.get("tai_bin:core_bin", np.nan)
    # 固定效应 β 取 change(post−pre) 方向（负值 = 改善）；
    # 事后比较报的是改善幅度（正值 = 改善），故对比项一律取反。
    return {
        "within_tai": -(b_core + b_int),     # 太极组内：核心 − 非核心（改善幅度）
        "within_ctrl": -b_core,              # 对照组内：核心 − 非核心（改善幅度）
        "between_core": -(b_tai + b_int),    # 组间·核心：太极 − 对照（改善幅度）
        "between_noncore": -b_tai,           # 组间·非核心：太极 − 对照（改善幅度）
        "_b0": b0,
    }


def cohens_d(delta, tai, core_mask, which):
    """|diff| / 合并 SD（条目层面 Δ）。"""
    if which in ("between_core", "between_noncore"):
        m = core_mask if which == "between_core" else ~core_mask
        v1 = delta[tai == 1][:, m].ravel()
        v0 = delta[tai == 0][:, m].ravel()
    else:                                   # 太极组内 / 对照组内
        g = 1 if which == "within_tai" else 0
        v1 = delta[tai == g][:, core_mask].ravel()
        v0 = delta[tai == g][:, ~core_mask].ravel()
    if len(v1) < 2 or len(v0) < 2:
        return np.nan
    sp = np.sqrt(((len(v1) - 1) * v1.var(ddof=1) + (len(v0) - 1) * v0.var(ddof=1))
                 / (len(v1) + len(v0) - 2))
    return float(abs(v1.mean() - v0.mean()) / sp) if sp > 0 else np.nan


# ------------------------------ bootstrap worker ------------------------------
_ST = {}


def _init_worker(delta, tai, core_mask, seed):
    _ST["delta"] = delta
    _ST["tai"] = tai
    _ST["core"] = core_mask
    _ST["seed"] = seed


def _one_iter(it):
    delta, tai, core = _ST["delta"], _ST["tai"], _ST["core"]
    n = delta.shape[0]
    rng = np.random.RandomState(_ST["seed"] + it)
    idx = rng.randint(0, n, size=n)
    long_df = build_long(delta, tai, core, subject_idx=idx)
    with contextlib.redirect_stdout(io.StringIO()):
        mdf = fit_ml(long_df)
    out = {k: float(mdf.params.get(k, np.nan)) for k in FE_TERMS}
    out.update(contrasts_from_params(mdf.params))
    d = delta[idx]
    for w in ["within_tai", "within_ctrl", "between_core", "between_noncore"]:
        out["d_" + w] = cohens_d(d, tai[idx], core, w)
    return out


def bootstrap(delta, tai, core_mask, iters, jobs, seed):
    with mp.Pool(jobs, initializer=_init_worker,
                 initargs=(delta, tai, core_mask, seed)) as pool:
        recs = []
        t0 = time.time()
        for k, r in enumerate(pool.imap(_one_iter, range(iters), chunksize=8), 1):
            recs.append(r)
            if k % 250 == 0 or k == iters:
                el = (time.time() - t0) / 60.0
                print(f"      {k}/{iters} 轮  已用 {el:.1f} min", flush=True)
    return pd.DataFrame(recs)


def _p_two_sided(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    return float(2.0 * min((v > 0).mean(), (v < 0).mean()))


def _ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


SIMPLE_LABELS = [
    ("within_tai", "太极组内：核心 − 非核心"),
    ("within_ctrl", "对照组内：核心 − 非核心"),
    ("between_core", "组间·核心：太极拳 − 对照"),
    ("between_noncore", "组间·非核心：太极拳 − 对照"),
]


def run_core_set(delta, tai, core_names, point, boot, iters):
    core_mask = np.array([nm in core_names for nm in ITEM_NAMES], dtype=bool)

    fe_rows = []
    for term in FE_TERMS:
        v = boot[term].values
        lo, hi = _ci(v)
        fe_rows.append({
            "term": term,
            "label": {"Intercept": "截距", "tai_bin": "分组（太极拳 vs 对照）",
                      "core_bin": "症状类别（核心 vs 非核心）",
                      "tai_bin:core_bin": "分组 × 症状类别"}[term],
            "beta": float(point[term]),
            "boot_mean": float(np.nanmean(v)),
            "se": float(np.nanstd(v, ddof=1)),
            "ci_low": lo, "ci_high": hi,
            "p": _p_two_sided(v),
        })
    fe = pd.DataFrame(fe_rows)

    sm_rows = []
    for w, label in SIMPLE_LABELS:
        v = boot[w].values
        lo, hi = _ci(v)
        sm_rows.append({
            "comparison": label,
            "diff": float(point[w]),
            "boot_mean": float(np.nanmean(v)),
            "ci_low": lo, "ci_high": hi,
            "p": _p_two_sided(v),
            # Cohen's d 取完整数据的点估计（|diff| / pooled SD），非 bootstrap 均值
            "cohens_d": float(cohens_d(delta, tai, core_mask, w)),
        })
    sm = pd.DataFrame(sm_rows)

    p_fe = OUT_DIR / "mixedlm_core_vs_noncore_bootstrap_FE.csv"
    p_sm = OUT_DIR / "mixedlm_core_vs_noncore_bootstrap_simple.csv"
    fe.to_csv(p_fe, index=False, encoding="utf-8-sig")
    sm.to_csv(p_sm, index=False, encoding="utf-8-sig")

    print(f"\n===== 核心症状：{core_names}  （bootstrap B={iters}）=====")
    print(fe[["label", "beta", "boot_mean", "ci_low", "ci_high", "p"]].to_string(index=False))
    print(sm[["comparison", "diff", "boot_mean", "ci_low", "ci_high", "p", "cohens_d"]].to_string(index=False))
    print(f"[导出] {p_fe.name}")
    print(f"[导出] {p_sm.name}")
    return fe, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=N_BOOT)
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    delta, tai = load_wide()
    n = delta.shape[0]
    print("=" * 78)
    print(f"RCT 混合效应模型 · n = {n}（太极组 {int((tai == 1).sum())} / 对照组 {int((tai == 0).sum())}）")
    print(f"ΔCES-D(后−前)：{delta.sum(axis=1).mean():+.3f} ± {delta.sum(axis=1).std(ddof=1):.3f}"
          f"   条目层面 {delta.mean():+.4f} ± {delta.std(ddof=1):.4f}")

    core_names = CORE_SYMPTOMS
    core_mask = np.array([nm in core_names for nm in ITEM_NAMES], dtype=bool)
    with contextlib.redirect_stdout(io.StringIO()):
        pt = fit_ml(build_long(delta, tai, core_mask))
    point = {k: float(pt.params.get(k, np.nan)) for k in FE_TERMS}
    point.update(contrasts_from_params(pt.params))
    print(f"\n--- 核心症状：{core_names}")
    print("    " + "  ".join(f"{k}={point[k]:+.4f}" for k in FE_TERMS))
    print(f"    B={args.iters} cluster bootstrap（受试者重抽样，{args.jobs} 进程）…")
    boot = bootstrap(delta, tai, core_mask, args.iters, args.jobs, args.seed)
    run_core_set(delta, tai, core_names, point, boot, args.iters)

    print("\n========== MIXEDLM DONE ==========")


if __name__ == "__main__":
    main()
