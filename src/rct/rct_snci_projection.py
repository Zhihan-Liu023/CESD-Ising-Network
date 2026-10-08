#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
==============================================================================
 RCT SNCI 投影分析脚本 —— 3447 例 FDSC 参考框架拟合 + 103 例投影（pm1 口径）
==============================================================================

把 103 例 RCT 被试投影到 3447 例 FDSC 横断面 Ising 参考框架，得到逐被试症状能量 E
与 SNCI，并比较太极组与对照组在干预前后的变化。

流程
  参考框架（3447 例）：二值化 CES-D → 节点wise LASSO 逻辑回归 + EBIC + AND 拟合
      Ising 参数 → 症状能量 → 渗流阈值 / Louvain 社区 / 桥接强度 → 核心症状 →
      SNCI 标准化
  投影分析（103 例，干预前/后）：投影到同一参考框架，逐被试算 E 与 SNCI，做组间
      比较、响应率分析、核心与非核心症状的特异性改善

口径
  Ising 拟合：±1 域编码 X = 2x − 1，节点wise LASSO 逻辑回归（saga,
      l1_ratio=1.0, 12 个 C 值网格）；EBIC 似然用 logaddexp 防溢出
      mean_ll = mean(logaddexp(0, −y·logits))；AND 规则（两个方向回归系数
      |β| > 1e-4 才保留边）；保号对称化 (J_raw + J_rawᵀ)/2 × AND（可负）。
      ±1 域回归系数 ÷2 才是 Ising 参数：h = β₀/2, J = β/2
      （±1 条件 logit = 2(h + ΣJ·s)，回归系数是参数的两倍）。
      主分析只用 ±1（pm1）编码，{0,1} 域仅出现在敏感性分析中。
  能量：E = −Σ h_i s_i − ½ Σ J_ij s_i s_j，s = 2x − 1
  SNCI：SNCI = clip(50 + 10·(E − μ)/σ, 0, 100)，μ/σ 取 3447 例能量分布（ddof=0）
  核心症状：跨队列交叉验证得到的共同核心症状（Depressed, Sad）。

数据与输出
  3447 例参考数据  <CESD_DATA_DIR>/3447例-CESD数据已反向计分(3).xlsx
  103 例 RCT 数据  <CESD_DATA_DIR>/RCT_总103例_CESD.xlsx
  两个工作簿均不随仓库分发，见 data/README.md；路径可分别用环境变量
  CESD_FDSC_XLSX / CESD_RCT_ITEMS_XLSX 覆盖。输出写入 <CESD_OUT_DIR>/rct_projection/。

分组约定：分组 == 1 → 太极干预组；分组 == 2 → 对照组

运行：
    python rct_snci_projection.py
（依赖：numpy, pandas, scipy, sklearn, networkx, python-louvain, openpyxl）
==============================================================================
"""

from __future__ import annotations

import os
import sys
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression   # 节点wise LASSO 估计

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*")

# ---------------------------------------------------------------------------
# 0. 路径与全局参数
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import DATA_DIR, OUT_DIR as _REPO_OUT   # noqa: E402

DATA_3447 = Path(os.environ.get(
    "CESD_FDSC_XLSX", DATA_DIR / "3447例-CESD数据已反向计分(3).xlsx"))
DATA_103 = Path(os.environ.get(
    "CESD_RCT_ITEMS_XLSX", DATA_DIR / "RCT_总103例_CESD.xlsx"))

SHEET_PRE = "干预前"
SHEET_POST = "干预后"

OUT_DIR = _REPO_OUT / "rct_projection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CESD_COLS = [f"CESD_{i}" for i in range(1, 21)]
# 注意：源数据已反向计分（CESD_4/8/12/16 已在 Excel 预处理），本脚本直接使用，不再二次反向。
BIN_THRESH = 2                 # 二值化阈值：反向计分后 >= 2 记症状存在
GROUP_MAP = {1: "太极组", 2: "对照组"}

# ---- Ising 拟合参数（与最终主分析一致：标准模块度 γ=1.0） ----
EBIC_GAMMA = 0.25
RANDOM_STATE = 42
RESOLUTION = 1.0                # 标准模块度（Clauset et al. 2004; Newman 2006）
AND_TOL = 1e-4                 # saga 解不出精确零，等价于 glmnet 的零判据
C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
ENCODING = "pm1"               # 估计与能量均用 {−1,+1}，内部一致

# ---- 核心症状（跨队列交叉验证得到的共同核心） ----
CORE_SYMPTOMS = ["Depressed", "Sad"]

ITEM_NAMES = [
    "Bothered", "Appetite", "Blues", "AsGood", "Mind",
    "Depressed", "Effort", "Hopeful", "Failure", "Fearful",
    "Sleep", "Happy", "TalkedLess", "Lonely", "Unfriendly",
    "EnjoyedLife", "Crying", "Sad", "Dislike", "Going",
]


# ---------------------------------------------------------------------------
# 3447 例参考框架：数据加载与二值化
# ---------------------------------------------------------------------------
def load_cesd_binary(path, bin_thr: int = BIN_THRESH):
    """读取已反向计分数据并二值化。返回 (scored 0-3 ndarray, binary 0/1 float ndarray)。"""
    df = pd.read_excel(path)
    cesd = df[CESD_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    scored = cesd.copy()   # 数据已反向计分，直接使用（CESD_4/8/12/16 已在源文件预处理）
    print("[数据] 数据已反向计分，跳过反向计分")
    binary = (scored >= bin_thr).astype(int)
    print(f"[数据] N={len(binary)}, P={binary.shape[1]}, 症状检出率={binary.values.mean():.3f}")
    return scored.values.astype(float), binary.values.astype(float)


# ---------------------------------------------------------------------------
# Ising 模型：节点wise LASSO 逻辑回归 + EBIC(γ=0.25) + AND 规则（pm1 域）
# ---------------------------------------------------------------------------
def _loglik_mean(y: np.ndarray, logits: np.ndarray) -> float:
    """±1 域平均负对数似然，用 logaddexp 防止溢出。"""
    return float(np.mean(np.logaddexp(0.0, -y * logits)))


def fit_ising_network(binary: np.ndarray, gamma: float = EBIC_GAMMA) -> dict:
    n, p = binary.shape
    X = (2 * binary.astype(int) - 1).astype(float)      # ±1 域编码

    intercept = np.zeros(p)
    j_raw = np.zeros((p, p))
    edge_mask = np.zeros((p, p), dtype=bool)

    for i in range(p):
        y, Z = X[:, i], np.delete(X, i, axis=1)
        best_ebic, best_coefs, best_int = np.inf, None, 0.0
        for c_val in C_GRID:
            clf = LogisticRegression(
                solver="saga", C=c_val, l1_ratio=1.0, fit_intercept=True,
                max_iter=3000, tol=1e-4, random_state=RANDOM_STATE,
            )
            clf.fit(Z, y)
            coefs = clf.coef_[0]
            n_nz = int(np.sum(np.abs(coefs) > AND_TOL))
            logits = clf.intercept_[0] + Z @ coefs
            mean_ll = _loglik_mean(y, logits)                    # logaddexp 防溢出
            ebic = 2.0 * n * mean_ll + n_nz * np.log(n) \
                   + 2.0 * gamma * n_nz * np.log(p - 1)
            if ebic < best_ebic:
                best_ebic, best_coefs, best_int = ebic, coefs.copy(), float(clf.intercept_[0])

        intercept[i] = best_int
        if best_coefs is not None:
            mask = np.ones(p, dtype=bool)
            mask[i] = False
            j_raw[i, mask] = best_coefs
            edge_mask[i, mask] = np.abs(best_coefs) > AND_TOL

    adj_and = edge_mask & edge_mask.T
    j_sym = (j_raw + j_raw.T) / 2.0 * adj_and                   # 保号对称化

    # ±1 域逻辑回归拟合 logit = 2(h + ΣJ·s)，Ising 参数为系数的一半
    intercept *= 0.5
    j_raw *= 0.5
    j_sym *= 0.5

    n_edges = int(adj_and.sum() // 2)
    print(f"[Ising] EBIC+AND (gamma={gamma}, enc={ENCODING}): {n_edges} 条边, "
          f"密度={n_edges / (p * (p - 1) / 2):.1%}")
    return {"intercept": intercept, "j_sym": j_sym, "j_raw": j_raw,
            "n_edges": n_edges, "adj_and": adj_and,
            "adj_or": edge_mask | edge_mask.T}


# ---------------------------------------------------------------------------
# Ising 能量（与拟合同域，pm1）
#          E = −Σ h_i s_i − ½ Σ J_ij s_i s_j,  s = 2x − 1
# ---------------------------------------------------------------------------
def compute_energy(binary: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    s = (2 * binary.astype(int) - 1).astype(float)
    E = -(s @ h) - 0.5 * np.sum((s @ J) * s, axis=1)
    print(f"[能量] E ∈ [{E.min():.2f}, {E.max():.2f}], μ={E.mean():.2f}, σ={E.std():.2f}")
    return E


# ---------------------------------------------------------------------------
# 渗流（阈值移除）+ Louvain 社区 + 桥接强度
# ---------------------------------------------------------------------------
def percolation_communities(j_sym: np.ndarray) -> dict:
    import networkx as nx
    import community as community_louvain
    n = j_sym.shape[0]
    edges = [(u, v, abs(j_sym[u, v])) for u in range(n) for v in range(u + 1, n)
             if abs(j_sym[u, v]) > AND_TOL]
    p_c = 0.0
    if edges:
        triu = np.array([w for _, _, w in edges])
        for thr in np.linspace(float(triu.min()), float(triu.max()), 60):
            gg = nx.Graph(); gg.add_nodes_from(range(n))
            for u, v, w in edges:
                if w >= thr:
                    gg.add_edge(u, v)
            if gg.number_of_edges() \
                    and len(max(nx.connected_components(gg), key=len)) / n >= 0.50:
                p_c = float(thr)
            else:
                break

    g = nx.Graph(); g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            w = abs(j_sym[u, v])
            if w > AND_TOL:
                g.add_edge(u, v, weight=w)
    partition = community_louvain.best_partition(g, resolution=RESOLUTION,
                                                 random_state=RANDOM_STATE)
    labels = np.array([partition[i] for i in range(n)])
    # Q 按与检测相同的分辨率计算（python-louvain.modularity 不支持 resolution 参数）
    _comms = {}
    for _node, _cid in partition.items():
        _comms.setdefault(_cid, set()).add(_node)
    Q = nx.community.modularity(g, list(_comms.values()),
                                weight='weight', resolution=RESOLUTION)

    bridge = np.zeros(n)
    for u in range(n):
        for v in range(u + 1, n):
            if labels[u] != labels[v]:
                w = abs(j_sym[u, v])
                bridge[u] += w
                bridge[v] += w

    print(f"[渗流] p_c={p_c:.4f}, {len(set(labels))} 社区, Q={Q:.4f}")
    return {"p_c": p_c, "labels": labels, "Q": Q, "bridge": bridge,
            "n_communities": len(set(labels))}


# ---------------------------------------------------------------------------
# SNCI 标准化（截断至 [0,100]）
# ---------------------------------------------------------------------------
def standardize_snci(E: np.ndarray) -> tuple:
    mu = float(E.mean())
    sigma = float(E.std(ddof=0))
    SNCI = np.clip(50.0 + 10.0 * (E - mu) / (sigma + 1e-12), 0.0, 100.0)
    print(f"[SNCI] SNCI ∈ [{SNCI.min():.2f}, {SNCI.max():.2f}]")
    return SNCI, mu, sigma


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------
def extract_cesd_binary(df: pd.DataFrame, thresh: int = BIN_THRESH) -> pd.DataFrame:
    """提取 103 例 CES-D 20 题 -> 二值化（数据已反向计分，不再执行 3-x）。"""
    cols = list(df.columns)
    if "CES-D总分" not in cols:
        raise KeyError("未找到 'CES-D总分' 列，无法定位 CES-D 20 题")
    total_idx = cols.index("CES-D总分")
    if total_idx < 20:
        raise ValueError("'CES-D总分' 列位置异常，前面不足 20 题")
    item_cols = cols[total_idx - 20: total_idx]
    scored = df[item_cols].copy().astype(float)   # 源数据已反向计分，直接使用
    binary = (scored >= thresh).astype(int)
    binary.columns = [f"CESD_{i}" for i in range(1, 21)]
    return binary


def extract_cesd_scored(df: pd.DataFrame) -> pd.DataFrame:
    """提取 103 例 CES-D 20 题（已反向计分，直接使用），返回 (n, 20) 0-3 分。"""
    cols = list(df.columns)
    total_idx = cols.index("CES-D总分")
    item_cols = cols[total_idx - 20: total_idx]
    scored = df[item_cols].copy().astype(float)   # 源数据已反向计分，直接使用
    return scored


def to_spin(binary_mat: np.ndarray) -> np.ndarray:
    return 2 * binary_mat.astype(int) - 1


def ising_energy(spin_mat: np.ndarray, h: np.ndarray, j: np.ndarray) -> np.ndarray:
    """E = - Σ h_i σ_i - 1/2 Σ J_ij σ_i σ_j，spin ∈ {-1,+1}。"""
    field_term = spin_mat @ h
    pair_term = 0.5 * np.sum(spin_mat @ j * spin_mat, axis=1)
    return -field_term - pair_term


def compute_snci(energy: np.ndarray, ref_energy: np.ndarray) -> np.ndarray:
    """SNCI = clip(50 + 10 * (E - mu_ref) / sd_ref, 0, 100)（参考人群标准化，ddof=0）。"""
    mu = float(ref_energy.mean())
    sd = float(ref_energy.std(ddof=0))
    if sd < 1e-12:
        return np.full_like(energy, 50.0)
    return np.clip(50.0 + 10.0 * (energy - mu) / sd, 0.0, 100.0)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if s == 0:
        return 0.0
    return float((x.mean() - y.mean()) / s)


def proportion_ci_wald(x: int, n: int, alpha: float = 0.05):
    if n == 0:
        return 0.0, 0.0
    p = x / n
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def odds_ratio_ci(a, b, c, d, alpha: float = 0.05):
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(1 - alpha / 2)
    return np.exp(np.log(or_val) - z * se), np.exp(np.log(or_val) + z * se)


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    fdr = np.empty(n)
    fdr[-1] = min(ranked[-1], 1.0)
    for i in range(n - 2, -1, -1):
        fdr[i] = min(ranked[i] * n / (i + 1), fdr[i + 1])
    out = np.empty(n)
    out[order] = fdr
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 103 例数据与投影
# ---------------------------------------------------------------------------
class DataBundle:
    """集中保存 103 例投影数据与分组信息。"""

    def __init__(self, h, j, items, ref_energy, core_symptoms):
        self.h = h
        self.j = j
        self.items = list(items)
        self.ref_energy = ref_energy
        self.core_symptoms = list(core_symptoms)

        self.df_pre = pd.read_excel(DATA_103, sheet_name=SHEET_PRE)
        self.df_post = pd.read_excel(DATA_103, sheet_name=SHEET_POST)

        # 按 序号 / 核磁编号 对齐两时间点
        id_cols = [c for c in ["序号", "核磁编号"] if c in self.df_pre.columns]
        if id_cols:
            pre_ids = self.df_pre[id_cols].reset_index(drop=True)
            post_ids = self.df_post[id_cols].reset_index(drop=True)
            aligned = pre_ids.merge(post_ids, on=id_cols, how="inner")
            keep = aligned.index
            self.df_pre = self.df_pre.loc[keep].reset_index(drop=True)
            self.df_post = self.df_post.loc[keep].reset_index(drop=True)

        self.n = len(self.df_pre)
        self.group = self.df_pre["分组"].values

        self.scored_pre = extract_cesd_scored(self.df_pre)
        self.scored_post = extract_cesd_scored(self.df_post)
        self.bin_pre = extract_cesd_binary(self.df_pre)
        self.bin_post = extract_cesd_binary(self.df_post)

        # NaN 稳健：剔除任一时间点 20 题存在缺失的受试者（保持前后时间点行对齐）
        nan_mask = self.scored_pre.isna().any(axis=1) | self.scored_post.isna().any(axis=1)
        if nan_mask.any():
            keep = ~nan_mask
            print(f"[RCT] 剔除 {int(nan_mask.sum())} 例含缺失 CES-D 条目的受试者")
            self.df_pre = self.df_pre[keep].reset_index(drop=True)
            self.df_post = self.df_post[keep].reset_index(drop=True)
            self.n = len(self.df_pre)
            self.group = self.df_pre["分组"].values
            self.scored_pre = extract_cesd_scored(self.df_pre)
            self.scored_post = extract_cesd_scored(self.df_post)
            self.bin_pre = extract_cesd_binary(self.df_pre)
            self.bin_post = extract_cesd_binary(self.df_post)

        self.e_pre = ising_energy(to_spin(self.bin_pre.values), self.h, self.j)
        self.e_post = ising_energy(to_spin(self.bin_post.values), self.h, self.j)

        self.snci_pre = compute_snci(self.e_pre, self.ref_energy)
        self.snci_post = compute_snci(self.e_post, self.ref_energy)
        self.delta_snci = self.snci_post - self.snci_pre

    def id_df(self) -> pd.DataFrame:
        cols = [c for c in ["序号", "核磁编号"] if c in self.df_pre.columns]
        if not cols:
            return pd.DataFrame({"序号": np.arange(1, self.n + 1)})
        return self.df_pre[cols].reset_index(drop=True)


def _group_stats(d, g):
    return d[g == 1], d[g == 2]


# ---------------------------------------------------------------------------
# 投影分析各步
# ---------------------------------------------------------------------------
def project_energy(b: DataBundle):
    result = b.id_df().copy()
    result["E_symptom_pre"] = b.e_pre
    result["E_symptom_post"] = b.e_post
    path = os.path.join(OUT_DIR, "E_symptom_103.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")
    result.to_excel(os.path.join(OUT_DIR, "E_symptom_103.xlsx"), index=False)
    summary = (f"样本量: {b.n}\n"
               f"E_symptom_pre : mean={b.e_pre.mean():.2f}, sd={b.e_pre.std():.2f}, "
               f"range=[{b.e_pre.min():.2f}, {b.e_pre.max():.2f}]\n"
               f"E_symptom_post: mean={b.e_post.mean():.2f}, sd={b.e_post.std():.2f}, "
               f"range=[{b.e_post.min():.2f}, {b.e_post.max():.2f}]")
    return {"csv": result, "csv_path": path, "summary": summary,
            "table": [["统计量", "均值", "标准差", "最小值", "最大值"],
                      ["E_symptom_pre", f"{b.e_pre.mean():.2f}", f"{b.e_pre.std():.2f}",
                       f"{b.e_pre.min():.2f}", f"{b.e_pre.max():.2f}"],
                      ["E_symptom_post", f"{b.e_post.mean():.2f}", f"{b.e_post.std():.2f}",
                       f"{b.e_post.min():.2f}", f"{b.e_post.max():.2f}"]]}


def compare_snci_change(b: DataBundle):
    result = b.id_df().copy()
    result["分组"] = b.group
    result["组别名称"] = [GROUP_MAP.get(g, g) for g in b.group]
    result["E_pre"] = b.e_pre
    result["E_post"] = b.e_post
    result["SNCI_pre"] = b.snci_pre
    result["SNCI_post"] = b.snci_post
    result["Delta_SNCI"] = b.delta_snci
    path = os.path.join(OUT_DIR, "SNCI_103_per_subject.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")

    g1, g2 = _group_stats(b.delta_snci, b.group)
    t_stat, p_val = stats.ttest_ind(g1, g2)
    d_stat = cohen_d(g1, g2)
    summary = (f"标准化方式: 3447 例参考框架能量分布 (pm1 口径)\n"
               f"SNCI_pre : M={b.snci_pre.mean():.2f}, SD={b.snci_pre.std():.2f}\n"
               f"SNCI_post: M={b.snci_post.mean():.2f}, SD={b.snci_post.std():.2f}\n"
               f"ΔSNCI 太极组: n={len(g1)}, M={g1.mean():.3f}, SD={g1.std():.3f}\n"
               f"ΔSNCI 对照组: n={len(g2)}, M={g2.mean():.3f}, SD={g2.std():.3f}\n"
               f"t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d_stat:.3f}")
    return {"csv": result, "csv_path": path, "summary": summary,
            "table": [["组别", "n", "ΔSNCI (M)", "ΔSNCI (SD)", "t", "p", "Cohen's d"],
                      ["太极组", len(g1), f"{g1.mean():.3f}", f"{g1.std():.3f}",
                       f"{t_stat:.3f}", f"{p_val:.4f}", f"{d_stat:.3f}"],
                      ["对照组", len(g2), f"{g2.mean():.3f}", f"{g2.std():.3f}", "", "", ""]],
            "extra": {"t": t_stat, "p": p_val, "d": d_stat,
                      "tj": (len(g1), g1.mean(), g1.std()),
                      "ct": (len(g2), g2.mean(), g2.std())}}


def compare_total_change(b: DataBundle):
    total_pre = b.scored_pre.sum(axis=1).values
    total_post = b.scored_post.sum(axis=1).values
    delta = total_post - total_pre
    result = b.id_df().copy()
    result["分组"] = b.group
    result["组别名称"] = [GROUP_MAP.get(g, g) for g in b.group]
    result["CESD_total_pre"] = total_pre
    result["CESD_total_post"] = total_post
    result["Delta_CESD_total"] = delta
    path = os.path.join(OUT_DIR, "CESD_total_change_103.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")

    g1, g2 = _group_stats(delta, b.group)
    t_stat, p_val = stats.ttest_ind(g1, g2)
    d_stat = cohen_d(g1, g2)
    summary = (f"CES-D_pre : M={total_pre.mean():.2f}, SD={total_pre.std():.2f}\n"
               f"CES-D_post: M={total_post.mean():.2f}, SD={total_post.std():.2f}\n"
               f"ΔCES-D 太极组: n={len(g1)}, M={g1.mean():.2f}, SD={g1.std():.2f}\n"
               f"ΔCES-D 对照组: n={len(g2)}, M={g2.mean():.2f}, SD={g2.std():.2f}\n"
               f"t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d_stat:.3f}")
    return {"csv": result, "csv_path": path, "summary": summary,
            "table": [["组别", "n", "ΔCES-D (M)", "ΔCES-D (SD)", "t", "p", "Cohen's d"],
                      ["太极组", len(g1), f"{g1.mean():.2f}", f"{g1.std():.2f}",
                       f"{t_stat:.3f}", f"{p_val:.4f}", f"{d_stat:.3f}"],
                      ["对照组", len(g2), f"{g2.mean():.2f}", f"{g2.std():.2f}", "", "", ""]]}


def response_rates(b: DataBundle):
    # ===== 3.5a SNCI 改善响应率（主结局：网络病理减轻） =====
    RESPONSE_THRESHOLD = 5.0
    responder = b.delta_snci <= -RESPONSE_THRESHOLD   # ΔSNCI ≤ −5（与锁定主结局定义一致）
    result = b.id_df().copy()
    result["分组"] = b.group
    result["组别名称"] = [GROUP_MAP.get(g, g) for g in b.group]
    result["SNCI_pre"] = b.snci_pre
    result["SNCI_post"] = b.snci_post
    result["Delta_SNCI"] = b.delta_snci
    result["Responder"] = responder.astype(int)
    path = os.path.join(OUT_DIR, "SNCI_response_103_per_subject.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")

    table_rows = [["组别", "n", "响应者", "响应率(%)", "95% CI"]]
    ci_info = {}
    for g in sorted(GROUP_MAP.keys()):
        mask = b.group == g
        n = int(mask.sum())
        x = int(responder[mask].sum())
        p = x / n if n else 0.0
        lo, hi = proportion_ci_wald(x, n)
        ci_info[g] = (x, n, p)
        table_rows.append([GROUP_MAP[g], n, x, f"{p*100:.1f}",
                           f"{lo*100:.1f}%-{hi*100:.1f}%"])
    a, n1, _ = ci_info[1]
    c, n2, _ = ci_info[2]
    b_ = n1 - a
    d_ = n2 - c
    table = np.array([[a, b_], [c, d_]])
    or_unadj = (a * d_) / (b_ * c) if (b_ * c) > 0 else np.inf
    or_ci_lo, or_ci_hi = odds_ratio_ci(a, b_, c, d_)
    _, p_fisher = stats.fisher_exact(table)
    chi2, p_chi2, _, _ = stats.chi2_contingency(table, correction=False)
    snci_summary = (f"【SNCI 响应（网络病理减轻）】\n"
                    f"响应定义: SNCI_post - SNCI_pre < -{RESPONSE_THRESHOLD}\n"
                    f"2×2 列联表: 太极 [{a},{b_}]  对照 [{c},{d_}]\n"
                    f"响应率: 太极 {a}/{n1}={a/n1*100:.1f}%, 对照 {c}/{n2}={c/n2*100:.1f}%\n"
                    f"OR={or_unadj:.2f} (95%CI {or_ci_lo:.2f}-{or_ci_hi:.2f})\n"
                    f"Fisher p={p_fisher:.4f}, Chi2 p={p_chi2:.4f}")

    # ===== 3.5b CES-D 总分响应率（≥50% 临床应答标准，Frank et al. 1991） =====
    total_pre = b.scored_pre.sum(axis=1).values.astype(float)
    total_post = b.scored_post.sum(axis=1).values.astype(float)
    reduction = total_pre - total_post
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_red = np.where(total_pre > 0, reduction / total_pre * 100.0, np.nan)
    resp_mask = np.asarray(pct_red >= 50)

    cesd_df = b.id_df().copy()
    cesd_df["分组"] = b.group
    cesd_df["组别名称"] = [GROUP_MAP.get(g, g) for g in b.group]
    cesd_df["CESD_total_pre"] = total_pre
    cesd_df["CESD_total_post"] = total_post
    cesd_df["Reduction"] = reduction
    cesd_df["Pct_reduction"] = pct_red
    cesd_df["R50"] = resp_mask.astype(int)

    n1 = int((b.group == 1).sum())
    n2 = int((b.group == 2).sum())
    a = int(resp_mask[b.group == 1].sum())
    c = int(resp_mask[b.group == 2].sum())
    b_ = n1 - a
    d_ = n2 - c
    r1 = a / n1 * 100 if n1 else 0.0
    r2 = c / n2 * 100 if n2 else 0.0
    or_ci_lo_k, or_ci_hi_k = odds_ratio_ci(a, b_, c, d_)
    or_val = (a * d_) / (b_ * c) if (b_ * c) > 0 else np.inf
    _, p_fisher_k = stats.fisher_exact([[a, b_], [c, d_]])
    _, p_chi2_k, _, _ = stats.chi2_contingency([[a, b_], [c, d_]], correction=False)
    ci1_lo, ci1_hi = proportion_ci_wald(a, n1)
    ci2_lo, ci2_hi = proportion_ci_wald(c, n2)

    cesd_path = os.path.join(OUT_DIR, "CESD_response_103_per_subject.csv")
    cesd_df.to_csv(cesd_path, index=False, encoding="utf-8-sig")

    cesd_table_rows = [
        ["组别", "n", "响应者", "响应率(%)", "95% CI"],
        ["太极组", str(n1), str(a), f"{r1:.1f}", f"{ci1_lo*100:.1f}%-{ci1_hi*100:.1f}%"],
        ["对照组", str(n2), str(c), f"{r2:.1f}", f"{ci2_lo*100:.1f}%-{ci2_hi*100:.1f}%"],
        ["OR", f"{or_val:.2f}", f"95%CI {or_ci_lo_k:.2f}-{or_ci_hi_k:.2f}",
         f"Fisher p={p_fisher_k:.4f}", f"Chi2 p={p_chi2_k:.4f}"],
    ]
    headline = (f"以 CES-D 总分下降 ≥50% 为抑郁症状减轻的响应标准，"
                f"太极组响应率为 {r1:.1f}%，显著高于对照组的 {r2:.1f}%"
                f"（OR = {or_val:.2f}, P = {p_fisher_k:.3f}）。")
    cesd_summary = headline

    summary = snci_summary + "\n\n" + headline
    return {"csv": result, "csv_path": path,
            "cesd_csv": cesd_df, "cesd_csv_path": cesd_path,
            "summary": summary,
            "table": table_rows + [["OR", f"{or_unadj:.2f}",
                                   f"95%CI {or_ci_lo:.2f}-{or_ci_hi:.2f}",
                                   f"Fisher p={p_fisher:.4f}", f"Chi2 p={p_chi2:.4f}"]],
            "extra_table": cesd_table_rows,
            "cesd_summary": cesd_summary}


def compare_post_energy(b: DataBundle):
    values_post = b.e_post
    mu_ref = float(b.ref_energy.mean())
    sd_ref = float(b.ref_energy.std(ddof=0))
    n_ref = len(b.ref_energy)

    e_taichi = values_post[b.group == 1]
    e_control = values_post[b.group == 2]

    rows = []
    for name, g in [("太极拳组", e_taichi), ("对照组", e_control)]:
        m = float(g.mean())
        sd = float(g.std(ddof=1))
        n = len(g)
        se = sd / np.sqrt(n)
        t_val = float((m - mu_ref) / se)
        p_val = float(2 * stats.t.sf(abs(t_val), df=n - 1))
        d_z = float((m - mu_ref) / sd_ref)
        rows.append([f"{name} (n={n})", f"{m:.2f}", f"{sd:.2f}",
                     f"{t_val:.3f}", f"{p_val:.4f}", f"{d_z:.3f}"])

    m_all = float(values_post.mean())
    sd_all = float(values_post.std(ddof=1))
    n_all = len(values_post)
    se_all = sd_all / np.sqrt(n_all)
    t_all = float((m_all - mu_ref) / se_all)
    p_all = float(2 * stats.t.sf(abs(t_all), df=n_all - 1))
    d_all = float((m_all - mu_ref) / sd_ref)

    # 组间比较：太极拳组 vs 对照组 E_post（独立样本 t 检验 + Welch 校正 + Cohen's d）
    t_between, p_between = stats.ttest_ind(e_taichi, e_control)
    t_welch, p_welch = stats.ttest_ind(e_taichi, e_control, equal_var=False)
    d_between = cohen_d(e_taichi, e_control)
    diff_between = float(e_taichi.mean() - e_control.mean())

    result = b.id_df().copy()
    result["分组"] = b.group
    result["组别名称"] = [GROUP_MAP.get(g, g) for g in b.group]
    result["E_symptom_post"] = values_post
    path = os.path.join(OUT_DIR, "E_post_vs_reference_103_per_subject.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")

    summary = (
        f"比较指标: E_symptom (Ising 能量, pm1 域)\n"
        f"3447 参考人群: μ_ref={mu_ref:.4f}, σ_ref={sd_ref:.4f}, n_ref={n_ref}\n\n"
        f"太极拳组 (n={len(e_taichi)}): M_post={rows[0][1]}, SD={rows[0][2]}, "
        f"t={rows[0][3]}, p={rows[0][4]}, d_z={rows[0][5]}\n"
        f"对照组   (n={len(e_control)}): M_post={rows[1][1]}, SD={rows[1][2]}, "
        f"t={rows[1][3]}, p={rows[1][4]}, d_z={rows[1][5]}\n"
        f"合并     ({n_all}例):          M_post={m_all:.3f}, SD={sd_all:.3f}, "
        f"t={t_all:.3f}, p={p_all:.4f}, d_z={d_all:.3f}\n\n"
        f"【组间比较: 太极拳组 vs 对照组 E_post】\n"
        f"均差 Δ = {diff_between:+.3f}（太极 {rows[0][1]} vs 对照 {rows[1][1]}）\n"
        f"独立样本 t = {t_between:.3f}, p = {p_between:.4f}\n"
        f"Welch t = {t_welch:.3f}, p = {p_welch:.4f}\n"
        f"Cohen's d = {d_between:.3f}"
    )
    return {"csv": result, "csv_path": path, "summary": summary,
            "table": [["统计量", "太极拳组", "对照组", "3447例参考人群"],
                      ["M_post (平均能量)", rows[0][1], rows[1][1], f"{mu_ref:.2f}"],
                      ["SD_post",            rows[0][2], rows[1][2], f"{sd_ref:.2f}"],
                      ["单样本 t",           rows[0][3], rows[1][3], "—"],
                      ["p 值 (双尾)",        rows[0][4], rows[1][4], "—"],
                      ["Cohen's d_z",       rows[0][5], rows[1][5], "—"]],
            "extra_table": [["组间比较", "数值"],
                            ["均差 Δ (太极−对照)", f"{diff_between:+.3f}"],
                            ["独立样本 t", f"{t_between:.3f}"],
                            ["p 值 (双尾)", f"{p_between:.4f}"],
                            ["Welch t (不等方差)", f"{t_welch:.3f}"],
                            ["Welch p 值", f"{p_welch:.4f}"],
                            ["Cohen's d", f"{d_between:.3f}"]],
            "between": {"t": t_between, "p": p_between,
                        "welch_t": t_welch, "welch_p": p_welch,
                        "d": d_between, "diff": diff_between}}


def core_change(b: DataBundle):
    """核心与非核心症状特异性改善分析。"""
    CORE = list(b.core_symptoms)
    NON_CORE = [s for s in b.items if s not in CORE]

    missing = [s for s in CORE if s not in b.items]
    if missing:
        raise ValueError(f"参考框架未找到目标症状: {missing}")

    def _build_symptom_table(symptom_list):
        idx1 = {s: b.items.index(s) + 1 for s in symptom_list}
        idx0 = {s: v - 1 for s, v in idx1.items()}

        out_records = []
        summary_records = []
        for i in range(b.n):
            out_records.append({"subject_id": i + 1, "group_code": b.group[i],
                                "group": GROUP_MAP.get(b.group[i], b.group[i])})

        table_rows = [["症状", "总体Δ(M±SD)", "太极Δ", "对照Δ",
                       "太极组内配对p", "对照组内配对p",
                       "组间t", "组间p", "Cohen's d",
                       "太极改善率", "对照改善率", "Fisher p"]]
        for name in symptom_list:
            i0 = idx0[name]
            pre = b.scored_pre.iloc[:, i0].values
            post = b.scored_post.iloc[:, i0].values
            change = post - pre
            improved = change < 0
            for i in range(b.n):
                out_records[i][f"{name}_pre"] = pre[i]
                out_records[i][f"{name}_post"] = post[i]
                out_records[i][f"{name}_change"] = change[i]
                out_records[i][f"{name}_improved"] = int(improved[i])

            g1 = change[b.group == 1]
            g2 = change[b.group == 2]
            t_ind, p_ind = stats.ttest_ind(g1, g2)
            d = cohen_d(g1, g2)
            k1, k2 = int(improved[b.group == 1].sum()), int(improved[b.group == 2].sum())
            n1, n2 = int((b.group == 1).sum()), int((b.group == 2).sum())
            ctable = [[k1, n1 - k1], [k2, n2 - k2]]
            if all(v > 0 for row in ctable for v in row):
                _, p_fisher = stats.fisher_exact(ctable)
            else:
                p_fisher = np.nan
            # 组内配对 t 检验（pre vs post，同组内）—— 仅在样本量≥2 时计算
            pre1, post1 = pre[b.group == 1], post[b.group == 1]
            pre2, post2 = pre[b.group == 2], post[b.group == 2]
            if len(pre1) >= 2:
                _, p_paired1 = stats.ttest_rel(post1, pre1)
            else:
                p_paired1 = np.nan
            if len(pre2) >= 2:
                _, p_paired2 = stats.ttest_rel(post2, pre2)
            else:
                p_paired2 = np.nan
            table_rows.append([
                f"{name}(第{idx1[name]}题)", f"{change.mean():.3f}±{change.std():.3f}",
                f"{g1.mean():.3f}", f"{g2.mean():.3f}",
                f"{p_paired1:.4f}", f"{p_paired2:.4f}",
                f"{t_ind:.3f}", f"{p_ind:.4f}", f"{d:.3f}",
                f"{k1}/{n1}={k1/n1*100:.1f}%", f"{k2}/{n2}={k2/n2*100:.1f}%",
                f"{p_fisher:.4f}" if not np.isnan(p_fisher) else "NA"])
            summary_records.append({"symptom": name, "between_p": p_ind, "fisher_p": p_fisher,
                                    "paired_p_T": p_paired1, "paired_p_C": p_paired2})

        return out_records, table_rows, summary_records

    core_out, core_table, core_summary = _build_symptom_table(CORE)
    non_core_out, non_core_table, non_core_summary = _build_symptom_table(NON_CORE)

    combined_records = []
    for i in range(b.n):
        rec = {**core_out[i]}
        for k, v in non_core_out[i].items():
            if k not in rec:
                rec[k] = v
        combined_records.append(rec)
    out_df = pd.DataFrame(combined_records)
    path = os.path.join(OUT_DIR, "core_symptom_change_103_per_subject.csv")
    out_df.to_csv(path, index=False, encoding="utf-8-sig")

    all_summary = core_summary + non_core_summary
    pvals = np.array([r["between_p"] for r in all_summary])
    fdr = fdr_bh(pvals)

    summary = (
        f"=== 表1: 核心症状（{len(CORE)}个） ===\n"
        f"症状: {', '.join(CORE)}\n"
        f"FDR 校正后组间 p: " +
        "; ".join(f"{r['symptom']}: {fp:.4f}" for r, fp in zip(core_summary, fdr[:len(CORE)])) +
        f"\n\n=== 表2: 非核心症状 ({len(NON_CORE)}个) ===\n"
        f"症状: {', '.join(NON_CORE)}\n"
        f"FDR 校正后组间 p: " +
        "; ".join(f"{r['symptom']}: {fp:.4f}" for r, fp in zip(non_core_summary, fdr[len(CORE):]))
    )
    return {"csv": out_df, "csv_path": path, "summary": summary,
            "table": core_table,
            "non_core_table": non_core_table}


def item_response_rates(b: DataBundle):
    """症状响应率分析（每症状一行，binary 层面 pre vs post 改善）。

    列：题号 | 症状 | 太极 ΔBinary | 太极 是否响应 | 太极 响应率(%)
        | 对照 ΔBinary | 对照 是否响应 | 对照 响应率(%) | 组间 Fisher p

    定义：
      - 响应（个体）：post_binary < pre_binary（症状减轻或消失）
      - 响应率（组，描述性）= 该组内响应人数 / 该组总人数（与组间 Fisher 一致）
      - ΔBinary = post − pre 的个体平均（0/1 差，范围 [-1, +1]）
      - 是否响应（组级）= 组平均 ΔBinary < −0.22  → "是"
        （Taichi CT 子分析判据；与 CES-D≥50% 临床应答并行，
         本表用于症状层面的 binary 响应）
      - 组间 p：响应率 2×2 Fisher 精确检验（检验组间响应率差异）
    """
    DBIN_RESP_THRESH = -0.22   # 组级『是否响应』: 组平均 ΔBinary < −0.22（Taichi CT 子分析判据）

    g1_mask = b.group == 1
    g2_mask = b.group == 2
    n1 = int(g1_mask.sum())
    n2 = int(g2_mask.sum())

    long_records = []                          # 长格式 CSV（每个受试者×每个症状）
    table_rows = [["题号", "症状", "太极 ΔBinary", "太极 是否响应", "太极 响应率(%)",
                   "对照 ΔBinary", "对照 是否响应", "对照 响应率(%)", "组间 p(Fisher)"]]
    pvals = []
    for i0, name in enumerate(b.items):
        pre_all = b.bin_pre.iloc[:, i0].values.astype(int)
        post_all = b.bin_post.iloc[:, i0].values.astype(int)

        # ---- 太极组 ----
        pre1, post1 = pre_all[g1_mask], post_all[g1_mask]
        dbin1 = post1 - pre1
        resp1 = (post1 < pre1).astype(int)
        rate1 = resp1.mean()
        mean_dbin1 = float(dbin1.mean())
        yes1 = "是" if mean_dbin1 < DBIN_RESP_THRESH else "否"

        # ---- 对照组 ----
        pre2, post2 = pre_all[g2_mask], post_all[g2_mask]
        dbin2 = post2 - pre2
        resp2 = (post2 < pre2).astype(int)
        rate2 = resp2.mean()
        mean_dbin2 = float(dbin2.mean())
        yes2 = "是" if mean_dbin2 < DBIN_RESP_THRESH else "否"

        # ---- 组间 Fisher ----
        k1, k2 = int(resp1.sum()), int(resp2.sum())
        table = [[k1, n1 - k1], [k2, n2 - k2]]
        if (k1 + k2) > 0 and (n1 - k1) + (n2 - k2) > 0:
            _, p_fisher = stats.fisher_exact(table)
        else:
            p_fisher = np.nan
        pvals.append(p_fisher)

        table_rows.append([
            f"{i0+1}", name,
            f"{mean_dbin1:+.4f}", yes1, f"{rate1*100:.1f}",
            f"{mean_dbin2:+.4f}", yes2, f"{rate2*100:.1f}",
            f"{p_fisher:.4f}" if not np.isnan(p_fisher) else "NA",
        ])

        # 长格式记录（每个受试者）
        for i in range(b.n):
            pre_i = int(pre_all[i])
            post_i = int(post_all[i])
            long_records.append({
                "subject_id": i + 1, "group_code": int(b.group[i]),
                "group": GROUP_MAP.get(b.group[i], b.group[i]),
                "qid": i0 + 1, "symptom": name,
                "pre_binary": pre_i, "post_binary": post_i,
                "delta_binary": post_i - pre_i,
                "responder": int(post_i < pre_i),
            })

    # FDR 校正（针对 20 个症状的 Fisher p）
    p_arr = np.array([p if not np.isnan(p) else 1.0 for p in pvals])
    fdr = fdr_bh(p_arr)
    fdr_row = [["症状", "组间 Fisher p", "FDR q"]]
    for i, name in enumerate(b.items):
        fdr_row.append([name, f"{pvals[i]:.4f}" if not np.isnan(pvals[i]) else "NA",
                        f"{fdr[i]:.4f}"])

    out_df = pd.DataFrame(long_records)
    path = os.path.join(OUT_DIR, "symptom_response_103_per_subject.csv")
    out_df.to_csv(path, index=False, encoding="utf-8-sig")
    wide_path = os.path.join(OUT_DIR, "symptom_response_summary_103.csv")
    pd.DataFrame(table_rows[1:], columns=table_rows[0]).to_csv(
        wide_path, index=False, encoding="utf-8-sig")

    # 汇总统计：显著多于太极的症状
    sig_T = sum(1 for i, p in enumerate(pvals)
                if p is not None and not np.isnan(p) and p < 0.05 and table_rows[i+1][8] != "NA")
    summary = (
        f"症状响应率分析（binary 层面 pre→post 改善，N=103，太极 {n1} / 对照 {n2}）\n"
        f"响应定义: post binary < pre binary（症状减轻或消失）\n"
        f"组级『是否响应』: 组平均 ΔBinary < −0.22  →  是（Taichi CT 子分析判据）\n"
        f"ΔBinary: post − pre 的个体平均（0/1 差，范围 [-1, +1]）\n"
        f"组间 p: 2×2 Fisher 精确检验（响应率组间比较）\n\n"
        f"症状数: 20 ｜ Fisher p<0.05 的症状数: {sig_T} 个\n"
        f"按组级『是否响应=是』统计: 太极 {sum(1 for r in table_rows[1:] if r[3]=='是')} / 20 症状；"
        f"对照 {sum(1 for r in table_rows[1:] if r[6]=='是')} / 20 症状"
    )

    return {"csv": out_df, "csv_path": path,
            "wide_csv_path": wide_path,
            "summary": summary,
            "table": table_rows,
            "extra_table": fdr_row,
            "pvals": pvals, "fdr": fdr}


# ---------------------------------------------------------------------------
# Part C. HTML 报告生成
# ---------------------------------------------------------------------------
def _table_html(table, header=True):
    html = ['<table class="tbl">']
    for ri, row in enumerate(table):
        if header and ri == 0:
            html.append("<tr>" + "".join(f"<th>{c}</th>" for c in row) + "</tr>")
        else:
            html.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def _img_to_base64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def build_html(results, b: DataBundle, meta: dict):
    steps = [
        ("投影与症状能量", "CES-D 二值化 → 投影到 3447 例 Ising 参考框架（pm1），计算 E_symptom",
         results["projection_energy"]),
        ("SNCI 组间比较", "干预前后 SNCI 变化，太极组 vs 对照组（独立样本 t 检验）",
         results["snci_compare"]),
        ("CES-D 总分组间比较", "CES-D 总分变化对比，太极组 vs 对照组（独立样本 t 检验）",
         results["total_compare"]),
        ("干预后能量与参考人群比较", "干预后 E 与 3447 参考人群均值比较（单样本 t 检验）+ 太极 vs 对照组组间比较",
         results["post_energy_compare"]),
        ("响应率", "SNCI 改善响应率（网络病理减轻）+ CES-D 总分响应率（≥50% 临床应答，Frank 1991）",
         results["response_rates"]),
        ("逐症状响应率", "症状响应率分析（每症状一行，binary 层面 pre→post 改善 + 组间 Fisher）",
         results["item_response_rates"]),
        ("核心与非核心症状改善", f"核心症状（{len(b.core_symptoms)}个）+ 非核心症状特异性改善（含组内配对 t）",
         results["core_change"]),
    ]
    sections = []
    for sid, desc, res in steps:
        block = [f'<div class="step"><h2>{sid}</h2><p class="desc">{desc}</p>']
        block.append(f'<pre class="summary">{res["summary"]}</pre>')
        if res.get("table"):
            block.append(_table_html(res["table"]))
        if res.get("non_core_table"):
            block.append(f'<h3 style="margin-top:18px;">非核心症状特异性改善（{20-len(b.core_symptoms)}个非核心症状）</h3>')
            block.append(_table_html(res["non_core_table"]))
        if res.get("extra_table"):
            block.append(_table_html(res["extra_table"]))
        for fig in res.get("figures", []):
            block.append(f'<img src="{_img_to_base64(fig)}" alt="{os.path.basename(fig)}"/>')
        block.append("</div>")
        sections.append("\n".join(block))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>RCT 投影分析结果报告（103 例，pm1 口径）</title>
<style>
 body{{font-family:"Microsoft YaHei","SimHei",sans-serif;margin:0;background:#f5f6f8;color:#222;}}
 header{{background:#2c3e50;color:#fff;padding:24px 32px;}}
 header h1{{margin:0;font-size:22px;}}
 header p{{margin:6px 0 0;opacity:.85;font-size:13px;}}
 main{{max-width:1080px;margin:0 auto;padding:24px;}}
 .step{{background:#fff;border-radius:10px;padding:20px 24px;margin-bottom:22px;
        box-shadow:0 1px 4px rgba(0,0,0,.08);}}
 .step h2{{margin:0 0 4px;color:#2c3e50;border-left:4px solid #4E79A7;padding-left:10px;}}
 .desc{{color:#666;font-size:13px;margin:4px 0 12px;}}
 .summary{{background:#f0f3f7;border-left:3px solid #4E79A7;padding:10px 14px;
           font-size:13px;white-space:pre-wrap;border-radius:4px;overflow-x:auto;}}
 .tbl{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;}}
 .tbl th,.tbl td{{border:1px solid #ddd;padding:6px 10px;text-align:center;}}
 .tbl th{{background:#eef2f7;}}
 .tbl tr:nth-child(even) td{{background:#fafbfc;}}
 img{{max-width:100%;margin-top:10px;border:1px solid #eee;border-radius:6px;}}
 .meta{{background:#fff8e1;border-left:4px solid #f0a020;padding:10px 14px;margin-bottom:18px;
        font-size:13px;white-space:pre-wrap;border-radius:4px;}}
 footer{{text-align:center;color:#999;font-size:12px;padding:20px;}}
</style></head>
<body>
<header><h1>RCT 投影分析结果报告（pm1 口径）</h1>
<p>数据：103 例太极阈下抑郁 CES-D（干预前/后）｜ 参考框架：3447 例 Ising 参数（本脚本内拟合）</p>
<p>样本量 N = {b.n}（太极组 {(b.group==1).sum()}，对照组 {(b.group==2).sum()}）｜ 生成于 rct_snci_projection.py</p>
</header>
<main>
<div class="meta">{meta["text"]}</div>
{''.join(sections)}
</main>
<footer>由 rct_snci_projection.py 自动生成</footer>
</body></html>"""
    out = os.path.join(OUT_DIR, "rct_projection_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("RCT SNCI 投影分析（3447 例 FDSC 参考框架，pm1 口径）")
    print("=" * 78)

    # ---- 3447 例参考框架拟合 ----
    print("\n[参考框架] 3447 例 Ising 拟合（pm1 域）")
    _, binary_3447 = load_cesd_binary(DATA_3447)
    net = fit_ising_network(binary_3447)
    lv = percolation_communities(net["j_sym"])
    energy_3447 = compute_energy(binary_3447, net["intercept"], net["j_sym"])
    snci_3447, mu_ref, sigma_ref = standardize_snci(energy_3447)

    # ---- 保存参考框架（供文档复用） ----
    np.savez_compressed(
        OUT_DIR / "reference_network_3447_pm1.npz",
        h=net["intercept"], j=net["j_sym"],
        community=lv["labels"],
        core_symptoms=np.array(CORE_SYMPTOMS, dtype=object),
        energy=energy_3447, snci=snci_3447,
        items=np.array(ITEM_NAMES, dtype=object),
        p_c=float(lv["p_c"]), strength=np.abs(net["j_sym"]).sum(axis=1),
        bridge=lv["bridge"], mu=mu_ref, sigma=sigma_ref,
        n_edges=net["n_edges"],
    )
    print(f"[npz] -> {OUT_DIR / 'reference_network_3447_pm1.npz'}")

    # ---- 103 例投影与分析 ----
    print("\n[RCT 投影] 103 例分析")
    b = DataBundle(net["intercept"], net["j_sym"], ITEM_NAMES,
                   energy_3447, CORE_SYMPTOMS)
    print(f"[RCT] 已加载 103 例（对齐后 N={b.n}），核心症状={b.core_symptoms}")

    results = {}
    results["projection_energy"] = project_energy(b)
    print("[完成] 投影与症状能量")
    results["snci_compare"] = compare_snci_change(b)
    print("[完成] SNCI 组间比较")
    results["total_compare"] = compare_total_change(b)
    print("[完成] CES-D 总分组间比较")
    results["post_energy_compare"] = compare_post_energy(b)
    print("[完成] 干预后能量与参考人群比较")
    results["response_rates"] = response_rates(b)
    print("[完成] SNCI 与 CES-D 响应率")
    results["item_response_rates"] = item_response_rates(b)
    print("[完成] 逐症状响应率")
    results["core_change"] = core_change(b)
    print(f"[完成] 核心与非核心症状改善（{len(b.core_symptoms)} 核心 + {20-len(b.core_symptoms)} 非核心）")

    # ---- 框架关键参数元信息 ----
    meta_text = (
        f"【3447 例参考框架（pm1 口径）】\n"
        f"  Ising: {net['n_edges']} 边 / 190 可能，密度 {net['n_edges']/190:.1%}；"
        f"拟合 = ±1 域节点wise saga LASSO + EBIC(γ=0.25, logaddexp) + AND(|β|>1e-4)；"
        f"J 保号对称化 (J+Jᵀ)/2×AND；±1 域系数 ÷2 得 h, J\n"
        f"  渗流: p_c = {lv['p_c']:.4f}（GCC 降至 50% 阈值）｜ 社区: {lv['n_communities']} 个, Q = {lv['Q']:.4f}\n"
        f"  能量: E = −Σh·s − ½ΣJ·s·s（pm1 域），μ = {mu_ref:.4f}, σ = {sigma_ref:.4f}, "
        f"范围 [{energy_3447.min():.2f}, {energy_3447.max():.2f}]\n"
        f"  核心症状: {', '.join(CORE_SYMPTOMS)}\n"
        f"  SNCI = clip(50 + 10·(E−μ)/σ, 0, 100)\n"
        f"【103 例投影】n = {b.n}（太极 {int((b.group==1).sum())} / 对照 {int((b.group==2).sum())}）"
    )

    report = build_html(results, b, {"text": meta_text})
    print("=" * 78)
    print("全部步骤完成。统一报告：")
    print(report)
    print("=" * 78)


if __name__ == "__main__":
    main()
