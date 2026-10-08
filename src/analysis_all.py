# -*- coding: utf-8 -*-
"""
3447 例 CES-D 症状网络 —— 主分析脚本（无绘图版本，单文件独立可运行）
================================================================
一次运行产出论文分析的全部结果：

  主分析    Ising 估计（节点wise LASSO+EBIC+AND）→ 能量 → 渗流/Louvain/桥接
            → 核心投票 → SNCI → 节点可预测性
  核心症状  以 5000 次病例重抽样入选率 ≥50% 定义核心症状（默认读
            bootstrap_pm1_n5000.csv；加 --core-boot 可在本脚本内现场重抽样，
            多进程并支持断点续跑）
  导出      npz 参数保存 + xlsx 导出（6 个 sheet；不含敏感性分析）

模型与口径（pm1）：
  ① ±1 域 EBIC 似然 logaddexp 防溢出
  ② j_sym 保号对称化 (j_raw + j_raw.T)/2
  ③ 估计与能量同域（s = 2x−1，E = −Σh·s − ½ΣJ·s·s）
  ④ ±1 域 sklearn 系数 ÷2：h = β₀/2, J = β/2（±1 条件 logit = 2(h+ΣJ·s)）
综合投票（4 指标 top-6 含并列、≥3 票）仅作为 bootstrap 判核心的引擎，
不作为最终结论输出。

依赖：numpy / pandas / networkx / scipy / scikit-learn /
      python-louvain / openpyxl
用法：
  python analysis_all.py --core-boot        # 先做 5000 次核心症状重抽样（长时），再跑全流程
  python analysis_all.py --core-boot --boot-iters 200 --boot-jobs 4   # 小规模试跑
  python analysis_all.py                     # 主分析全流程（约 1 分钟）

敏感性分析（编码/γ/渗流操作化/分辨率、GCC 判据、判定阈值、p_c 复现）
已全部移至 sensitivity_analysis.py。
输出（均在 results/）：
  bootstrap_pm1_n{n}.csv（仅 --core-boot 时产出或续写）
  ising_params_3447_pm1.npz / 3447例_分析结果_pm1.xlsx（6 个 sheet）
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import warnings
from pathlib import Path

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 只屏蔽两类常见噪音：FutureWarning（库版本兼容提示）+ sklearn 收敛提示
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

# ============================================================
# 0. 配置
# ============================================================
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
DATA_DIR = Path(CONFIG_DATA_DIR)          # 数据置于 data/（可用环境变量 CESD_DATA_DIR 覆盖）
DATA_PATH = DATA_DIR / "3447例-CESD数据已反向计分(3).xlsx"
OUT_DIR = Path(CONFIG_OUT_DIR)            # 结果输出目录（可用环境变量 CESD_OUT_DIR 覆盖）
OUT_DIR.mkdir(parents=True, exist_ok=True)

CESD_COLS = [f"CESD_{i}" for i in range(1, 21)]
REVERSE_COLS = ["CESD_4", "CESD_8", "CESD_12", "CESD_16"]  # 数据已反向，仅文档可追溯
REVERSED_ALREADY = True

ITEM_NAMES = [
    "Bothered", "Appetite", "Blues", "AsGood", "Mind",
    "Depressed", "Effort", "Hopeful", "Failure", "Fearful",
    "Sleep", "Happy", "TalkedLess", "Lonely", "Unfriendly",
    "EnjoyedLife", "Crying", "Sad", "Dislike", "Going",
]

C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
EBIC_GAMMA = 0.25
RANDOM_STATE = 42
DEFAULT_BIN_THR = 2
AND_TOL = 1e-4
DEFAULT_ENCODING = "pm1"

# 核心症状重抽样（--core-boot 时现场执行）
N_CORE_BOOT = 5000                       # 论文正式版次数；小规模试跑可用 200
N_BOOT_JOBS = 6                          # 并行进程数
RNG_BASE = 20240825                      # 与既有 bootstrap CSV 一致的基种子，勿改（否则无法续跑）
BOOT_CSV_TPL = OUT_DIR / "bootstrap_pm1_n{n_iter}.csv"

# ============================================================
# 1.1 数据准备
# ============================================================
def load_cesd_binary(path: Path, bin_thr: int = DEFAULT_BIN_THR):
    """读取已反向计分数据并二值化。返回 (scored 0-3, binary 0/1 float)。"""
    df = pd.read_excel(path)
    cesd = df[CESD_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    scored = cesd.copy()
    if not REVERSED_ALREADY:
        for col in REVERSE_COLS:
            scored[col] = 3 - scored[col]
        print("[1.1] 已执行反向计分 (CESD_4/8/12/16)")
    else:
        print("[1.1] 数据已反向计分，跳过反向计分")
    binary = (scored >= bin_thr).astype(int)
    print(f"[1.1] N={len(binary)}, P={binary.shape[1]}, 症状检出率={binary.values.mean():.3f}")
    return scored, binary.values.astype(float)


# ============================================================
# 1.2 Ising 估计：节点wise LASSO 逻辑回归 + EBIC + AND
# ============================================================
def _loglik_mean(y, logits, encoding):
    """给定编码下的平均负对数似然（用于 EBIC），logaddexp 防溢出。"""
    if encoding == "01":
        y_pm1 = 2.0 * y - 1.0
        return float(np.mean(np.logaddexp(0.0, -y_pm1 * logits)))
    return float(np.mean(np.logaddexp(0.0, -y * logits)))


def fit_ising_network(binary: np.ndarray, gamma: float = EBIC_GAMMA,
                encoding: str = DEFAULT_ENCODING) -> dict:
    n, p = binary.shape
    X = binary.astype(float) if encoding == "01" else (2 * binary.astype(int) - 1).astype(float)

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
            mean_ll = _loglik_mean(y, logits, encoding)
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
    j_sym = (j_raw + j_raw.T) / 2.0 * adj_and                  # 保号对称化（保留负向边）

    if encoding == "pm1":                                      # ±1 域回归系数 ÷2 得 Ising 参数
        intercept *= 0.5
        j_raw *= 0.5
        j_sym *= 0.5

    n_edges = int(adj_and.sum() // 2)
    print(f"[1.2] EBIC+AND (gamma={gamma}, enc={encoding}): {n_edges} 条边, "
          f"密度={n_edges / (p * (p - 1) / 2):.1%}")
    return {"intercept": intercept, "j_sym": j_sym, "j_raw": j_raw,
            "n_edges": n_edges, "adj_and": adj_and, "adj_or": edge_mask | edge_mask.T}


# ============================================================
# 1.3 Ising 能量（与估计同域）
# ============================================================
def compute_energy(binary: np.ndarray, h: np.ndarray, J: np.ndarray,
                 encoding: str = DEFAULT_ENCODING) -> np.ndarray:
    if encoding == "01":
        s = binary.astype(float)
    else:
        s = (2 * binary.astype(int) - 1).astype(float)
    E = -(s @ h) - 0.5 * np.sum((s @ J) * s, axis=1)
    print(f"[1.3] E ∈ [{E.min():.2f}, {E.max():.2f}], μ={E.mean():.2f}, σ={E.std():.2f}")
    return E


# ============================================================
# 1.4 渗流（参数化 GCC）+ Louvain（参数化 resolution）+ 桥接
# ============================================================
def percolation_communities(j_sym: np.ndarray, gcc_frac: float = 0.50,
                       resolution: float = 1.0) -> dict:
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
                    and len(max(nx.connected_components(gg), key=len)) / n >= gcc_frac:
                p_c = float(thr)
            else:
                break

    g = nx.Graph(); g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            w = abs(j_sym[u, v])
            if w > AND_TOL:
                g.add_edge(u, v, weight=w)
    partition = community_louvain.best_partition(g, resolution=resolution,
                                                 random_state=RANDOM_STATE)
    labels = np.array([partition[i] for i in range(n)])
    # python-louvain 的 modularity() 不接受 resolution 参数，这里用 networkx 的
    # resolution-aware modularity 计算 Q，与社区检测保持同一分辨率。
    _comms = {}
    for _node, _cid in partition.items():
        _comms.setdefault(_cid, set()).add(_node)
    Q = nx.community.modularity(g, list(_comms.values()),
                                weight='weight', resolution=resolution)

    bridge = np.zeros(n)
    for u in range(n):
        for v in range(u + 1, n):
            if labels[u] != labels[v]:
                w = abs(j_sym[u, v])
                bridge[u] += w
                bridge[v] += w

    print(f"[1.4] p_c={p_c:.4f}, {len(set(labels))} 社区, Q={Q:.4f}")
    for cid in sorted(set(labels)):
        mem = [ITEM_NAMES[i] for i in range(n) if labels[i] == cid]
        print(f"      C{cid}: {mem}")
    return {"p_c": p_c, "labels": labels, "Q": Q, "bridge": bridge,
            "n_communities": len(set(labels))}


# ============================================================
# 1.5 核心投票（并列扩展 top-6，仅作 bootstrap 引擎）
# ============================================================
def top_k_with_ties(vals: np.ndarray, k: int = 6) -> set:
    order = np.argsort(vals)[::-1]
    thr_val = vals[order[k - 1]]
    return set(np.where(vals >= thr_val - 1e-12)[0].tolist())


def point_estimate_core(j_sym: np.ndarray, h: np.ndarray, p_c: float,
                 bridge: np.ndarray, top_k: int = 6) -> dict:
    n = j_sym.shape[0]
    strength = np.abs(j_sym).sum(axis=1)
    percolation = (np.abs(j_sym) >= p_c).sum(axis=1) if p_c > 0 else np.zeros(n)
    field = np.abs(h)
    metrics = {"strength": strength, "bridge": bridge,
               "percolation": percolation, "field": field}
    top_sets = {name: top_k_with_ties(vals, top_k) for name, vals in metrics.items()}
    votes = np.zeros(n, dtype=int)
    for s in top_sets.values():
        for idx in s:
            votes[idx] += 1
    pillars = [ITEM_NAMES[i] for i in range(n) if votes[i] >= 3]
    # 返回 votes / top_sets / pillars，供 bootstrap 直接复用
    return {"votes": votes, "top_sets": top_sets, "pillars": pillars}


def pillar_names(net, lv, top_k=6):
    pl = point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"], top_k=top_k)
    n = net["j_sym"].shape[0]
    return [ITEM_NAMES[i] for i in range(n) if pl["votes"][i] >= 3]


# ============================================================
# 1.6 SNCI 标准化
# ============================================================
def standardize_snci(E: np.ndarray) -> tuple:
    mu = float(E.mean())
    sigma = float(E.std(ddof=0))
    SNCI = np.clip(50.0 + 10.0 * (E - mu) / (sigma + 1e-12), 0.0, 100.0)
    print(f"[1.6] SNCI ∈ [{SNCI.min():.2f}, {SNCI.max():.2f}]")
    return SNCI, mu, sigma


# ============================================================
# 1.7 节点可预测性（Haslbeck & Waldorp 2018）
# ============================================================
def step7_predictability(binary: np.ndarray, net: dict,
                         encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    n, p = binary.shape
    X = binary.astype(float) if encoding == "01" else (2 * binary.astype(int) - 1).astype(float)
    rows = []
    for i in range(p):
        y, Z = X[:, i], np.delete(X, i, axis=1)
        z = net["intercept"][i] + Z @ net["j_raw"][i, np.arange(p) != i]
        if encoding == "pm1":
            z = 2.0 * z                                 # 恢复 sklearn 的 logit 尺度
        if encoding == "01":
            # logaddexp 替代 log1p(exp(z))，避免 z 大时上溢
            ll_fit = np.sum(y * z - np.logaddexp(0.0, z))
            p1 = y.mean()
            ll_null = n * (p1 * np.log(p1 + 1e-12)
                           + (1 - p1) * np.log(1 - p1 + 1e-12))
        else:
            # logaddexp 防止 −y·z 大负值时 exp 溢出
            ll_fit = -np.sum(np.logaddexp(0.0, -y * z))
            p1 = (y.mean() + 1.0) / 2.0
            z0 = np.log(p1 / (1 - p1 + 1e-12))
            ll_null = -np.sum(np.logaddexp(0.0, -y * z0))
        mcf = 1 - ll_fit / ll_null
        prob = 1 / (1 + np.exp(-z))
        r2 = float(np.corrcoef(y, prob)[0, 1] ** 2)
        rows.append({"症状": ITEM_NAMES[i], "McFadden_R2": round(float(mcf), 3),
                     "corr2_obs_pred": round(r2, 3)})
    df = pd.DataFrame(rows)
    print(f"[1.7] 可预测性: McFadden R² 均值={df['McFadden_R2'].mean():.3f} "
          f"({df['McFadden_R2'].min():.3f}–{df['McFadden_R2'].max():.3f}), "
          f"corr² 均值={df['corr2_obs_pred'].mean():.3f}")
    return df


# ============================================================
# 1.8 核心症状判定（读重抽样入选率，≥ 50%）
# ============================================================
def core_boot_csv(n_iter=N_CORE_BOOT):
    """核心症状重抽样结果 CSV 路径（文件名带次数，避免不同规模互相覆盖）。"""
    return Path(str(BOOT_CSV_TPL).replace("{n_iter}", str(n_iter)))


def load_robust_core(n_iter=N_CORE_BOOT):
    """从重抽样 CSV 读核心症状：入选率 ≥ 50% 判定。缺失时返回空并给出补跑提示。"""
    csv_path = core_boot_csv(n_iter)
    if not csv_path.exists():
        print(f"[1.8] 未找到 {csv_path.name}——可加 --core-boot 现场重抽样"
              f"（python analysis_all.py --core-boot --boot-iters {n_iter}）；本次以候选症状占位。")
        return [], set(), {}
    bdf = pd.read_csv(csv_path)
    core = bdf.loc[bdf["inclusion_pct"] >= 50.0, "症状"].tolist()
    pct = dict(zip(bdf["症状"], bdf["inclusion_pct"]))
    print(f"[1.8] 核心症状(≥50%) = {[(s, pct[s]) for s in core]}")
    return core, set(core), pct


# ============================================================
# 1.9 核心症状：5 000 次病例重抽样（入选率 ≥ 50%；可断点续跑）
# ============================================================
_BOOT_BINARY = None          # 由各子进程的 initializer 设置，避免每轮重复传参


def _boot_init(binary):
    global _BOOT_BINARY
    _BOOT_BINARY = binary


def boot_one_iter(iter_idx):
    """单轮：有放回抽病例 → 重估 Ising → 渗流/Louvain → 四指标投票 → 本轮核心症状集合。

    重抽样单位是受试者（病例），每轮完整重跑 step2→step4→step5，与主分析同口径。
    """
    rng = np.random.default_rng(RNG_BASE + iter_idx)
    n = _BOOT_BINARY.shape[0]
    bs = _BOOT_BINARY[rng.integers(0, n, size=n)]
    with contextlib.redirect_stdout(io.StringIO()):      # 屏蔽 step2/4/5 打印
        net = fit_ising_network(bs, encoding=DEFAULT_ENCODING)
        lv = percolation_communities(net["j_sym"])
        pillar = point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])
    return set(pillar["pillars"])


def run_core_bootstrap(binary, n_iter=N_CORE_BOOT, n_jobs=N_BOOT_JOBS, csv_path=None):
    """病例重抽样 n_iter 次，统计各症状入选核心（≥3 票）的比例，落盘 CSV。

    每 50 轮增量保存；若 CSV 已存在且 done < n_iter，则从 done 处续跑。
    """
    csv_path = Path(csv_path) if csv_path else core_boot_csv(n_iter)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 点估计核心（仅用于 CSV 的 is_main_core 标记）
    with contextlib.redirect_stdout(io.StringIO()):
        main_net = fit_ising_network(binary, gamma=EBIC_GAMMA, encoding=DEFAULT_ENCODING)
        main_lv = percolation_communities(main_net["j_sym"])
        main_pillar = point_estimate_core(main_net["j_sym"], main_net["intercept"],
                                   main_lv["p_c"], main_lv["bridge"])

    counts = {it: 0 for it in ITEM_NAMES}
    done = 0

    def save_snapshot():
        rows = sorted(((it, 100.0 * counts[it] / max(done, 1), it in main_pillar["pillars"])
                       for it in ITEM_NAMES), key=lambda r: -r[1])
        pd.DataFrame([{"症状": it, "inclusion_pct": round(p, 1),
                       "is_main_core": c, "done": done}
                      for it, p, c in rows]).to_csv(csv_path, index=False)

    if csv_path.exists():                      # 断点续跑
        try:
            prev = pd.read_csv(csv_path)
            prev_done = int(prev["done"].iloc[0]) if "done" in prev.columns else n_iter
            for _, row in prev.iterrows():
                if row["症状"] in counts:
                    counts[row["症状"]] = int(round(row["inclusion_pct"] / 100.0 * prev_done))
            done = prev_done
            if done >= n_iter:
                save_snapshot()
                print(f"[1.9] 已有完成档 done={done} ≥ {n_iter} -> {csv_path}")
                return csv_path
            print(f"[1.9] 检测到残档 done={done}/{n_iter}，从第 {done + 1} 轮续跑")
        except Exception as e:
            print(f"[1.9] 读取残档失败（{e}），从头开始")
            counts = {it: 0 for it in ITEM_NAMES}
            done = 0

    print(f"[1.9] 病例重抽样 {done}→{n_iter} 次，{n_jobs} 进程并行；每 50 轮落盘")
    with mp.Pool(n_jobs, initializer=_boot_init, initargs=(binary,)) as pool:
        for core_set in pool.imap_unordered(boot_one_iter, range(done, n_iter)):
            for it in core_set:
                counts[it] += 1
            done += 1
            if done % 50 == 0:
                save_snapshot()
                print(f"[1.9] 完成 {done}/{n_iter}")
    save_snapshot()
    print(f"[1.9] 完成 {done}/{n_iter} -> {csv_path}")
    return csv_path


# ============================================================
# 4. xlsx 导出（6 个 sheet；敏感性分析见 sensitivity_analysis.py）
# ============================================================
def _cohen_d(snci, high):
    g1, g0 = snci[high], snci[~high]
    sp = np.sqrt(((len(g1) - 1) * g1.var(ddof=1) + (len(g0) - 1) * g0.var(ddof=1))
                 / (len(g1) + len(g0) - 2))
    return round((g1.mean() - g0.mean()) / sp, 4)


def export_excel(scored, binary, net, lv, energy, snci_arr, cesd_total,
                 mu_ref, sigma_ref, pred_df, encoding, robust_core, core_set, pct):
    n, p = binary.shape
    excel_path = OUT_DIR / "3447例_分析结果_pm1.xlsx"
    notes = pd.DataFrame({
        "项目": ["分析对象", "二值化阈值", "编码方式", "Ising估计方法",
                 "EBIC惩罚参数", "AND规则阈值", "临界渗流阈值p_c",
                 "Louvain分辨率", "核心症状(≥50%入选率)", "SNCI标准化", "可预测性"],
        "说明": [
            f"N={n}，P={p} 个CES-D条目（已反向计分，0-3分）",
            "条目得分 ≥ 2 → 1（症状存在），否则 0",
            f"{encoding} 域估计与能量同域（s=2x-1，E=−Σh·s−½ΣJ·s·s）",
            "节点wise LASSO逻辑回归 + EBIC + AND规则（van Borkulo 2014）",
            f"γ = {EBIC_GAMMA}",
            f"两个方向回归系数 |β| > {AND_TOL} 才保留该边",
            f"GCC降至50%时的边权阈值 p_c = {lv['p_c']:.4f}",
            "resolution = 1.0（标准模块度）",
            "、".join(f"{s}({pct[s]:.1f}%)" for s in robust_core),
            f"SNCI = 50 + 10×(E-μ)/σ，μ={mu_ref:.3f}，σ={sigma_ref:.3f}，截断至[0,100]",
            "McFadden R² 与 corr(y,p̂)²（Haslbeck & Waldorp 2018）",
        ],
    })
    high = (cesd_total >= 16)
    main_summary = pd.DataFrame({
        "指标": ["样本量N", "节点数P", "网络边数", "网络密度", "临界渗流阈值p_c",
                 "社区数", "模块度Q", "核心症状(≥50%入选率)", "能量均值μ", "能量标准差σ",
                 "SNCI范围", "r(E, CESD)", "AUC(≥16)", "Cohen's d(≥16)"],
        "数值": [n, p, net["n_edges"],
                 f"{net['n_edges'] / (p * (p - 1) / 2):.1%}",
                 f"{lv['p_c']:.4f}", lv["n_communities"], f"{lv['Q']:.4f}",
                 "、".join(f"{s}({pct[s]:.1f}%)" for s in robust_core),
                 f"{mu_ref:.3f}", f"{sigma_ref:.3f}",
                 f"[{snci_arr.min():.2f}, {snci_arr.max():.2f}]",
                 f"{stats.pearsonr(energy, cesd_total)[0]:.4f}",
                 f"{roc_auc_score(high.astype(int), snci_arr):.4f}",
                 _cohen_d(snci_arr, high)],
    })
    Jdf = pd.DataFrame(net["j_sym"], columns=ITEM_NAMES)
    Jdf.insert(0, "症状", ITEM_NAMES)
    strength = np.abs(net["j_sym"]).sum(axis=1)
    field = np.abs(net["intercept"])
    votes = point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])["votes"]
    node_df = pd.DataFrame({
        "症状": ITEM_NAMES, "外场h": net["intercept"], "|h|外场强度": field,
        "强度Strength(Σ|J|)": strength,
        "桥接强度Bridge(跨社区Σ|J|)": lv["bridge"],
        "渗流度Percolation(≥p_c边数)":
            (np.abs(net["j_sym"]) >= lv["p_c"]).sum(axis=1) if lv["p_c"] > 0 else np.zeros(p),
        "总票数Votes": votes,
        "是否核心症状(≥50%)": ["是" if ITEM_NAMES[i] in core_set else "否" for i in range(p)],
        "社区Community": lv["labels"],
    })
    subj_df = pd.DataFrame({
        "CESD总分": cesd_total, "二值化症状总数": binary.sum(axis=1).astype(int),
        "Ising能量E": energy, "SNCI": snci_arr,
    })
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="说明", index=False)
        main_summary.to_excel(writer, sheet_name="主分析汇总", index=False)
        Jdf.to_excel(writer, sheet_name="J邻接矩阵", index=False)
        node_df.to_excel(writer, sheet_name="节点中心性与社区", index=False)
        subj_df.to_excel(writer, sheet_name="被试能量与SNCI", index=False)
        pred_df.to_excel(writer, sheet_name="节点可预测性", index=False)
    print(f"[Excel] -> {excel_path}")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-boot", action="store_true",
                        help="核心症状病例重抽样（长时，可续跑）")
    parser.add_argument("--boot-iters", type=int, default=N_CORE_BOOT,
                        help=f"重抽样次数（默认 {N_CORE_BOOT}）")
    parser.add_argument("--boot-jobs", type=int, default=N_BOOT_JOBS,
                        help=f"重抽样并行进程数（默认 {N_BOOT_JOBS}）")
    args = parser.parse_args()

    scored, binary = load_cesd_binary(DATA_PATH)
    cesd_total = scored.sum(axis=1).values

    # ---- 主分析 ----
    net = fit_ising_network(binary, gamma=EBIC_GAMMA, encoding=DEFAULT_ENCODING)
    energy = compute_energy(binary, net["intercept"], net["j_sym"], encoding=DEFAULT_ENCODING)
    lv = percolation_communities(net["j_sym"])
    snci_arr, mu_ref, sigma_ref = standardize_snci(energy)
    pred_df = step7_predictability(binary, net, encoding=DEFAULT_ENCODING)

    # ---- 核心症状（重抽样入选率 ≥50%）----
    if args.core_boot:
        run_core_bootstrap(binary, n_iter=args.boot_iters, n_jobs=args.boot_jobs)
    robust_core, core_set, pct = load_robust_core(args.boot_iters)

    # ---- 保存与导出 ----
    # 负向边按对称化后的 j_sym 判断（j_raw 两个方向可能符号不一致）
    neg_pairs = [(ITEM_NAMES[u], ITEM_NAMES[v])
                 for u in range(binary.shape[1]) for v in range(u + 1, binary.shape[1])
                 if net["adj_and"][u, v] and net["j_sym"][u, v] < 0]
    print(f"[边符号] 负向边 {len(neg_pairs)} 对: {neg_pairs}")

    np.savez_compressed(
        OUT_DIR / "ising_params_3447_pm1.npz",
        h=net["intercept"], J=net["j_sym"], j_raw=net["j_raw"],
        adj_and=net["adj_and"], adj_or=net["adj_or"],
        community=lv["labels"], pillars=np.array(robust_core if robust_core else []),
        energy=energy, snci=snci_arr, items=np.array(ITEM_NAMES),
        p_c=lv["p_c"], strength=np.abs(net["j_sym"]).sum(axis=1),
        bridge=lv["bridge"], mu_ref=mu_ref, sigma_ref=sigma_ref,
    )
    print(f"[npz] -> {OUT_DIR / 'ising_params_3447_pm1.npz'}")

    export_excel(scored, binary, net, lv, energy, snci_arr, cesd_total,
                 mu_ref, sigma_ref, pred_df, DEFAULT_ENCODING,
                 robust_core, core_set, pct)

    print("\n========== ANALYSIS ALL DONE ==========")
    print(f"  N={len(binary)}, P={binary.shape[1]}")
    print(f"  边={net['n_edges']}, p_c={lv['p_c']:.4f}, Q={lv['Q']:.4f}")
    print(f"  核心症状(≥50%) = {[(s, pct[s]) for s in robust_core]}")
    print(f"  E∈[{energy.min():.2f}, {energy.max():.2f}], μ={mu_ref:.3f}, σ={sigma_ref:.3f}")
    print(f"  SNCI∈[{snci_arr.min():.2f}, {snci_arr.max():.2f}]")
    print(f"  r(E,CESD)={stats.pearsonr(energy, cesd_total)[0]:.4f}, "
          f"AUC(≥16)={roc_auc_score((cesd_total >= 16).astype(int), snci_arr):.4f}, "
          f"d={_cohen_d(snci_arr, cesd_total >= 16)}")
    print("=======================================")


if __name__ == "__main__":
    main()
