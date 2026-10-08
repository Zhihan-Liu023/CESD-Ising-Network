# -*- coding: utf-8 -*-
"""
跨队列验证 · 纯数据分析脚本（无画图代码）
================================================================
功能（全部为数值计算，不含任何 matplotlib / 绘图）：
  (1) 判别效度：各队列独立估计 Ising 网络 → 计算 SNCI →
      以 SNCI 为连续预测变量对高症状组（CES-D 总分 ≥ 16）进行 ROC 曲线分析，
      报告 AUC（5000 次分层 Bootstrap 95% CI）、Youden 最佳 SNCI 截断值、
      灵敏度、特异度、PPV、NPV；
      同时报告 SNCI–总分 Pearson r（Fisher z 95% CI）、Cohen's d、Welch t。
  (2) 网络结构：模块度 Q、渗流阈值 p_c、社区数(Louvain γ=1.0)、
      核心症状以重抽样入选率 ≥50% 为准（点估计入选 ≥3 票仅作对照）；
      外部队列额外报告相对 FDSC 发现队列的 ARI 与边权相关 r_J。
  (3) 核心症状 Bootstrap 稳定性（每轮重抽样→重估网络→投票，
      入选率 ≥50% 定为核心症状；带断点续跑，部分完成快照可直接填表）。
      默认开启；CFPS 全年龄等大样本长时任务建议在用户本机跑。
  (4) 网络比较测试（NCT，van Borkulo et al. 2022, Psychological Methods）：
      对 FDSC vs CFPS、FDSC vs MIDUS 两两做置换检验，报告
      全局强度不变性（S = |GS1-GS2|）与网络结构不变性（M = max|ΔJ_ij|）
      的 P 值；作为 SNCI 跨样本可比性的逻辑基础（证明网络结构足够相似）。
      带断点续跑；P 值越大越支持"网络不变"，P<0.05 表示两网络在该方面显著不同。

关键口径（与已发表分析完全一致）：
  - Ising：nodewise LASSO 逻辑回归 + EBIC(γ=0.25) + AND 对称化，pm1 编码
  - SNCI = clip(50 + 10(E-μ)/σ, 0, 100)，μ/σ 为本队列能量 E 的均值/标准差
  - 高症状：CES-D 总分 ≥ 16（单一定义）
  - 渗流：按 |J| 阈值递增，GCC 降至 50% 时的阈值 p_c
  - Louvain：标准模块度 resolution=1.0；Q 用 networkx resolution-aware modularity
  - 核心症状：点估计为强度/桥接/渗流度/外场 各取前 6（含并列）得票 ≥3 作对照；
    最终判定以重抽样（每轮重估网络→投票）入选率 ≥50% 为准。
  - 青年亚组(15–30)仅进入判别段；网络结构段按最新决定默认不含青年，
    可在 STRUCTURE_COHORTS 中自行加入 "CFPS_youth"。

依赖：numpy, pandas, scipy, scikit-learn, networkx, python-louvain(community)
用法示例：
  # 全量（4 队列判别 + 3 队列结构 + NCT；核心症状默认基于重抽样）
  python cross_cohort_analysis.py
  # 仅 FDSC 冒烟测试（快速）
  python cross_cohort_analysis.py --discrim-cohorts FDSC --no-structure
  # 把 CFPS 全年龄的部分快照(3100/5000)续跑到 5000（大样本，建议本机跑）
  python cross_cohort_analysis.py --bootstrap --boot-iters 5000
  # NCT 小规模试跑（100 次置换，约数分钟）
  python cross_cohort_analysis.py --no-discrim --no-structure --nct-iters 100
  # 跳过 NCT（如只想看重抽样/判别）
  python cross_cohort_analysis.py --no-nct
输出：results/cross_cohort_discriminant_results.csv
      results/cross_cohort_structure_results.csv
      results/cross_cohort_nct_results.csv
      results/nct_<t1>_vs_<t2>_n<iters>.csv （NCT 置换快照，断点续跑）
      results/cross_boot_<tag>_n<iters>.csv （<tag>=fdsc/cfps/midus，重抽样入选率）
"""

import argparse
import io
import os
import contextlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, adjusted_rand_score
from sklearn.linear_model import LogisticRegression   # fit_ising_network 的 nodewise 估计依赖
import community as community_louvain
import networkx as nx

# ============================== CONFIG（换机器只改这里） ==============================
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
OUT_DIR = Path(CONFIG_OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(CONFIG_DATA_DIR)   # 数据置于 data/
FDSC_XLSX = DATA_DIR / "3447例-CESD数据已反向计分(3).xlsx"
CFPS_CSV = DATA_DIR / "CFPS_cesd20_scores.csv"
MIDUS_XL = DATA_DIR / "MIDUS_CESD20.xlsx"

# 模型与标准化口径
EBIC_GAMMA = 0.25
AND_TOL = 1e-4
RANDOM_STATE = 42
C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
THRESHOLDS = [16]          # 高症状组界定（CES-D 总分）
N_BOOT_AUC = 5000          # AUC 的 95%CI 分层 Bootstrap 次数

# 队列选择（青年仅进判别段）
DISCRIM_COHORTS = ["FDSC", "CFPS_all", "CFPS_youth", "MIDUS"]
STRUCTURE_COHORTS = ["FDSC", "CFPS_all", "MIDUS"]   # 不含青年（如需加入改此处）

# 核心症状 Bootstrap（默认开启：核心症状判定基于重抽样；部分快照可直接填表）
N_BOOT_CORE = 5000
N_JOBS_CORE = 6
INCLUSION_THRESHOLD = 50.0          # 重抽样入选率阈值（%）
# 快照文件名标签（与已产出的 cross_boot_cfps/midus_*.csv 对齐，支持断点续跑）
BOOT_FILE_TAG = {"FDSC": "fdsc", "CFPS_all": "cfps", "MIDUS": "midus"}
BOOT_COHORTS = ["FDSC", "CFPS_all", "MIDUS"]   # 全部结构队列都基于重抽样

# 网络比较测试（NCT，van Borkulo et al. 2022, Psychological Methods）
NCT_PERMS = 1000          # 置换次数（正式 1000；试跑可 --nct-iters 100）
NCT_JOBS = 6              # 并行进程数（12 核机器可调至 10）
NCT_SEED = 20240826       # 置换随机种子（独立于模型 seed=42，勿改以保可续跑）
NCT_PAIRS = [("FDSC", "MIDUS"), ("FDSC", "CFPS_all")]   # 两两比较；先快(MIDUS n=1255)后慢(CFPS n=31033)

ITEM_NAMES = [
    "Bothered", "Appetite", "Blues", "AsGood", "Mind",
    "Depressed", "Effort", "Hopeful", "Failure", "Fearful",
    "Sleep", "Happy", "TalkedLess", "Lonely", "Unfriendly",
    "EnjoyedLife", "Crying", "Sad", "Dislike", "Going",
]

# ============================== 数据载入 ==============================
def load_fdsc():
    df = pd.read_excel(FDSC_XLSX)
    cols = [f"CESD_{i}" for i in range(1, 21)]
    scored = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    X = scored.values.astype(float)
    return "FDSC 发现队列", X, (X >= 2).astype(float), len(X)


def load_cfps():
    df = pd.read_csv(CFPS_CSV)
    cols = [f"qq601{i}_sc" for i in range(1, 21)]
    keep = df[cols].notna().all(axis=1)
    X = df.loc[keep, cols].values.astype(float)
    age = df.loc[keep, "age"].astype(float).values
    youth = (age >= 15) & (age <= 30)
    out = [("CFPS 全年龄", X, (X >= 2).astype(float), int(keep.sum()))]
    out.append(("CFPS 青年(15-30)", X[youth], (X[youth] >= 2).astype(float), int(youth.sum())))
    return out


def load_midus():
    df = pd.read_excel(MIDUS_XL)
    cols = [c for c in df.columns if c.startswith("CESD") and "(" in c]
    cols = sorted(cols, key=lambda c: int(c.split(" ")[0].replace("CESD", "")))
    if len(cols) != 20:
        total = next(c for c in df.columns if "total" in c.lower())
        ti = list(df.columns).index(total)
        cols = list(df.columns[ti - 20:ti])
    keep = df[cols].notna().all(axis=1)
    X = df.loc[keep, cols].values.astype(float)
    return "MIDUS M2", X, (X >= 2).astype(float), int(keep.sum())


def load_all():
    """返回 {cohort_key: (label, scored 0-3, binary 0/1, N)}。"""
    data = {}
    lbl, sc, bn, n = load_fdsc()
    data["FDSC"] = (lbl, sc, bn, n)
    for lbl, sc, bn, n in load_cfps():
        key = "CFPS_all" if "全年龄" in lbl else "CFPS_youth"
        data[key] = (lbl, sc, bn, n)
    lbl, sc, bn, n = load_midus()
    data["MIDUS"] = (lbl, sc, bn, n)
    return data


# ============================== Ising 估计 ==============================
def _loglik_mean(y, logits, encoding):
    if encoding == "01":
        y_pm1 = 2.0 * y - 1.0
        return float(np.mean(np.logaddexp(0.0, -y_pm1 * logits)))
    return float(np.mean(np.logaddexp(0.0, -y * logits)))


def fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="pm1"):
    n, p = binary.shape
    X = binary.astype(float) if encoding == "01" else (2 * binary.astype(int) - 1).astype(float)
    intercept = np.zeros(p)
    j_raw = np.zeros((p, p))
    edge_mask = np.zeros((p, p), dtype=bool)
    for i in range(p):
        y, Z = X[:, i], np.delete(X, i, axis=1)
        best_ebic, best_coefs, best_int = np.inf, None, 0.0
        for c_val in C_GRID:
            clf = LogisticRegression(solver="saga", C=c_val, l1_ratio=1.0,
                                     fit_intercept=True, max_iter=3000, tol=1e-4,
                                     random_state=RANDOM_STATE)
            clf.fit(Z, y)
            coefs = clf.coef_[0]
            n_nz = int(np.sum(np.abs(coefs) > AND_TOL))
            logits = clf.intercept_[0] + Z @ coefs
            mean_ll = _loglik_mean(y, logits, encoding)
            ebic = 2.0 * n * mean_ll + n_nz * np.log(n) + 2.0 * gamma * n_nz * np.log(p - 1)
            if ebic < best_ebic:
                best_ebic, best_coefs, best_int = ebic, coefs.copy(), float(clf.intercept_[0])
        intercept[i] = best_int
        if best_coefs is not None:
            mask = np.ones(p, dtype=bool)
            mask[i] = False
            j_raw[i, mask] = best_coefs
            edge_mask[i, mask] = np.abs(best_coefs) > AND_TOL
    adj_and = edge_mask & edge_mask.T
    j_sym = (j_raw + j_raw.T) / 2.0 * adj_and
    if encoding == "pm1":
        intercept *= 0.5
        j_raw *= 0.5
        j_sym *= 0.5
    n_edges = int(adj_and.sum() // 2)
    return {"intercept": intercept, "j_sym": j_sym, "n_edges": n_edges}


def compute_snci(binary, net):
    h, J = net["intercept"], net["j_sym"]
    s = 2 * binary - 1
    E = -(s @ h) - 0.5 * np.sum((s @ J) * s, axis=1)
    mu, sigma = float(E.mean()), float(E.std(ddof=1))
    snci = np.clip(50 + 10 * (E - mu) / sigma, 0, 100)
    return E, snci, mu, sigma


# ============================== 判别统计量 ==============================
def corr_ci(x, y):
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(x) - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), p


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def auc_ci(y, score, n_boot=N_BOOT_AUC, seed=RANDOM_STATE):
    y = np.asarray(y); score = np.asarray(score)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        idx = np.concatenate([b0, b1])
        bs.append(roc_auc_score(y[idx], score[idx]))
    bs = np.array(bs)
    return roc_auc_score(y, score), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def youden_metrics(y, score):
    y = np.asarray(y); score = np.asarray(score, dtype=float)
    fpr, tpr, thr = roc_curve(y, score)
    j = tpr - fpr
    k = int(np.argmax(j))
    cut = float(thr[k])
    pred = (score >= cut).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return cut, sens, spec, ppv, npv


def analyze_discriminative(name, scored, binary):
    net = fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="pm1")
    E, SNCI, mu, sigma = compute_snci(binary, net)
    total = scored.sum(axis=1)
    res = {"队列": name, "N": len(binary), "边": net["n_edges"],
           "能量mu": round(mu, 4), "能量sigma": round(sigma, 4),
           "SNCI_min": round(float(SNCI.min()), 2), "SNCI_max": round(float(SNCI.max()), 2),
           "SNCI_mean": round(float(SNCI.mean()), 2), "SNCI_sd": round(float(SNCI.std(ddof=1)), 2),
           "E_min": round(float(E.min()), 4), "E_max": round(float(E.max()), 4),
           "E_mean": round(float(E.mean()), 4), "E_sd": round(float(E.std(ddof=1)), 4),
           "总分_min": float(total.min()), "总分_max": float(total.max())}
    for thr in THRESHOLDS:
        tag = f">= {thr}"
        high = (total >= thr).astype(int)
        pos, neg = high == 1, ~ (high == 1)
        # 判别效度：以 SNCI 为连续预测变量直接做 ROC 曲线分析
        auc, lo, hi = auc_ci(high, SNCI, n_boot=N_BOOT_AUC)
        r, rlo, rhi, _ = corr_ci(SNCI, total)
        d = cohens_d(SNCI[pos], SNCI[neg])
        t, _ = stats.ttest_ind(SNCI[pos], SNCI[neg], equal_var=False)
        cut, sens, spec, ppv, npv = youden_metrics(high, SNCI)
        res[f"AUC_{tag}"] = round(auc, 4)
        res[f"AUC_{tag}_lo"] = round(lo, 4)
        res[f"AUC_{tag}_hi"] = round(hi, 4)
        res[f"r_{tag}"] = round(r, 4)
        res[f"r_{tag}_lo"] = round(rlo, 4)
        res[f"r_{tag}_hi"] = round(rhi, 4)
        res[f"d_{tag}"] = round(d, 3)
        res[f"Welch_t_{tag}"] = round(float(t), 2)
        res[f"SNCI截断_{tag}"] = round(cut, 2)
        res[f"灵敏度_{tag}"] = round(sens, 4)
        res[f"特异度_{tag}"] = round(spec, 4)
        res[f"PPV_{tag}"] = round(ppv, 4)
        res[f"NPV_{tag}"] = round(npv, 4)
        print(f"  [{name}] {tag}: AUC={auc:.4f} (95%CI {lo:.4f}-{hi:.4f}) | "
              f"r={r:.3f} | d={d:.3f} | Welch t={t:.1f} | "
              f"Youden SNR截断={cut:.2f} 灵敏={sens:.3f} 特异={spec:.3f}")
    return res


# ============================== 网络结构 ==============================
def top_k_with_ties(vals, k=6):
    order = np.argsort(vals)[::-1]
    thr_val = vals[order[k - 1]]
    return set(np.where(vals >= thr_val - 1e-12)[0].tolist())


def percolation_communities(j_sym, gcc_frac=0.50, resolution=1.0):
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
            if gg.number_of_edges() and len(max(nx.connected_components(gg), key=len)) / n >= gcc_frac:
                p_c = float(thr)
            else:
                break
    g = nx.Graph(); g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            w = abs(j_sym[u, v])
            if w > AND_TOL:
                g.add_edge(u, v, weight=w)
    partition = community_louvain.best_partition(g, resolution=resolution, random_state=RANDOM_STATE)
    labels = np.array([partition[i] for i in range(n)])
    comms = {}
    for node, cid in partition.items():
        comms.setdefault(cid, set()).add(node)
    Q = nx.community.modularity(g, list(comms.values()), weight="weight", resolution=resolution)
    bridge = np.zeros(n)
    for u in range(n):
        for v in range(u + 1, n):
            if labels[u] != labels[v]:
                w = abs(j_sym[u, v]); bridge[u] += w; bridge[v] += w
    return {"p_c": p_c, "labels": labels, "Q": Q, "bridge": bridge,
            "n_communities": len(set(labels))}


def point_estimate_core(j_sym, h, p_c, bridge, top_k=6):
    n = j_sym.shape[0]
    strength = np.abs(j_sym).sum(axis=1)
    percolation = (np.abs(j_sym) >= p_c).sum(axis=1) if p_c > 0 else np.zeros(n)
    field = np.abs(h)
    metrics = {"strength": strength, "bridge": bridge, "percolation": percolation, "field": field}
    top_sets = {name: top_k_with_ties(vals, top_k) for name, vals in metrics.items()}
    votes = np.zeros(n, dtype=int)
    for s in top_sets.values():
        for idx in s:
            votes[idx] += 1
    return votes


def analyze_structure(binary, label, ref_labels=None, ref_j=None):
    net = fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="pm1")
    j, h = net["j_sym"], net["intercept"]
    n = j.shape[0]
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if abs(j[u, v]) > AND_TOL]
    n_edges = len(pairs)
    n_neg = sum(1 for (u, v) in pairs if j[u, v] < 0)
    density = n_edges / (n * (n - 1) / 2)
    lv = percolation_communities(j, resolution=1.0)
    votes = point_estimate_core(j, h, lv["p_c"], lv["bridge"])
    core = [ITEM_NAMES[i] for i in range(n) if votes[i] >= 3]
    res = {"样本": label, "n": len(binary), "边数": n_edges, "密度": round(density, 4),
           "负向边": n_neg, "p_c": round(lv["p_c"], 4), "社区数": lv["n_communities"],
           "Q": round(lv["Q"], 4), "核心症状(重抽样≥50%)": "（重抽样中）",
           "点估计核心(≥3票)": "、".join(core) if core else "无",
           "得票详情": "; ".join(f"{ITEM_NAMES[i]}={int(votes[i])}"
                                  for i in np.argsort(-votes) if votes[i] > 0)}
    if ref_labels is not None:
        res["ARI vs 主分析"] = round(adjusted_rand_score(ref_labels, lv["labels"]), 4)
    if ref_j is not None:
        tri = np.triu_indices(20, k=1)
        res["r_J vs 主分析"] = round(float(np.corrcoef(ref_j[tri], j[tri])[0, 1]), 4)
    print(f"  [{label}] 边数={n_edges} 密度={density:.4f} p_c={lv['p_c']:.4f} "
          f"Q={lv['Q']:.4f} 社区数={lv['n_communities']} | 核心: {res['点估计核心(≥3票)']}")
    return res, lv["labels"], j


# ============================== 核心症状 Bootstrap（可选，带断点续跑） ==============================
def boot_one_iter(binary, iter_idx, main_labels):
    rng = np.random.default_rng(RANDOM_STATE * 1000003 + iter_idx)
    bs = binary[rng.integers(0, binary.shape[0], size=binary.shape[0])]
    with contextlib.redirect_stdout(io.StringIO()):
        net = fit_ising_network(bs, gamma=EBIC_GAMMA, encoding="pm1")
        lv = percolation_communities(net["j_sym"], resolution=1.0)
        votes = point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])
    core = {ITEM_NAMES[i] for i in range(len(votes)) if votes[i] >= 3}
    ari = float(adjusted_rand_score(main_labels, lv["labels"]))
    return {"core": core, "p_c": lv["p_c"], "Q": lv["Q"],
            "n_comm": lv["n_communities"], "ari": ari}


def _atomic_write_csv(df, path, retries=6, delay=0.8):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{int(time.time()*1000)}.tmp")
    for k in range(retries):
        try:
            df.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, path)
            return True
        except (PermissionError, OSError):
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(delay * (k + 1))
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return False


def run_core_bootstrap(data, cohorts, n_iter, n_jobs, force=False):
    """对指定结构队列取核心症状重抽样结果（每轮重抽样→重估网络→投票，入选率≥50% 定核心）。

    策略（兼顾用户本机已跑的快照）：
      - 快照存在且 done>=n_iter：直接读满轮结果填表；
      - 快照存在但 0<done<n_iter：默认用现有 done 轮结果填表（不强制续跑），
        仅当 force=True（显式 --bootstrap）时才从 done 续跑到 n_iter；
      - 无快照：必须重估 n_iter 轮（如 FDSC 发现队列）。
    返回 {label: set(核心症状)}，供结构 CSV 填“核心症状(重抽样≥50%)”列。
    快照文件名按 BOOT_FILE_TAG 映射（fdsc/cfps/midus），与已产出文件对齐。
    """
    _, main_bin = data["FDSC"][1], data["FDSC"][2]
    with contextlib.redirect_stdout(io.StringIO()):
        net0 = fit_ising_network(main_bin, gamma=EBIC_GAMMA, encoding="pm1")
        lv0 = percolation_communities(net0["j_sym"], resolution=1.0)
    main_labels = lv0["labels"]
    items = list(ITEM_NAMES)
    cores = {}

    for key in cohorts:
        label, _, binary, _ = data[key]
        tag = BOOT_FILE_TAG.get(key, key.lower())
        csv_path = OUT_DIR / f"cross_boot_{tag}_n{n_iter}.csv"
        counts = {it: 0 for it in items}
        done = 0
        if csv_path.exists():
            prev = pd.read_csv(csv_path)
            done = int(prev["done"].iloc[0])
            for _, row in prev.iterrows():
                if row["症状"] in counts:
                    counts[row["症状"]] = int(round(row["inclusion_pct"] / 100.0 * done))

        if done >= n_iter:
            core = {s for s in items if 100.0 * counts[s] / done >= INCLUSION_THRESHOLD}
            cores[label] = core
            print(f"[{label}] 重抽样已完成 (done={done})，核心症状(≥50%) = {sorted(core)}")
            continue

        if 0 < done < n_iter and not force:
            # 用户本机仍在跑的大样本：用现有 done 轮结果填表，不阻塞
            core = {s for s in items if 100.0 * counts[s] / done >= INCLUSION_THRESHOLD}
            cores[label] = core
            print(f"[{label}] 重抽样仅 {done}/{n_iter}（部分完成），核心症状(基于{done}轮, ≥50%) "
                  f"= {sorted(core)}；显式 --bootstrap 可从 {done} 续跑至 {n_iter}")
            continue

        # 无快照，或 force 续跑
        t0 = time.time()
        for it in range(done, n_iter):
            res = boot_one_iter(binary, it, main_labels)
            for it_ in res["core"]:
                counts[it_] += 1
            done = it + 1
            if done % 100 == 0 or done == n_iter:
                rows = sorted(((s, 100.0 * counts[s] / done) for s in items), key=lambda r: -r[1])
                _atomic_write_csv(pd.DataFrame(
                    [{"症状": s, "inclusion_pct": round(p, 1), "done": done} for s, p in rows]), csv_path)
                el = (time.time() - t0) / 60.0
                print(f"[{label}] {done}/{n_iter}  已用 {el:.1f} min")
        core = {s for s in items if 100.0 * counts[s] / max(done, 1) >= INCLUSION_THRESHOLD}
        cores[label] = core
        print(f"[{label}] 核心症状(≥50%) = {sorted(core)}")
    return cores


# ============================== 网络比较测试 (NCT) ==============================
# 参考文献：van Borkulo, C. D., et al. (2022). Comparing network structures on
# three aspects: A permutation test. Psychological Methods.
#
# 用途：证明 CFPS / MIDUS 网络与 FDSC 发现队列足够相似，作为 SNCI 跨样本
#       可比性的逻辑基础（否则审稿人会质疑独立估计网络的跨样本比较有效性）。
# 两个检验统计量（每对比较报告双侧置换 P 值）：
#   S  全局强度不变性：S = |GS1 - GS2|，GS = Σ_{i<j} |J_ij|（网络总连接强度）
#   M  网络结构不变性：M = max_{i<j} |J1_ij - J2_ij|（最大边权差，含一侧为零的边）
# 置换方案：合并两组受试者 → 随机重排 → 按原组样本量拆分 → 重估两个网络
#           → 重算 S/M；P 值 =（重排统计量 ≥ 观测值 的次数 + 1）/（置换次数 + 1）。
# 口径：与主分析一致（pm1 编码、EBIC γ=0.25、AND 对称化），保证可比性。


def _global_strength(j_sym):
    """网络全局强度：上三角 |J| 之和（= 所有节点强度之和 / 2）。"""
    triu = np.triu_indices(j_sym.shape[0], k=1)
    return float(np.abs(j_sym[triu]).sum())


def _max_edge_diff(j1, j2):
    """网络结构差异统计量 M = max_{i<j} |J1_ij - J2_ij|（含一侧为零的边）。"""
    triu = np.triu_indices(j1.shape[0], k=1)
    return float(np.abs(j1[triu] - j2[triu]).max())


_NCT_STATE = None


def _nct_init(bin1, bin2, n1, n2):
    global _NCT_STATE
    _NCT_STATE = (bin1, bin2, n1, n2)


def _nct_one_perm(iter_idx):
    """单次置换：合并→重排→按原组样本量拆分→重估两网络→返回 (S, M)。"""
    rng = np.random.default_rng(NCT_SEED + iter_idx)
    bin1, bin2, n1, n2 = _NCT_STATE
    combined = np.vstack([bin1, bin2])
    perm = rng.permutation(combined.shape[0])
    g1 = combined[perm[:n1]]
    g2 = combined[perm[n1:n1 + n2]]
    with contextlib.redirect_stdout(io.StringIO()):
        net1 = fit_ising_network(g1, gamma=EBIC_GAMMA, encoding="pm1")
        net2 = fit_ising_network(g2, gamma=EBIC_GAMMA, encoding="pm1")
    S = abs(_global_strength(net1["j_sym"]) - _global_strength(net2["j_sym"]))
    M = _max_edge_diff(net1["j_sym"], net2["j_sym"])
    return float(S), float(M)


def run_nct_pair(data, key1, key2, n_perm=NCT_PERMS, n_jobs=NCT_JOBS):
    """对两队列做 NCT：报告全局强度不变性(P_S)与网络结构不变性(P_M)。

    观测网络用 fit_ising_network 估计一次（确定），随后 n_perm 次置换并行重估；
    每 50 轮原子落盘快照，支持断点续跑（seed 固定，续跑结果一致）。
    """
    label1, _, bin1, n1 = data[key1]
    label2, _, bin2, n2 = data[key2]

    with contextlib.redirect_stdout(io.StringIO()):
        net1 = fit_ising_network(bin1, gamma=EBIC_GAMMA, encoding="pm1")
        net2 = fit_ising_network(bin2, gamma=EBIC_GAMMA, encoding="pm1")
    GS1 = _global_strength(net1["j_sym"])
    GS2 = _global_strength(net2["j_sym"])
    S_obs = abs(GS1 - GS2)
    M_obs = _max_edge_diff(net1["j_sym"], net2["j_sym"])

    t1 = BOOT_FILE_TAG.get(key1, key1.lower())
    t2 = BOOT_FILE_TAG.get(key2, key2.lower())
    csv_path = OUT_DIR / f"nct_{t1}_vs_{t2}_n{n_perm}.csv"

    S_ge = 0
    M_ge = 0
    done = 0
    if csv_path.exists():
        try:
            prev = pd.read_csv(csv_path)
            done = int(prev["done"].iloc[0])
            S_ge = int(prev["S_ge"].iloc[0])
            M_ge = int(prev["M_ge"].iloc[0])
        except Exception:
            done = S_ge = M_ge = 0

    def snapshot():
        _atomic_write_csv(pd.DataFrame([{
            "done": done, "S_obs": round(S_obs, 6), "M_obs": round(M_obs, 6),
            "GS1": round(GS1, 6), "GS2": round(GS2, 6),
            "S_ge": S_ge, "M_ge": M_ge}]), csv_path)

    if done >= n_perm:
        print(f"[NCT] {label1} vs {label2}: 已完成 done={done}/{n_perm}")
    else:
        t0 = time.time()
        print(f"[NCT] {label1} vs {label2}: 置换 {done}→{n_perm} 次, "
              f"{n_jobs} 进程（长时，支持断点续跑）")
        with mp.Pool(n_jobs, initializer=_nct_init, initargs=(bin1, bin2, n1, n2)) as pool:
            for S, M in pool.imap_unordered(_nct_one_perm, range(done, n_perm)):
                if S >= S_obs - 1e-12:
                    S_ge += 1
                if M >= M_obs - 1e-12:
                    M_ge += 1
                done += 1
                if done % 50 == 0 or done == n_perm:
                    snapshot()
                    print(f"[NCT] {done}/{n_perm}  已用 {(time.time() - t0) / 60.0:.1f} min")

    p_S = (S_ge + 1) / (done + 1)
    p_M = (M_ge + 1) / (done + 1)
    print(f"[NCT] {label1} vs {label2}: GS1={GS1:.3f} GS2={GS2:.3f} "
          f"S_obs={S_obs:.3f} P_全局强度={p_S:.4f}; M_obs={M_obs:.3f} "
          f"P_结构={p_M:.4f} (Nperm={done})")
    return {"比较对": f"{label1} vs {label2}", "N1": n1, "N2": n2,
            "GS1": round(GS1, 4), "GS2": round(GS2, 4),
            "S_obs(全局强度差)": round(S_obs, 4),
            "P_全局强度不变性": round(p_S, 4),
            "M_obs(最大边权差)": round(M_obs, 4),
            "P_网络结构不变性": round(p_M, 4),
            "置换次数": done}


# ============================== 主流程 ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discrim-cohorts", nargs="*", default=None,
                    help="判别效度要计算的队列键，默认 DISCRIM_COHORTS")
    ap.add_argument("--structure-cohorts", nargs="*", default=None,
                    help="网络结构要计算的队列键，默认 STRUCTURE_COHORTS")
    ap.add_argument("--no-structure", action="store_true")
    ap.add_argument("--no-discrim", action="store_true", help="跳过判别效度段（仅验证网络结构/重抽样）")
    ap.add_argument("--bootstrap", action="store_true",
                    help="核心症状重抽样默认已开启；此开关用于把部分完成的快照续跑到目标轮数")
    ap.add_argument("--boot-iters", type=int, default=N_BOOT_CORE)
    ap.add_argument("--boot-jobs", type=int, default=N_JOBS_CORE)
    ap.add_argument("--boot-cohorts", nargs="*", default=None)
    ap.add_argument("--no-nct", action="store_true", help="跳过网络比较测试(NCT)")
    ap.add_argument("--nct-iters", type=int, default=NCT_PERMS,
                    help=f"NCT 置换次数（默认 {NCT_PERMS}）")
    ap.add_argument("--nct-jobs", type=int, default=NCT_JOBS,
                    help=f"NCT 并行进程数（默认 {NCT_JOBS}）")
    args = ap.parse_args()

    data = load_all()
    dkeys = args.discrim_cohorts or DISCRIM_COHORTS
    skeys = args.structure_cohorts or STRUCTURE_COHORTS

    # ---- (1) 判别效度 ----
    drows = []
    if not args.no_discrim:
        print("=" * 80 + "\n[1] 判别效度（SNCI 对高症状组的区分能力）\n" + "=" * 80)
        for key in dkeys:
            if key not in data:
                print(f"[跳过] 未知队列键 {key}")
                continue
            label, scored, binary, n = data[key]
            print(f"\n===== {label} (N={n}) =====")
            drows.append(analyze_discriminative(label, scored, binary))
        if drows:
            dout = pd.DataFrame(drows)
            dout.to_csv(OUT_DIR / "cross_cohort_discriminant_results.csv", index=False, encoding="utf-8-sig")
            print(f"\n[导出] {OUT_DIR / 'cross_cohort_discriminant_results.csv'}")
    else:
        print("[判别效度] 已跳过（--no-discrim）")

    # ---- (2) 网络结构 ----
    if not args.no_structure:
        print("\n" + "=" * 80 + "\n[2] 网络结构（跨样本稳健性）\n" + "=" * 80)
        srows = []
        ref_labels = None
        ref_j = None
        for key in skeys:
            if key not in data:
                print(f"[跳过] 未知队列键 {key}")
                continue
            label, _, binary, n = data[key]
            row, labels, j = analyze_structure(binary, label, ref_labels, ref_j)
            srows.append(row)
            if key == "FDSC":
                ref_labels, ref_j = labels, j

        # ---- (3) 核心症状重抽样（默认开启），回填结构 CSV 的“核心症状(重抽样≥50%)”列 ----
        bcohorts = args.boot_cohorts or BOOT_COHORTS
        cores = run_core_bootstrap(data, bcohorts, args.boot_iters, args.boot_jobs, force=args.bootstrap)
        for row in srows:
            lbl = row["样本"]
            if lbl in cores:
                cs = cores[lbl]
                row["核心症状(重抽样≥50%)"] = "、".join(sorted(cs)) if cs else "无"

        scols = ["样本", "n", "边数", "密度", "负向边", "p_c", "社区数", "Q",
                 "核心症状(重抽样≥50%)", "点估计核心(≥3票)",
                 "ARI vs 主分析", "r_J vs 主分析"]
        if srows:
            stbl = pd.DataFrame(srows)[scols]
            stbl.to_csv(OUT_DIR / "cross_cohort_structure_results.csv", index=False, encoding="utf-8-sig")
            print(f"\n[导出] {OUT_DIR / 'cross_cohort_structure_results.csv'}")
            print(stbl.to_string(index=False))
        else:
            print("[网络结构] 无有效队列，跳过导出")

    # ---- (3) 网络比较测试 (NCT) ----
    if not args.no_nct:
        print("\n" + "=" * 80 + "\n[3] 网络比较测试（NCT，van Borkulo 2022）\n" + "=" * 80)
        nct_rows = []
        for k1, k2 in NCT_PAIRS:
            if k1 in data and k2 in data:
                nct_rows.append(run_nct_pair(data, k1, k2, args.nct_iters, args.nct_jobs))
            else:
                print(f"[NCT] 跳过未知队列对 {k1} vs {k2}")
        if nct_rows:
            nout = pd.DataFrame(nct_rows)
            nout.to_csv(OUT_DIR / "cross_cohort_nct_results.csv", index=False, encoding="utf-8-sig")
            print(f"\n[导出] {OUT_DIR / 'cross_cohort_nct_results.csv'}")
            print(nout.to_string(index=False))

    print("\n===== 跨队列验证分析完成 =====")


if __name__ == "__main__":
    main()
