# -*- coding: utf-8 -*-
"""三队列 Ising 网络「自身一致性」分析（回应劲松老师「网络本身对比」）。
复用主分析同款 fit_ising_network / percolation_communities / point_estimate_core / ITEM_NAMES。
输出：共识骨架网络图 + 共识边明细 + 综合一致性总表。
"""
import sys, time
import numpy as np
import pandas as pd

# --- 路径配置（config.py 统一提供，可用环境变量覆盖）----------------------
import sys as _sys
from pathlib import Path as _Path
_FIGDIR = _Path(__file__).resolve().parent
_SRC = _FIGDIR.parent
_REPO = _SRC.parent
for _p in (str(_SRC), str(_FIGDIR), str(_REPO)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from cross_cohort_analysis import (load_all, fit_ising_network, percolation_communities,  # noqa: E402
                                   point_estimate_core, ITEM_NAMES, EBIC_GAMMA, AND_TOL, OUT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

KEYS = ["FDSC", "CFPS_all", "MIDUS"]
P = 20
tri = [(u, v) for u in range(P) for v in range(u + 1, P)]

# ---------- 载入/估计三队列观测网络（优先缓存）----------
cache = OUT_DIR / "net_compare_j.npz"
J, H = {}, {}
try:
    arr = np.load(cache)
    for k in KEYS:
        J[k] = arr[f"j_{k}"]; H[k] = arr[f"h_{k}"]
    print("[缓存] 命中 net_compare_j.npz", flush=True)
except Exception as e:
    print("[缓存] 未命中，重新估计:", e, flush=True)
    data = load_all()
    for key in KEYS:
        lbl, sc, bn, n = data[key]
        t0 = time.time()
        net = fit_ising_network(bn, gamma=EBIC_GAMMA, encoding="pm1")
        J[key], H[key] = net["j_sym"], net["intercept"]
        print(f"[估计] {key} N={n} 边数={net['n_edges']} {time.time()-t0:.1f}s", flush=True)
    np.savez(cache, **{f"j_{k}": J[k] for k in KEYS}, **{f"h_{k}": H[k] for k in KEYS})

# ---------- 主分析指标（四中心性 + 投票核心 + 社区）----------
MAIN = {}
for k in KEYS:
    j, h = J[k], H[k]
    lv = percolation_communities(j, resolution=1.0)
    votes = point_estimate_core(j, h, lv["p_c"], lv["bridge"])
    MAIN[k] = dict(
        strength=np.abs(j).sum(axis=1),
        bridge=lv["bridge"],
        percolation=(np.abs(j) >= lv["p_c"]).sum(axis=1) if lv["p_c"] > 0 else np.zeros(P),
        field=h.copy(),
        votes=votes,
        core=set(ITEM_NAMES[i] for i in range(P) if votes[i] >= 3),
        p_c=lv["p_c"], Q=lv["Q"], labels=lv["labels"], n_comm=len(set(lv["labels"])),
    )
    print(f"[主分析指标] {k}: p_c={lv['p_c']:.3f} Q={lv['Q']:.3f} 社区数={len(set(lv['labels']))} "
          f"核心(≥3票)={sorted(MAIN[k]['core'])}", flush=True)

# ---------- 三队列共有边（共识骨架）----------
presence = {k: (np.abs(J[k]) > AND_TOL) for k in KEYS}
consensus = [uv for uv in tri if all(presence[k][uv[0], uv[1]] for k in KEYS)]
print(f"\n[共识边] 三队列共有边 = {len(consensus)} / {len(tri)}", flush=True)

# 共识边明细
rows = []
for u, v in consensus:
    jv = {k: float(J[k][u, v]) for k in KEYS}
    signs = {k: (1 if jv[k] > 0 else -1) for k in KEYS}
    same_sign = len(set(signs.values())) == 1
    rows.append(dict(
        症状u=ITEM_NAMES[u], 症状v=ITEM_NAMES[v],
        J_FDSC=round(jv["FDSC"], 3), J_CFPS=round(jv["CFPS_all"], 3), J_MIDUS=round(jv["MIDUS"], 3),
        absJ_mean=round(np.mean([abs(jv[k]) for k in KEYS]), 3),
        方向一致="是" if same_sign else "否",
    ))
cons_df = pd.DataFrame(rows).sort_values("absJ_mean", ascending=False)
cons_df.to_csv(OUT_DIR / "net_compare_consensus.csv", index=False, encoding="utf-8-sig")

# ---------- 权重一致性（共识边内 |J| 两两相关）----------
absJ = {k: np.array([abs(J[k][u, v]) for u, v in consensus]) for k in KEYS}
rho_list = []
for a, b in [("FDSC", "CFPS_all"), ("FDSC", "MIDUS"), ("CFPS_all", "MIDUS")]:
    r, _ = spearmanr(absJ[a], absJ[b])
    rho_list.append(r)
mean_rho = float(np.mean(rho_list))
dir_consistent = cons_df["方向一致"].eq("是").mean()

# ---------- 两两边重叠 Jaccard（重算，自洽）----------
jac = {}
for a, b in [("FDSC", "CFPS_all"), ("FDSC", "MIDUS"), ("CFPS_all", "MIDUS")]:
    m1, m2 = presence[a], presence[b]
    inter = float((m1 & m2).sum() // 2)
    uni = float((m1 | m2).sum() // 2)
    jac[f"{a} vs {b}"] = inter / uni if uni else np.nan

# 四中心性相关均值（两两）
cent_r = {cn: [] for cn in ["强度", "桥接", "渗流", "外部场"]}
for a, b in [("FDSC", "CFPS_all"), ("FDSC", "MIDUS"), ("CFPS_all", "MIDUS")]:
    for cn, key in [("强度", "strength"), ("桥接", "bridge"), ("渗流", "percolation"), ("外部场", "field")]:
        cent_r[cn].append(float(np.corrcoef(MAIN[a][key], MAIN[b][key])[0, 1]))
cent_mean = {cn: float(np.mean(v)) for cn, v in cent_r.items()}

# 社区 ARI（以 FDSC 为参照）
from sklearn.metrics import adjusted_rand_score
ari = {b: float(adjusted_rand_score(MAIN["FDSC"]["labels"], MAIN[b]["labels"])) for b in ["CFPS_all", "MIDUS"]}

# 三队列共同稳健核心（读 overlap csv）
ov = pd.read_csv(OUT_DIR / "cross_cohort_core_overlap.csv")
common_core = None
for _, r in ov.iterrows():
    if "三队列" in str(r.get("比较", "")) or str(r.get("比较对", "")).find("三队列") >= 0:
        common_core = r.get("共同核心")
print("三队列共同稳健核心:", common_core, flush=True)

# ---------- 综合一致性总表 ----------
overview = pd.DataFrame([
    dict(维度="共识边（三队列共有）", 值=f"{len(consensus)}/{len(tri)} ({len(consensus)/len(tri)*100:.0f}%)"),
    dict(维度="边权方向一致比例（共识边内）", 值=f"{dir_consistent*100:.0f}%"),
    dict(维度="边权|J|相关（共识边内两两Spearman均值）", 值=round(mean_rho, 3)),
    dict(维度="边重叠Jaccard (FDSC vs CFPS)", 值=round(jac["FDSC vs CFPS_all"], 3)),
    dict(维度="边重叠Jaccard (FDSC vs MIDUS)", 值=round(jac["FDSC vs MIDUS"], 3)),
    dict(维度="边重叠Jaccard (CFPS vs MIDUS)", 值=round(jac["CFPS_all vs MIDUS"], 3)),
    dict(维度="强度中心性相关均值", 值=round(cent_mean["强度"], 3)),
    dict(维度="桥接中心性相关均值", 值=round(cent_mean["桥接"], 3)),
    dict(维度="渗流中心性相关均值", 值=round(cent_mean["渗流"], 3)),
    dict(维度="外部场相关均值", 值=round(cent_mean["外部场"], 3)),
    dict(维度="社区结构ARI (FDSC vs CFPS)", 值=round(ari["CFPS_all"], 3)),
    dict(维度="社区结构ARI (FDSC vs MIDUS)", 值=round(ari["MIDUS"], 3)),
    dict(维度="三队列共同稳健核心症状", 值=common_core if common_core else "见 overlap.csv"),
])
overview.to_csv(OUT_DIR / "net_compare_consistency_overview.csv", index=False, encoding="utf-8-sig")
print("\n===== 网络一致性总览 =====")
print(overview.to_string(index=False))

# ---------- 共识骨架网络图 ----------
# 环形布局：节点按 FDSC Louvain 社区分组排列（同社区相邻、社区内按强度降序）
order = []
for c in sorted(set(MAIN["FDSC"]["labels"])):
    members = [i for i in range(P) if MAIN["FDSC"]["labels"][i] == c]
    members.sort(key=lambda i: -MAIN["FDSC"]["strength"][i])
    order.extend(members)
n_all = len(order)
pos = {i: (float(np.cos(2 * np.pi * idx / n_all)), float(np.sin(2 * np.pi * idx / n_all)))
       for idx, i in enumerate(order)}
mean_strength = np.mean([MAIN[k]["strength"] for k in KEYS], axis=0)

G = nx.Graph()
G.add_nodes_from(range(P))
for u, v in consensus:
    signs = [1 if J[k][u, v] > 0 else -1 for k in KEYS]
    same_sign = len(set(signs)) == 1
    G.add_edge(u, v, w=np.mean([abs(J[k][u, v]) for k in KEYS]), same=same_sign)

fig, ax = plt.subplots(figsize=(9, 9))
maxw = max(d["w"] for _, _, d in G.edges(data=True))
for u, v, d in G.edges(data=True):
    col = "#c0392b" if not d["same"] else "#2c6fbb"
    ls = "--" if not d["same"] else "-"
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=col,
            lw=0.6 + 5.5 * d["w"] / maxw, ls=ls, alpha=0.8, zorder=1,
            solid_capstyle="round")
nsize = 260 + 1100 * (mean_strength - mean_strength.min()) / (mean_strength.max() - mean_strength.min())
for i in range(P):
    x, y = pos[i]
    ax.scatter([x], [y], s=nsize[i], c="#e8eef5", edgecolors="#34495e", linewidths=1.4, zorder=2)
    lx, ly = x * 1.17, y * 1.17
    ha = "center"
    if x > 0.3:
        ha = "left"
    elif x < -0.3:
        ha = "right"
    ax.text(lx, ly, ITEM_NAMES[i], ha=ha, va="center", fontsize=9, zorder=3)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "net_compare_consensus.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n[导出] {OUT_DIR / 'net_compare_consensus.png'}")
print(f"[导出] {OUT_DIR / 'net_compare_consensus.csv'}")
print(f"[导出] {OUT_DIR / 'net_compare_consistency_overview.csv'}")
