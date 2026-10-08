# -*- coding: utf-8 -*-
"""
敏感性分析汇总脚本（单文件，无绘图）
================================================================
按第一个位置参数选择任务：

  scenarios    5 个情景（编码 01、γ=0、γ=0.5、Louvain 分辨率 1.5 与 2.0）
               各跑 n 次病例重抽样，记录各症状获选候选症状（≥3 票）的比例
  cutoff       GCC 崩解判据 20%~75% 八档 × n 次（每轮只重估一次网络）
  threshold    主判据（GCC=50%）下三项阈值交叉：排序范围 k=4/6/8 ×
               得票门槛 ≥2/3/4 × 入选率门槛 50/60/70%
  topk         k=4/6/8 × GCC 八档（备用交叉网格，稿件未引用）
  table        汇总主分析与各情景入选率，输出补充材料表与摘要行
  point        点估计敏感性表（方法段 4 维度：编码/γ/渗流操作化/分辨率），
               输出 sens_results.csv
  percolation  p_c 复现：网格口径（与主分析一致）+ 精确秩移除口径 + τ_c(S2)
  all          依次执行 point → scenarios → cutoff → threshold → table → percolation

用法
----
    python sensitivity_analysis.py scenarios 5000 6
    python sensitivity_analysis.py cutoff 5000 6
    python sensitivity_analysis.py threshold 5000 6
    python sensitivity_analysis.py topk 5000 6
    python sensitivity_analysis.py table 5000
    python sensitivity_analysis.py point          # 点估计敏感性表（不重抽样）
    python sensitivity_analysis.py percolation
    python sensitivity_analysis.py all 5000 6
    python sensitivity_analysis.py scenarios 20 4 enc01 res20   # 指定情景

第 2、3 个位置参数分别为 iterations、n_jobs；scenarios 之后可跟情景键。
所有重抽样任务都带断点续跑：结果 CSV 含 done 列，重跑同一命令即从断点继续。
所有任务与主分析共用同一种子体系（rng = default_rng(RNG_BASE + iter_idx)），
故各情景、各判据的第 k 次迭代使用同一批重抽样样本，可逐列配对比较。

依赖：numpy / pandas（逐轮计算复用 analysis_all.py）
"""

import sys
import io
import time
import contextlib
import importlib.util
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# ============================================================
# 0. 配置（换机器只改这里）
# ============================================================
MAIN_PATH = Path(__file__).with_name("analysis_all.py")      # 主分析脚本（同目录）
# --- 路径配置（config.py 统一提供，可用环境变量覆盖）----------------------
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parent
_REPO = _SRC.parent
for _p in (str(_SRC), str(_REPO)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from pathlib import Path  # noqa: E402
from config import (DATA_DIR as CONFIG_DATA_DIR, OUT_DIR as CONFIG_OUT_DIR,  # noqa: E402
                    PERC_OUT as CONFIG_PERC_OUT)
# ---------------------------------------------------------------------------
DATA_PATH = Path(CONFIG_DATA_DIR) / "3447例-CESD数据已反向计分(3).xlsx"
OUT_DIR = Path(CONFIG_OUT_DIR)   # 结果目录（与主分析、跨队列共用）

# p_c 复现所需的两个输入
J_XLSX = str(Path(CONFIG_DATA_DIR) / "J_correct.xlsx")
NPZ = str(Path(CONFIG_OUT_DIR) / "ising_params_3447_pm1.npz")
PERC_OUT = Path(CONFIG_PERC_OUT)
N_GRID = 60                                                   # 网格口径的阈值点数

ENCODING = "pm1"
EBIC_GAMMA = 0.25                   # 主分析口径，勿改
PERC_TOL = 1e-4                     # 边保留阈值，与主分析 AND_TOL 一致
RNG_BASE = 20240825                 # 与主分析一致；勿改，否则无法与既有结果配对
INCLUSION_THRESHOLD = 50.0          # 入选率 ≥ 50% 判为核心症状
SAVE_EVERY = 50                     # 每多少轮落盘快照

CUTOFFS = [0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75]    # GCC 目标占比
K_LIST = [4, 6, 8]                  # 20 节点 → 前 20% / 30% / 40%
VOTE_THRS = [2, 3, 4]               # 得票门槛；3 = 主分析
INCLUSIONS = [50.0, 60.0, 70.0]     # 入选率门槛(%)；50 = 主分析

# 情景名 → (维度标签, 参数)。主分析口径 = γ0.25 / pm1 / res1.0，不在此列。
SCENARIOS = {
    "enc01":   ("1.编码",   dict(gamma=0.25, encoding="01",  resolution=1.0)),
    "gamma0":  ("2.正则化γ", dict(gamma=0.0,  encoding="pm1", resolution=1.0)),
    "gamma05": ("2.正则化γ", dict(gamma=0.5,  encoding="pm1", resolution=1.0)),
    "res15":   ("4.分辨率",  dict(gamma=0.25, encoding="pm1", resolution=1.5)),
    "res20":   ("4.分辨率",  dict(gamma=0.25, encoding="pm1", resolution=2.0)),
}
SCEN_ORDER = ["enc01", "gamma0", "gamma05", "res15", "res20"]

# table 任务用：情景键 → (维度, 参数描述)
SCEN_TABLE = [
    ("main",    "主分析",    "γ=0.25，±1 编码，分辨率 1.0"),
    ("enc01",   "编码方式",  "0/1 编码"),
    ("gamma0",  "正则化强度", "EBIC γ=0"),
    ("gamma05", "正则化强度", "EBIC γ=0.5"),
    ("res15",   "社区分辨率", "Louvain 分辨率 1.5"),
    ("res20",   "社区分辨率", "Louvain 分辨率 2.0"),
]

spec = importlib.util.spec_from_file_location("analysis_all", MAIN_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ITEM_NAMES = list(m.ITEM_NAMES)
NODE = len(ITEM_NAMES)


# ============================================================
# 1. 公共：数据载入与快照
# ============================================================
_BIN = None


def _init_worker(binary):
    """Windows 为 spawn 模式：数据经 initializer 注入子进程，避免逐任务 pickle。"""
    global _BIN
    _BIN = binary


def load_binary():
    _, binary = m.load_cesd_binary(DATA_PATH if DATA_PATH.exists() else m.DATA_PATH)
    return binary


def load_scored():
    """返回 (scored_df, binary)：point 任务需要 CES-D 总分。"""
    return m.load_cesd_binary(DATA_PATH if DATA_PATH.exists() else m.DATA_PATH)


def snapshot_rows(rows, out_csv, encoding="utf-8-sig"):
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding=encoding)


def main_core_from_csv():
    """主分析核心症状（用于 CSV 的 is_main_core 标记）。

    固定以主分析的 5000 轮结果为准（与合并前一致）；缺失时退回点估计。
    """
    f = OUT_DIR / "bootstrap_pm1_n5000.csv"
    if f.exists():
        d = pd.read_csv(f)
        core = set(d.loc[d["inclusion_pct"] >= INCLUSION_THRESHOLD, "症状"])
        if core:
            return core
    with contextlib.redirect_stdout(io.StringIO()):
        net = m.fit_ising_network(load_binary()[1])
        lv = m.percolation_communities(net["j_sym"])
        pl = m.point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])
    return set(pl["pillars"])


# ============================================================
# 2. 任务：scenarios —— 5 个情景的病例重抽样
# ============================================================
def _scen_one(args):
    """单轮：按情景参数重抽样 → 重估网络 → 渗流/Louvain → 投票 → 本轮核心症状。"""
    iter_idx, key = args
    cfg = SCENARIOS[key][1]
    n = _BIN.shape[0]
    rng = np.random.default_rng(RNG_BASE + iter_idx)
    bs = _BIN[rng.integers(0, n, size=n)]
    with contextlib.redirect_stdout(io.StringIO()):
        net = m.fit_ising_network(bs, gamma=cfg["gamma"], encoding=cfg["encoding"])
        lv = m.percolation_communities(net["j_sym"], resolution=cfg["resolution"])
        pl = m.point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])
    return set(pl["pillars"])


def task_scenarios(n_iter, n_jobs, keys=None):
    keys = keys or SCEN_ORDER
    for k in keys:
        if k not in SCENARIOS:
            raise SystemExit(f"未知情景 {k}，可选：{SCEN_ORDER}")

    binary = load_binary()
    main_core = main_core_from_csv()
    print(f"[sens-boot] N={len(binary)}, P={binary.shape[1]}, "
          f"iterations={n_iter}, n_jobs={n_jobs}, 情景={keys}")
    print(f"[sens-boot] 主分析核心症状: {sorted(main_core)}")

    summary = {}
    for key in keys:
        label, cfg = SCENARIOS[key]
        out_csv = OUT_DIR / f"sens_bootstrap_{key}_n{n_iter}.csv"
        counts = {it: 0 for it in ITEM_NAMES}
        done = 0

        def save_snapshot(cur_done=0):
            rows = sorted(((it, 100.0 * counts[it] / max(cur_done, 1)) for it in ITEM_NAMES),
                          key=lambda r: -r[1])
            snapshot_rows([{"症状": it, "inclusion_pct": round(p, 1),
                            "is_main_core": it in main_core, "done": cur_done}
                           for it, p in rows], out_csv, encoding="utf-8")

        if out_csv.exists():                                   # 断点续跑
            try:
                prev = pd.read_csv(out_csv)
                prev_done = int(prev["done"].iloc[0])
                if prev_done >= n_iter:
                    print(f"[{key}] 已完成 (done={prev_done})，跳过")
                    summary[key] = set(
                        prev.loc[prev["inclusion_pct"] >= INCLUSION_THRESHOLD, "症状"])
                    continue
                for _, row in prev.iterrows():
                    if row["症状"] in counts:
                        counts[row["症状"]] = int(
                            round(row["inclusion_pct"] / 100.0 * prev_done))
                done = prev_done
                print(f"[{key}] 断点续跑，从 {done + 1}/{n_iter} 继续")
            except Exception as e:
                print(f"[{key}] 残档读取失败（{e}），从头开始")

        t0 = time.time()
        with mp.Pool(n_jobs, initializer=_init_worker, initargs=(binary,)) as pool:
            for core_set in pool.imap_unordered(
                    _scen_one, [(k, key) for k in range(done, n_iter)], chunksize=10):
                for it in core_set:
                    counts[it] += 1
                done += 1
                if done % SAVE_EVERY == 0:
                    save_snapshot(done)
                    el = time.time() - t0
                    eta = el / max(done, 1) * (n_iter - done) / 60.0
                    print(f"[{key}] {done}/{n_iter}  已用 {el/60:.1f} min，"
                          f"预计剩余 {eta:.1f} min")

        save_snapshot(done)
        core = {it for it in ITEM_NAMES
                if 100.0 * counts[it] / max(done, 1) >= INCLUSION_THRESHOLD}
        summary[key] = core
        print(f"[{key}] 完成 {done}/{n_iter} -> 核心症状 {sorted(core)} -> {out_csv}")

    print("\n===== 敏感性 bootstrap 汇总（入选率 ≥ 50%） =====")
    print(f"{'情景':<10} {'维度':<10} {'参数':<34} 核心症状")
    for key in keys:
        label, cfg = SCENARIOS[key]
        param = f"γ={cfg['gamma']}, {cfg['encoding']}, res={cfg['resolution']}"
        print(f"{key:<10} {label:<10} {param:<34} {sorted(summary[key])}")
    print("===== SENS BOOTSTRAP DONE =====")


# ============================================================
# 3. 任务：cutoff —— GCC 崩解判据敏感性
# ============================================================
def _cutoff_one(args):
    """单轮：重估网络一次，对每个 GCC 判据各算一次核心症状集合。"""
    k, binary, cutoffs = args
    rng = np.random.default_rng(RNG_BASE + k)
    n = binary.shape[0]
    bs = binary[rng.integers(0, n, size=n)]
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        net = m.fit_ising_network(bs, encoding=ENCODING)
        j, h = net["j_sym"], net["intercept"]
        for c in cutoffs:
            lv = m.percolation_communities(j, gcc_frac=c)
            pl = m.point_estimate_core(j, h, lv["p_c"], lv["bridge"])
            out[round(c, 4)] = set(pl["pillars"])
    return out


def _cutoff_pivot(df):
    pv = df.pivot(index="symptom", columns="cutoff", values="inclusion_pct")
    return pv.sort_values(by=0.5, ascending=False).to_string()


def task_cutoff(n_iter, n_jobs):
    out_csv = OUT_DIR / f"boot_cutoff_sensitivity_n{n_iter}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary = load_binary()
    print(f"[boot×cutoff] N={len(binary)}  P={binary.shape[1]}  iters={n_iter}  jobs={n_jobs}")
    print(f"[boot×cutoff] cutoffs = {CUTOFFS}")

    counts = {(round(c, 4), it): 0 for c in CUTOFFS for it in ITEM_NAMES}
    done = 0

    if out_csv.exists():                                       # 断点续跑
        try:
            prev = pd.read_csv(out_csv)
            prev_done = int(prev["done"].iloc[0])
            if prev_done >= n_iter:
                print(f"[boot×cutoff] 已完成 (done={prev_done})，直接读取 -> {out_csv}")
                print(_cutoff_pivot(prev))
                print("[boot×cutoff] FINISHED_MARKER")
                return
            if prev_done > 0:
                print(f"[boot×cutoff] 残档 done={prev_done}/{n_iter}，续跑")
                for _, r in prev.iterrows():
                    key = (round(float(r["cutoff"]), 4), r["symptom"])
                    if key in counts:
                        counts[key] = int(r["n_core"])
                done = prev_done
        except Exception as e:
            print(f"[boot×cutoff] 读取残档失败（{e}），从头开始")

    def snap(cur_done):
        rows = []
        for c in CUTOFFS:
            c = round(c, 4)
            for it in ITEM_NAMES:
                rows.append({"cutoff": c, "symptom": it,
                             "inclusion_pct": round(100.0 * counts[(c, it)] / max(cur_done, 1), 1),
                             "n_core": counts[(c, it)], "done": cur_done})
        snapshot_rows(rows, out_csv)

    with mp.Pool(n_jobs) as pool:
        for core_map in pool.imap_unordered(
                _cutoff_one, [(k, binary, CUTOFFS) for k in range(done, n_iter)]):
            for c, syms in core_map.items():
                for it in syms:
                    counts[(round(c, 4), it)] += 1
            done += 1
            if done % SAVE_EVERY == 0:
                snap(done)
                print(f"[boot×cutoff] {done}/{n_iter}")

    snap(done)
    print(_cutoff_pivot(pd.read_csv(out_csv)))
    print(f"\n[boot×cutoff] 完成 {done}/{n_iter} -> {out_csv}")
    print("[boot×cutoff] FINISHED_MARKER")


# ============================================================
# 4. 任务：threshold —— 主判据下三项阈值的一致性
# ============================================================
def _threshold_one(args):
    """单轮：重估网络一次，返回每个 (GCC 判据, k) 组合下各节点的得票向量（长度 20，值 0–4）。"""
    it, binary, cutoffs, klist = args
    rng = np.random.default_rng(RNG_BASE + it)
    n = binary.shape[0]
    bs = binary[rng.integers(0, n, size=n)]
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        net = m.fit_ising_network(bs, encoding=ENCODING)
        j, h = net["j_sym"], net["intercept"]
        for c in cutoffs:
            lv = m.percolation_communities(j, gcc_frac=c)
            for K in klist:
                pl = m.point_estimate_core(j, h, lv["p_c"], lv["bridge"], top_k=K)
                out[(round(c, 4), K)] = np.asarray(pl["votes"], dtype=np.int8)
    return out


def _hist_from_snapshot(prev, cells):
    """从快照恢复各节点得票直方图（只需 2/3/4 三档，0/1 不影响任何门槛）。"""
    hist = {cell + (i,): np.zeros(5) for cell in cells for i in range(NODE)}
    bykey = {}
    for _, r in prev.iterrows():
        bykey[(round(float(r["cutoff"]), 4), int(r["k"]), int(r["vote_thr"]), r["symptom"])] = int(r["n_core"])
    for (c, K) in cells:
        for i, s in enumerate(ITEM_NAMES):
            n2 = bykey.get((c, K, 2, s), 0)
            n3 = bykey.get((c, K, 3, s), 0)
            n4 = bykey.get((c, K, 4, s), 0)
            hh = hist[(c, K, i)]
            hh[4] = n4
            hh[3] = max(n3 - n4, 0)
            hh[2] = max(n2 - n3, 0)
    return hist


def task_threshold(n_iter, n_jobs, cutoffs=None, tag="maincut"):
    """主判据（GCC = 50%）下三项阈值交叉；tag=maincut 对应稿件表 S6 的口径。"""
    cutoffs = cutoffs or [0.50]
    cells = [(round(c, 4), K) for c in cutoffs for K in K_LIST]
    out_csv = OUT_DIR / (f"boot_threshold_consistency_{tag}_n{n_iter}.csv"
                         if tag == "maincut" else f"boot_threshold_consistency_n{n_iter}.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary = load_binary()
    print(f"[thr] N={len(binary)}  P={binary.shape[1]}  iters={n_iter}  jobs={n_jobs}")
    print(f"[thr] cutoffs={cutoffs}")
    print(f"[thr] k={K_LIST} (前 20%/30%/40%)  vote_thr={VOTE_THRS}  inclusion={INCLUSIONS}")

    hist = {cell + (i,): np.zeros(5) for cell in cells for i in range(NODE)}
    done = 0

    if out_csv.exists():                                       # 断点续跑
        try:
            prev = pd.read_csv(out_csv)
            prev_done = int(prev["done"].iloc[0])
            if prev_done >= n_iter:
                print(f"[thr] 已完成 (done={prev_done}) -> {out_csv}")
                print("[thr] FINISHED_MARKER")
                return
            if prev_done > 0:
                print(f"[thr] 残档 done={prev_done}/{n_iter}，续跑")
                hist = _hist_from_snapshot(prev, cells)
                done = prev_done
        except Exception as e:
            print(f"[thr] 读取残档失败（{e}），从头开始")

    def snap(cur_done):
        rows = []
        for (c, K) in cells:
            for vt in VOTE_THRS:
                for i, s in enumerate(ITEM_NAMES):
                    ncore = int(hist[(c, K, i)][vt:].sum())
                    rows.append({"k": K, "cutoff": c, "vote_thr": vt, "symptom": s,
                                 "inclusion_pct": round(100.0 * ncore / max(cur_done, 1), 1),
                                 "n_core": ncore, "done": cur_done})
        snapshot_rows(rows, out_csv)

    with mp.Pool(n_jobs) as pool:
        for vote_map in pool.imap_unordered(
                _threshold_one, [(i, binary, cutoffs, K_LIST) for i in range(done, n_iter)]):
            for (c, K), v in vote_map.items():
                for i in range(NODE):
                    hist[(round(c, 4), K, i)][int(v[i])] += 1
            done += 1
            if done % SAVE_EVERY == 0:
                snap(done)
                print(f"[thr] {done}/{n_iter}")

    snap(done)
    print(f"[thr] 完成 {done}/{n_iter} -> {out_csv}")
    print("[thr] FINISHED_MARKER")


# ============================================================
# 5. 任务：topk —— k × GCC 交叉网格（备用）
# ============================================================
def _topk_one(args):
    """单轮：重估网络一次，对 (GCC 判据 × k) 每个组合各算一次核心症状集合。"""
    it, binary, cutoffs, klist = args
    rng = np.random.default_rng(RNG_BASE + it)
    n = binary.shape[0]
    bs = binary[rng.integers(0, n, size=n)]
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        net = m.fit_ising_network(bs, encoding=ENCODING)
        j, h = net["j_sym"], net["intercept"]
        for c in cutoffs:
            lv = m.percolation_communities(j, gcc_frac=c)
            for K in klist:
                pl = m.point_estimate_core(j, h, lv["p_c"], lv["bridge"], top_k=K)
                out[(round(c, 4), K)] = set(pl["pillars"])
    return out


def task_topk(n_iter, n_jobs):
    cells = [(round(c, 4), K) for c in CUTOFFS for K in K_LIST]
    out_csv = OUT_DIR / f"boot_topk_consistency_n{n_iter}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary = load_binary()
    print(f"[topk×cutoff] N={len(binary)}  P={binary.shape[1]}  iters={n_iter}  jobs={n_jobs}")
    print(f"[topk×cutoff] cutoffs={CUTOFFS}")
    print(f"[topk×cutoff] k_list={K_LIST}  (20 节点 -> 前 20%/30%/40%)")

    counts = {(c, K, it): 0 for c, K in cells for it in ITEM_NAMES}
    done = 0

    if out_csv.exists():                                       # 断点续跑
        try:
            prev = pd.read_csv(out_csv)
            prev_done = int(prev["done"].iloc[0])
            if prev_done >= n_iter:
                print(f"[topk×cutoff] 已完成 (done={prev_done}) -> {out_csv}")
                print("[topk×cutoff] FINISHED_MARKER")
                return
            if prev_done > 0:
                print(f"[topk×cutoff] 残档 done={prev_done}/{n_iter}，续跑")
                for _, r in prev.iterrows():
                    key = (round(float(r["cutoff"]), 4), int(r["k"]), r["symptom"])
                    if key in counts:
                        counts[key] = int(r["n_core"])
                done = prev_done
        except Exception as e:
            print(f"[topk×cutoff] 读取残档失败（{e}），从头开始")

    def snap(cur_done):
        rows = []
        for c, K in cells:
            for it in ITEM_NAMES:
                rows.append({"k": K, "cutoff": c, "symptom": it,
                             "inclusion_pct": round(100.0 * counts[(c, K, it)] / max(cur_done, 1), 1),
                             "n_core": counts[(c, K, it)], "done": cur_done})
        snapshot_rows(rows, out_csv)

    with mp.Pool(n_jobs) as pool:
        for core_map in pool.imap_unordered(
                _topk_one, [(i, binary, CUTOFFS, K_LIST) for i in range(done, n_iter)]):
            for (c, K), syms in core_map.items():
                for s in syms:
                    counts[(round(c, 4), K, s)] += 1
            done += 1
            if done % SAVE_EVERY == 0:
                snap(done)
                print(f"[topk×cutoff] {done}/{n_iter}")

    snap(done)
    print(f"[topk×cutoff] 完成 {done}/{n_iter} -> {out_csv}")
    print("[topk×cutoff] FINISHED_MARKER")


# ============================================================
# 6. 任务：table —— 跨情景入选率汇总（补充材料表）
# ============================================================
def perc_removal(j_sym, mode="ascending", gcc_frac=0.50, seed=42):
    """GCC 降至 gcc_frac 时已移除边的比例 r_c（ascending=按权重递增，random=随机）。

    与原 analysis_all.py 中的实现一致（已随敏感性一并迁入本文件）。
    """
    n = j_sym.shape[0]
    edges = [(u, v, abs(j_sym[u, v])) for u in range(n) for v in range(u + 1, n)
             if abs(j_sym[u, v]) > PERC_TOL]
    if not edges:
        return None
    if mode == "ascending":
        edges.sort(key=lambda e: e[2])
    else:
        rng = np.random.default_rng(seed)
        edges = [edges[i] for i in rng.permutation(len(edges))]
    g = nx.Graph(); g.add_nodes_from(range(n))
    g.add_edges_from([(u, v) for u, v, _ in edges])
    total = len(edges)
    for i, (u, v, _) in enumerate(edges):
        g.remove_edge(u, v)
        # 无边图上 connected_components 仍返回单点集合（GCC=1/n），显式计算更稳健
        gcc = len(max(nx.connected_components(g), key=len)) / n
        if gcc < gcc_frac:
            return round((i + 1) / total, 4)
    return 1.0


def task_point():
    """点估计敏感性表（方法段 4 维度：编码 / γ / 渗流操作化 / 分辨率）。

    不做重抽样，只比较各设定下的网络结构指标与候选核心症状（≥3 票），
    输出 sens_results.csv。
    """
    scored, binary = load_scored()
    cesd_total = scored.sum(axis=1).values
    print(f"[point] N={len(binary)}  P={binary.shape[1]}")

    net = m.fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="pm1")
    lv = m.percolation_communities(net["j_sym"])
    core0 = m.pillar_names(net, lv)
    robust = main_core_from_csv()

    rows = []

    def mk(scenario, param, net_, lv_, core_, extra=None, bin_ref=None):
        p_ = net_["j_sym"].shape[0]
        bin_ = bin_ref if bin_ref is not None else binary
        d = {
            "情景": scenario, "参数": param,
            "检出率": round(float(bin_.mean()), 3), "边数": net_["n_edges"],
            "密度": f"{net_['n_edges'] / (p_ * (p_ - 1) / 2):.1%}",
            "p_c": round(lv_["p_c"], 4),
            "社区数": lv_["n_communities"], "Q": round(lv_["Q"], 4),
            "候选症状(≥3票)": "、".join(core_) if core_ else "—",
            "与主分析核心症状交集": f"{len(set(core_) & robust)}/{len(robust)}",
            "r_c(GCC→0.5)": perc_removal(net_["j_sym"], "ascending"),
        }
        if extra:
            d.update(extra)
        return d

    rows.append(mk("基准", "pm1, γ=0.25, 阈值≥2, 递增移除", net, lv, core0))

    # 维度1：编码 01
    net01 = m.fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="01")
    lv01 = m.percolation_communities(net01["j_sym"])
    core01 = m.pillar_names(net01, lv01)
    e01 = m.compute_energy(binary, net01["intercept"], net01["j_sym"], encoding="01")
    r01 = round(float(stats.pearsonr(e01, cesd_total)[0]), 4)
    rows.append(mk("1.编码", "encoding=01（vs pm1）", net01, lv01, core01,
                   {"E均值(01)": round(float(e01.mean()), 3), "r(E,CESD)(01)": r01}))

    # 维度2：EBIC γ（主分析为 0.25，此处只考察 0 与 0.5 两侧）
    for g in [0.0, 0.5]:
        net_g = m.fit_ising_network(binary, gamma=g, encoding="pm1")
        lv_g = m.percolation_communities(net_g["j_sym"])
        core_g = m.pillar_names(net_g, lv_g)
        rows.append(mk("2.正则化γ", f"γ={g}（主分析 0.25）", net_g, lv_g, core_g))

    # 维度3：渗流操作化 递增 vs 随机
    r_asc = perc_removal(net["j_sym"], "ascending")
    r_rand = perc_removal(net["j_sym"], "random")
    rows.append(mk("3.渗流策略",
                   f"递增移除 r_c={r_asc} vs 随机移除 r_c={r_rand}",
                   net, lv, core0,
                   {"r_c(GCC→0.5)": f"递增{r_asc} / 随机{r_rand}"}))

    # 维度4：Louvain 分辨率（主分析为 γ=1.0 标准模块度，此处只考察更细粒度）
    for res, tag in [(1.5, "γ=1.5"), (2.0, "γ=2.0")]:
        lv_r = m.percolation_communities(net["j_sym"], resolution=res)
        core_r = m.pillar_names(net, lv_r)
        rows.append(mk("4.分辨率", f"Louvain {tag}", net, lv_r, core_r))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = OUT_DIR / "sens_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[point] -> {out}")
    print(df.to_string(index=False))
    return df


def task_table(n_iter):
    def load(key):
        f = (OUT_DIR / f"bootstrap_pm1_n{n_iter}.csv" if key == "main"
             else OUT_DIR / f"sens_bootstrap_{key}_n{n_iter}.csv")
        if not f.exists():
            return None, None
        d = pd.read_csv(f)
        done = int(d["done"].iloc[0]) if "done" in d.columns else n_iter
        return d.set_index("症状")["inclusion_pct"], done

    series, dones, missing = {}, {}, []
    for key, _, _ in SCEN_TABLE:
        s, done = load(key)
        if s is None:
            missing.append(key)
            continue
        series[key], dones[key] = s, done

    if missing:
        print(f"[警告] 缺少情景结果：{missing}")
    incomplete = {k: v for k, v in dones.items() if v < n_iter}
    if incomplete:
        print(f"[警告] 未跑满 {n_iter} 次的情景：{incomplete}")

    keys = [k for k, _, _ in SCEN_TABLE if k in series]
    core = {k: sorted(series[k][series[k] >= INCLUSION_THRESHOLD].index) for k in keys}

    print("\n===== (a) 逐情景核心症状（入选率 ≥ 50%） =====")
    for key, dim, desc in SCEN_TABLE:
        if key not in core:
            continue
        mc = set(core.get("main", []))
        ov = f"{len(set(core[key]) & mc)}/{len(mc)}" if mc else "—"
        print(f"{dim:<6} {desc:<22} n={dones[key]:<5} 与主分析重叠 {ov:<5} {core[key]}")

    union = sorted({s for k in keys for s in series[k][series[k] >= 10.0].index},
                   key=lambda s: -series.get("main", series[keys[0]]).get(s, 0))
    hdr = "| 症状 | " + " | ".join(d if k == "main" else f"{v}" for k, d, v in SCEN_TABLE
                                   if k in series) + " |"
    print("\n===== (b) 补充材料表：跨情景入选率（%） =====")
    print(hdr)
    print("|:---|" + "---:|" * len(keys))
    for s in union:
        cells = []
        for k in keys:
            v = series[k].get(s, 0.0)
            cells.append(f"**{v:.1f}**" if v >= INCLUSION_THRESHOLD else f"{v:.1f}")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print(f"\n注：粗体为入选率 ≥ {INCLUSION_THRESHOLD:.0f}% 的核心症状；"
          f"各情景均为 {n_iter} 次病例重抽样，仅列出至少在一个情景中入选率 ≥ 10% 的症状。")


# ============================================================
# 7. 任务：percolation —— p_c 两种口径复现
# ============================================================
def _perc_load_J(path):
    df = pd.read_excel(path, header=0)
    J = df.iloc[:, 1:].to_numpy(dtype=float)
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return J


def _perc_comp_sizes(n, adj):
    seen = np.zeros(n, bool)
    sizes = []
    for s in range(n):
        if seen[s]:
            continue
        st, c = [s], 0
        seen[s] = True
        while st:
            u = st.pop()
            c += 1
            for v in np.where(adj[u] & ~seen)[0]:
                seen[v] = True
                st.append(int(v))
        sizes.append(c)
    return sorted(sizes, reverse=True)


def _perc_gcc(n, adj):
    c = _perc_comp_sizes(n, adj) + [0, 0]
    return c[0], c[1]


def _perc_adj_from(J, thr, keep_ge=True):
    A = np.abs(J)
    M = (A >= thr) if keep_ge else (A > thr)
    np.fill_diagonal(M, False)
    return M


def _perc_method_author(J, n_grid=N_GRID):
    """(A) 网格口径：等距 n_grid 个 |J| 阈值，取仍使 GCC ≥ 50% 的最大阈值。"""
    n = J.shape[0]
    w = np.abs(J[np.triu_indices(n, 1)])
    w = w[w > 1e-4]
    thr = np.linspace(w.min(), w.max(), n_grid)
    g = np.array([_perc_gcc(n, _perc_adj_from(J, t, keep_ge=True))[0] / n for t in thr])
    above = np.where(g >= 0.50)[0]
    p_c = float(thr[above[-1]]) if len(above) else np.nan
    kept = int((np.abs(J[np.triu_indices(n, 1)]) >= p_c).sum())
    tot = int((np.abs(J[np.triu_indices(n, 1)]) > 1e-4).sum())
    return p_c, thr, g, kept, tot


def _perc_method_exact(J):
    """(B) 逐条边移除口径：取 GCC 首次跌破 50% 时的阈值，并给 τ_c(S2)。"""
    n = J.shape[0]
    iu = np.triu_indices(n, 1)
    w = np.abs(J[iu])
    order = np.argsort(w)
    ws = w[order]
    rows = []
    for k in range(len(ws) + 1):
        keep = np.zeros_like(w, bool)
        if k < len(ws):
            keep[order[k:]] = True
        tau = float(ws[k - 1]) if k > 0 else 0.0
        A = np.zeros((n, n), bool)
        ii, jj = iu[0][keep], iu[1][keep]
        A[ii, jj] = A[jj, ii] = True
        g, s2 = _perc_gcc(n, A)
        rows.append(dict(tau=tau, kept=int(keep.sum()), GCC=g, GCC_frac=g / n, S2=s2))
    df = pd.DataFrame(rows)
    hit = df[df.GCC_frac <= 0.50]
    tau_c = float(hit.iloc[0].tau) if len(hit) else np.nan
    k2 = int(df.S2.idxmax())
    return df, tau_c, float(df.iloc[k2].tau), int(df.iloc[k2].S2)


def task_percolation(out_dir=None):
    out = Path(out_dir) if out_dir else PERC_OUT
    out.mkdir(parents=True, exist_ok=True)

    J = _perc_load_J(J_XLSX)
    n = J.shape[0]
    iu = np.triu_indices(n, 1)
    tot = int((np.abs(J[iu]) > 1e-4).sum())
    print(f"[FDSC symptom network] N={n}, edges(off-diag, |J|>0)={tot}, "
          f"|J| range=[{np.abs(J[iu]).min():.4f}, {np.abs(J[iu]).max():.4f}]")

    try:
        P = np.load(NPZ, allow_pickle=True)
        print(f"[npz] stored p_c = {float(P['p_c']):.6f}  "
              f"(J match: {np.allclose(J, P['J'].astype(float))})")
    except Exception as e:
        print("[npz] skip:", e)

    p_c, thr, g, kept, tot2 = _perc_method_author(J)
    print(f"\n(A) 网格口径 gcc_prune (n_grid={N_GRID})  ->  p_c = {p_c:.6f}")
    print(f"    |J| >= p_c keeps {kept}/{tot2} edges "
          f"({kept/tot2:.1%}); removed {1-kept/tot2:.1%}")

    df, tau_c, tau_s2, s2max = _perc_method_exact(J)
    at = df[df.tau == tau_c].iloc[0]
    print(f"\n(B) 精确秩移除口径（GCC 首次 ≤ 50%） ->  tau_c = {tau_c:.6f}")
    print(f"    at tau_c: GCC = {int(at.GCC)}/{n}, kept edges = {int(at.kept)}/{tot2} "
          f"({at.kept/tot2:.1%}); removed {1-at.kept/tot2:.1%}")
    print(f"    tau_c_S2 (max S2 = {s2max}) = {tau_s2:.6f}")

    # 仅为对照打印：严格跌破口径（GCC < 50%），对应稿件"需移除 83.3% 的边"
    hit_strict = df[df.GCC_frac < 0.50]
    if len(hit_strict):
        r = hit_strict.iloc[0]
        print(f"\n(B') 严格跌破口径（GCC < 50%，仅打印、不写入 CSV） -> tau = {r.tau:.6f}")
        print(f"    at tau: GCC = {int(r.GCC)}/{n}, kept edges = {int(r.kept)}/{tot2} "
              f"({r.kept/tot2:.1%}); removed {1-r.kept/tot2:.1%}")
        print("    注：主分析 perc_removal(ascending) 即此口径，其 r_c = 0.8333，"
              "与稿件「需移除约 83% 的边」一致；网格口径 (A) 则为 82.1%。")

    df.to_csv(out / "FDSC_percolation_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(N=n, edges=tot, p_c_author=p_c, tau_c_exact=tau_c,
                       tau_c_S2=tau_s2, edges_kept_at_pc=kept,
                       frac_removed_at_pc=1 - kept / tot2)]).to_csv(
        out / "FDSC_percolation_thresholds.csv", index=False, encoding="utf-8-sig")
    print(f"\n[out] {out/'FDSC_percolation_curve.csv'}")
    print(f"[out] {out/'FDSC_percolation_thresholds.csv'}")


# ============================================================
# 8. 入口
# ============================================================
TASKS = ["point", "scenarios", "cutoff", "threshold", "topk", "table", "percolation", "all"]


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task not in TASKS:
        raise SystemExit(f"未知任务 {task}，可选：{TASKS}\n"
                         f"例：python sensitivity_analysis.py scenarios 5000 6")
    rest = sys.argv[2:]

    # percolation 不涉及重抽样，第二个参数是输出目录
    if task == "percolation":
        task_percolation(rest[0] if rest else None)
        return
    if task == "point":                 # 不涉及重抽样
        task_point()
        return
    if task == "table":
        task_table(int(rest[0]) if rest else 5000)
        return

    n_iter = int(rest[0]) if len(rest) > 0 else 5000
    n_jobs = int(rest[1]) if len(rest) > 1 else 6
    extra = rest[2:]

    if task == "scenarios":
        task_scenarios(n_iter, n_jobs, extra or None)
    elif task == "cutoff":
        task_cutoff(n_iter, n_jobs)
    elif task == "threshold":
        # 默认只跑主判据（与稿件表 S6 同口径）；加 "full" 跑 GCC 全网格交叉（备用）
        if extra and extra[0] == "full":
            task_threshold(n_iter, n_jobs, cutoffs=CUTOFFS, tag="full")
        else:
            task_threshold(n_iter, n_jobs)
    elif task == "topk":
        task_topk(n_iter, n_jobs)
    else:                                   # all
        task_point()
        task_scenarios(n_iter, n_jobs)
        task_cutoff(n_iter, n_jobs)
        task_threshold(n_iter, n_jobs)
        task_table(n_iter)
        task_percolation()


if __name__ == "__main__":
    main()
