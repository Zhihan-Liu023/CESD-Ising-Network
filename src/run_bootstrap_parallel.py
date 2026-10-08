"""核心症状 Bootstrap（并行版，带断点续跑）。

与主分析 cross_cohort_analysis.run_core_bootstrap 同口径：
  - 每轮：有放回重抽样(样本量=n) -> fit_ising_network(pm1, EBIC γ=0.25, AND) 重估网络
         -> percolation_communities(resolution=1.0) -> point_estimate_core 四中心性 Top6 投票
         -> 得票 ≥3 记为该轮核心症状集
  - 入选率 = 某症状入选核心的轮数 / 完成轮数；≥50% 定为稳健核心
  - RNG 种子公式与 boot_one_iter 完全一致：42*1000003 + iter_idx（保证可复现/可续跑）
  - 输出 cross_boot_<tag>_n<iters>.csv，列=症状,inclusion_pct,done（与现有 cfps/midus 快照同格式）

主代码 run_core_bootstrap 为单线程，FDSC(n=3447) 跑 5000 次要 ~14h；
本脚本用 multiprocessing.Pool 并行，10 进程约 1.5h。CFPS 全年龄 n=31033 单次重估
~2min，若需重跑请在本机放长时任务（本脚本同样适用，n_jobs 可调）。

用法：
  python run_bootstrap_parallel.py --cohorts FDSC --iters 5000 --jobs 10
  python run_bootstrap_parallel.py --cohorts FDSC CFPS_all MIDUS --iters 5000 --jobs 10
"""
import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

# 让脚本在代码目录外也能导入主模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_cohort_analysis import (load_all, fit_ising_network, percolation_communities,  # noqa: E402
                                   point_estimate_core, ITEM_NAMES, EBIC_GAMMA,
                                   OUT_DIR, RANDOM_STATE, INCLUSION_THRESHOLD)
BOOT_FILE_TAG = {"FDSC": "fdsc", "CFPS_all": "cfps", "MIDUS": "midus"}

# ---- 多进程 worker 全局状态（避免每轮 pickle 大数据）----
_BIN = None
_MLABELS = None


def _init(binary, main_labels):
    global _BIN, _MLABELS
    _BIN = binary
    _MLABELS = main_labels


def _one_iter(iter_idx):
    """单次 bootstrap 迭代，返回该轮核心症状集（与 boot_one_iter 同口径）。"""
    rng = np.random.default_rng(RANDOM_STATE * 1000003 + iter_idx)
    bs = _BIN[rng.integers(0, _BIN.shape[0], size=_BIN.shape[0])]
    with contextlib.redirect_stdout(io.StringIO()):
        net = fit_ising_network(bs, gamma=EBIC_GAMMA, encoding="pm1")
        lv = percolation_communities(net["j_sym"], resolution=1.0)
        votes = point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])
    core = {ITEM_NAMES[i] for i in range(len(votes)) if votes[i] >= 3}
    return core


def _dump(counts, done, csv_path):
    rows = sorted(((s, 100.0 * counts[s] / done) for s in counts), key=lambda r: -r[1])
    df = pd.DataFrame([{"症状": s, "inclusion_pct": round(p, 1), "done": done}
                      for s, p in rows])
    tmp = csv_path.with_name(f"{csv_path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, csv_path)


def run_one(key, n_iter, n_jobs):
    data = load_all()
    label, _, binary, _ = data[key]
    tag = BOOT_FILE_TAG.get(key, key.lower())
    csv_path = OUT_DIR / f"cross_boot_{tag}_n{n_iter}.csv"
    items = list(ITEM_NAMES)

    # 主分析观测网络标签（仅用于 ARI，核心计算实际只用 votes；此处保留以对齐原逻辑）
    with contextlib.redirect_stdout(io.StringIO()):
        net0 = fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="pm1")
        lv0 = percolation_communities(net0["j_sym"], resolution=1.0)
    main_labels = lv0["labels"]

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
        print(f"[{tag}] 已完成 done={done}，稳健核心(≥50%) = {sorted(core)}")
        return

    t0 = time.time()
    remaining = list(range(done, n_iter))
    print(f"[{tag}] 续跑 {done} -> {n_iter}（{len(remaining)} 轮，{n_jobs} 进程）", flush=True)
    with mp.Pool(n_jobs, initializer=_init, initargs=(binary, main_labels)) as pool:
        for k, core in enumerate(pool.imap(_one_iter, remaining), start=1):
            for s in core:
                counts[s] += 1
            cur = done + k
            if cur % 500 == 0 or cur == n_iter:
                _dump(counts, cur, csv_path)
                el = (time.time() - t0) / 60.0
                print(f"[{tag}] {cur}/{n_iter}  已用 {el:.1f} min", flush=True)

    core = {s for s in items if 100.0 * counts[s] / max(done, 1) >= INCLUSION_THRESHOLD}
    print(f"[{tag}] 稳健核心(≥50%) = {sorted(core)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=["FDSC"],
                    help="队列键，如 FDSC CFPS_all MIDUS")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    for key in args.cohorts:
        run_one(key, args.iters, args.jobs)


if __name__ == "__main__":
    # Windows spawn 必须放在 if __name__=='__main__' 保护下
    mp.freeze_support()
    main()
