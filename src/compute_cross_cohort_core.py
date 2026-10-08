"""三队列 bootstrap 稳健核心症状 · 跨样本重叠分析。

读取 cross_boot_{fdsc,cfps,midus}_n5000.csv（症状,inclusion_pct,done），
按 ≥50% 入选率提取各队列稳健核心，计算：
  - 各队列稳健核心清单（含入选率）
  - 两两重叠（交集 / 并集 / Jaccard）
  - 三队列共同核心（交集）
  - 与点估计核心(≥3票)的差异（点估计来自 analyze_structure 的"点估计核心(≥3票)"，
    此处用同一 observed 网络重新计算，确保可比）
输出：cross_cohort_core_overlap.csv
"""
import contextlib
import io
import os
import sys
from itertools import combinations

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_cohort_analysis import (load_all, fit_ising_network, percolation_communities,  # noqa: E402
                                   ITEM_NAMES, EBIC_GAMMA, OUT_DIR)

TAGS = {"FDSC": "fdsc", "CFPS_all": "cfps", "MIDUS": "midus"}


def robust_core(path, thr=50.0):
    """读取快照，按入选率 ≥ thr% 提取稳健核心。inclusion_pct 已是百分比。"""
    d = pd.read_csv(path)
    done = int(d["done"].iloc[0])
    core = {r["症状"]: float(r["inclusion_pct"])
            for _, r in d.iterrows() if float(r["inclusion_pct"]) >= thr}
    return core, done


def point_estimate_core(key):
    """用 observed 网络重算四中心性投票核心(≥3票)，与主分析口径对齐。"""
    data = load_all()
    label, _, binary, _ = data[key]
    with contextlib.redirect_stdout(io.StringIO()):
        net = fit_ising_network(binary, gamma=EBIC_GAMMA, encoding="pm1")
        lv = percolation_communities(net["j_sym"], resolution=1.0)
        votes = point_estimate_core(net["j_sym"], net["intercept"], lv["p_c"], lv["bridge"])
    core = [ITEM_NAMES[i] for i in range(len(votes)) if votes[i] >= 3]
    return core


def main():
    data = load_all()
    cores = {}
    dones = {}
    for key, tag in TAGS.items():
        p = OUT_DIR / f"cross_boot_{tag}_n5000.csv"
        if not p.exists():
            print(f"[跳过] {tag} 快照缺失：{p}")
            continue
        core, done = robust_core(p, thr=50.0)
        cores[key] = core
        dones[key] = done
        print(f"\n=== {key}（N={data[key][3]}，bootstrap done={done}）稳健核心(≥50%) ===")
        for s in sorted(core, key=lambda x: -core[x]):
            print(f"  {s:12s} {core[s]:.0f}%")

    if len(cores) < 2:
        print("队列不足，无法计算重叠")
        return

    # 两两重叠
    print("\n=== 两两重叠（稳健核心, ≥50%）===")
    rows = []
    for a, b in combinations(cores, 2):
        sa, sb = set(cores[a]), set(cores[b])
        inter = sa & sb
        union = sa | sb
        jac = len(inter) / len(union) if union else 0.0
        print(f"  {a} ∩ {b}: {sorted(inter)} | Jaccard={jac:.2f} (|∩|={len(inter)},|∪|={len(union)})")
        rows.append({"比较对": f"{a}∩{b}", "交集数": len(inter), "并集数": len(union),
                     "Jaccard": round(jac, 3), "共同核心": "、".join(sorted(inter))})

    # 三队列共同
    if len(cores) == 3:
        triple = set(cores["FDSC"]) & set(cores["CFPS_all"]) & set(cores["MIDUS"])
        print(f"\n=== 三队列共同稳健核心 ===\n  {sorted(triple) if triple else '（无）'}")
        rows.append({"比较对": "三队列交集", "交集数": len(triple), "并集数": "",
                     "Jaccard": "", "共同核心": "、".join(sorted(triple)) if triple else "无"})

    # 与点估计对比
    print("\n=== 稳健核心 vs 点估计核心(≥3票) ===")
    for key in cores:
        pe = set(point_estimate_core(key))
        rb = set(cores[key])
        print(f"  {key}: 点估计={sorted(pe)} | 稳健={sorted(rb)} | "
              f"稳健⊆点估计? {rb <= pe} | 差异={sorted(pe ^ rb)}")

    out = OUT_DIR / "cross_cohort_core_overlap.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[导出] {out}")


if __name__ == "__main__":
    main()
