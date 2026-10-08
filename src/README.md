# src —— 主分析、敏感性分析与跨队列验证

本目录是稿件**数值层**的主体代码。统计口径、参数与判据见
[`../AnalysisPlan.md`](../AnalysisPlan.md)。

## 主分析（FDSC 发现队列）

- `analysis_all.py` —— 一次运行给出主分析全部结果：Ising 估计（节点wise LASSO + EBIC γ = 0.25 + AND）
  → 能量 E → 渗流 p_c / Louvain 社区 / 桥接强度 → 核心症状四指标投票 → SNCI → 节点可预测性。
  输出 `ising_params_3447_pm1.npz` 与 `3447例_分析结果_pm1.xlsx`（6 个 sheet）。
  加 `--core-boot` 可在脚本内现场跑 5000 次核心症状重抽样（多进程、断点续跑，输出 `bootstrap_pm1_n5000.csv`）。
- `sensitivity_analysis.py` —— 敏感性分析，六个维度：编码 / EBIC γ / 渗流操作化 / Louvain 分辨率、
  GCC 判据、判定阈值、p_c 复现。子命令 `point`、`scenarios`、`cutoff`、`threshold`、`topk`、`percolation`。
- `compute_cross_cohort_core.py` —— 三队列核心症状交集。
- `run_bootstrap_parallel.py` —— 跨队列核心症状重抽样（并行版）。
- `cross_cohort_logistic.py` —— 共享函数库（`fit_ising_network` / `compute_snci`），
  供 `tables/` 与 `src/rct/` 下的脚本 import。

## 跨队列验证（CFPS 2012 / MIDUS M2）

- `cross_cohort_analysis.py` —— 判别效度（ROC / AUC / Youden）、网络结构比较、核心症状重抽样、
  NCT 网络比较。分段开关：`--discrim-cohorts`、`--no-structure`、`--no-nct`、
  `--bootstrap --boot-iters`。
- `consensus_network.py` —— 三队列共识骨架网络与一致性指标（Table S7、Fig. S1）。

## 混合效应模型

- `rct_mixedlm_core_priority.py` —— 太极对核心症状优先改善的混合效应模型（Table S14；受试者随机
  截距，cluster bootstrap B = 5000）。核心症状取跨队列交叉验证得到的共同核心症状（Depressed、Sad）。

## 运行

```bash
python src/analysis_all.py                                     # 主分析（约 20 秒）
python src/analysis_all.py --core-boot --boot-iters 5000 --boot-jobs 8
python src/sensitivity_analysis.py point                       # 点估计敏感性表（秒级）
python src/sensitivity_analysis.py scenarios 5000 6
python src/sensitivity_analysis.py cutoff 5000 6
python src/sensitivity_analysis.py threshold 5000 6
python src/sensitivity_analysis.py topk 5000 6
python src/sensitivity_analysis.py percolation                 # p_c 三种读法复现
python src/cross_cohort_analysis.py                            # 判别 + 结构 + 重抽样 + NCT
python src/compute_cross_cohort_core.py
python src/rct_mixedlm_core_priority.py --iters 5000 --jobs 8   # Table S14
```

输出全部写入 `results/`。

RCT 干预验证的代码在 [`rct/`](rct/README.md)。
