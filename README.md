# CES-D 症状网络 · Ising 模型与 SNCI 分析代码

本仓库是论文 **Network-derived clinical indicator of symptom system stability and core symptoms in depression: a cross-cultural validation study**（网络衍生的症状系统稳定性与核心症状临床指标：一项跨文化验证研究）的完整分析代码。

**当前版本：v1.0（二值化 Ising）** ｜ 分析协议见 [`AnalysisPlan.md`](AnalysisPlan.md)

- 四个队列：**FDSC n = 3447**（发现）、**CFPS 2012 n = 31033**、**MIDUS M2 n = 1255**、**RCT n = 103**（干预）
- 模型：20 个 CES-D 条目 → 二值化（≥2）→ Ising 网络（节点wise LASSO 逻辑回归 + EBIC γ = 0.25 + AND 对称化）
- 衍生指标：网络能量 E 与标准化网络临床指标 **SNCI**
- 分析协议：逐节的统计口径、参数与判据见 [`AnalysisPlan.md`](AnalysisPlan.md)（英文）
- 稿件正文 / 补充材料中 **Fig. 1–5、Fig. S1 的图中数值与 Table S1–S14** 均可由本仓库脚本复现

---

## 分析流程

| 阶段 | 内容 | 代码 | 说明 |
|---|---|---|---|
| 数据准备 | 四队列量表数据、二值化与反向计分 | `data/` | [`data/README.md`](data/README.md) |
| 主分析 | Ising 网络 → 能量 → 渗流 / Louvain / 桥接 → 核心症状投票 → SNCI | [`src/`](src/README.md) | [`src/README.md`](src/README.md) |
| 敏感性分析 | 六个维度（编码 / γ / 渗流 / 分辨率 / 阈值 / p_c）× 5000 次重抽样 | `src/sensitivity_analysis.py` | [`src/README.md`](src/README.md) |
| 跨队列验证 | CFPS 2012 / MIDUS M2 的判别效度、网络结构与共识骨架 | `src/cross_cohort_analysis.py` | [`src/README.md`](src/README.md) |
| RCT 干预 | 103 例太极 RCT 投影到 FDSC 参考网络 | [`src/rct/`](src/rct/README.md) | [`src/rct/README.md`](src/rct/README.md) |
| 补充表 | Table S1 / S2 | [`tables/`](tables/README.md) | [`tables/README.md`](tables/README.md) |

每个代码目录另有自己的 README，写明该阶段的脚本、运行顺序与输入输出；逐节的统计口径与参数
集中在 [`AnalysisPlan.md`](AnalysisPlan.md)。

---

## 目录结构

```
config.py                     集中路径配置（唯一需要按机器修改的文件）
requirements.txt
src/
  analysis_all.py             ★ 主分析（FDSC）：Ising → 能量 → 渗流/Louvain/桥接 → 投票 → SNCI → 节点可预测性
  sensitivity_analysis.py     ★ 敏感性分析（6 个维度 + p_c 复现）
  cross_cohort_analysis.py    ★ 跨队列验证（判别效度 ROC、网络结构、核心症状重抽样、NCT 网络比较）
  compute_cross_cohort_core.py  跨队列核心症状重叠（三队列交集）
  run_bootstrap_parallel.py     跨队列核心症状重抽样（并行版）
  rct_mixedlm_core_priority.py  Table S14 混合效应模型（cluster bootstrap B = 5000）
  cross_cohort_logistic.py      共享函数库（fit_ising_network / compute_snci），供 tables/ 与 rct/ 下的脚本 import
  consensus_network.py          三队列共识骨架网络与一致性指标（Fig. S1 数值 / Table S7）
  rct/                        ★ RCT 干预验证（103 例），见 rct/README.md
    README.md                     本阶段脚本说明
    rct_intervention.py           基线均衡（Table S2 性别行）/ 前后变化与组间比较（Fig. 5a 数值）/ 逐条目响应率 + BH（Fig. 5e 数值、Table S10）
    rct_snci_projection.py  3447 例参考网络投影 → E、SNCI、症状响应与核心症状改善（Fig. 5b–d/5f 数值、Table S11–S13）
tables/
  README.md                        本目录表清单与 Table S2 口径说明
  export_fdsc_snci_groups.py       FDSC 逐被试 SNCI 导出（Fig. 1e 数据源）
  table_s1_demographics.py         Table S1 三队列人口学特征（性别 χ² / 年龄 t）
  table_s2_rct_baseline.py         Table S2 RCT 基线特征（median (IQR) + Mann–Whitney；`--check` 比对刊出值）
data/                         数据放置处（见 data/README.md）
results/                      所有脚本的输出目录
```

---

## 安装与运行

```bash
pip install -r requirements.txt
```

数据放到 `data/`（或用环境变量 `CESD_DATA_DIR` 指向别处），结果默认写入 `results/`。

```bash
# 主分析：约 20 秒
python src/analysis_all.py

# 主分析 + 现场跑 5000 次核心症状重抽样（长时，多进程）
python src/analysis_all.py --core-boot --boot-iters 5000 --boot-jobs 8

# 敏感性分析
python src/sensitivity_analysis.py point          # 点估计敏感性表（秒级）
python src/sensitivity_analysis.py scenarios 5000 6
python src/sensitivity_analysis.py cutoff 5000 6
python src/sensitivity_analysis.py threshold 5000 6
python src/sensitivity_analysis.py topk 5000 6
python src/sensitivity_analysis.py percolation    # p_c 三种读法复现

# 跨队列
python src/cross_cohort_analysis.py                                                   # 判别 + 结构 + 重抽样 + NCT
python src/cross_cohort_analysis.py --discrim-cohorts FDSC --no-structure --no-nct    # 快速自检
python src/compute_cross_cohort_core.py                                               # 三队列核心症状重叠

# RCT
python src/rct_mixedlm_core_priority.py --iters 5000 --jobs 8                         # Table S14
python src/rct/rct_intervention.py                                                    # Fig. 5a / 5e、Table S10（约 3 分钟）
python src/rct/rct_snci_projection.py                                                  # 3447 例参考框架拟合 + 103 例投影

# 补充表
python tables/table_s1_demographics.py                    # Table S1 三队列人口学特征
python tables/table_s2_rct_baseline.py --check            # Table S2 RCT 基线特征（并打印与刊出表的差异）
```

---

## 关键口径

| 项目 | 设定 |
|---|---|
| 二值化 | CES-D 条目 0–3 分，得分 **≥ 2** 记为症状存在 |
| 反向计分 | 条目 4 / 8 / 12 / 16；FDSC 数据文件已反向计分（`REVERSED_ALREADY = True`） |
| Ising | 节点wise L1 逻辑回归（saga，12 个 C 值网格）+ EBIC（γ = 0.25）+ AND 对称化 |
| 编码 | `s = 2x − 1 ∈ {−1, +1}`（pm1）；±1 域回归系数 ÷2 得 Ising 参数：h = β₀/2，J = β/2 |
| 能量 | `E = −Σ hᵢsᵢ − ½ Σ Jᵢⱼsᵢsⱼ`（与拟合同域） |
| SNCI | `clip(50 + 10(E − μ)/σ, 0, 100)`，**各队列用本队列能量分布独立标准化** |
| 渗流 | 60 个等距 \|J\| 阈值，取仍使最大连通成分 ≥ 50% 节点的最大阈值 p_c |
| Louvain | 主分析 resolution = 1.0；Q 用 networkx resolution-aware modularity |
| 核心症状 | 点估计四指标（Strength / Bridge / Percolation / External field）各取前 6（含并列）、得票 ≥ 3，仅作重抽样的**引擎**；最终判据为 5000 次病例重抽样入选率 **≥ 50%** |
| 高症状组 | CES-D 总分 **≥ 16** |
| 随机种子 | 模型 / Louvain / ROC / 网络布局 seed = 42；NCT 置换 seed = 20240826；重抽样 `RNG_BASE + iter_idx`（主分析 `RNG_BASE = 20240825`；跨队列 / RCT `RANDOM_STATE × 1000003`，`RANDOM_STATE = 42`） |

---

## 软件环境

Python 3.13（Windows）。依赖与版本区间见 [`requirements.txt`](requirements.txt)：numpy ≥ 2.0、
pandas ≥ 2.2、scipy ≥ 1.13、scikit-learn ≥ 1.5（网络估计）、networkx ≥ 3.3 与 python-louvain ≥ 0.16
（图论与社区检测）、statsmodels ≥ 0.14（混合效应模型）、matplotlib ≥ 3.9（共识网络与 ROC 面板）、
openpyxl ≥ 3.1 与 python-docx ≥ 1.1（表格输出）。

---

## 数据

| 队列 | 文件名 | N |
|---|---|---|
| FDSC 发现队列 | `3447例-CESD数据已反向计分(3).xlsx` | 3447 |
| CFPS 2012 | `CFPS_cesd20_scores.csv` | 31033（青年 15–30 亚组 7281） |
| CFPS 2012 / 2016 两期匹配（纵向） | `CFPS_2012_2016_两期完整20项匹配(5).xlsx`（sheet `CFPS2012` / `CFPS2016`） | 3976 |
| MIDUS M2 | `MIDUS_CESD20.xlsx` | 1255 |
| RCT | `RCT_总103例_CESD.xlsx`（sheet `干预前` / `干预后`） | 103（太极 49 / 对照 54） |
| 渗流复现 | `J_correct.xlsx` | — |

---

## 引用

见 [`CITATION.cff`](CITATION.cff)。许可见 [`LICENSE`](LICENSE)。
