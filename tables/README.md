# tables —— 补充材料表

本目录产出补充材料中**由数据实算**的表格：Table S1、Table S2，以及 Fig. 1e 的逐被试数据源。
Table S7–S9 的数值分别由
[`../src/cross_cohort_analysis.py`](../src/cross_cohort_analysis.py) 与
[`../src/consensus_network.py`](../src/consensus_network.py) 产出。

| 脚本 | 表 | 口径与数值来源 |
|---|---|---|
| `table_s1_demographics.py` | Table S1 | 三队列人口学。性别：未校正 Pearson χ²（最小**期望**频数 < 5 用 Fisher）；年龄：Levene P ≥ 0.05 用合并方差 t，否则 Welch（仅 MIDUS 需 Welch，t = 5.16） |
| `table_s2_rct_baseline.py` | Table S2 | RCT 两组基线均衡（`--check` 比对刊出值）。median (IQR) + Mann–Whitney Z；性别行未校正 Pearson χ² |
| `export_fdsc_snci_groups.py` | Fig. 1e 数据源 | FDSC 逐被试 SNCI |

## Table S2 的 Mann–Whitney 口径

脚本按行复刻刊出表的两种正态近似：**20 个 CES-D 条目**用**未带结点校正**的
$z=(U-n_1n_2/2)/\sqrt{n_1n_2(n+1)/12}$，**年龄 / 受教育年限 / 总分**用**带结点校正**的近似
（scipy 的 `method="asymptotic"` 即带结点校正）；median (IQR) 用 numpy 默认线性插值（= R type 7）。
符号取原始 U 的方向，故年龄与总分的 Z 符号与刊出表相反，属分组次序造成的任意符号，只作提示。

实跑 24 行逐格复现 21 行；3 处不符均疑为刊出手工录入误差，`--check` 会逐条打印。

## 运行

```bash
python tables/table_s1_demographics.py
python tables/table_s2_rct_baseline.py --check
python tables/export_fdsc_snci_groups.py
```

输出写入 `results/`。
