# src/rct —— 太极 RCT 干预验证（103 例）

本目录是 103 例（太极 49 / 对照 54）随机对照试验的干预验证代码，对应
Fig. 5、Table S2、Table S10–S13。统计口径见 [`../../AnalysisPlan.md`](../../AnalysisPlan.md) §9。

> ⚠ **SNCI 的参考框架决定 Fig. 5b–5d 的数值。** 手稿用的 SNCI / 能量是把 103 例
> **投影到 FDSC 3447 例参考网络**（冻结 h、J）得到的；而 `rct_intervention.py` 内置的
> 那套 SNCI 是**用合并 RCT 基线网络**（13 条边）标准化的。两者不可互换：ΔSNCI 组间
> 前者 t = −2.65（P = 0.009，= 稿件 5b），后者 t = −1.95（P = 0.054）。

| 脚本 | 产物 | 说明 |
|---|---|---|
| `rct_intervention.py` | Fig. 5a / 5e 数值、Table S2 性别行、Table S10 | 基线均衡；前后变化与组间比较（配对 t / 合并方差 t、Cohen d、d_z）；逐条目响应率（≥ 50 % 改善，未校正 Pearson χ² / Fisher + BH 校正 20 项） |
| `rct_snci_projection.py` | Fig. 5b–d / 5f 数值、Table S11–S13 | 3447 例参考网络投影 → E、SNCI、症状响应与核心症状改善（投影与能量 → 核心与非核心症状改善） |

Fig. 5 六个面板的**数值**由 `rct_intervention.py` 与 `rct_snci_projection.py` 产出。

Table S14（太极对核心症状优先改善的混合效应模型）在
[`../rct_mixedlm_core_priority.py`](../rct_mixedlm_core_priority.py)，核心症状取跨队列核心
（Depressed、Sad）。

## 运行

```bash
python src/rct/rct_intervention.py                 # 约 3 分钟
python src/rct/rct_snci_projection.py
```
