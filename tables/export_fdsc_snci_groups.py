# -*- coding: utf-8 -*-
"""导出 FDSC 每个被试的 SNCI 数据（高/低症状组，用于复核或画图）。

口径与 cross_cohort_logistic.py 完全一致：
  - 高症状组 = CES-D 总分 >= 16；低 = <16
  - SNCI = clip(50 + 10*(E - mu)/sigma, 0, 100)，mu/sigma 为 FDSC 本队列能量分布的均值/标准差
输出：results/FDSC_SNCI_groups.csv（utf-8-sig，Excel 可直接打开）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
CODE = _REPO / "src"
sys.path.insert(0, str(CODE))
import cross_cohort_logistic as ccl  # noqa: E402

FDSC_XLSX = str(Path(CONFIG_DATA_DIR) / "3447例-CESD数据已反向计分(3).xlsx")
OUT = Path(CONFIG_OUT_DIR)

df = pd.read_excel(FDSC_XLSX)
CESD_COLS = [f"CESD_{i}" for i in range(1, 21)]
fd = df[CESD_COLS].apply(pd.to_numeric, errors="coerce").dropna()
scored = fd.values.astype(float)
binary = (scored >= 2).astype(float)
total = scored.sum(axis=1)

net = ccl.fit_ising_network(binary)
E, SNCI, mu, sigma = ccl.compute_snci(binary, net)

grp = np.where(total >= 16, "高症状组", "低症状组")
out = pd.DataFrame({
    "编号": np.arange(1, len(scored) + 1),
    "原Excel行号": fd.index.values + 2,          # 数据首行对应 Excel 第 2 行（第 1 行为表头）
    "CESD总分": total,
    "组别": grp,
    "能量E": E,
    "SNCI": SNCI,
})
out = out.sort_values(["组别", "CESD总分"], kind="stable").reset_index(drop=True)

csv = OUT / "FDSC_SNCI_groups.csv"
out.to_csv(csv, index=False, encoding="utf-8-sig")
print(f"[导出] {csv}  共 {len(out)} 行")
print(f"低症状组 n = {(grp == '低症状组').sum()}  高症状组 n = {(grp == '高症状组').sum()}")
print(f"组均值: 低 = {SNCI[total < 16].mean():.2f} ± {SNCI[total < 16].std(ddof=1):.2f}"
      f" | 高 = {SNCI[total >= 16].mean():.2f} ± {SNCI[total >= 16].std(ddof=1):.2f}")
print(out.head(8).to_string(index=False))
