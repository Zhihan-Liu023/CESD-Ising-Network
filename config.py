# -*- coding: utf-8 -*-
"""
集中路径配置
================================================================
全部脚本的输入 / 输出位置都在这里定义。换机器时**不必改任何分析脚本**：

  方式一（推荐）：设置环境变量
      CESD_DATA_DIR   数据目录   默认 <repo>/data
      CESD_OUT_DIR    结果目录   默认 <repo>/results
      CESD_PERC_OUT   渗流复现输出目录  默认 <CESD_OUT_DIR>/percolation_output

  方式二：直接修改本文件中的默认值

数据文件清单见 data/README.md。CSV 结果含 `done` 列，支持断点续跑。
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("CESD_DATA_DIR", REPO_ROOT / "data"))
OUT_DIR = Path(os.environ.get("CESD_OUT_DIR", REPO_ROOT / "results"))
PERC_OUT = Path(os.environ.get("CESD_PERC_OUT", OUT_DIR / "percolation_output"))

for _d in (DATA_DIR, OUT_DIR, PERC_OUT):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = ["REPO_ROOT", "DATA_DIR", "OUT_DIR", "PERC_OUT"]
