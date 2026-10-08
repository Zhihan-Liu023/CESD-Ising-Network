# Input data

Place the four cohort files here.

```
data/
├── 3447例-CESD数据已反向计分(3).xlsx      FDSC discovery cohort, n = 3,447   (restricted)
├── CFPS_cesd20_scores.csv                 CFPS 2012, n = 31,033              (public)
├── MIDUS_CESD20.xlsx                      MIDUS M2, n = 1,255                (public)
├── RCT_总103例_CESD.xlsx                  Tai Chi RCT, n = 103               (restricted)
└── J_correct.xlsx                         FDSC |J| matrix, for the p_c reproduction
```

`RCT_总103例_CESD.xlsx` carries both off-treatment and on-treatment CES-D item scores
(sheets `干预前` / `干预后`) and is the single RCT input for all three RCT scripts:
`src/rct/rct_intervention.py`, `src/rct/rct_snci_projection.py`
and `src/rct_mixedlm_core_priority.py`.  The same workbook is also distributed under
the name `太极阈下CES-D青基103例_RCT-已反向计分.xlsx`; the two files are cell-for-cell
identical in the 20 CES-D items, so either name can be placed here (the scripts accept
`CESD_RCT_XLSX` / `CESD_RCT_ITEMS_XLSX` overrides).

## Access

**CFPS 2012** — publicly available from the China Family Panel Studies,
https://cfpsdata.pku.edu.cn

**MIDUS M2** — publicly available from the Midlife in the United States study,
https://midus.wisc.edu

**FDSC** and the **RCT** data are not distributed with this repository. They
contain potentially identifying information and were collected under
informed-consent agreements that do not permit public deposition. They are
available from the corresponding author on reasonable request, subject to
approval by the Ethics Committee of the Rehabilitation Hospital Affiliated to
Fujian University of Traditional Chinese Medicine.

## Redirecting the data directory

To keep the data outside the repository, set an environment variable rather than
editing `config.py`:

```bash
# Windows (PowerShell)
$env:CESD_DATA_DIR = "D:\path\to\your\data"

# macOS / Linux
export CESD_DATA_DIR=/path/to/your/data
```

Files marked *restricted* are excluded by `.gitignore` and will not be committed.
