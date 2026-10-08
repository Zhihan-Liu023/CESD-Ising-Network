# -*- coding: utf-8 -*-
"""
共享函数库：CES-D 二值化数据的 Ising 网络估计与 SNCI 计算。
================================================================

供仓库内其他脚本 import 使用：

  fit_ising_network(binary, gamma=0.25, encoding="pm1")
      节点wise LASSO 逻辑回归（saga, l1_ratio=1.0）+ EBIC 选 C + AND 对称化；
      ±1 域回归系数 ÷2 得 Ising 参数 h = β₀/2、J = β/2。
      返回 {"intercept": h, "j_sym": J, "n_edges": 边数}。

  compute_snci(binary, net)
      按传入网络的 h、J 算能量 E = −Σh·s − ½ΣJ·s·s（s = 2x − 1），
      以本队列能量分布的均值与样本标准差标准化
      SNCI = clip(50 + 10(E − μ)/σ, 0, 100)，返回 (E, SNCI, mu, sigma)。

各队列的判别效度、网络结构与核心症状分析分别在 cross_cohort_analysis.py、
consensus_network.py 与 src/rct/ 下的脚本中实现。
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

EBIC_GAMMA = 0.25
AND_TOL = 1e-4
RANDOM_STATE = 42
C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


# ====================== Ising 估计（与主分析同一口径） ======================
def _loglik_mean(y, logits, encoding):
    if encoding == "01":
        y_pm1 = 2.0 * y - 1.0
        return float(np.mean(np.logaddexp(0.0, -y_pm1 * logits)))
    return float(np.mean(np.logaddexp(0.0, -y * logits)))


def fit_ising_network(binary: np.ndarray, gamma: float = EBIC_GAMMA, encoding: str = "pm1") -> dict:
    n, p = binary.shape
    X = binary.astype(float) if encoding == "01" else (2 * binary.astype(int) - 1).astype(float)
    intercept = np.zeros(p)
    j_raw = np.zeros((p, p))
    edge_mask = np.zeros((p, p), dtype=bool)
    for i in range(p):
        y, Z = X[:, i], np.delete(X, i, axis=1)
        best_ebic, best_coefs, best_int = np.inf, None, 0.0
        for c_val in C_GRID:
            clf = LogisticRegression(solver="saga", C=c_val, l1_ratio=1.0,
                                     fit_intercept=True, max_iter=3000, tol=1e-4,
                                     random_state=RANDOM_STATE)
            clf.fit(Z, y)
            coefs = clf.coef_[0]
            n_nz = int(np.sum(np.abs(coefs) > AND_TOL))
            logits = clf.intercept_[0] + Z @ coefs
            mean_ll = _loglik_mean(y, logits, encoding)
            ebic = 2.0 * n * mean_ll + n_nz * np.log(n) + 2.0 * gamma * n_nz * np.log(p - 1)
            if ebic < best_ebic:
                best_ebic, best_coefs, best_int = ebic, coefs.copy(), float(clf.intercept_[0])
        intercept[i] = best_int
        if best_coefs is not None:
            mask = np.ones(p, dtype=bool)
            mask[i] = False
            j_raw[i, mask] = best_coefs
            edge_mask[i, mask] = np.abs(best_coefs) > AND_TOL
    adj_and = edge_mask & edge_mask.T
    j_sym = (j_raw + j_raw.T) / 2.0 * adj_and
    if encoding == "pm1":
        intercept *= 0.5
        j_raw *= 0.5
        j_sym *= 0.5
    n_edges = int(adj_and.sum() // 2)
    return {"intercept": intercept, "j_sym": j_sym, "n_edges": n_edges}


# ====================== SNCI 计算（本队列参数） ======================
def compute_snci(binary, net):
    h, J = net["intercept"], net["j_sym"]
    s = 2 * binary - 1
    E = -(s @ h) - 0.5 * np.sum((s @ J) * s, axis=1)
    mu, sigma = float(E.mean()), float(E.std(ddof=1))
    snci = np.clip(50 + 10 * (E - mu) / sigma, 0, 100)
    return E, snci, mu, sigma
