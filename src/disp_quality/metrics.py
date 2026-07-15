from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


def safe_binary_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    y_true = y_true.astype(np.uint8)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
    }


def safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def quality_coverage_curve(reliability: np.ndarray, bad_target: np.ndarray, coverages: np.ndarray | None = None) -> list[dict[str, float]]:
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 10)
    order = np.argsort(-reliability)
    bad_sorted = bad_target[order]
    n = len(order)
    rows = []
    for coverage in coverages:
        k = max(1, int(math.ceil(float(coverage) * n)))
        rows.append({
            "coverage": float(coverage),
            "retained_pixels": int(k),
            "bad_pixel_ratio": float(bad_sorted[:k].mean()),
        })
    return rows
