from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def ks_statistic(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def expected_calibration_error(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_proba >= left) & (y_proba <= right)
        else:
            mask = (y_proba >= left) & (y_proba < right)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_pred = float(y_proba[mask].mean())
        avg_true = float(y_true[mask].mean())
        ece += (count / len(y_true)) * abs(avg_pred - avg_true)
    return float(ece)


def top_decile_capture_lift(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0

    top_n = max(1, int(np.ceil(0.10 * n)))
    order = np.argsort(-y_proba)
    top_idx = order[:top_n]

    capture = _safe_divide(float(y_true[top_idx].sum()), float(y_true.sum()))
    overall_rate = float(y_true.mean())
    top_rate = float(y_true[top_idx].mean())
    lift = _safe_divide(top_rate, overall_rate) if overall_rate > 0 else 0.0
    return capture, lift


def confusion_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = _safe_divide(tn, tn + fp)
    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    npv = _safe_divide(tn, tn + fn)
    approval_rate = _safe_divide(int((y_pred == 0).sum()), len(y_pred))
    rejection_rate = 1.0 - approval_rate
    bad_rate_approved = _safe_divide(fn, fn + tn)
    default_capture_rate = _safe_divide(tp, tp + fn)
    balanced_accuracy = 0.5 * (specificity + default_capture_rate)

    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "npv": npv,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "bad_rate_approved": bad_rate_approved,
        "default_capture_rate": default_capture_rate,
        "balanced_accuracy": balanced_accuracy,
    }


def classification_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float, calibration_bins: int = 10
) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    has_two_classes = len(np.unique(y_true_np)) >= 2
    y_pred = (y_proba >= threshold).astype(int)
    top_capture, top_lift = top_decile_capture_lift(y_true, y_proba)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if has_two_classes else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "ks_statistic": ks_statistic(y_true, y_proba) if has_two_classes else float("nan"),
        "ece": expected_calibration_error(y_true, y_proba, n_bins=calibration_bins),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "top_decile_capture_rate": top_capture,
        "top_decile_lift": top_lift,
    }
    metrics.update(confusion_metrics(y_true, y_pred))
    return metrics


def calibration_curve_table(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) == 0:
        return pd.DataFrame(columns=["bin", "count", "mean_predicted", "observed_rate"])

    frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    # Quantile bins keep similar support per bucket on skewed probability distributions.
    frame["bin"] = pd.qcut(frame["y_proba"], q=n_bins, labels=False, duplicates="drop")
    grouped = (
        frame.groupby("bin", dropna=False)
        .agg(
            count=("y_true", "size"),
            mean_predicted=("y_proba", "mean"),
            observed_rate=("y_true", "mean"),
            min_predicted=("y_proba", "min"),
            max_predicted=("y_proba", "max"),
        )
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def _robust_bins(reference: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.quantile(reference, quantiles)
    bins = np.unique(bins)
    if len(bins) < 3:
        ref_min = float(np.min(reference))
        ref_max = float(np.max(reference))
        if ref_min == ref_max:
            ref_max = ref_min + 1e-6
        bins = np.linspace(ref_min, ref_max, n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> float:
    reference = np.asarray(reference)
    current = np.asarray(current)
    bins = _robust_bins(reference, n_bins=n_bins)

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_ratio = ref_hist / max(ref_hist.sum(), 1)
    cur_ratio = cur_hist / max(cur_hist.sum(), 1)

    ref_ratio = np.clip(ref_ratio, eps, None)
    cur_ratio = np.clip(cur_ratio, eps, None)

    psi = np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio))
    return float(psi)


@dataclass
class BootstrapConfig:
    n_iterations: int = 300
    random_state: int = 42


def bootstrap_confidence_intervals(
    y_true: pd.Series,
    y_proba: np.ndarray,
    cfg: BootstrapConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    n = len(y)

    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": lambda a, b: float(roc_auc_score(a, b)),
        "average_precision": lambda a, b: float(average_precision_score(a, b)),
        "ks_statistic": lambda a, b: ks_statistic(a, b),
        "brier_score": lambda a, b: float(brier_score_loss(a, b)),
        "log_loss": lambda a, b: float(log_loss(a, b, labels=[0, 1])),
    }
    metric_samples: dict[str, list[float]] = {k: [] for k in metric_functions}

    for _ in range(cfg.n_iterations):
        idx = rng.integers(0, n, size=n)
        y_s = y[idx]
        if len(np.unique(y_s)) < 2:
            continue
        p_s = p[idx]
        for name, fn in metric_functions.items():
            metric_samples[name].append(fn(y_s, p_s))

    rows = []
    for name, samples in metric_samples.items():
        if not samples:
            rows.append(
                {
                    "metric": name,
                    "mean": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "n_bootstrap_samples": 0,
                }
            )
            continue
        arr = np.asarray(samples)
        rows.append(
            {
                "metric": name,
                "mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5)),
                "n_bootstrap_samples": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)


def segment_performance_report(
    eval_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    group_columns: list[str],
    min_segment_size: int,
    calibration_bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in model_specs:
        model_key = str(spec["model_key"])
        proba_col = str(spec["proba_col"])
        threshold = float(spec["threshold"])

        overall_metrics = classification_metrics(
            eval_df["y_true"], eval_df[proba_col].to_numpy(), threshold, calibration_bins=calibration_bins
        )
        for group_col in group_columns:
            if group_col not in eval_df.columns:
                continue
            for group_value, group_df in eval_df.groupby(group_col, dropna=False):
                if len(group_df) < min_segment_size:
                    continue
                group_metrics = classification_metrics(
                    group_df["y_true"],
                    group_df[proba_col].to_numpy(),
                    threshold,
                    calibration_bins=calibration_bins,
                )
                # Fairness calculations 
                disparate_impact = group_metrics["approval_rate"] / max(overall_metrics["approval_rate"], 1e-6)
                fpr_parity_gap = group_metrics["false_positive_rate"] - overall_metrics["false_positive_rate"]
                fnr_parity_gap = group_metrics["false_negative_rate"] - overall_metrics["false_negative_rate"]
                
                group_mean_pred = group_df[proba_col].mean()
                group_observed_rate = group_df["y_true"].mean()
                calibration_error = abs(group_mean_pred - group_observed_rate)
                
                row = {
                    "model_key": model_key,
                    "group_column": group_col,
                    "group_value": str(group_value),
                    "segment_count": int(len(group_df)),
                    "segment_default_rate": float(group_observed_rate),
                    "disparate_impact_ratio": float(disparate_impact),
                    "fpr_parity_gap": float(fpr_parity_gap),
                    "fnr_parity_gap": float(fnr_parity_gap),
                    "calibration_error": float(calibration_error),
                }
                for metric_name in [
                    "approval_rate",
                    "bad_rate_approved",
                    "false_positive_rate",
                    "false_negative_rate",
                ]:
                    row[metric_name] = group_metrics[metric_name]
                rows.append(row)

    return pd.DataFrame(rows)
