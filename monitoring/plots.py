from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency at runtime
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency at runtime
    plt = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _can_plot() -> bool:
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skipping metric plots.")
        return False
    return True


def _metric_value(comparison_df: pd.DataFrame, model_key: str, metric: str) -> float:
    row = comparison_df.loc[comparison_df["model_key"] == model_key, metric]
    if row.empty:
        return float("nan")
    return float(row.iloc[0])


def _base_name(output_dir: Path, dataset_name: str, suffix: str) -> Path:
    return output_dir / f"{dataset_name}_{suffix}.png"


def plot_roc_pr(
    eval_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    comparison_df: pd.DataFrame,
    output_path: Path,
    y_col: str = "y_true",
    title_prefix: str = "Validation",
) -> Path | None:
    if not _can_plot():
        return None

    _ensure_parent(output_path)
    y_true = eval_df[y_col].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_roc, ax_pr = axes

    for spec in model_specs:
        model_key = str(spec["model_key"])
        proba_col = str(spec["proba_col"])
        y_proba = eval_df[proba_col].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = _metric_value(comparison_df, model_key, "roc_auc")
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{model_key} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = _metric_value(comparison_df, model_key, "average_precision")
        ax_pr.plot(recall, precision, linewidth=2, label=f"{model_key} (AP={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    ax_roc.set_title(f"{title_prefix} ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(alpha=0.3)
    ax_roc.legend()

    ax_pr.set_title(f"{title_prefix} Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(alpha=0.3)
    ax_pr.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration(
    eval_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    output_path: Path,
    n_bins: int = 10,
    y_col: str = "y_true",
    title_prefix: str = "Validation",
) -> Path | None:
    if not _can_plot():
        return None

    _ensure_parent(output_path)
    y_true = eval_df[y_col].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for spec in model_specs:
        model_key = str(spec["model_key"])
        proba_col = str(spec["proba_col"])
        frame = pd.DataFrame({"y_true": y_true, "y_proba": eval_df[proba_col].to_numpy()})
        frame["bin"] = pd.qcut(frame["y_proba"], q=n_bins, labels=False, duplicates="drop")
        grouped = (
            frame.groupby("bin", dropna=False)
            .agg(mean_pred=("y_proba", "mean"), obs_rate=("y_true", "mean"))
            .reset_index()
        )
        ax.plot(grouped["mean_pred"], grouped["obs_rate"], marker="o", linewidth=2, label=model_key)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax.set_title(f"{title_prefix} Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Default Rate")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusions(
    eval_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    output_path: Path,
    y_col: str = "y_true",
    title_prefix: str = "Validation",
) -> Path | None:
    if not _can_plot():
        return None

    _ensure_parent(output_path)
    y_true = eval_df[y_col].to_numpy()
    n_models = len(model_specs)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, spec in zip(axes, model_specs):
        model_key = str(spec["model_key"])
        proba_col = str(spec["proba_col"])
        threshold = float(spec["threshold"])
        y_pred = (eval_df[proba_col].to_numpy() >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title(f"{title_prefix} Confusion\n{model_key} @ {threshold:.2f}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ks_curves(
    eval_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    output_path: Path,
    y_col: str = "y_true",
    title_prefix: str = "Validation",
) -> Path | None:
    if not _can_plot():
        return None

    _ensure_parent(output_path)
    y_true = eval_df[y_col].to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for spec in model_specs:
        model_key = str(spec["model_key"])
        proba_col = str(spec["proba_col"])
        y_proba = eval_df[proba_col].to_numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        ks_values = tpr - fpr
        max_idx = int(np.argmax(ks_values))
        ax.plot(thresholds, ks_values, linewidth=2, label=f"{model_key} (KS={ks_values[max_idx]:.3f})")
        ax.scatter([thresholds[max_idx]], [ks_values[max_idx]], s=30)

    ax.set_title(f"{title_prefix} KS by Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("TPR - FPR")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_threshold_cost(
    eval_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    output_path: Path,
    false_positive_cost: float,
    false_negative_cost: float,
    y_col: str = "y_true",
    title_prefix: str = "Validation",
) -> Path | None:
    if not _can_plot():
        return None

    _ensure_parent(output_path)
    y_true = eval_df[y_col].to_numpy()
    thresholds = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for spec in model_specs:
        model_key = str(spec["model_key"])
        proba_col = str(spec["proba_col"])
        y_proba = eval_df[proba_col].to_numpy()
        costs = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            _ = tn + tp
            cost = fp * false_positive_cost + fn * false_negative_cost
            costs.append(cost)
        costs_arr = np.asarray(costs, dtype=float)
        best_idx = int(np.argmin(costs_arr))
        ax.plot(thresholds, costs_arr, linewidth=2, label=f"{model_key}")
        ax.scatter([thresholds[best_idx]], [costs_arr[best_idx]], s=28)

    ax.set_title(f"{title_prefix} Threshold vs Business Cost")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Business Cost")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_bars(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    if not _can_plot():
        return None

    _ensure_parent(output_path)
    metrics = ["roc_auc", "average_precision", "ks_statistic", "ece", "business_cost"]
    available = [m for m in metrics if m in comparison_df.columns]
    if not available:
        return None

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    model_keys = comparison_df["model_key"].astype(str).tolist()
    x = np.arange(len(model_keys))

    for ax, metric in zip(axes, available):
        values = comparison_df[metric].astype(float).to_numpy()
        ax.bar(x, values, color=["#1f77b4", "#ff7f0e"][: len(values)])
        ax.set_xticks(x, labels=model_keys, rotation=20)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for idx, val in enumerate(values):
            ax.text(idx, val, f"{val:.3f}" if abs(val) < 1000 else f"{val:.1f}", ha="center", va="bottom")

    fig.suptitle("Baseline vs Champion Metric Comparison", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_metric_plots(
    artifacts_dir: Path,
    comparison_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model_specs: list[dict[str, str | float]],
    false_positive_cost: float,
    false_negative_cost: float,
    calibration_bins: int = 10,
    oot_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    if not _can_plot():
        return {}

    out: dict[str, str] = {}
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    validation_outputs = {
        "validation_roc_pr": plot_roc_pr(
            validation_df,
            model_specs,
            comparison_df,
            _base_name(plots_dir, "validation", "roc_pr"),
            title_prefix="Validation",
        ),
        "validation_calibration": plot_calibration(
            validation_df,
            model_specs,
            _base_name(plots_dir, "validation", "calibration"),
            n_bins=calibration_bins,
            title_prefix="Validation",
        ),
        "validation_confusion": plot_confusions(
            validation_df,
            model_specs,
            _base_name(plots_dir, "validation", "confusion"),
            title_prefix="Validation",
        ),
        "validation_ks": plot_ks_curves(
            validation_df,
            model_specs,
            _base_name(plots_dir, "validation", "ks"),
            title_prefix="Validation",
        ),
        "validation_threshold_cost": plot_threshold_cost(
            validation_df,
            model_specs,
            _base_name(plots_dir, "validation", "threshold_cost"),
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            title_prefix="Validation",
        ),
        "model_comparison_bars": plot_metric_bars(
            comparison_df,
            _base_name(plots_dir, "comparison", "bars"),
        ),
    }
    for key, path in validation_outputs.items():
        if path is not None:
            out[key] = str(path)

    if oot_df is not None and len(oot_df) > 0:
        oot_outputs = {
            "oot_roc_pr": plot_roc_pr(
                oot_df,
                model_specs,
                comparison_df,
                _base_name(plots_dir, "oot", "roc_pr"),
                title_prefix="OOT",
            ),
            "oot_calibration": plot_calibration(
                oot_df,
                model_specs,
                _base_name(plots_dir, "oot", "calibration"),
                n_bins=calibration_bins,
                title_prefix="OOT",
            ),
            "oot_confusion": plot_confusions(
                oot_df,
                model_specs,
                _base_name(plots_dir, "oot", "confusion"),
                title_prefix="OOT",
            ),
            "oot_ks": plot_ks_curves(
                oot_df,
                model_specs,
                _base_name(plots_dir, "oot", "ks"),
                title_prefix="OOT",
            ),
            "oot_threshold_cost": plot_threshold_cost(
                oot_df,
                model_specs,
                _base_name(plots_dir, "oot", "threshold_cost"),
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
                title_prefix="OOT",
            ),
        }
        for key, path in oot_outputs.items():
            if path is not None:
                out[key] = str(path)

    return out
