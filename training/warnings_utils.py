from __future__ import annotations

import warnings


def suppress_known_non_actionable_warnings() -> None:
    """Silence noisy warnings that do not affect correctness in this pipeline."""
    warnings.filterwarnings(
        "ignore",
        message=(
            r"X does not have valid feature names, but "
            r"LGBMClassifier was fitted with feature names"
        ),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"LightGBM binary classifier with TreeExplainer shap values "
            r"output has changed to a list of ndarray"
        ),
        category=UserWarning,
    )
