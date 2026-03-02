from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


def _feature_names(preprocessor) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def _importance_vector(classifier) -> np.ndarray | None:
    if hasattr(classifier, "feature_importances_"):
        return np.asarray(classifier.feature_importances_)

    if hasattr(classifier, "coef_"):
        coef = np.asarray(classifier.coef_)
        if coef.ndim == 2 and coef.shape[0] >= 1:
            return coef[0]
        return coef.ravel()

    return None


def export_global_importance(
    pipeline: Pipeline, output_dir: Path, top_n: int = 30
) -> Path | None:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    names = _feature_names(preprocessor)
    values = _importance_vector(classifier)
    if values is None or not len(names):
        LOGGER.warning("No model-native feature importance available.")
        return None

    n = min(len(names), len(values))
    df = pd.DataFrame(
        {
            "feature": names[:n],
            "importance": values[:n],
            "abs_importance": np.abs(values[:n]),
        }
    ).sort_values("abs_importance", ascending=False)

    path = output_dir / "global_feature_importance.csv"
    df.head(top_n).to_csv(path, index=False)
    return path


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def export_shap_reports(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    output_dir: Path,
    background_size: int = 2000,
    explain_size: int = 1000,
    top_n: int = 30,
    random_state: int = 42,
) -> dict[str, str]:
    try:
        import shap
    except ImportError:
        LOGGER.warning("SHAP is not installed; skipping SHAP outputs.")
        return {}

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    model_name = classifier.__class__.__name__.lower()
    if not any(token in model_name for token in ("lgbm", "forest", "boost", "tree", "xgb", "catboost")):
        LOGGER.info("Skipping SHAP for non-tree model=%s", classifier.__class__.__name__)
        return {}

    if len(X_background) == 0 or len(X_explain) == 0:
        LOGGER.warning("No data available for SHAP export.")
        return {}

    bg = X_background.sample(min(background_size, len(X_background)), random_state=random_state)
    explain = X_explain.sample(min(explain_size, len(X_explain)), random_state=random_state)

    X_bg_t = _to_dense(preprocessor.transform(bg))
    X_explain_t = _to_dense(preprocessor.transform(explain))
    names = _feature_names(preprocessor)

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_explain_t)
    except Exception as exc:
        LOGGER.warning("SHAP generation failed: %s", exc)
        return {}

    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[-1])
    else:
        shap_matrix = np.asarray(shap_values)

    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, -1]

    if shap_matrix.shape[1] != len(names):
        LOGGER.warning(
            "SHAP dimension mismatch (values=%s, names=%s). Skipping report.",
            shap_matrix.shape,
            len(names),
        )
        return {}

    global_importance = np.abs(shap_matrix).mean(axis=0)
    global_df = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": global_importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    global_path = output_dir / "shap_global_importance.csv"
    global_df.head(top_n).to_csv(global_path, index=False)

    score = pipeline.predict_proba(explain)[:, 1]
    local_row = int(np.argmax(score))
    local_values = shap_matrix[local_row]
    local_df = pd.DataFrame(
        {
            "feature": names,
            "shap_value": local_values,
            "abs_shap": np.abs(local_values),
        }
    ).sort_values("abs_shap", ascending=False)

    local_payload = {
        "predicted_default_pd": float(score[local_row]),
        "reason_codes": [],
        "xai_disclaimer": (
            "LIMITATIONS: These explanations are post-hoc approximations (SHAP), "
            "are not strictly causal, and are provided for audit/transparency support "
            "rather than absolute ground truth."
        )
    }
    
    def map_feature_to_reason(feature_name: str) -> str:
        name = feature_name.lower()
        if "ext_source" in name: return "External Credit Bureau Score"
        if "days_employed" in name or "employed" in name: return "Employment History Length"
        if "amt_annuity" in name or "annuity" in name: return "Debt Burden (Annuity)"
        if "days_birth" in name or "age" in name: return "Applicant Age Profile"
        if "amt_credit" in name or "credit" in name: return "Total Credit Amount Requested"
        if "amt_goods_price" in name or "goods" in name: return "Asset Value to Loan Ratio"
        if "payment" in name or "dpd" in name: return "Historical Payment Behavior"
        return "Applicant Financial/Demographic Profile"

    # Output top 3 to 5 business-friendly reasons
    for _, row in local_df.head(5).iterrows():
        direction = "INCREASED RISK" if row["shap_value"] > 0 else "DECREASED RISK"
        reason = map_feature_to_reason(str(row["feature"]))
        
        reason_str = f"{reason} ({direction})"
        if reason_str not in local_payload["reason_codes"]:
            local_payload["reason_codes"].append(reason_str)
        if len(local_payload["reason_codes"]) >= 5:
            break

    local_path = output_dir / "shap_local_explanation.json"
    with local_path.open("w", encoding="utf-8") as fp:
        json.dump(local_payload, fp, indent=2)

    return {
        "shap_global_importance": str(global_path),
        "shap_local_explanation": str(local_path),
    }


def generate_reason_codes(
    explainer,
    model,
    X_processed,
    feature_names: list[str],
    reason_mapper: dict,
    max_reasons: int = 5,
) -> list[str]:
    """
    Generates the top reason codes for a given model prediction using SHAP.
    
    Args:
        explainer: Fitted SHAP explainer
        model: Trained model (classifier)
        X_processed: Preprocessed input features (single instance, usually 2D array-like)
        feature_names: List of all feature names matching X_processed columns
        reason_mapper: Dictionary mapping raw feature names to business-friendly descriptions
        max_reasons: Maximum number of reason codes to return
        
    Returns:
        List of formatted reason strings (e.g. "External Credit Bureau Score (INCREASED RISK)")
    """
    X_dense = _to_dense(X_processed)
    
    try:
        shap_values = explainer.shap_values(X_dense)
        
        if isinstance(shap_values, list):
            shap_matrix = np.asarray(shap_values[-1])
        else:
            shap_matrix = np.asarray(shap_values)
            
        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, -1]
            
        if len(shap_matrix) > 0:
            local_values = shap_matrix[0]
        else:
            return []
            
    except Exception as exc:
        LOGGER.warning("SHAP generation failed during reason code generation: %s", exc)
        return []

    if len(local_values) != len(feature_names):
        LOGGER.warning("SHAP dimension mismatch in reason codes.")
        return []

    local_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": local_values,
            "abs_shap": np.abs(local_values),
        }
    ).sort_values("abs_shap", ascending=False)
    
    reasons = []
    
    for _, row in local_df.iterrows():
        feature_name = str(row["feature"])
        
        if abs(row["shap_value"]) < 1e-6:
            continue
            
        direction = "INCREASED RISK" if row["shap_value"] > 0 else "DECREASED RISK"
        
        reason_text = reason_mapper.get(feature_name)
        
        if not reason_text:
            feature_lower = feature_name.lower()
            for key, val in reason_mapper.items():
                if key.lower() in feature_lower:
                    reason_text = val
                    break
                    
        if not reason_text:
            reason_text = "Applicant Financial/Demographic Profile"
            
        reason_str = f"{reason_text} ({direction})"
        
        if reason_str not in reasons:
            reasons.append(reason_str)
            
        if len(reasons) >= max_reasons:
            break
            
    if not reasons:
        reasons.append("Base risk profile")
        
    return reasons
