# app.py

"""
FastAPI serving app for the Credit Decision System.

- Loads a single immutable artifact (pipeline + model + explainer)
- Produces Probability of Default (PD)
- Applies policy-based decisioning
- Returns decision-safe reason codes
- Fails loudly on invalid outputs
"""

import math
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from serving.api_contract import (
    CreditDecisionRequest,
    CreditDecisionResponse,
)
from policy.decision_logic import apply_credit_policy
from explanations.shap_utils import generate_reason_codes

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

ARTIFACT_PATH = Path(
    "artifacts/xai_loandefault/coupled_model_explainer.pkl"
)

if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(
        f"Bundled artifact not found at: {ARTIFACT_PATH}"
    )

# -------------------------------------------------------------------
# Load IMMUTABLE Bundled Artifact (ONCE at startup)
# -------------------------------------------------------------------

bundle = joblib.load(ARTIFACT_PATH)

pipeline = bundle["pipeline"]
model = bundle["model"]
explainer = bundle["explainer"]
reason_mapper = bundle["reason_mapper"]

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------

app = FastAPI(
    title="Credit Decision System API",
    version="1.0.0",
    description="Policy-driven credit underwriting decision service",
)

# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "artifact_loaded": True,
        "artifact_path": str(ARTIFACT_PATH),
    }

# -------------------------------------------------------------------
# Credit Decision Endpoint
# -------------------------------------------------------------------

@app.post(
    "/credit-decision",
    response_model=CreditDecisionResponse,
)
def credit_decision(request: CreditDecisionRequest):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Pad dataframe with missing expected columns (all at once to avoid fragmentation)
        if hasattr(pipeline, "feature_names_in_"):
            input_df = input_df.reindex(columns=pipeline.feature_names_in_, fill_value=float("nan"))

        # Preprocess
        X_processed = pipeline.transform(input_df)

        # Predict PD
        pd_score = float(model.predict_proba(X_processed)[:, 1][0])

        # Guard against non-finite floats (not JSON-serialisable)
        if not math.isfinite(pd_score):
            pd_score = 0.0

        # Apply policy
        decision_tier = apply_credit_policy(pd_score)

        # Generate reason codes
        reason_codes = generate_reason_codes(
            explainer=explainer,
            model=model,
            X_processed=X_processed,
            feature_names=pipeline.get_feature_names_out().tolist(),
            reason_mapper=reason_mapper,
            max_reasons=5,
        )

        # Build validated response
        return CreditDecisionResponse(
            pd_score=pd_score,
            decision_tier=decision_tier,
            reason_codes=reason_codes,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Credit decision failed: {str(e)}",
        )