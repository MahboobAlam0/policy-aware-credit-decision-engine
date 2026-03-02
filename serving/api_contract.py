# serving/api_contract.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class CreditDecisionRequest(BaseModel):
    """
    Input contract for credit decision requests.
    Expects raw applicant features AFTER offline feature engineering.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Feature-name to value mapping used by the PD model"
    )


class CreditDecisionResponse(BaseModel):
    """
    Output contract for credit decision responses.
    """
    pd_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted Probability of Default"
    )
    decision_tier: str = Field(
        ...,
        description="Policy decision: AUTO_APPROVE | MANUAL_REVIEW | AUTO_DECLINE"
    )
    reason_codes: List[str] = Field(
        ...,
        min_items=0,
        max_items=5,
        description="Top business-friendly decision reason codes"
    )