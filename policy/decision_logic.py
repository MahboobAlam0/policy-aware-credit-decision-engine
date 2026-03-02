from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

def compute_expected_loss(pd_scores: np.ndarray, lgd: float, ead: np.ndarray | float) -> np.ndarray:
    """Expected Loss = Probability of Default * Loss Given Default * Exposure at Default."""
    return pd_scores * lgd * ead

def generate_policy_frontier(
    y_true: pd.Series,
    pd_scores: np.ndarray,
    lgd: float = 0.45,
    ead: float = 1.0,
) -> pd.DataFrame:
    """
    Sweeps a single Auto-Approve threshold to generate an approval-rate vs. expected-loss frontier.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    for thr in thresholds:
        approved = pd_scores < thr
        num_approved = int(approved.sum())
        approval_rate = float(num_approved / len(y_true))
        
        if num_approved == 0:
            continue
            
        bad_rate = float(y_true[approved].mean())
        
        # Expected portfolio loss of the approved cohort
        el_portfolio = float(compute_expected_loss(pd_scores[approved], lgd, ead).sum())
        # Actual loss for evaluation mapping
        actual_portfolio_loss = float((y_true[approved] * lgd * ead).sum())
        
        results.append({
            "threshold": float(thr),
            "approval_rate": approval_rate,
            "bad_rate": bad_rate,
            "expected_portfolio_loss": el_portfolio,
            "actual_portfolio_loss": actual_portfolio_loss,
        })
        
    return pd.DataFrame(results)

def search_dual_threshold_policy(
    y_true: pd.Series,
    pd_scores: np.ndarray,
    lgd: float = 0.45,
    ead: float = 1.0,
    review_cost_per_applicant: float = 0.02,
    review_approval_rate: float = 0.35,
    min_approval_rate: float = 0.35,
    max_bad_rate: float = 0.12,
    sensitive_features: pd.DataFrame | None = None,
    min_disparate_impact: float | None = None,
) -> dict[str, float]:
    """
    Finds the optimal (approve_thr, review_thr) combination minimizing total portfolio loss + review costs,
    subject to minimum approval rate and maximum bad rate constraints.
    """
    best_cost = float("inf")
    best_policy = None
    
    # Grid search for thresholds
    grid = np.linspace(0.01, 0.95, 40)
    for auto_approve_thr in grid:
        for review_thr in grid:
            if auto_approve_thr >= review_thr:
                continue
                
            # Simulate policy
            auto_approved_mask = pd_scores < auto_approve_thr
            review_mask = (pd_scores >= auto_approve_thr) & (pd_scores < review_thr)
            
            # Simple assumption: manual review approves `review_approval_rate` at random from the review bucket
            # In a real system, we'd multiply the review bucket's bad rate by a human expert modifier.
            num_auto_approved = int(auto_approved_mask.sum())
            num_reviewed = int(review_mask.sum())
            num_review_approved = int(num_reviewed * review_approval_rate)
            
            total_approved = num_auto_approved + num_review_approved
            total_approval_rate = total_approved / len(y_true) if len(y_true) > 0 else 0
            
            if total_approved == 0:
                continue
                
            auto_approve_bads = y_true[auto_approved_mask].sum()
            review_pool_bad_rate = y_true[review_mask].mean() if num_reviewed > 0 else 0
            review_approved_bads = num_review_approved * review_pool_bad_rate
            
            total_bads = auto_approve_bads + review_approved_bads
            total_bad_rate = total_bads / total_approved
            
            # Constraints check
            if total_approval_rate < min_approval_rate or total_bad_rate > max_bad_rate:
                continue
                
            # Optional Fairness Constraint Check
            # Explicitly modeled as a surfaced tradeoff constraint, not a baseline optimization.
            if min_disparate_impact is not None and sensitive_features is not None:
                di_failed = False
                for col in sensitive_features.columns:
                    for val, group_idx in sensitive_features.groupby(col).groups.items():
                        if len(group_idx) < 100: continue
                        
                        group_total = len(group_idx)
                        auto_app_group = auto_approved_mask[group_idx].sum()
                        rev_pool_group = review_mask[group_idx].sum()
                        
                        # Uniform review approval rate assumption per group for simplicity
                        group_approved = auto_app_group + (rev_pool_group * review_approval_rate)
                        group_approval_rate_val = group_approved / group_total
                        
                        di = group_approval_rate_val / max(total_approval_rate, 1e-6)
                        if di < min_disparate_impact:
                            di_failed = True
                            break
                    if di_failed: break
                
                if di_failed:
                    continue
                    
            # Cost = Expected Loss of auto-approvals + Expected Loss of review-approvals + Review Operational Cost
            el_auto = compute_expected_loss(pd_scores[auto_approved_mask], lgd, ead).sum()
            el_review = compute_expected_loss(pd_scores[review_mask], lgd, ead).mean() * num_review_approved if num_reviewed > 0 else 0
            operational_cost = num_reviewed * review_cost_per_applicant
            
            total_policy_cost = el_auto + el_review + operational_cost
            
            if total_policy_cost < best_cost:
                best_cost = total_policy_cost
                best_policy = {
                    "auto_approve_threshold": float(auto_approve_thr),
                    "manual_review_threshold": float(review_thr),
                    "total_approval_rate": float(total_approval_rate),
                    "portfolio_bad_rate": float(total_bad_rate),
                    "total_policy_cost": float(total_policy_cost),
                    "operational_review_cost": float(operational_cost),
                    "auto_approved_count": num_auto_approved,
                    "manual_reviewed_count": num_reviewed,
                }
                
    return best_policy or {"error": "No policy meets constraints"}


def generate_fairness_tradeoff_table(
    y_true: pd.Series,
    pd_scores: np.ndarray,
    sensitive_features: pd.DataFrame,
    lgd: float = 0.45,
    ead: float = 1.0,
) -> pd.DataFrame:
    """
    Produces a tradeoff table showing expected loss vs fairness (Disparate Impact).
    DO NOT CLAIM OPTIMIZATION: this explicitly frames fairness as a surfaced policy tradeoff.
    """
    thresholds = np.linspace(0.01, 0.99, 40)
    results = []
    
    n_total = len(y_true)
    if n_total == 0:
        return pd.DataFrame()
        
    for thr in thresholds:
        approved_mask = pd_scores < thr
        overall_approval_rate = approved_mask.sum() / n_total
        el_portfolio = float(compute_expected_loss(pd_scores[approved_mask], lgd, ead).sum())
        
        min_di = 1.0
        
        for col in sensitive_features.columns:
            for val, group_idx in sensitive_features.groupby(col).groups.items():
                if len(group_idx) < 100: continue
                group_approved = approved_mask[group_idx].sum()
                group_approval_rate = group_approved / len(group_idx)
                di = group_approval_rate / max(overall_approval_rate, 1e-6)
                min_di = min(min_di, di)
        
        results.append({
            "threshold": float(thr),
            "expected_portfolio_loss": el_portfolio,
            "overall_approval_rate": float(overall_approval_rate),
            "min_disparate_impact": float(min_di),
        })
        
    return pd.DataFrame(results)

def evaluate_el_sensitivity_analysis(
    y_true: pd.Series,
    pd_scores: np.ndarray,
    lgd_assumptions: list[float] | None = None,
    ead: float = 1.0,
    policy_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Evaluates Expected Portfolio Loss under different Loss Given Default (LGD) assumptions 
    for a fixed approval threshold policy.
    Highlights stability of policy outcomes strictly for interview/audit credibility.
    """
    if lgd_assumptions is None:
        lgd_assumptions = [0.35, 0.45, 0.60]
        
    results = []
    
    approved_mask = pd_scores < policy_threshold
    num_approved = int(approved_mask.sum())
    
    if num_approved == 0:
        return pd.DataFrame()
        
    for lgd in lgd_assumptions:
        el_portfolio = float(compute_expected_loss(pd_scores[approved_mask], lgd, ead).sum())
        actual_portfolio_loss = float((y_true[approved_mask] * lgd * ead).sum())
        
        results.append({
            "lgd_assumption": float(lgd),
            "ead_assumption": float(ead),
            "policy_threshold": float(policy_threshold),
            "approved_count": num_approved,
            "expected_portfolio_loss": el_portfolio,
            "actual_portfolio_loss": actual_portfolio_loss,
            "el_variance_vs_lgd_0_45_base": el_portfolio - float(compute_expected_loss(pd_scores[approved_mask], 0.45, ead).sum())
        })
        
    return pd.DataFrame(results)


def apply_credit_policy(
    pd_score: float,
    auto_approve_threshold: float = 0.15,
    manual_review_threshold: float = 0.35,
) -> str:
    """
    Applies the credit decision policy to a given Probability of Default (PD) score.

    Args:
        pd_score: Predicted Probability of Default [0.0, 1.0].
        auto_approve_threshold: Scores below this are automatically approved.
        manual_review_threshold: Scores between auto_approve and this are sent to manual review.
                                 Scores above this are automatically declined.

    Returns:
        A string representing the decision tier.
        One of: 'AUTO_APPROVE', 'MANUAL_REVIEW', 'AUTO_DECLINE'.
    """
    if pd_score < auto_approve_threshold:
        return "AUTO_APPROVE"
    elif pd_score < manual_review_threshold:
        return "MANUAL_REVIEW"
    else:
        return "AUTO_DECLINE"
