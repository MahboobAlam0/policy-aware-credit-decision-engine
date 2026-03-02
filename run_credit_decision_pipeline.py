import argparse
import logging
from pathlib import Path

from training.config import PipelineConfig
from data.pipeline import load_and_preprocess_data
from training.model import build_pipeline, _lightgbm_optuna_search
from training.serialization import save_bundled_artifact
from explanations.shap_utils import export_global_importance, export_shap_reports
from policy.decision_logic import (
    search_dual_threshold_policy,
    generate_fairness_tradeoff_table,
    evaluate_el_sensitivity_analysis
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def main(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    
    # Optional override for faster testing
    if args.fast:
        LOGGER.info("Running in FAST mode (limited rows, no tuning)")
        config.baseline_max_train_rows = 50000
        config.optuna_enabled = False
        config.ensemble_seeds = (42,)

    artifacts_dir = config.ensure_artifacts_dir()
    
    LOGGER.info("=== PHASE 1: Data Processing (Respecting ECOA Protected Attributes) ===")
    LOGGER.info("Loading features and automatically excluding AGE/GENDER from training arrays...")
    X_train, X_valid, X_oot, y_train, y_valid, y_oot, ids_train, ids_valid, ids_oot, protected_valid = load_and_preprocess_data(config)
    
    LOGGER.info("Train shape: %s, Valid shape: %s", X_train.shape, X_valid.shape)
    
    LOGGER.info("=== PHASE 2: ML Probability of Default (PD) Estimation ===")
    tuned_params = None
    if config.optuna_enabled:
        LOGGER.info("Running Optuna hyperparameter optimization...")
        tuned_params, tuning_score, _ = _lightgbm_optuna_search(X_train, y_train, config)
        LOGGER.info(f"Tuning complete. Best ROC-AUC: {tuning_score:.4f}")
    
    LOGGER.info("Training Single Champion LightGBM Model for 1:1 Explanation Consistency...")
    champion_pipeline, _ = build_pipeline(
        model_key="champion",
        features=X_train,
        config=config,
        params_override=tuned_params,
        seed_override=config.random_state
    )
    
    champion_pipeline.fit(X_train, y_train)
    pd_scores_valid = champion_pipeline.predict_proba(X_valid)[:, 1]
    
    LOGGER.info("=== PHASE 3: Expected Loss & Policy Frontier Search ===")
    LOGGER.info("Sweeping for optimal Auto-Approve & Manual-Review dual thresholds...")
    
    policy_result = search_dual_threshold_policy(
        y_true=y_valid,
        pd_scores=pd_scores_valid,
        lgd=config.lgd_assumption,
        ead=config.ead_assumption,
        review_cost_per_applicant=config.policy_manual_review_unit_cost,
        review_approval_rate=config.policy_manual_review_approval_rate,
        min_approval_rate=config.policy_min_approval_rate,
        max_bad_rate=config.policy_max_bad_rate,
    )
    
    if "error" in policy_result:
        LOGGER.error("Policy Search Failed: %s", policy_result["error"])
        return
        
    thr_approve = policy_result["auto_approve_threshold"]
    thr_review = policy_result["manual_review_threshold"]
    
    LOGGER.info(f"Optimal Policy Constraints Met:")
    LOGGER.info(f" -> Auto-Approve applicants with PD < {thr_approve:.4f}")
    LOGGER.info(f" -> Route to Manual Review for {thr_approve:.4f} <= PD < {thr_review:.4f}")
    LOGGER.info(f" -> Auto-Decline applicants with PD >= {thr_review:.4f}")
    LOGGER.info(f" -> Projected Portfolio Bad Rate: {policy_result['portfolio_bad_rate']:.4f}")
    
    LOGGER.info("=== PHASE 4: Fairness vs Expected Loss Tradeoff Evaluation ===")
    # Extract sensitive attributes from the subset we kept for tracking
    LOGGER.info("Generating Disparate Impact vs Portfolio Cost tradeoff table over threshold sweeps...")
    fair_table = generate_fairness_tradeoff_table(
        y_true=y_valid,
        pd_scores=pd_scores_valid,
        sensitive_features=protected_valid[list(config.fairness_monitor_columns)],
        lgd=config.lgd_assumption,
        ead=config.ead_assumption
    )
    LOGGER.info("\nTradeoff Frontier (Sample):\n%s", fair_table.iloc[::10].to_string())
    
    LOGGER.info("=== PHASE 5: Model-Explainer Artifact Coupling ===")
    LOGGER.info("Generating TreeExplainer and Reason Code mappings...")
    export_global_importance(champion_pipeline, artifacts_dir, top_n=20)
    
    # We load SHAP in background payload to bundle the explainer
    import shap
    explainer = shap.TreeExplainer(champion_pipeline.named_steps["classifier"])
    
    reason_mapper = {
        "ext_source_1": "External Credit Bureau Score",
        "ext_source_2": "External Credit Bureau Score",
        "ext_source_3": "External Credit Bureau Score",
        "days_employed": "Employment History Length",
        "amt_annuity": "Debt Burden (Annuity)",
        "days_birth": "Applicant Age Profile",
        "amt_credit": "Total Credit Amount Requested",
        "amt_goods_price": "Asset Value to Loan Ratio",
    }
    
    LOGGER.info("Saving IMMUTABLE Bundled Artifact (Pipeline + Model + Explainer)...")
    save_bundled_artifact(
        pipeline=champion_pipeline.named_steps["preprocessor"],
        model=champion_pipeline.named_steps["classifier"],
        explainer=explainer,
        reason_mapper=reason_mapper,
        output_dir=artifacts_dir,
        version="1.0.0"
    )
    
    LOGGER.info("=== PIPELINE COMPLETION ===")
    LOGGER.info("All final artifacts written to: %s", artifacts_dir)
    LOGGER.info(f"Run `python -m serving.test_failure_mode` to test API rejection safety.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Decision System Runner")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (limit rows, disable tuning) for quick testing")
    args = parser.parse_args()
    main(args)
