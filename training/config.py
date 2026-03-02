from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Central configuration for training, evaluation, and explainability."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"

    random_state: int = 42
    validation_size: float = 0.2
    include_auxiliary_tables: bool = True
    temporal_join_enforced: bool = True
    baseline_max_train_rows: int = 0
    baseline_epochs: int = 25
    cv_n_splits: int = 5
    optuna_enabled: bool = True
    optuna_n_trials: int = 60
    optuna_timeout_seconds: int = 3600
    tuning_max_rows: int = 120000
    ensemble_seeds: tuple[int, ...] = (42,)
    oot_fraction: float = 0.15
    oot_time_column: str = "DAYS_ID_PUBLISH"
    calibration_bins: int = 10
    bootstrap_iterations: int = 300
    min_segment_size: int = 1000

    # ECOA / FAIR LENDING GOVERNANCE:
    # Age (DAYS_BIRTH, AGE_YEARS) is heavily predictive but presents severe disparate impact risks.
    # While ECOA theoretically allows age in empirical systems under strict monotonic rules,
    # we explicitly EXCLUDE it from training features to structurally prevent age discrimination
    # and safely prioritize regulatory compliance over marginal AUC lift. 
    # These attributes are retained strictly for post-hoc disparate impact monitoring.
    protected_attributes: tuple[str, ...] = (
        "CODE_GENDER",
        "NAME_FAMILY_STATUS",
        "DAYS_BIRTH",
        "AGE_YEARS",
        "FLAG_OWN_REALTY",
    )
    fairness_monitor_columns: tuple[str, ...] = (
        "CODE_GENDER",
        "NAME_FAMILY_STATUS",
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
    )

    false_positive_cost: float = 1.0
    false_negative_cost: float = 5.0

    # Expected-loss assumptions for policy optimization.
    lgd_assumption: float = 0.45
    ead_assumption: float = 1.0
    policy_manual_review_approval_rate: float = 0.35
    policy_manual_review_unit_cost: float = 0.02
    policy_decline_opportunity_cost: float = 0.0
    policy_min_approval_rate: float = 0.35
    policy_max_bad_rate: float = 0.12
    policy_grid_step: float = 0.01

    top_n_features: int = 30
    shap_background_size: int = 2000
    shap_explain_size: int = 1000
    local_explanations_count: int = 5
    n_jobs: int = -1

    lightgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "objective": "binary",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "home-credit-default-risk"

    @property
    def train_path(self) -> Path:
        return self.data_dir / "application_train.csv"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "application_test.csv"

    @property
    def bureau_path(self) -> Path:
        return self.data_dir / "bureau.csv"

    @property
    def previous_application_path(self) -> Path:
        return self.data_dir / "previous_application.csv"

    @property
    def pos_cash_path(self) -> Path:
        return self.data_dir / "POS_CASH_balance.csv"

    @property
    def installments_path(self) -> Path:
        return self.data_dir / "installments_payments.csv"

    @property
    def credit_card_path(self) -> Path:
        return self.data_dir / "credit_card_balance.csv"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts" / "xai_loandefault"

    def ensure_artifacts_dir(self, override_path: Path | None = None) -> Path:
        out_dir = override_path or self.artifacts_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
