# Model Card - Loan Default XAI

## Business Objective
Predict probability of repayment difficulty to support underwriting decisions, reduce default losses, and keep approvals fair and explainable.

## Target
TARGET=1 means client had payment difficulties; TARGET=0 means no observed payment difficulty in the defined observation window.

## Data Profile
- Train rows: 307511
- Test rows: 48744
- Target rate: 0.0807

## Validation Strategy
- Development split: stratified holdout with validation_size=0.2
- CV folds for tuning/selection: 5
- Optuna enabled: True
- Ensemble seeds: [42, 52, 62]
- OOT enabled: True
- OOT time column: DAYS_ID_PUBLISH
- OOT effective rows: 46164

## Winner
- Model: LightGBMEnsemble(3)
- Threshold: 0.5800
- False Positive Cost: 1.0
- False Negative Cost: 5.0

## Comparison Snapshot
- LightGBMEnsemble(3) (champion) | ROC-AUC=0.7921, PR-AUC=0.2770, KS=0.4509, BusinessCost=16039.0
- SGDLogisticBaseline (baseline) | ROC-AUC=0.5564, PR-AUC=0.0879, KS=0.1129, BusinessCost=25905.0

## Report Files
- Comparison: `model_comparison.csv`
- Calibration: `calibration_report.csv`
- Bootstrap CIs: `bootstrap_confidence_intervals.csv`
- Drift PSI: `drift_psi_report.csv`
- Segment Performance: `segment_performance_report.csv`
- Metric Plots:
  - model_comparison_bars: `comparison_bars.png`
  - oot_calibration: `oot_calibration.png`
  - oot_confusion: `oot_confusion.png`
  - oot_ks: `oot_ks.png`
  - oot_roc_pr: `oot_roc_pr.png`
  - oot_threshold_cost: `oot_threshold_cost.png`
  - validation_calibration: `validation_calibration.png`
  - validation_confusion: `validation_confusion.png`
  - validation_ks: `validation_ks.png`
  - validation_roc_pr: `validation_roc_pr.png`
  - validation_threshold_cost: `validation_threshold_cost.png`