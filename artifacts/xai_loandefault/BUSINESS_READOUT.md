# Business Readout - Loan Default Decisioning

## Executive Summary
- Champion model: **LightGBMEnsemble(3)** at threshold **0.58**. This model reduces expected misclassification cost by **38.09%** versus baseline on validation.
- Cost impact: baseline=25905.0, winner=16039.0, delta=9866.0.
- Default capture: baseline=0.399, winner=0.483, delta=0.084.

## Why It Matters To Business
- Better capture of risky applicants reduces charge-offs and expected credit losses.
- Controlled approval policy helps maintain growth while reducing portfolio risk.
- Explainability artifacts support risk committees and model governance.

## Decision Policy Translation (Per 1,000 Applicants)
- Baseline approvals: 705.2, rejections: 294.8
- Winner approvals: 855.3, rejections: 144.7
- Winner expected false negatives: 39.9 per 1k
- Winner expected false positives: 107.5 per 1k

## Interview Pitch (60-90 seconds)
I built a production-style credit-risk decisioning system for loan default prediction. I compared a baseline logistic regression against a LightGBM champion, optimized threshold using business costs, and validated both holdout and out-of-time behavior. The champion improved discriminatory power, reduced expected misclassification cost, and shipped with governance outputs including explainability, calibration, drift, segment fairness, and a model card.

## Resume Bullet (Example)
Built end-to-end explainable loan default decisioning pipeline (baseline vs champion), adding cost-optimized thresholding, OOT validation, calibration/drift/fairness monitoring, and business policy bands; reduced expected validation misclassification cost by 38.09% vs baseline.