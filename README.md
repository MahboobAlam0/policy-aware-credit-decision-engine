---
title: Policy-Aware Credit Decisioning Engine
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Policy-Aware Credit Decisioning Engine — Explainable AI for Loan Default Risk Under Fairness Constraints

## Overview

This repository implements a **policy-aware credit decision system**, not just a loan default prediction model.

In real financial institutions, machine learning models **do not approve or decline loans directly**.  
They estimate risk, which is then evaluated by business rules, regulatory constraints, and human governance.

This project demonstrates how to design such a system end-to-end:

- Probability of Default (PD) estimation
- Policy-based decisioning (approve / review / decline)
- Expected Loss optimization
- Decision-safe explainability
- Fairness monitoring and tradeoff analysis

The system is intentionally designed as an **Applied ML portfolio project** that mirrors how credit decisioning works in regulated environments.

---

## Problem Statement

When a customer applies for credit, a lender must answer three distinct questions:

1. **How risky is this applicant?**  
   → What is the probability they will default?

2. **What action should be taken?**  
   → Approve, route to manual review, or decline?

3. **How can the decision be explained and audited?**  
   → For regulators, internal governance, and applicants.

Most ML projects stop at the first question.

**This system explicitly separates and solves all three.**

---

## High-Level System Flow
``` mermaid
Applicant Features
↓
Probability of Default (PD)
↓
Policy & Expected Loss Rules
↓
Decision (Approve / Review / Decline)
↓
Decision-Safe Explanations + Fairness Monitoring
```


---

## System Design

### 1. Feature Construction & Governance

Applicant features include credit bureau attributes, repayment history, employment data, and loan characteristics.

Protected attributes (e.g., age, gender):

- are **excluded from model training**
- are retained **only for post-hoc monitoring**

This design prioritizes governance clarity and reduces regulatory risk in a portfolio context.  
No claim is made that exclusion alone removes bias.

---

### 2. Risk Estimation (Probability of Default)

A single **LightGBM** model estimates the **Probability of Default (PD)**.

Key properties:

- Outputs a calibrated probability (0–1)
- No business thresholds are applied inside the model
- A single champion model is used (no ensembling)

The model answers exactly one question:

> “How likely is this applicant to default?”

---

### 3. Policy-Based Decisioning

Decisions are made by a **separate policy layer**, not by the ML model.

The policy evaluates **Expected Loss (EL)**:
 **Expected Loss = PD × LGD × EAD**


Assumptions (explicitly documented):

- LGD is fixed (default: 0.45)
- EAD is normalized for demonstration purposes

The system evaluates a wide range of threshold combinations to **surface tradeoffs** between:

- expected portfolio loss
- approval rate
- fairness metrics

A realistic **dual-threshold policy** is applied:

| PD Range | Decision |
|--------|----------|
| Low risk | Auto-Approve |
| Medium risk | Manual Review |
| High risk | Auto-Decline |

Thresholds are policy choices informed by analysis, not claims of optimality.

---

### 4. Explainability (XAI)

Each decision includes **3–5 business-friendly reason codes**, generated using SHAP.

Design principles:

- The explainer is **bundled immutably** with the trained model
- The scoring model and explainer can never drift apart
- Outputs are limited to decision-safe explanations

Important clarification:

> SHAP explains model behavior, not causal relationships.

Explanations are intended for underwriting review, adverse action communication, and monitoring.

---

### 5. Fairness & Governance

The system does **not** automatically “fix” bias.

Instead, it:

- computes approval and error-rate disparities across demographic groups
- surfaces **fairness vs. profitability tradeoffs**
- leaves final decisions to human governance

This reflects real-world credit risk practice, where fairness is a **policy decision**, not a purely technical one.

---

## What Makes This Project Different

| Typical ML Project | This System |
|-------------------|-------------|
| Binary classification | Probability-based risk scoring |
| Fixed thresholds | Policy-driven decisioning |
| Model = decision | Model feeds a policy engine |
| XAI as an afterthought | XAI bundled with the model |
| Fairness ignored or hidden | Fairness surfaced as tradeoffs |

---

## Project Structure

```
data/           →  Loads and processes raw applicant data
training/       →  Trains the AI model and tunes its parameters
policy/         →  Business rules: thresholds, expected loss, fairness checks
explanations/   →  Generates human-readable reason codes from SHAP values
monitoring/     →  Evaluation metrics, calibration, fairness reports
serving/        →  API contracts and failure safety tests
app.py          →  The backend server (FastAPI)
frontend.py     →  The interactive demo dashboard (Streamlit)
```

The **FastAPI service** is the core system.  
The Streamlit dashboard is optional and used only for demonstration.

---

## Running the System

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model and Generate Artifacts

```bash
python run_credit_decision_pipeline.py
```

For a faster test run (limited data, no hyperparameter tuning):

```bash
python run_credit_decision_pipeline.py --fast
```

### 3. Launch the Backend API

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can explore the endpoints interactively at `http://127.0.0.1:8000/docs`.

### 4. Launch the Demo Dashboard

In a separate terminal:

```bash
streamlit run frontend.py
```

Open `http://localhost:8501` to interact with the credit decision system.

---

## Generated Artifacts

When the pipeline runs, it produces a full audit trail:

| Artifact | What It Contains |
|---|---|
| `coupled_model_explainer.pkl` | Immutable bundle: preprocessor + model + explainer (locked together) |
| `model_comparison.csv` | Side-by-side comparison of baseline vs. champion model |
| `segment_performance_report.csv` | Fairness metrics by group |
| `bootstrap_confidence_intervals.csv` | Statistical uncertainty estimates |
| `decision_policy_bands.csv` | The auto-approve / review / decline thresholds |
| `shap_global_importance.csv` | Which features matter most across all applicants |
| `plots/` | ROC curves, calibration plots, confusion matrices, cost curves |

---

## Known Limitations (Intentionally Documented)

This project is a realistic prototype, not a production-certified system. 
**Key limitations:**

- **Temporal leakage:** Auxiliary feature aggregation is not strictly point-in-time
- **Proxy time split:** Out-of-time validation uses a proxy timestamp
- **Fixed loss assumptions:** LGD and EAD are fixed assumptions
- **Post-hoc explanations:** SHAP explanations are post-hoc and non-causal

These limitations are documented intentionally. In regulated systems, acknowledging limitations is mandatory.

---