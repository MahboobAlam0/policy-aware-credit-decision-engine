# frontend.py
# Professional dark-themed dashboard for the Credit Decision System API.

import streamlit as st
import requests
import joblib
import numpy as np
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Policy-Aware Credit Decisioning Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Cohesive dark fintech dashboard
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Force dark background everywhere */
    .stApp, .main, section[data-testid="stSidebar"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide default streamlit header/footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Top banner ── */
    .top-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .top-banner h1 {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
    }
    .top-banner p {
        font-size: 0.88rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
        max-width: 720px;
    }
    .badge {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        color: #60a5fa;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* ── Architecture pipeline ── */
    .pipeline-container {
        display: flex;
        align-items: stretch;
        gap: 0;
        margin: 1rem 0;
    }
    .pipeline-step {
        flex: 1;
        background: #1e293b;
        border: 1px solid #334155;
        padding: 0.9rem 0.75rem;
        text-align: center;
        position: relative;
    }
    .pipeline-step:first-child {
        border-radius: 8px 0 0 8px;
    }
    .pipeline-step:last-child {
        border-radius: 0 8px 8px 0;
    }
    .pipeline-step .step-num {
        font-size: 0.62rem;
        font-weight: 700;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }
    .pipeline-step .step-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.15rem;
    }
    .pipeline-step .step-desc {
        font-size: 0.68rem;
        color: #64748b;
        line-height: 1.35;
    }
    .pipeline-arrow {
        display: flex;
        align-items: center;
        color: #475569;
        font-size: 1rem;
        padding: 0 0.15rem;
    }

    /* ── Result cards ── */
    .result-section {
        margin-top: 1rem;
    }
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
    }
    .card-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    /* PD Score */
    .pd-score {
        font-size: 3rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .pd-bar-track {
        width: 100%;
        height: 6px;
        background: #334155;
        border-radius: 3px;
        margin: 0.75rem 0 0.5rem 0;
        overflow: hidden;
    }
    .pd-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .pd-bar-low { background: linear-gradient(90deg, #059669, #34d399); }
    .pd-bar-mid { background: linear-gradient(90deg, #d97706, #fbbf24); }
    .pd-bar-high { background: linear-gradient(90deg, #dc2626, #f87171); }

    .pd-scale {
        display: flex;
        justify-content: space-between;
        font-size: 0.62rem;
        color: #64748b;
    }

    /* Decision */
    .decision-outcome {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .decision-approve { color: #34d399; }
    .decision-review { color: #fbbf24; }
    .decision-decline { color: #f87171; }

    .decision-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    .indicator-approve { background: #059669; box-shadow: 0 0 8px rgba(5,150,105,0.4); }
    .indicator-review { background: #d97706; box-shadow: 0 0 8px rgba(217,119,6,0.4); }
    .indicator-decline { background: #dc2626; box-shadow: 0 0 8px rgba(220,38,38,0.4); }

    .decision-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.55;
        margin-top: 0.5rem;
    }
    .decision-tier-label {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
        margin-top: 0.75rem;
    }
    .tier-approve { background: rgba(5,150,105,0.15); color: #34d399; }
    .tier-review { background: rgba(217,119,6,0.15); color: #fbbf24; }
    .tier-decline { background: rgba(220,38,38,0.15); color: #f87171; }

    /* Reason codes */
    .reason-row {
        display: flex;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(51,65,85,0.5);
    }
    .reason-row:last-child { border-bottom: none; }
    .reason-tag {
        font-size: 0.6rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        letter-spacing: 0.04em;
        min-width: 85px;
        text-align: center;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }
    .tag-risk-up {
        background: rgba(220,38,38,0.12);
        color: #f87171;
    }
    .tag-risk-down {
        background: rgba(5,150,105,0.12);
        color: #34d399;
    }
    .reason-text {
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    /* Disclaimer */
    .footer-note {
        font-size: 0.7rem;
        color: #475569;
        border-top: 1px solid #1e293b;
        padding-top: 0.75rem;
        margin-top: 1.5rem;
        line-height: 1.5;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1rem 0 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/credit-decision")

# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_expected_features():
    try:
        bundle = joblib.load("artifacts/xai_loandefault/coupled_model_explainer.pkl")
        return bundle["pipeline"].feature_names_in_.tolist()
    except Exception:
        return []

expected_features = load_expected_features()

# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="top-banner">
    <div>
        <span class="badge">LightGBM Champion</span>
        <span class="badge">SHAP TreeExplainer</span>
        <span class="badge">Dual-Threshold Policy</span>
    </div>
    <h1>Policy-Aware Credit Decisioning Engine</h1>
    <p>
        End-to-end credit underwriting prototype that separates probability-of-default
        estimation from business decision logic. Uses coupled ML artifacts with
        SHAP-based reason codes for regulatory audit transparency.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Architecture Pipeline
# ---------------------------------------------------------------------------

with st.expander("System Architecture", expanded=False):
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="step-num">Step 1</div>
            <div class="step-title">Feature Input</div>
            <div class="step-desc">Raw applicant data. Protected attributes excluded from model training.</div>
        </div>
        <div class="pipeline-arrow">&rarr;</div>
        <div class="pipeline-step">
            <div class="step-num">Step 2</div>
            <div class="step-title">Preprocessing</div>
            <div class="step-desc">ColumnTransformer: median imputation for numerics, OHE for categoricals.</div>
        </div>
        <div class="pipeline-arrow">&rarr;</div>
        <div class="pipeline-step">
            <div class="step-num">Step 3</div>
            <div class="step-title">PD Estimation</div>
            <div class="step-desc">Single champion LightGBM classifier outputs calibrated default probability.</div>
        </div>
        <div class="pipeline-arrow">&rarr;</div>
        <div class="pipeline-step">
            <div class="step-num">Step 4</div>
            <div class="step-title">Policy Layer</div>
            <div class="step-desc">Expected-loss dual-threshold: approve, manual review, or decline.</div>
        </div>
        <div class="pipeline-arrow">&rarr;</div>
        <div class="pipeline-step">
            <div class="step-num">Step 5</div>
            <div class="step-title">XAI Reason Codes</div>
            <div class="step-desc">TreeExplainer SHAP values mapped to business-friendly explanations.</div>
        </div>
    </div>
    <div style="font-size:0.75rem;color:#64748b;margin-top:0.25rem;">
        The pipeline, classifier, and explainer are bundled into a single immutable artifact to prevent model-explainer drift.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Applicant Profile")

st.sidebar.markdown('<div class="sidebar-section-label">External Bureau Scores</div>', unsafe_allow_html=True)
ext_source_1 = st.sidebar.slider("External Source 1", 0.0, 1.0, 0.5, key="es1")
ext_source_2 = st.sidebar.slider("External Source 2", 0.0, 1.0, 0.6, key="es2")
ext_source_3 = st.sidebar.slider("External Source 3", 0.0, 1.0, 0.4, key="es3")

st.sidebar.markdown('<div class="sidebar-section-label">Employment & Demographics</div>', unsafe_allow_html=True)
years_employed = st.sidebar.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
age_years = st.sidebar.number_input("Age (years)", min_value=18.0, max_value=100.0, value=35.0, step=1.0)

days_employed = -1.0 * years_employed * 365
days_birth = -1.0 * age_years * 365

st.sidebar.markdown('<div class="sidebar-section-label">Loan Parameters</div>', unsafe_allow_html=True)
amt_credit = st.sidebar.number_input("Credit Amount ($)", min_value=1000, value=150000, step=5000)
amt_annuity = st.sidebar.number_input("Annuity Amount ($)", min_value=100, value=15000, step=1000)
amt_goods_price = st.sidebar.number_input("Asset Value ($)", min_value=1000, value=150000, step=5000)

st.sidebar.markdown('<div class="sidebar-section-label">Applicant Details</div>', unsafe_allow_html=True)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
own_car = st.sidebar.selectbox("Owns Vehicle", ["Yes", "No"])
own_realty = st.sidebar.selectbox("Owns Property", ["Yes", "No"])

# ---------------------------------------------------------------------------
# Build Payload
# ---------------------------------------------------------------------------

gender_map = {"Female": 0.0, "Male": 1.0, "Other": 2.0}

payload_features = {
    "EXT_SOURCE_1": float(ext_source_1),
    "EXT_SOURCE_2": float(ext_source_2),
    "EXT_SOURCE_3": float(ext_source_3),
    "DAYS_EMPLOYED": float(days_employed),
    "AMT_ANNUITY": float(amt_annuity),
    "DAYS_BIRTH": float(days_birth),
    "AMT_CREDIT": float(amt_credit),
    "AMT_GOODS_PRICE": float(amt_goods_price),
    "CODE_GENDER": gender_map.get(gender, 1.0),
    "FLAG_OWN_CAR": 1.0 if own_car == "Yes" else 0.0,
    "FLAG_OWN_REALTY": 1.0 if own_realty == "Yes" else 0.0,
}

# Only send features we have values for.
# The backend pads any missing columns with NaN internally (app.py L85-89).
payload = {"features": payload_features}

# ---------------------------------------------------------------------------
# Action Bar
# ---------------------------------------------------------------------------

col_btn, col_inspect = st.columns([2, 1])
with col_btn:
    evaluate = st.button("Run Credit Decision Pipeline", type="primary", use_container_width=True)
with col_inspect:
    with st.expander("Inspect Payload", expanded=False):
        visible = {k: v for k, v in payload_features.items() if not (isinstance(v, float) and np.isnan(v))}
        st.json({"features": visible})

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if evaluate:
    with st.spinner("Processing..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            decision = result.get("decision_tier", "UNKNOWN")
            pd_score = result.get("pd_score", 0.0)
            reason_codes = result.get("reason_codes", [])

            # Determine color scheme
            if decision == "AUTO_APPROVE":
                bar_class = "pd-bar-low"
                dec_class = "decision-approve"
                ind_class = "indicator-approve"
                tier_class = "tier-approve"
                dec_text = "Approved"
                dec_desc = "Applicant risk is below the auto-approve threshold. The expected loss falls within acceptable portfolio bounds, requiring no further underwriter intervention."
            elif decision == "MANUAL_REVIEW":
                bar_class = "pd-bar-mid"
                dec_class = "decision-review"
                ind_class = "indicator-review"
                tier_class = "tier-review"
                dec_text = "Manual Review"
                dec_desc = "Applicant falls within the indeterminate risk band between auto-approve and auto-decline thresholds. Routed to a human underwriter for subjective assessment."
            else:
                bar_class = "pd-bar-high"
                dec_class = "decision-decline"
                ind_class = "indicator-decline"
                tier_class = "tier-decline"
                dec_text = "Declined"
                dec_desc = "Applicant risk exceeds the maximum acceptable expected-loss boundary. Application is automatically declined per policy constraints."

            pd_pct = pd_score * 100

            st.markdown("---")

            col_pd, col_decision, col_reasons = st.columns([1, 1.3, 1.5])

            with col_pd:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">Probability of Default</div>
                    <div class="pd-score">{pd_pct:.1f}%</div>
                    <div class="pd-bar-track">
                        <div class="pd-bar-fill {bar_class}" style="width:{min(pd_pct, 100):.1f}%"></div>
                    </div>
                    <div class="pd-scale">
                        <span>0%</span>
                        <span>Low Risk</span>
                        <span>50%</span>
                        <span>High Risk</span>
                        <span>100%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_decision:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">Policy Decision</div>
                    <div class="decision-outcome {dec_class}">
                        <span class="decision-indicator {ind_class}"></span>
                        {dec_text}
                    </div>
                    <div class="decision-desc">{dec_desc}</div>
                    <div class="decision-tier-label {tier_class}">{decision}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_reasons:
                st.markdown('<div class="card"><div class="card-label">Top Risk Factors (SHAP)</div></div>', unsafe_allow_html=True)
                if reason_codes:
                    for reason in reason_codes:
                        if "INCREASED" in reason:
                            tag_html = '<span style="font-size:0.6rem;font-weight:700;padding:0.15rem 0.5rem;border-radius:3px;letter-spacing:0.04em;min-width:85px;text-align:center;margin-right:0.75rem;background:rgba(220,38,38,0.12);color:#f87171;">RISK UP</span>'
                        else:
                            tag_html = '<span style="font-size:0.6rem;font-weight:700;padding:0.15rem 0.5rem;border-radius:3px;letter-spacing:0.04em;min-width:85px;text-align:center;margin-right:0.75rem;background:rgba(5,150,105,0.12);color:#34d399;">RISK DOWN</span>'
                        factor = reason.split("(")[0].strip() if "(" in reason else reason
                        st.markdown(f'{tag_html}<span style="font-size:0.82rem;color:#cbd5e1;">{factor}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#64748b;font-size:0.85rem;">No significant risk factors identified.</span>', unsafe_allow_html=True)

            st.markdown("""
            <div class="footer-note">
                Explanations are post-hoc local approximations generated by SHAP TreeExplainer.
                They are provided for adverse-action transparency and audit support, not as causal
                attributions or regulatory compliance certifications. The model-explainer pair is
                immutably coupled to prevent drift between scoring and explanation artifacts.
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error(
                f"Unable to reach the backend at {API_URL}. "
                "Ensure the FastAPI server is running with: uvicorn app:app --reload"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"Server error: {e.response.status_code} -- {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
