import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_pdf_report, CreditDataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css(os.path.join("assets", "custom.css"))

# Helper to load model and preprocessor
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load(os.path.join("model", "best_model.pkl"))
        preprocessor = joblib.load(os.path.join("model", "preprocessor.pkl"))
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Ensure you have run train.py first.")
        return None, None

@st.cache_data
def load_metrics():
    metrics_path = os.path.join("model", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset():
    data_path = os.path.join("data", "credit_risk_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

# Load resources
model, preprocessor = load_model_artifacts()
metrics = load_metrics()
df = load_dataset()

# Sidebar Navigation
st.sidebar.markdown(
    '<div style="text-align: center; margin-bottom: 20px;">'
    '<h2 class="gradient-text-blue" style="margin-bottom: 5px; font-size: 26px;">CodeAlpha</h2>'
    '<p style="color: #64748b; font-size: 13px; font-weight: 500;">Credit Scoring System</p>'
    '</div>', 
    unsafe_allow_html=True
)

st.sidebar.markdown("<hr style='margin-top: 0; border-color: rgba(255,255,255,0.05);' />", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation", 
    ["Dashboard", "Credit Prediction", "Analytics"],
    index=0
)

st.sidebar.markdown("<br/><br/><br/>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="background-color: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">'
    '<h4 style="margin: 0; font-size: 14px; color: #f8fafc;">System Status</h4>'
    '<p style="margin: 5px 0 0 0; font-size: 12px; color: #10b981;">● Online & Secured</p>'
    f'<p style="margin: 3px 0 0 0; font-size: 11px; color: #94a3b8;">Active Model: {metrics["best_model_name"] if metrics else "None"}</p>'
    '</div>',
    unsafe_allow_html=True
)

# PAGE A: DASHBOARD
if page == "Dashboard":
    st.markdown('<h1 class="gradient-text-blue" style="font-size: 38px; margin-bottom: 5px;">Credit Risk Management Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 16px; margin-bottom: 25px;">Automated underwriting engine and model benchmarking suite</p>', unsafe_allow_html=True)
    
    if df is not None and metrics is not None:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_apps = len(df)
            st.markdown(
                f'<div class="metric-card">'
                f'<p style="margin: 0; color: #94a3b8; font-size: 14px; font-weight: 600;">TOTAL PORTFOLIO CLIENTS</p>'
                f'<h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: 700; color: #f8fafc;">{total_apps:,}</h2>'
                f'<p style="margin: 5px 0 0 0; color: #64748b; font-size: 12px;">Active credit files analyzed</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with col2:
            avg_income = df['person_income'].mean()
            st.markdown(
                f'<div class="metric-card">'
                f'<p style="margin: 0; color: #94a3b8; font-size: 14px; font-weight: 600;">AVERAGE ANNUAL INCOME</p>'
                f'<h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: 700; color: #f8fafc;">₹{avg_income:,.0f}</h2>'
                f'<p style="margin: 5px 0 0 0; color: #64748b; font-size: 12px;">Pre-tax household earnings</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with col3:
            default_rate = df['loan_status'].mean() * 100
            st.markdown(
                f'<div class="metric-card">'
                f'<p style="margin: 0; color: #94a3b8; font-size: 14px; font-weight: 600;">HISTORICAL DEFAULT RATE</p>'
                f'<h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: 700; color: #ef4444;">{default_rate:.2f}%</h2>'
                f'<p style="margin: 5px 0 0 0; color: #64748b; font-size: 12px;">Bad risk segment size</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with col4:
            avg_loan = df['loan_amnt'].mean()
            st.markdown(
                f'<div class="metric-card">'
                f'<p style="margin: 0; color: #94a3b8; font-size: 14px; font-weight: 600;">AVERAGE REQUESTED LOAN</p>'
                f'<h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: 700; color: #f8fafc;">₹{avg_loan:,.0f}</h2>'
                f'<p style="margin: 5px 0 0 0; color: #64748b; font-size: 12px;">Average disbursement size</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br/><br/>", unsafe_allow_html=True)

        # Overview & Model performance splits
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("### System Description & Objectives")
            st.markdown(
                "This Credit Scoring System employs machine learning to evaluate the risk profiles of credit applicants. "
                "By analyzing personal financial details, loan structures, and historical relationships, the engine generates "
                "risk indicators and supports instant loan underwriting decisions."
            )
            st.markdown(
                "#### Key Capabilities:\n"
                "- **Instant Risk Assessments**: Calculates default risk and approval confidence scores.\n"
                "- **Explainable Recommendations**: Identifies risk drivers like high Debt-to-Income (DTI) or default histories.\n"
                "- **Dynamic Policy Simulation**: Allows underwriters to evaluate parameter sensitivities interactively.\n"
                "- **Professional Audit Exports**: Generates standard PDF documentation for credit files."
            )
            
        with right_col:
            st.markdown("### Model Comparison Benchmark")
            
            # Format comparison metrics
            comparison_rows = []
            for m_name, m_metrics in metrics['metrics'].items():
                comparison_rows.append({
                    'Model': m_name,
                    'Accuracy': f"{m_metrics['Accuracy']:.2%}",
                    'Precision': f"{m_metrics['Precision']:.2%}",
                    'Recall': f"{m_metrics['Recall']:.2%}",
                    'F1 Score': f"{m_metrics['F1 Score']:.2%}",
                    'ROC-AUC': f"{m_metrics['ROC-AUC']:.2%}"
                })
            
            st.table(pd.DataFrame(comparison_rows))
            
            st.info(
                f"**Auto-Selection Result:** The system has chosen **{metrics['best_model_name']}** "
                f"as the active scoring algorithm based on the highest F1 Score to manage credit default class imbalances."
            )

    else:
        st.warning("Model metrics or dataset files are missing. Please execute the training script (`train.py`) first.")

# PAGE B: CREDIT PREDICTION
elif page == "Credit Prediction":
    st.markdown('<h1 class="gradient-text-blue" style="font-size: 38px; margin-bottom: 5px;">Credit Risk Underwriting Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 16px; margin-bottom: 25px;">Enter applicant metrics to evaluate approval probability and risk scores</p>', unsafe_allow_html=True)
    
    if model is None or preprocessor is None:
        st.error("Missing prediction artifacts. Make sure you run `train.py` before simulating.")
    else:
        # Form columns
        col1, col2 = st.columns([2, 1.3])
        
        with col1:
            st.markdown("#### Applicant Personal & Financial Profile")
            
            # Setup interactive input widgets
            form_age = st.slider("Applicant Age (Years)", min_value=18, max_value=90, value=28, step=1)
            form_income = st.number_input("Annual Income (₹)", min_value=4000, max_value=50000000, value=55000, step=1000)
            form_emp_length = st.slider("Employment Length (Years)", min_value=0, max_value=min(40, max(0, form_age - 16)), value=4, step=1)
            
            st.markdown("#### Requested Loan Details")
            form_loan_amt = st.number_input("Requested Loan Amount (₹)", min_value=500, max_value=10000000, value=10000, step=500)
            form_int_rate = st.slider("Loan Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.5, step=0.1)
            form_intent = st.selectbox("Loan Purpose/Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
            
            st.markdown("#### Credit History Profile")
            form_cred_hist_length = st.slider("Credit History Length (Years)", min_value=2, max_value=min(30, max(2, form_age - 15)), value=6, step=1)
            form_home_ownership = st.selectbox("Home Ownership Status", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
            form_prev_default = st.radio("Has Historical Default on File?", ['N', 'Y'], horizontal=True)

            form_cibil = st.slider(
                                "CIBIL Score",
                                min_value=300,
                                max_value=900,
                                value=750,
                                step=1
                            )
            
            
            # Automatic Debt-to-Income Calculation
            dti = round(form_loan_amt / form_income, 2)
            st.markdown(f"**Calculated Debt-to-Income (DTI) Ratio:** `{dti:.0%}` (Loan Amount / Annual Income)")
            
            # Predict Button
            predict_clicked = st.button("Evaluate Credit Risk")
            
        with col2:
            st.markdown("#### Credit Risk Evaluation Report")
            
            if predict_clicked:
                # Prepare single row DataFrame matching the training layout
                input_data = pd.DataFrame({
                    'person_age': [form_age],
                    'person_income': [form_income],
                    'person_home_ownership': [form_home_ownership],
                    'person_emp_length': [float(form_emp_length)],
                    'loan_intent': [form_intent],
                    'loan_amnt': [form_loan_amt],
                    'loan_int_rate': [form_int_rate],
                    'loan_percent_income': [dti],
                    'cb_person_default_on_file': [form_prev_default],
                    'cb_person_cred_hist_length': [form_cred_hist_length]
                })
                
                # Transform data
                transformed_input = preprocessor.transform(input_data)
                
                # Predict probability of default (Class 1)
                default_prob = model.predict_proba(transformed_input)[0, 1]
                
                # Calculations
                approval_prob = (1 - default_prob) * 100

                ml_score = int(
                    300 + ((1 - default_prob) * 600)
                )

                credit_score = int(
                    (0.7 * ml_score) + (0.3 * form_cibil)
                )                            
                # Risk Threshold classification
                if credit_score >= 750:
                    status = "Excellent Credit"
                    status_class = "risk-good"

                elif credit_score >= 700:
                    status = "Good Credit"
                    status_class = "risk-good"

                elif credit_score >= 650:
                    status = "Fair Credit"
                    status_class = "risk-good"

                elif credit_score >= 600:
                    status = "Poor Credit"
                    status_class = "risk-bad"

                else:
                    status = "Very Poor Credit"
                    status_class = "risk-bad"

                if credit_score >= 650:
                    recommendation = (
                        f"Applicant demonstrates strong repayment capacity and qualifies for lending. "
                        f"The calculated Credit Score is {credit_score}, placing the applicant in the {status} category."
                        )
                else:
                    recommendation = (
                        f"Applicant falls below underwriting threshold. "
                        f"The calculated Credit Score is {credit_score}, indicating elevated default risk."
                        )
                # Display output cards
                st.markdown(
                    f'<div class="prediction-card">'
                    f'<h3 style="margin: 0 0 15px 0; color: #94a3b8; font-size: 16px;">ASSESSMENT RESULT</h3>'
                    f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">'
                    f'<span style="font-size: 20px; font-weight: 600; color: #f8fafc;">Risk Category</span>'
                    f'<span class="risk-badge {status_class}">{status}</span>'
                    f'</div>'
                    f'<div class="detail-item">'
                    f'<span class="detail-label">Approval Probability</span>'
                    f'<span class="detail-value" style="color: {"#10b981" if default_prob < 0.35 else "#ef4444"}; font-size: 20px;">{approval_prob:.1f}%</span>'
                    f'</div>'
                    
                    f'<div class="detail-item">'
                    f'<span class="detail-label">Credit Score</span>'
                    f'<span class="detail-value">{credit_score} / 900</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Render Risk Score Gauge
                st.markdown("##### Credit Risk Score Meter")
                color_fill = "gauge-fill-good" if default_prob < 0.35 else "gauge-fill-bad"
                gauge_percent = ((credit_score - 300) / 600) * 100
                st.markdown(
                    f'<div class="gauge-container">'
                    f'<div class="gauge-fill {color_fill}" style="width: {gauge_percent}%;"></div>'
                    f'</div>'
                    f'<div style="display: flex; justify-content: space-between; color: #64748b; font-size: 12px; font-weight: 600; margin-bottom: 25px;">'
                    f'<span>LOW CREDIT (300)</span>'
                    f'<span>FAIR (600)</span>'
                    f'<span>HIGH CREDIT (900)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("##### System Underwriting Notes")
                st.info(recommendation)
                
                # Calculate eligible loan amount
                if credit_score >= 750:
                        eligible_amount = int(form_income * 5)

                elif credit_score >= 700:
                        eligible_amount = int(form_income * 4)

                elif credit_score >= 650:
                    eligible_amount = int(form_income * 3)

                else:
                    eligible_amount = int(form_income * 1.5)

                st.success(
                    f"Estimated Eligible Loan Amount: ₹{eligible_amount:,.0f}"
                )
                
                # PDF Generation Setup
                client_dict = {
                    'person_age': form_age,
                    'person_income': form_income,
                    'person_home_ownership': form_home_ownership,
                    'person_emp_length': form_emp_length,
                    'loan_intent': form_intent,
                    'loan_amnt': form_loan_amt,
                    'loan_int_rate': form_int_rate,
                    'cb_person_cred_hist_length': form_cred_hist_length,
                    'cb_person_default_on_file': form_prev_default,
                    'loan_percent_income': dti
                }
                risk_results_dict = {
                    'status': status,
                    'score': credit_score,
                    'probability': round(default_prob * 100, 2),
                    'recommendation': recommendation
                }
                
                pdf_filename = "credit_report.pdf"
                generate_pdf_report(client_dict, risk_results_dict, pdf_filename)
                
                with open(pdf_filename, "rb") as f:
                    st.download_button(
                        label="Download PDF Assessment Report",
                        data=f,
                        file_name=f"Credit_Report_{form_age}_{form_loan_amt}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("Fill out the profile on the left and click **Evaluate Credit Risk** to run model predictions.")

# PAGE C: ANALYTICS
elif page == "Analytics":
    st.markdown('<h1 class="gradient-text-blue" style="font-size: 38px; margin-bottom: 5px;">Credit Risk Analytics Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 16px; margin-bottom: 25px;">Deep dive analysis of underlying portfolio distributions and feature importances</p>', unsafe_allow_html=True)
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Portfolio Distributions", "Feature Importance", "Correlation Structure"])
        
        with tab1:
            st.markdown("### Portfolio Demographics & Loan Distribution")
            
            c1, c2 = st.columns(2)
            with c1:
                # Age distribution vs default
                fig_age = px.histogram(
                    df, x="person_age", color="loan_status", 
                    marginal="box", nbins=30, 
                    title="Age Distribution vs Loan Default Status",
                    color_discrete_map={0: "#10b981", 1: "#ef4444"},
                    labels={"person_age": "Applicant Age", "loan_status": "Default Status"}
                )
                fig_age.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_age, use_container_width=True)
                
            with c2:
                # Debt-to-income vs default box plot
                fig_dti = px.box(
                    df, x="loan_status", y="loan_percent_income", 
                    color="loan_status",
                    title="Debt-to-Income (DTI) Ratio by Credit Status",
                    color_discrete_map={0: "#10b981", 1: "#ef4444"},
                    labels={"loan_percent_income": "DTI Ratio", "loan_status": "Default Status"}
                )
                fig_dti.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dti, use_container_width=True)
                
            # Loan Intent Distribution count plot
            fig_intent = px.histogram(
                df, x="loan_intent", color="loan_status",
                barmode="group",
                title="Loan Purpose Categories vs Risk Status",
                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                labels={"loan_intent": "Loan Intent", "loan_status": "Default Status"}
            )
            fig_intent.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_intent, use_container_width=True)

        with tab2:
            st.markdown("### Model Feature Importance")
            st.markdown("The chart below showcases the predictive weights assigned to each feature in identifying defaults.")
            
            importance_img = os.path.join("model", "feature_importance.png")
            if os.path.exists(importance_img):
                st.image(importance_img, caption="Pre-computed Relative Feature Importance Scores")
            else:
                st.warning("Feature importance image not found. Ensure training script has executed successfully.")

        with tab3:
            st.markdown("### Correlation Structures")
            st.markdown("Visualizes linear correlations between variables. Highlights multicollinearity and risk indicators.")
            
            heatmap_img = os.path.join("model", "correlation_heatmap.png")
            if os.path.exists(heatmap_img):
                st.image(heatmap_img, caption="Correlation Matrix of Pre-Processed Data")
            else:
                st.warning("Correlation heatmap not found. Run train.py first.")
    else:
        st.warning("Dataset not loaded. Make sure the dataset resides at data/credit_risk_dataset.csv.")
