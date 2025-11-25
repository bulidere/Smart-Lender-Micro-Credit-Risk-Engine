import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# --- 1. Load Resources ---
model = joblib.load('credit_risk_model.pkl')
explainer = joblib.load('shap_explainer.pkl')

# --- 2. Sidebar: User Inputs ---
st.title("🏦 Smart-Lender Risk Engine")
st.markdown("### Machine Learning Credit Assessment Tool")

st.sidebar.header("Applicant Information")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, value=60000)
loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000, value=15000)
fico = st.sidebar.slider("FICO Credit Score", 300, 850, 700)
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2)

# --- 3. Processing ---
if st.sidebar.button("Assess Risk"):
    # Create a dataframe matching the model's training input
    input_data = pd.DataFrame({
        'Income': [income],
        'Loan_Amount': [loan_amount],
        'FICO_Score': [fico],
        'DTI': [dti]
    })
    
    # Prediction
    prediction = model.predict(input_data)[0] # 0 = Pay, 1 = Default
    probability = model.predict_proba(input_data)[0][1] # Probability of Default (1)
    
    # --- 4. Display Results ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decision Recommendation")
        if probability > 0.4: # Threshold for high risk
            st.error("❌ REJECT APPLICATION")
            st.metric(label="Risk Tier", value="High Risk")
        else:
            st.success("✅ APPROVE APPLICATION")
            st.metric(label="Risk Tier", value="Low Risk")
            
    with col2:
        st.subheader("Model Confidence")
        st.metric(label="Probability of Default", value=f"{probability:.2%}")
        
    # --- 5. Explainability (SHAP) ---
    st.divider()
    st.subheader("Why did the AI make this decision?")
    st.write("The plot below shows which factors pushed the risk score UP (Red) or DOWN (Blue).")
    
    # Calculate SHAP for this specific instance
    shap_val = explainer.shap_values(input_data)
    
    # Plotting
    # Note: Streamlit requires matplotlib figures
    fig, ax = plt.subplots()
    shap.summary_plot(shap_val, input_data, plot_type="bar", show=False)
    
    st.pyplot(fig)
    # streamlit run app.py