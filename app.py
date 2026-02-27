import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f9fbfd;
    }
    .stMetric {
        font-size:20px !important;
        font-weight:600;
    }
    .grade-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
try:
    model = joblib.load("startup_success_model.pkl")
    feature_list = joblib.load("model_features.pkl")
except:
    st.error("❌ Model files missing.")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🚀 Startup Success Prediction System")
st.caption("AI-powered startup acquisition probability estimator")
st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    relationships = st.number_input("Number of Relationships", min_value=0)
    funding_total_usd = st.number_input("Total Funding (USD)", min_value=0.0)
    milestones = st.number_input("Total Milestones Achieved", min_value=0)
    funding_rounds = st.number_input("Funding Rounds", min_value=0)
    team_size = st.number_input("Team Size", min_value=1)

with col2:
    avg_participants = st.number_input("Average Investors per Round", min_value=0.0)
    age_first_funding_year = st.number_input("Age at First Funding (Years)", min_value=0.0)
    age_last_funding_year = st.number_input("Age at Last Funding (Years)", min_value=0.0)
    age_first_milestone_year = st.number_input("Age at First Milestone (Years)", min_value=0.0)
    age_last_milestone_year = st.number_input("Age at Last Milestone (Years)", min_value=0.0)

is_top500 = st.selectbox("Recognized as Top 500 Startup?", ["No", "Yes"])
usp_defined = st.selectbox("USP Clearly Defined?", ["No", "Yes"])

is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Startup Outcome"):

    input_dict = dict.fromkeys(feature_list, 0)

    input_dict.update({
        "relationships": relationships,
        "funding_total_usd": funding_total_usd,
        "milestones": milestones,
        "funding_rounds": funding_rounds,
        "team_size": team_size,
        "avg_participants": avg_participants,
        "age_first_funding_year": age_first_funding_year,
        "age_last_funding_year": age_last_funding_year,
        "age_first_milestone_year": age_first_milestone_year,
        "age_last_milestone_year": age_last_milestone_year,
        "is_top500": is_top500_value,
        "usp_defined": usp_defined_value
    })

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    risk_score = 1 - probability
    confidence = probability * 100

    # --------------------------------------------------
    # RESULTS SECTION
    # --------------------------------------------------

    st.write("---")
    st.subheader("📊 Prediction Results")

    colA, colB = st.columns([1, 1])

    with colA:
        st.metric("Startup Success Probability", f"{confidence:.2f}%")
        st.progress(int(confidence))

        if confidence >= 85:
            st.success("🏆 Investor Grade: A+ (High Confidence)")
        elif confidence >= 70:
            st.success("✅ Investor Grade: A (Strong Potential)")
        elif confidence >= 50:
            st.warning("⚠ Investor Grade: B (Moderate Risk)")
        else:
            st.error("❌ Investor Grade: C (High Risk)")

    with colB:
        st.metric("Estimated Failure Risk", f"{risk_score*100:.2f}%")

        fig, ax = plt.subplots()
        ax.barh(["Success", "Failure"], [probability, risk_score])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

    # --------------------------------------------------
    # FEATURE IMPORTANCE GRAPH
    # --------------------------------------------------

    if hasattr(model, "feature_importances_"):
        st.write("### 🔍 Top 10 Influential Features")

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_list,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig2, ax2 = plt.subplots()
        ax2.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1]
        )
        ax2.set_xlabel("Importance Score")
        st.pyplot(fig2)

    # --------------------------------------------------
    # SMARTER SUGGESTIONS
    # --------------------------------------------------

    st.write("### 💡 Strategic Recommendations")

    if confidence < 70:
        st.write("- Strengthen investor relationships and funding stability.")
        st.write("- Increase milestone delivery frequency.")
        st.write("- Optimize burn rate and runway management.")
        st.write("- Clearly refine and communicate your USP.")
    else:
        st.write("- Focus on sustainable scaling strategies.")
        st.write("- Improve operational efficiency.")
        st.write("- Maintain strong investor engagement.")
        st.write("- Expand into larger addressable markets.")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📘 Project Overview")
st.sidebar.write("""
Machine Learning based Startup Success Prediction System.
""")

st.sidebar.markdown("### 🧠 Model Details")
st.sidebar.write("""
- Algorithm: Random Forest  
- Features Used: 41  
- Accuracy: ~80%  
- Explainable AI Enabled  
""")

st.sidebar.caption("Developed by Khushi Rupareliya 🚀")
