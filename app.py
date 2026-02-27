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
        background-color: #f4f7fa;
    }
    .stMetric {
        font-size:20px !important;
        font-weight:600;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Features
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
# Core Financial Inputs
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    relationships = st.number_input("Number of Relationships", min_value=0)
    funding_total_usd = st.number_input("Total Funding (USD)", min_value=0.0)
    age_last_milestone_year = st.number_input("Age at Last Milestone (Years)", min_value=0.0)
    age_last_funding_year = st.number_input("Age at Last Funding (Years)", min_value=0.0)
    age_first_funding_year = st.number_input("Age at First Funding (Years)", min_value=0.0)

with col2:
    age_first_milestone_year = st.number_input("Age at First Milestone (Years)", min_value=0.0)
    avg_participants = st.number_input("Average Investors per Round", min_value=0.0)
    milestones = st.number_input("Total Milestones Achieved", min_value=0)
    funding_rounds = st.number_input("Funding Rounds", min_value=0)
    is_top500 = st.selectbox("Recognized as Top 500 Startup?", ["No", "Yes"])

# --------------------------------------------------
# Startup Profile
# --------------------------------------------------
st.subheader("📊 Startup Profile")

team_size = st.number_input("Team Size", min_value=1)
usp_defined = st.selectbox("USP Clearly Defined?", ["No", "Yes"])

industry_type = st.selectbox(
    "Industry Type",
    ["software", "web", "mobile", "enterprise",
     "advertising", "gamesvideo", "ecommerce",
     "biotech", "consulting", "othercategory"]
)

market_size = st.selectbox("Market Size", ["Small", "Medium", "Large"])
startup_stage = st.selectbox("Startup Stage", ["MVP", "Revenue", "Scaling"])

# Binary conversion
is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔍 Predict Startup Outcome"):

    input_dict = dict.fromkeys(feature_list, 0)

    # Numeric Inputs
    input_dict["relationships"] = relationships
    input_dict["funding_total_usd"] = funding_total_usd
    input_dict["age_last_milestone_year"] = age_last_milestone_year
    input_dict["age_last_funding_year"] = age_last_funding_year
    input_dict["age_first_funding_year"] = age_first_funding_year
    input_dict["age_first_milestone_year"] = age_first_milestone_year
    input_dict["avg_participants"] = avg_participants
    input_dict["milestones"] = milestones
    input_dict["funding_rounds"] = funding_rounds
    input_dict["is_top500"] = is_top500_value
    input_dict["team_size"] = team_size
    input_dict["usp_defined"] = usp_defined_value

    # Industry Encoding
    industry_column = f"is_{industry_type}"
    if industry_column in input_dict:
        input_dict[industry_column] = 1

    # Market Size Encoding
    market_column = f"market_size_{market_size}"
    if market_column in input_dict:
        input_dict[market_column] = 1

    # Startup Stage Encoding
    stage_column = f"startup_stage_{startup_stage}"
    if stage_column in input_dict:
        input_dict[stage_column] = 1

    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except:
        st.error("❌ Prediction failed. Feature mismatch detected.")
        st.stop()

    risk_score = 1 - probability
    confidence = probability * 100

    # --------------------------------------------------
    # Results Section
    # --------------------------------------------------
    st.write("---")
    st.subheader("📊 Prediction Results")

    colA, colB = st.columns([1, 1])

    with colA:
        st.metric("Startup Success Probability", f"{confidence:.2f}%")
        st.progress(int(confidence))

        if confidence >= 85:
            st.success("🏆 Investor Grade: A+ (Very Strong)")
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
    # Feature Importance (Graph instead of table)
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
    # Smart Suggestions
    # --------------------------------------------------
    st.write("### 💡 Strategic Recommendations")

    if confidence < 70:
        st.write("- Strengthen investor relationships and funding depth.")
        st.write("- Deliver milestones more consistently.")
        st.write("- Improve capital efficiency and runway planning.")
        st.write("- Refine and clearly communicate USP.")
    else:
        st.write("- Focus on scalable growth strategies.")
        st.write("- Optimize operational performance.")
        st.write("- Expand into larger markets strategically.")
        st.write("- Maintain strong investor engagement.")

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
