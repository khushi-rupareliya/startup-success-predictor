import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# Load Model & Feature List
# --------------------------------------------------
try:
    model = joblib.load("startup_success_model.pkl")
    feature_list = joblib.load("model_features.pkl")
except Exception:
    st.error("❌ Error loading model files. Ensure both .pkl files exist.")
    st.stop()

if not hasattr(model, "predict"):
    st.error("❌ Loaded file is not a trained model.")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🚀 Startup Success Prediction System")
st.caption("AI-powered startup acquisition probability estimator.")
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
# Startup Profile Section
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

market_size = st.selectbox(
    "Market Size",
    ["Small", "Medium", "Large"]
)

startup_stage = st.selectbox(
    "Startup Stage",
    ["MVP", "Revenue", "Scaling"]
)

# Convert binary fields
is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Startup Outcome"):

    # Initialize all features as 0
    input_dict = dict.fromkeys(feature_list, 0)

    # Fill numeric core features
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

    # Industry dummy handling
    industry_column = f"is_{industry_type}"
    if industry_column in input_dict:
        input_dict[industry_column] = 1

    # Market Size dummy handling
    if market_size == "Medium" and "market_size_Medium" in input_dict:
        input_dict["market_size_Medium"] = 1
    elif market_size == "Small" and "market_size_Small" in input_dict:
        input_dict["market_size_Small"] = 1
    # If Large → do nothing (base category)

    # Startup Stage dummy handling
    stage_column = f"startup_stage_{startup_stage}"
    if stage_column in input_dict:
        input_dict[stage_column] = 1

    # Convert to DataFrame in correct order
    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except Exception:
        st.error("❌ Prediction failed. Check feature alignment.")
        st.stop()

    risk_score = 1 - probability

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------
    st.write("---")
    st.subheader("📊 Prediction Result")

    st.metric("Startup Success Probability", f"{probability*100:.2f}%")
    st.progress(min(int(probability * 100), 100))

    # Risk classification
    if probability >= 0.7:
        st.success("🟢 Strong Growth Potential")
        rating = "⭐⭐⭐⭐⭐"
    elif probability >= 0.4:
        st.warning("🟡 Moderate Risk – Needs Strategic Improvement")
        rating = "⭐⭐⭐"
    else:
        st.error("🔴 High Risk – Financial & Growth Concerns")
        rating = "⭐⭐"

    st.write(f"### ⭐ Startup Health Rating: {rating}")
    st.metric("Estimated Failure Risk", f"{risk_score*100:.2f}%")

    st.info(
        "Prediction is based on financial strength, milestones, investor backing, industry positioning, and startup stage."
    )

    st.write("### 💡 Suggestions to Improve Success Probability")

    if probability < 0.7:
        st.write("- Improve funding stability and runway.")
        st.write("- Achieve more measurable milestones.")
        st.write("- Strengthen investor participation.")
        st.write("- Clarify and strengthen USP.")
    else:
        st.write("- Focus on sustainable scaling.")
        st.write("- Optimize capital efficiency.")
        st.write("- Maintain strong investor relations.")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Project Overview")
st.sidebar.write("""
Random Forest model trained on startup ecosystem data.
""")

st.sidebar.markdown("### Model Details")
st.sidebar.write("""
- Algorithm: Random Forest  
- Features Used: 41  
- Accuracy: ~80%  
""")

st.sidebar.caption("Developed for Capstone Project")
