import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# Dark Fintech Styling
# --------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: white;
}
[data-testid="stHeader"] {
    background-color: #0E1117;
}
.stMetric {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Features
# --------------------------------------------------
try:
    model = joblib.load("startup_success_model.pkl")
    feature_list = joblib.load("model_features.pkl")
except Exception:
    st.error("❌ Model files missing.")
    st.stop()

# Default values
probability = 0
confidence = 0
risk_percent = 0
funding_percentile = 0
milestone_percentile = 0
relationships_percentile = 0

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("startupdata.csv")
successful_df = df[df["status"] == "acquired"]
failed_df = df[df["status"] == "closed"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🩺 Startup X-Ray")
st.caption("AI-Driven Startup Diagnostics & Investment Intelligence Platform")
st.divider()

# --------------------------------------------------
# TABS (FIXED)
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Startup Diagnostic",
    "🧪 Scenario Simulator",
    "📈 Market Insights",
    "📄 X-Ray Report"
])

# --------------------------------------------------
# Mode Selector
# --------------------------------------------------
mode = st.radio(
    "Select Analysis Mode",
    ["🎯 Quick Prediction", "🩺 Full Startup X-Ray"],
    horizontal=True
)

# --------------------------------------------------
# Inputs
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    relationships = st.number_input("Strategic Partnerships", min_value=0)
    funding_total_usd = st.number_input("Total Capital Raised ($)", min_value=0.0)
    age_first_funding_year = st.number_input("Years Until First Funding", min_value=0.0)
    age_last_funding_year = st.number_input("Years Until Most Recent Funding", min_value=0.0)
    age_last_milestone_year = st.number_input("Years Until Latest Milestone", min_value=0.0)

with col2:
    age_first_milestone_year = st.number_input("Years Until First Major Milestone", min_value=0.0)
    avg_participants = st.number_input("Average Investors", min_value=0.0)
    milestones = st.number_input("Milestones", min_value=0)
    funding_rounds = st.number_input("Funding Rounds", min_value=0)
    is_top500 = st.selectbox("Top Startup?", ["No", "Yes"])

team_size = st.number_input("Team Size", min_value=1)
usp_defined = st.selectbox("USP Defined?", ["No", "Yes"])
industry_type = st.selectbox("Industry", ["Software","Web","Mobile","Enterprise","Other"])
market_size = st.selectbox("Market Size", ["Small","Medium","Large"])
startup_stage = st.selectbox("Stage", ["MVP","Revenue","Scaling"])

is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# --------------------------------------------------
# TAB 1
# --------------------------------------------------
with tab1:
    if st.button("🔍 Run Startup X-Ray Analysis"):

        input_dict = dict.fromkeys(feature_list, 0)

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

        industry_column = f"is_{industry_type.lower()}"
        if industry_column in input_dict:
            input_dict[industry_column] = 1

        market_column = f"market_size_{market_size}"
        if market_column in input_dict:
            input_dict[market_column] = 1

        stage_column = f"startup_stage_{startup_stage}"
        if stage_column in input_dict:
            input_dict[stage_column] = 1

        input_df = pd.DataFrame([input_dict])

        probability = model.predict_proba(input_df)[0][1]
        confidence = probability * 100

        st.metric("Startup Success Score", f"{confidence:.2f}%")

# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:

    st.subheader("🧪 Startup Scenario Simulator")

    funding_increase = st.slider("Increase Funding (%)", 0, 100, 20)
    milestone_increase = st.slider("Increase Milestones", 0, 10, 2)

    if st.button("Run Simulation"):

        input_dict = dict.fromkeys(feature_list, 0)

        input_dict["relationships"] = relationships
        input_dict["funding_total_usd"] = funding_total_usd * (1 + funding_increase/100)
        input_dict["age_last_milestone_year"] = age_last_milestone_year
        input_dict["age_last_funding_year"] = age_last_funding_year
        input_dict["age_first_funding_year"] = age_first_funding_year
        input_dict["age_first_milestone_year"] = age_first_milestone_year
        input_dict["avg_participants"] = avg_participants
        input_dict["milestones"] = milestones + milestone_increase
        input_dict["funding_rounds"] = funding_rounds
        input_dict["is_top500"] = is_top500_value
        input_dict["team_size"] = team_size
        input_dict["usp_defined"] = usp_defined_value

        sim_df = pd.DataFrame([input_dict])
        sim_prob = model.predict_proba(sim_df)[0][1]

        st.metric("Simulated Success Probability", f"{sim_prob*100:.2f}%")

# --------------------------------------------------
# TAB 3 (Market Insights)
# --------------------------------------------------
with tab3:

    st.subheader("📈 Startup Ecosystem Insights")

    outcome_counts = df["status"].value_counts()

    pie = go.Figure(data=[
        go.Pie(labels=outcome_counts.index, values=outcome_counts.values)
    ])

    st.plotly_chart(pie)

# --------------------------------------------------
# TAB 4 (Report)
# --------------------------------------------------
with tab4:

    st.subheader("📄 Startup X-Ray Report")

    if st.button("Generate X-Ray Report"):
        st.success("Report Ready")
