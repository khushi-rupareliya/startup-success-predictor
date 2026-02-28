import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# Page Config (Collapsible Sidebar)
# --------------------------------------------------
st.set_page_config(
    page_title="Startup X-Ray",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
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

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="collapsedControl"] {
    color: white;
}

.stMetric {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
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
# Load Dataset for X-Ray
# --------------------------------------------------
df = pd.read_csv("startupdata.csv")
successful_df = df[df["status"] == "acquired"]
failed_df = df[df["status"] == "closed"]

# ==================================================
# Sidebar Navigation (SaaS Style)
# ==================================================
st.sidebar.title("🩺 Startup X-Ray")

navigation = st.sidebar.radio(
    "Navigation",
    ["🎯 Quick Prediction", "🩺 Full Startup X-Ray"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 About Platform")
st.sidebar.write("""
AI-powered Startup Due Diligence & Investment Intelligence System.
""")

st.sidebar.markdown("### 🧠 Model Engine")
st.sidebar.write("""
- Algorithm: Random Forest  
- Dataset Size: 923 Startups  
- Acquired: 597  
- Closed: 326  
- Features: 41  
- Accuracy: ~80%
""")

st.sidebar.markdown("### ⚙ Analysis Modes")
st.sidebar.write("""
🎯 Quick Prediction  
🩺 Full Startup X-Ray  
""")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Khushi Rupareliya 🚀")

# ==================================================
# Main Header
# ==================================================
st.title("🚀 Startup X-Ray Intelligence Platform")
st.caption("AI-powered Startup Due Diligence & Investment Analytics")
st.divider()

# ==================================================
# Input Section (UNCHANGED)
# ==================================================
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

is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# ==================================================
# Prediction
# ==================================================
if st.button("🔍 Run Analysis"):

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

    industry_column = f"is_{industry_type}"
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
    risk_percent = (1 - probability) * 100

    st.divider()

    # ==================================================
    # 🎯 QUICK PREDICTION MODE
    # ==================================================
    if navigation == "🎯 Quick Prediction":

        st.subheader("📊 Investment Dashboard")

        colA, colB = st.columns(2)

        with colA:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={'suffix': "%"},
                title={'text': "Startup Success Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00F5A0"},
                    'steps': [
                        {'range': [0, 40], 'color': "#3A0D0D"},
                        {'range': [40, 70], 'color': "#3A330D"},
                        {'range': [70, 100], 'color': "#0D3A22"}
                    ]
                }
            ))
            gauge.update_layout(paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(gauge, use_container_width=True)

        with colB:
            donut = go.Figure(data=[go.Pie(
                labels=["Success Probability", "Failure Risk"],
                values=[confidence, risk_percent],
                hole=0.6,
                marker=dict(colors=["#00F5A0", "#FF4B4B"])
            )])
            donut.update_layout(paper_bgcolor="#0E1117", font=dict(color="white"))
            st.plotly_chart(donut, use_container_width=True)

    # ==================================================
    # 🩺 FULL STARTUP X-RAY MODE
    # ==================================================
    if navigation == "🩺 Full Startup X-Ray":

        st.subheader("🩺 Startup X-Ray Diagnostic Report")

        funding_percentile = (df["funding_total_usd"] < funding_total_usd).mean() * 100
        milestone_percentile = (df["milestones"] < milestones).mean() * 100
        relationships_percentile = (df["relationships"] < relationships).mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Funding Percentile", f"{funding_percentile:.1f}%")
        col2.metric("Milestone Percentile", f"{milestone_percentile:.1f}%")
        col3.metric("Network Strength Percentile", f"{relationships_percentile:.1f}%")

        st.markdown("### 🏦 Investor Readiness")

        if probability > 0.75 and funding_percentile > 60:
            st.success("🟢 Growth Stage Investment Profile")
        elif probability > 0.5:
            st.warning("🟡 Early Stage / Angel Investment Profile")
        else:
            st.error("🔴 High Risk / Pre-Validation Stage")

        st.markdown("### 📋 X-Ray Summary")
        st.write(f"""
        • Survival Probability: {confidence:.2f}%  
        • Funding Strength Percentile: {funding_percentile:.1f}%  
        • Execution Strength Percentile: {milestone_percentile:.1f}%  
        • Network Strength Percentile: {relationships_percentile:.1f}%  
        """)
