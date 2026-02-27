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
# Load Model & Features (SAFE)
# --------------------------------------------------
try:
    model = joblib.load("startup_success_model.pkl")
    feature_list = joblib.load("model_features.pkl")
except Exception:
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

# Binary Conversion
is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Startup Outcome"):

    input_dict = dict.fromkeys(feature_list, 0)

    # Numeric Inputs (SAFE – no removal)
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
        probability = model.predict_proba(input_df)[0][1]
    except Exception:
        st.error("❌ Feature mismatch detected.")
        st.stop()

    risk_score = 1 - probability
    confidence = probability * 100
    risk_percent = risk_score * 100

    st.divider()
    st.subheader("📊 AI Investment Dashboard")

    colA, colB = st.columns(2)

    # --------------------------------------------------
    # Gauge Chart
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Donut Chart
    # --------------------------------------------------
    with colB:
        donut = go.Figure(data=[go.Pie(
            labels=["Success Probability", "Failure Risk"],
            values=[confidence, risk_percent],
            hole=0.6,
            marker=dict(colors=["#00F5A0", "#FF4B4B"])
        )])
        donut.update_layout(paper_bgcolor="#0E1117", font=dict(color="white"))
        st.plotly_chart(donut, use_container_width=True)

    # --------------------------------------------------
    # Investor Grade
    # --------------------------------------------------
    if confidence >= 85:
        st.success("🏆 Investor Grade: A+ | Exceptional Potential")
    elif confidence >= 70:
        st.success("✅ Investor Grade: A | Strong Growth Signals")
    elif confidence >= 50:
        st.warning("⚠ Investor Grade: B | Moderate Risk Profile")
    else:
        st.error("❌ Investor Grade: C | High Investment Risk")

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------
    if hasattr(model, "feature_importances_"):

        st.markdown("### 🔍 Key Decision Drivers")

        importance_df = pd.DataFrame({
            "Feature": feature_list,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        fig = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation='h',
            marker_color="#00F5A0"
        ))

        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white"),
            yaxis=dict(autorange="reversed")
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # AI Strategic Insights
    # --------------------------------------------------
    st.markdown("### 🧠 AI Strategic Insights")

    if confidence < 70:
        st.write("• Increase funding depth and investor backing.")
        st.write("• Accelerate milestone achievement frequency.")
        st.write("• Improve capital efficiency and runway planning.")
        st.write("• Strengthen unique value proposition.")
    else:
        st.write("• Strong funding-health indicators detected.")
        st.write("• High investor engagement pattern.")
        st.write("• Positioned for scalable market expansion.")
        st.write("• Maintain capital discipline for long-term growth.")

    st.caption("Prediction powered by Random Forest ensemble learning across 41 engineered features.")

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
