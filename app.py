import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Startup Success Predictor", layout="wide")

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("startup_success_model.pkl")
feature_list = joblib.load("model_features.pkl")
df = pd.read_csv("startupdata.csv")

successful_df = df[df["status"] == "acquired"]
failed_df = df[df["status"] == "closed"]

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔎 Predictor",
    "🧪 Simulator",
    "📈 Insights"
])

# -------------------------------
# 🔎 TAB 1: PREDICTOR
# -------------------------------
with tab1:

    st.subheader("Startup Success Prediction")

    col1, col2 = st.columns(2)

    with col1:
        funding = st.number_input("Funding ($)", min_value=0.0)
        milestones = st.number_input("Milestones", min_value=0)
        relationships = st.number_input("Partnerships", min_value=0)

    with col2:
        team_size = st.number_input("Team Size", min_value=1)
        funding_rounds = st.number_input("Funding Rounds", min_value=0)

    if st.button("Predict"):

        input_dict = dict.fromkeys(feature_list, 0)

        input_dict["funding_total_usd"] = funding
        input_dict["milestones"] = milestones
        input_dict["relationships"] = relationships
        input_dict["team_size"] = team_size
        input_dict["funding_rounds"] = funding_rounds

        input_df = pd.DataFrame([input_dict])

        prob = model.predict_proba(input_df)[0][1]

        st.metric("Success Probability", f"{prob*100:.2f}%")

# -------------------------------
# 🧪 TAB 2: SIMULATOR
# -------------------------------
with tab2:

    st.subheader("What-If Scenario Simulator")

    st.markdown("Adjust only key drivers 👇")

    funding_change = st.slider("Funding Change (%)", -50, 200, 0)
    milestone_change = st.slider("Milestone Change", -5, 10, 0)

    if st.button("Run Simulation"):

        base_funding = funding
        base_milestones = milestones

        sim_dict = dict.fromkeys(feature_list, 0)

        sim_dict["funding_total_usd"] = base_funding * (1 + funding_change/100)
        sim_dict["milestones"] = base_milestones + milestone_change
        sim_dict["relationships"] = relationships
        sim_dict["team_size"] = team_size
        sim_dict["funding_rounds"] = funding_rounds

        sim_df = pd.DataFrame([sim_dict])

        sim_prob = model.predict_proba(sim_df)[0][1]

        st.metric("Simulated Success", f"{sim_prob*100:.2f}%")

# -------------------------------
# 📈 TAB 3: INSIGHTS
# -------------------------------
with tab3:

    st.subheader("Market Insights Dashboard")

    # Success Rate
    success_rate = (len(successful_df) / len(df)) * 100
    st.metric("Market Success Rate", f"{success_rate:.1f}%")

    # Funding comparison
    avg_success = successful_df["funding_total_usd"].mean()
    avg_fail = failed_df["funding_total_usd"].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Success", "Failed"], y=[avg_success, avg_fail]))
    st.plotly_chart(fig, use_container_width=True)

    st.info("Successful startups generally have higher funding and execution.")
