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

# Default values to prevent NameError
probability = 0
confidence = 0
risk_percent = 0
funding_percentile = 0
milestone_percentile = 0
relationships_percentile = 0

# --------------------------------------------------
# Load Historical Dataset (For X-Ray Mode)
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
# Platform Navigation
# --------------------------------------------------
tab1, tab2, tab4, tab5 = st.tabs([
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
# 💰 Funding & Growth Profile
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    relationships = st.number_input(
        "Strategic Partnerships",
        min_value=0,
        help="Number of key partnerships, collaborations, or strategic relationships."
    )

    funding_total_usd = st.number_input(
        "Total Capital Raised ($)",
        min_value=0.0,
        help="Total capital raised across all funding rounds."
    )

    age_first_funding_year = st.number_input(
        "Years Until First Funding",
        min_value=0.0,
        help="Number of years from founding until the startup secured its first funding."
    )

    age_last_funding_year = st.number_input(
        "Years Until Most Recent Funding",
        min_value=0.0,
        help="Number of years from founding until the latest funding round."
    )

    age_last_milestone_year = st.number_input(
        "Years Until Latest Milestone",
        min_value=0.0,
        help="Years from founding until the most recent major product or business milestone."
    )


with col2:

    age_first_milestone_year = st.number_input(
        "Years Until First Major Milestone",
        min_value=0.0,
        help="Years from founding until the first significant milestone or product launch."
    )

    avg_participants = st.number_input(
        "Average Investors per Funding Round",
        min_value=0.0,
        help="Average number of investors participating in each funding round."
    )

    milestones = st.number_input(
        "Product / Business Milestones Achieved",
        min_value=0,
        help="Major product launches, partnerships, or market achievements."
    )

    funding_rounds = st.number_input(
        "Number of Funding Rounds",
        min_value=0,
        help="Total number of investment rounds completed by the startup."
    )

    is_top500 = st.selectbox(
        "Recognized in Major Startup Rankings?",
        ["No", "Yes"],
        help="Indicates whether the startup has appeared in major startup rankings or industry lists."
    )
# --------------------------------------------------
# 🧭 Company Overview
# --------------------------------------------------
st.subheader("🧭 Company Overview")

team_size = st.number_input(
    "Team Size",
    min_value=1,
    help="Total number of core team members actively working in the startup."
)

usp_defined = st.selectbox(
    "Unique Value Proposition Clearly Defined?",
    ["No", "Yes"],
    help="Indicates whether the startup has a clearly defined unique value proposition."
)

industry_type = st.selectbox(
    "Industry Sector",
    ["Software", "Web", "Mobile", "Enterprise",
     "Advertising", "Gaming & Video", "E-Commerce",
     "Biotech", "Consulting", "Other"],
    help="Primary industry sector in which the startup operates."
)

market_size = st.selectbox(
    "Target Market Size",
    ["Small", "Medium", "Large"],
    help="Estimated size of the addressable market for the startup."
)

startup_stage = st.selectbox(
    "Current Startup Stage",
    ["MVP", "Revenue", "Scaling"],
    help="Current maturity level of the startup based on product development and revenue generation."
)

# Binary conversions (do not change)
is_top500_value = 1 if is_top500 == "Yes" else 0
usp_defined_value = 1 if usp_defined == "Yes" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
with tab1:

    if st.button("🔍 Run Startup X-Ray Analysis"):
        
        probability = model.predict_proba(input_df)[0][1]
        confidence = probability * 100

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix': "%"},
            title={'text': "Startup Success Score"}
        ))

        st.plotly_chart(gauge)

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

        # Industry Encoding
        industry_column = f"is_{industry_type.lower()}"
        if industry_column in input_dict:
            input_dict[industry_column] = 1

        # Market Encoding
        market_column = f"market_size_{market_size}"
        if market_column in input_dict:
            input_dict[market_column] = 1

        # Stage Encoding
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

        # --------------------------------------------------
        # Startup Health Score
        # --------------------------------------------------
        funding_percentile = (df["funding_total_usd"] < funding_total_usd).mean() * 100
        milestone_percentile = (df["milestones"] < milestones).mean() * 100
        relationships_percentile = (df["relationships"] < relationships).mean() * 100

        health_score = (
            0.4 * funding_percentile +
            0.3 * milestone_percentile +
            0.2 * relationships_percentile +
            0.1 * confidence
        )

        st.metric("Startup Health Index", f"{health_score:.1f} / 100")

with tab2:

    st.subheader("🧪 Startup Scenario Simulator")

    st.markdown("### 🎛 Adjust Key Factors")

    # 👇 Only tweak a few important variables
    funding_change = st.slider("Change Funding (%)", -50, 200, 0)
    milestone_change = st.slider("Change Milestones", -5, 10, 0)
    relationship_change = st.slider("Change Partnerships", -5, 10, 0)

    if st.button("🚀 Run Simulation"):

        # 👇 Start from ORIGINAL input
        sim_dict = dict.fromkeys(feature_list, 0)

        sim_dict["relationships"] = relationships + relationship_change
        sim_dict["funding_total_usd"] = funding_total_usd * (1 + funding_change / 100)
        sim_dict["age_last_milestone_year"] = age_last_milestone_year
        sim_dict["age_last_funding_year"] = age_last_funding_year
        sim_dict["age_first_funding_year"] = age_first_funding_year
        sim_dict["age_first_milestone_year"] = age_first_milestone_year
        sim_dict["avg_participants"] = avg_participants
        sim_dict["milestones"] = milestones + milestone_change
        sim_dict["funding_rounds"] = funding_rounds
        sim_dict["is_top500"] = is_top500_value
        sim_dict["team_size"] = team_size
        sim_dict["usp_defined"] = usp_defined_value

        # Encoding stays SAME
        industry_column = f"is_{industry_type.lower()}"
        if industry_column in sim_dict:
            sim_dict[industry_column] = 1

        market_column = f"market_size_{market_size}"
        if market_column in sim_dict:
            sim_dict[market_column] = 1

        stage_column = f"startup_stage_{startup_stage}"
        if stage_column in sim_dict:
            sim_dict[stage_column] = 1

        sim_df = pd.DataFrame([sim_dict])

        sim_prob = model.predict_proba(sim_df)[0][1]

        st.metric(
            "📈 Simulated Success Probability",
            f"{sim_prob * 100:.2f}%"
        )
        
with tab4:

    st.subheader("📈 Startup Ecosystem Intelligence")

    st.markdown("Gain deep insights into what separates successful startups from failed ones.")

    # --------------------------------------------------
    # 📊 1. Success vs Failure Overview
    # --------------------------------------------------
    st.markdown("### 🏁 Market Outcome Distribution")

    outcome_counts = df["status"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Startups", len(df))

    with col2:
        success_rate = (outcome_counts.get("acquired", 0) / len(df)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    pie = go.Figure(data=[
        go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            hole=0.5
        )
    ])

    pie.update_layout(paper_bgcolor="#0E1117", font=dict(color="white"))
    st.plotly_chart(pie, use_container_width=True)

    # --------------------------------------------------
    # 💰 2. Funding Insights
    # --------------------------------------------------
    st.markdown("### 💰 Funding Intelligence")

    col1, col2 = st.columns(2)

    avg_success_funding = successful_df["funding_total_usd"].mean()
    avg_failed_funding = failed_df["funding_total_usd"].mean()

    col1.metric("Avg Funding (Successful)", f"${avg_success_funding:,.0f}")
    col2.metric("Avg Funding (Failed)", f"${avg_failed_funding:,.0f}")

    funding_bar = go.Figure()
    funding_bar.add_trace(go.Bar(
        x=["Successful", "Failed"],
        y=[avg_success_funding, avg_failed_funding]
    ))

    funding_bar.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(funding_bar, use_container_width=True)

    # --------------------------------------------------
    # 🚀 3. Execution (Milestones)
    # --------------------------------------------------
    st.markdown("### 🚀 Execution Strength")

    avg_success_milestones = successful_df["milestones"].mean()
    avg_failed_milestones = failed_df["milestones"].mean()

    milestone_bar = go.Figure()
    milestone_bar.add_trace(go.Bar(
        x=["Successful", "Failed"],
        y=[avg_success_milestones, avg_failed_milestones]
    ))

    milestone_bar.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(milestone_bar, use_container_width=True)

    # --------------------------------------------------
    # 🤝 4. Network Strength
    # --------------------------------------------------
    st.markdown("### 🤝 Network & Partnerships")

    avg_success_rel = successful_df["relationships"].mean()
    avg_failed_rel = failed_df["relationships"].mean()

    rel_bar = go.Figure()
    rel_bar.add_trace(go.Bar(
        x=["Successful", "Failed"],
        y=[avg_success_rel, avg_failed_rel]
    ))

    rel_bar.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(rel_bar, use_container_width=True)

    # --------------------------------------------------
    # 📊 5. Distribution Analysis
    # --------------------------------------------------
    st.markdown("### 📊 Funding Distribution Across Market")

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=df["funding_total_usd"]))

    hist.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(hist, use_container_width=True)

    # --------------------------------------------------
    # 🧠 6. AI Insights Summary (VERY IMPORTANT)
    # --------------------------------------------------
    st.markdown("### 🧠 Key Market Insights")

    st.info(f"""
📊 Key Observations from Market Data:

• Successful startups raise ~{avg_success_funding/1e6:.1f}M USD on average  
• Failed startups raise ~{avg_failed_funding/1e6:.1f}M USD  

• Execution matters: successful startups achieve ~{avg_success_milestones:.1f} milestones vs {avg_failed_milestones:.1f}  

• Strong networks correlate with success: {avg_success_rel:.1f} vs {avg_failed_rel:.1f} partnerships  

💡 Insight: Funding alone is not enough — execution + network strength are critical drivers.
""")

with tab5:

    st.subheader("📄 Startup X-Ray Report")

    st.write(
        "Generate a downloadable summary of the startup diagnostic analysis."
    )

    if st.button("Generate X-Ray Report"):
        st.success("Report module ready. PDF export can be integrated here.")


    # ==================================================
    # 🎯 QUICK PREDICTION MODE (UNCHANGED)
    # ==================================================
    if mode == "🎯 Quick Prediction":

        st.divider()
        st.subheader("📊 AI Investment Dashboard")

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

        if confidence >= 85:
            st.success("🏆 Investor Grade: A+ | Exceptional Potential")
        elif confidence >= 70:
            st.success("✅ Investor Grade: A | Strong Growth Signals")
        elif confidence >= 50:
            st.warning("⚠ Investor Grade: B | Moderate Risk Profile")
        else:
            st.error("❌ Investor Grade: C | High Investment Risk")

    # ==================================================
    # 🩺 FULL STARTUP X-RAY MODE (NEW)
    # ==================================================
    if mode == "🩺 Full Startup X-Ray":

        st.divider()
        st.subheader("🩺 Startup X-Ray Diagnostic Report")

        funding_percentile = (df["funding_total_usd"] < funding_total_usd).mean() * 100
        milestone_percentile = (df["milestones"] < milestones).mean() * 100
        relationships_percentile = (df["relationships"] < relationships).mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Funding Percentile", f"{funding_percentile:.1f}%")
        col2.metric("Milestone Percentile", f"{milestone_percentile:.1f}%")
        col3.metric("Network Strength Percentile", f"{relationships_percentile:.1f}%")

st.markdown("### 🧭 Startup vs Acquired Startup Benchmark")

avg_success = [
    successful_df["funding_total_usd"].mean(),
    successful_df["milestones"].mean(),
    successful_df["relationships"].mean(),
    successful_df["funding_rounds"].mean()
]

startup_values = [
    funding_total_usd,
    milestones,
    relationships,
    funding_rounds
]

categories = [
    "Funding",
    "Milestones",
    "Partnerships",
    "Funding Rounds"
]

radar = go.Figure()

radar.add_trace(go.Scatterpolar(
    r=startup_values,
    theta=categories,
    fill='toself',
    name="Your Startup"
))

radar.add_trace(go.Scatterpolar(
    r=avg_success,
    theta=categories,
    fill='toself',
    name="Acquired Startup Average"
))

st.plotly_chart(radar, use_container_width=True)

st.markdown("### 📊 Comparison with Acquired Startups")

avg_success_funding = successful_df["funding_total_usd"].mean()
avg_success_milestones = successful_df["milestones"].mean()
avg_success_relationships = successful_df["relationships"].mean()

if funding_total_usd < avg_success_funding:
    st.error("🔴 Funding below acquired startup average.")
else:
    st.success("🟢 Funding above acquired startup average.")

if milestones < avg_success_milestones:
    st.error("🔴 Milestones below acquired startup average.")
else:
    st.success("🟢 Strong milestone execution.")

if relationships < avg_success_relationships:
    st.warning("🟡 Network relationships below acquired average.")
else:
    st.success("🟢 Strong investor/network connectivity.")

st.markdown("### 🏦 Investor Readiness Assessment")

if probability > 0.75 and funding_percentile > 60:
    tier = "🟢 Growth Stage Investment Profile"
elif probability > 0.5:
    tier = "🟡 Early Stage / Angel Investment Profile"
else:
    tier = "🔴 High Risk / Pre-Validation Stage"

st.info(f"Investment Tier: {tier}")

st.markdown("### 📋 X-Ray Summary")
st.write(f"""
• Survival Probability: {confidence:.2f}%  
• Funding Strength Percentile: {funding_percentile:.1f}%  
• Execution Strength Percentile: {milestone_percentile:.1f}%  
• Network Strength Percentile: {relationships_percentile:.1f}%  
• Investment Classification: {tier}
""")











