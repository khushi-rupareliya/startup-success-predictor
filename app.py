import streamlit as st
import joblib
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# Custom Minimal Styling
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    font-size: 16px;
    color: #8e8e8e;
    margin-top: 0;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    font-size: 16px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load("startup_success_model.pkl")

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.title("🚀 Startup Success Prediction System")
st.caption("Predict acquisition probability using funding and growth indicators.")
st.divider()

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    relationships = st.number_input("Number of Relationships", min_value=0)
    funding_total_usd = st.number_input("Total Funding (USD)", min_value=0.0, format="%.2f")
    age_last_milestone_year = st.number_input("Age at Last Milestone (Years)", min_value=0.0)
    age_last_funding_year = st.number_input("Age at Last Funding (Years)", min_value=0.0)
    age_first_funding_year = st.number_input("Age at First Funding (Years)", min_value=0.0)

with col2:
    age_first_milestone_year = st.number_input("Age at First Milestone (Years)", min_value=0.0)
    avg_participants = st.number_input("Average Investors per Round", min_value=0.0)
    milestones = st.number_input("Total Milestones Achieved", min_value=0)
    funding_rounds = st.number_input("Funding Rounds", min_value=0)
    is_top500 = st.selectbox("Recognized as Top 500 Startup?", ["No", "Yes"])

st.write("")
st.write("")

# Convert Top 500 selection
is_top500_value = 1 if is_top500 == "Yes" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Startup Outcome"):

    input_data = np.array([[relationships,
                            funding_total_usd,
                            age_last_milestone_year,
                            age_last_funding_year,
                            age_first_funding_year,
                            age_first_milestone_year,
                            avg_participants,
                            milestones,
                            funding_rounds,
                            is_top500_value]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")

    if prediction[0] == 1:
        st.success("✅ High Probability of Startup Success")
        st.metric("Predicted Success Probability", f"{probability*100:.2f}%")
        st.progress(min(int(probability*100), 100))
    else:
        st.error("❌ High Risk of Startup Failure")
        st.metric("Predicted Success Probability", f"{probability*100:.2f}%")
        st.progress(min(int(probability*100), 100))

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Project Overview")
st.sidebar.write("""
This application uses a Random Forest Machine Learning model
trained on real-world startup ecosystem data from Kaggle.
""")

st.sidebar.write("")

st.sidebar.markdown("### Model Details")
st.sidebar.write("""
- Algorithm: Random Forest Classifier  
- Features Used: 10 Key Financial & Growth Indicators  
- Dataset Size: 923 Startups  
""")

st.sidebar.write("---")
st.sidebar.caption("Developed for Capstone Project")


