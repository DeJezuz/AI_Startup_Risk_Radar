import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/startup_risks.csv")

# Train model
X = df.drop("risk_level", axis=1)
y = df["risk_level"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.title("🚀 AI Startup Risk Radar")
st.write("Predict startup risk levels and visualize vulnerabilities.")

# User input
budget = st.slider("Budget (USD)", 10000, 500000, 100000)
team_size = st.slider("Team Size", 1, 50, 5)
deadline = st.slider("Deadline (months)", 1, 24, 6)
runway = st.slider("Runway (months)", 1, 24, 12)

# Make prediction
input_data = pd.DataFrame([[budget, team_size, deadline, runway]],
                          columns=["budget", "team_size", "deadline_months", "runway_months"])
prediction = model.predict(input_data)[0]

st.subheader("Predicted Risk Level:")
st.write(f"**{prediction}**")

# Feature importance chart
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.bar(X.columns, importances)
ax.set_title("Feature Importance")
st.pyplot(fig)
