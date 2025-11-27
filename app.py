import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("startup_risks.csv")
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Plot feature importance
def plot_feature_importance():
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    return "feature_importance.png"

# Plot radar chart
def plot_radar(values):
    labels = ["Operational", "Financial", "Team", "Market"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.fill(angles, values, color="blue", alpha=0.4)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Risk Radar")
    plt.tight_layout()
    plt.savefig("radar.png")
    return "radar.png"

# Prediction function
def predict_risk(budget, team_size, deadline_months, runway_months, business_domain):
    input_df = pd.DataFrame([[budget, team_size, deadline_months, runway_months]],
                            columns=["budget", "team_size", "deadline_months", "runway_months"])
    prediction = model.predict(input_df)[0]

    # Simulated radar values (scale to 0–1)
    radar_values = [
        deadline_months / 24,       # Operational
        budget / 500000,            # Financial
        team_size / 50,             # Team
        runway_months / 24          # Market proxy
    ]

    radar_img = plot_radar(radar_values)
    feature_img = plot_feature_importance()

    return f"Predicted Risk Level: {prediction}", radar_img, feature_img

# Gradio interface
demo = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(10000, 500000, label="Funding (USD)"),
        gr.Slider(1, 50, label="Team Size"),
        gr.Slider(1, 24, label="Deadline (Months)"),
        gr.Slider(1, 24, label="Runway (Months)"),
        gr.Dropdown(["Healthcare", "Finance", "Education", "Retail"], label="Business Domain")
    ],
    outputs=[
        gr.Text(label="Predicted Risk Level"),
        gr.Image(label="Risk Radar Chart"),
        gr.Image(label="Feature Importance Chart")
    ],
    title="🚀 AI Startup Risk Radar",
    description="Predict startup risk level and visualize vulnerabilities across key dimensions."
)

demo.launch()
