import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("startup_risks.csv")  # make sure this file has 'risk_level'
X = df.drop("risk_level", axis=1)      # features
y = df["risk_level"]                   # target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Prediction function
def predict_risk(budget, team_size, deadline_months, runway_months):
    input_df = pd.DataFrame([[budget, team_size, deadline_months, runway_months]],
                            columns=["budget", "team_size", "deadline_months", "runway_months"])
    prediction = model.predict(input_df)[0]
    return f"Predicted Risk Level: {prediction}"

# Gradio interface
demo = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(10000, 500000, label="Budget (USD)"),
        gr.Slider(1, 50, label="Team Size"),
        gr.Slider(1, 24, label="Deadline (Months)"),
        gr.Slider(1, 24, label="Runway (Months)")
    ],
    outputs="text",
    title="🚀 AI Startup Risk Radar",
    description="Predict startup risk level based on key attributes."
)

demo.launch()
