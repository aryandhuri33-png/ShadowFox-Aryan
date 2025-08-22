import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
try:
    df = pd.read_csv("HousingData.csv")
except FileNotFoundError:
    raise FileNotFoundError("❌ 'HousingData.csv' not found in the project directory.")

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Convert feature values to numeric
X = X.apply(pd.to_numeric, errors='coerce')
df = df.loc[X.dropna().index]
X = X.dropna()
y = y.loc[X.index]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot feature coefficients
def plot_coefficients():
    coefs = model.coef_
    features = X.columns
    plt.figure(figsize=(8, 4))
    colors = ['#1f77b4' if c < 0 else '#ff7f0e' for c in coefs]
    sns.barplot(x=coefs, y=features, hue=features, palette=colors, dodge=False, legend=False)
    plt.xlabel("Coefficient Value")
    plt.title("Feature Coefficients (Linear Regression)")
    plt.tight_layout()
    return plt.gcf()

# Plot actual vs predicted values
def plot_predictions():
    preds = model.predict(X_test)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, preds, alpha=0.6, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price (in Lakhs)")
    plt.ylabel("Predicted Price (in Lakhs)")
    plt.title("Actual vs Predicted Prices")
    plt.tight_layout()
    return plt.gcf()

# Predict house price
def predict_price(*inputs):
    try:
        input_df = pd.DataFrame([inputs], columns=X.columns)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        return f"**💰 Predicted Price:** ₹ {round(pred, 2)} Lakhs"
    except Exception as e:
        return f"Error: {str(e)}"

# Generate both plots
def generate_plots():
    return plot_coefficients(), plot_predictions()

# UI layout
default_vals = X.mean().to_dict()
group1_feats = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM"]
group2_feats = [feat for feat in X.columns if feat not in group1_feats]

with gr.Blocks(
    theme=gr.themes.Default(primary_hue="indigo"),
    css="""
    #pred-output {
        font-size: 1.4em;
        font-weight: bold;
        text-align: right;
        padding: 6px 0;
        color: #1a202c;
    }
    #section-heading {
        font-weight: 800;
        margin-top: 30px;
    }
    """
) as demo:

    # Set browser tab title
    gr.HTML("<script>document.title = 'Boston Housing Price Estimator';</script>")

    gr.Markdown("## 🏠 Boston House Price Prediction System (Linear Regression)")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🧮 Enter Feature Values:")

            with gr.Row():
                with gr.Column():
                    inputs_group1 = [
                        gr.Number(label=feat, value=round(default_vals[feat], 2))
                        for feat in group1_feats
                    ]
                with gr.Column():
                    inputs_group2 = [
                        gr.Number(label=feat, value=round(default_vals[feat], 2))
                        for feat in group2_feats
                    ]

            with gr.Row():
                with gr.Column():
                    output = gr.Markdown(elem_id="pred-output")
                    predict_btn = gr.Button("Predict Price")

    predict_btn.click(fn=predict_price, inputs=inputs_group1 + inputs_group2, outputs=output)

    with gr.Row():
        with gr.Column():
            gr.Markdown("📉 Coefficients of Features", elem_id="section-heading")
            coef_plot = gr.Plot()
        with gr.Column():
            gr.Markdown("📈 Actual vs Predicted", elem_id="section-heading")
            prediction_plot = gr.Plot()

    gen_plot_btn = gr.Button("🔄 Generate Plots")
    gen_plot_btn.click(fn=generate_plots, outputs=[coef_plot, prediction_plot])

demo.launch()