import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import os

# ----------------------------
# 1. Create synthetic data
# ----------------------------
X, y = make_classification(
    n_samples=1000, n_features=10,
    n_informative=5, n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------
# 2. Fit XGBoost model
# ----------------------------
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Predict probabilities (we’ll threshold manually)
y_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------
# 3. Initialize Dash app
# ----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Threshold Dashboard for XGBoost Classifier"),

    # Slider for threshold
    html.Label("Decision Threshold:"),
    dcc.Slider(
        id="threshold-slider",
        min=0.0, max=1.0, step=0.01, value=0.5,
        marks={0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1.0"},
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    html.Div(id="accuracy-text", style={"margin-top": "20px", "font-size": "18px"}),

    dcc.Graph(id="confusion-matrix-heatmap")
])

# ----------------------------
# 4. Define callbacks
# ----------------------------
@app.callback(
    [Output("accuracy-text", "children"),
     Output("confusion-matrix-heatmap", "figure")],
    [Input("threshold-slider", "value")]
)
def update_outputs(threshold):
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Create dataframe for heatmap
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    # Plot heatmap
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    acc_text = f"Accuracy at threshold {threshold:.2f}: {accuracy:.3f} | TP={tp}, TN={tn}, FP={fp}, FN={fn}"

    return acc_text, fig


# ----------------------------
# 5. Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))
