#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/
try:
    import cml.utils_v1 as utils
    cdsw = utils._emulate_cdsw()
except ImportError:
    import cdsw

import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
from sklearn.metrics import confusion_matrix

from src.api import ApiUtility
import cmlapi

# ----------------------------
# Load and flatten metrics
# ----------------------------
USERNAME = os.environ["PROJECT_OWNER"]
project_id = os.environ["CDSW_PROJECT_ID"]

client = cmlapi.default_client()
apiUtil = ApiUtility()
listModelsResponse = client.list_models(project_id)
all_models_data = []

for model in range(len(listModelsResponse.models)):
    modelName = listModelsResponse.models[model].name
    Model_CRN = apiUtil.get_latest_deployment_details(model_name=modelName)["model_crn"]
    Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=modelName)["latest_deployment_crn"]

    model_metrics = cdsw.read_metrics(model_crn=Model_CRN, model_deployment_crn=Deployment_CRN)

    record = {
        "model_name": modelName,
        "model_crn": Model_CRN,
        "deployment_crn": Deployment_CRN
    }
    if isinstance(model_metrics, dict):
        record.update(model_metrics)
    all_models_data.append(record)

df = pd.json_normalize(all_models_data)

records = []
for _, row in df.iterrows():
    model_name = row.get("model_name")
    model_crn = row.get("model_crn")
    metrics_list = row.get("metrics", [])

    if not isinstance(metrics_list, list):
        continue

    for request in metrics_list:
        flat = {
            "model_name": model_name,
            "final_label": request.get("metrics", {}).get("final_label"),
            "probability": request.get("metrics", {}).get("probability"),
            "y_pred": request.get("metrics", {}).get("y_pred")
        }
        records.append(flat)

metrics_flat_df = pd.DataFrame(records)
metrics_flat_df["final_label"] = pd.to_numeric(metrics_flat_df["final_label"], errors='coerce').fillna(0).astype(int)
metrics_flat_df["probability"] = pd.to_numeric(metrics_flat_df["probability"], errors='coerce').fillna(0)

# ----------------------------
# Initialize Dash
# ----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Model ROI Dashboard",
            style={"font-size": "32px", "font-family": "Arial, sans-serif", "text-align": "center"}),

    html.P(
    "This dashboard allows you to compare the financial ROI of each model. "
    "Increasing the decision threshold makes the model more conservative, predicting fewer positives, which may reduce false positives but also may miss true positives. "
    "Lowering the threshold makes the model more aggressive, predicting more positives, which may capture more true positives but also increases false positives. "
    "This has a direct impact on the number of records classified in each of the four categories, and because each is associated with a different income or penalty, this ultimately impacts Net Revenue as well. "
    "Net Revenue = TP Revenue + TN Revenue - FP Penalty - Cost to Operate Model.",
    style={"font-size": "18px", "color": "gray", "margin-bottom": "30px", "text-align": "center"}
    ),


    html.Div([
        html.Div([html.Label("Financial Revenue (Actual Target=1):"), dcc.Input(id="revenue-class-1", type="number", value=100, step=10)]),
        html.Div([html.Label("Financial Revenue (Actual Target=0):"), dcc.Input(id="revenue-class-0", type="number", value=10, step=10)]),
        html.Div([html.Label("Penalty per False Positive:"), dcc.Input(id="penalty-fp", type="number", value=50, step=10)]),
        html.Div([html.Label("Cost to Operate Model:"), dcc.Input(id="cost-operate", type="number", value=0, step=10)])
    ], style={"display": "flex", "gap": "20px", "margin-bottom": "30px"}),

    html.Div([
        html.Div([dcc.Graph(id="confusion-matrix-heatmap")], style={"width": "70%"}),
        html.Div([
            html.Label("Decision Threshold:"),
            dcc.Slider(
                id="threshold-slider",
                min=0.0, max=1.0, step=0.01, value=0.5,
                marks={0:"0.0", 0.25:"0.25",0.5:"0.5",0.75:"0.75",1:"1.0"},
                tooltip={"placement": "left", "always_visible": True},
                vertical=True, verticalHeight=400
            )
        ], style={"width": "10%", "margin-left": "30px", "margin-top": "50px"})
    ], style={"display": "flex"}),

    html.Div(id="selected-model-display", style={"margin-top": "20px", "font-size": "20px"}),

    html.H3("Model Summary", style={"margin-top": "40px"}),
    dash_table.DataTable(
        id="model-summary-table",
        columns=[
            {"name":"Model Name","id":"model_name"},
            {"name":"TP","id":"TP"},
            {"name":"TN","id":"TN"},
            {"name":"FP","id":"FP"},
            {"name":"FN","id":"FN"},
            {"name":"Net Revenue","id":"Net_Revenue", "type":"numeric", "format": {"locale": {"symbol":["$",""]}, "specifier": "$,.2f"}}
        ],
        row_selectable="single",
        selected_rows=[0],
        style_cell={"textAlign": "center", "font-family": "Arial"},
        style_header={"backgroundColor":"#f0f0f0", "fontWeight":"bold"}
    ),

    html.H3("Net Revenue Comparison", style={"margin-top": "40px"}),
    dcc.Graph(id="net-revenue-bar")
])

# ----------------------------
# Callbacks
# ----------------------------
@app.callback(
    Output("model-summary-table", "data"),
    Input("threshold-slider", "value"),
    Input("revenue-class-1", "value"),
    Input("revenue-class-0", "value"),
    Input("penalty-fp", "value"),
    Input("cost-operate", "value")
)
def update_model_table(threshold, revenue_1, revenue_0, penalty_fp, cost_operate):
    table_data = []
    for model in metrics_flat_df["model_name"].unique():
        df_model = metrics_flat_df[metrics_flat_df["model_name"]==model]
        y_true = df_model["final_label"]
        y_pred = (df_model["probability"] >= threshold).astype(int)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
        except:
            tn = fp = fn = tp = 0
        net_revenue = tp*revenue_1 + tn*revenue_0 - fp*penalty_fp - cost_operate
        table_data.append({
            "model_name": model,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Net_Revenue": net_revenue
        })
    return table_data

@app.callback(
    [Output("confusion-matrix-heatmap", "figure"),
     Output("selected-model-display", "children"),
     Output("net-revenue-bar", "figure")],
    [Input("model-summary-table", "selected_rows"),
     Input("threshold-slider", "value"),
     Input("revenue-class-1", "value"),
     Input("revenue-class-0", "value"),
     Input("penalty-fp", "value"),
     Input("cost-operate", "value")]
)
def update_confusion_and_revenue(selected_rows, threshold, revenue_1, revenue_0, penalty_fp, cost_operate):
    # --- Confusion Matrix ---
    if not selected_rows:
        cm_fig = px.imshow([[0,0],[0,0]], text_auto=False, color_continuous_scale=px.colors.sequential.Greens)
        display_text = "No model selected"
    else:
        selected_model = metrics_flat_df["model_name"].unique()[selected_rows[0]]
        df_model = metrics_flat_df[metrics_flat_df["model_name"]==selected_model]
        y_true = df_model["final_label"]
        y_pred = (df_model["probability"] >= threshold).astype(int)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
        except:
            tn = fp = fn = tp = 0
        cm_df = pd.DataFrame([[tn, fp],[fn, tp]], index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"])
        cm_fig = px.imshow(cm_df, color_continuous_scale=px.colors.sequential.Greens, text_auto=False)

        # Add only TP/TN/FP/FN labels inside
        annotations = []
        labels_map = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}
        for i, row in enumerate(cm_df.index):
            for j, col in enumerate(cm_df.columns):
                value = cm_df.iloc[i,j]
                label = labels_map[(i,j)]
                annotations.append(dict(
                    x=j, y=i, text=f"{label}={value}", showarrow=False,
                    font=dict(color="black", size=16)
                ))
        cm_fig.update_layout(annotations=annotations, title=f"Confusion Matrix for {selected_model}", title_x=0.5)
        display_text = f"Selected Model: {selected_model} | TP={tp}, TN={tn}, FP={fp}, FN={fn}"

    # --- Net Revenue Bar Chart ---
    net_rev_data = []
    for model in metrics_flat_df["model_name"].unique():
        df_model = metrics_flat_df[metrics_flat_df["model_name"]==model]
        y_true = df_model["final_label"]
        y_pred = (df_model["probability"] >= threshold).astype(int)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
        except:
            tn = fp = fn = tp = 0
        net_revenue = tp*revenue_1 + tn*revenue_0 - fp*penalty_fp - cost_operate
        net_rev_data.append({"model_name": model, "Net_Revenue": net_revenue})

    net_rev_df = pd.DataFrame(net_rev_data)
    net_rev_fig = px.bar(net_rev_df, x="model_name", y="Net_Revenue", color="Net_Revenue",
                         color_continuous_scale=px.colors.sequential.Greens,
                         text="Net_Revenue")
    net_rev_fig.update_layout(title="Net Revenue Comparison Across Models", title_x=0.5, xaxis_title="Model", yaxis_title="Net Revenue")

    return cm_fig, display_text, net_rev_fig

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("CDSW_APP_PORT", 8050)))
