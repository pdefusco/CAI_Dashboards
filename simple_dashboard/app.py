import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample dataset
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Value_1": [10, 15, 13, 17],
    "Value_2": [16, 5, 11, 9]
})

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Simple Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Simple Interactive Dashboard"),

    dcc.Dropdown(
        id="value-selector",
        options=[
            {"label": "Value 1", "value": "Value_1"},
            {"label": "Value 2", "value": "Value_2"}
        ],
        value="Value_1",
        clearable=False
    ),

    dcc.Graph(id="bar-chart")
])

# Callback to update graph
@app.callback(
    Output("bar-chart", "figure"),
    Input("value-selector", "value")
)
def update_chart(selected_value):
    fig = px.bar(df, x="Category", y=selected_value,
                 title=f"Bar Chart of {selected_value}")
    return fig
