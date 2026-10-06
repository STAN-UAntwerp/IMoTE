"""
TODO: This page is a work in progress
"""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from imote import ids

dash.register_page(__name__, path="/explain", name="Explain")

layout = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Button(
                    "Show global explanation plot",
                    id=ids.EXPLAIN_COMPUTE_BUTTON,
                    class_name="mb-3 w-100",
                ),
                dcc.Dropdown(
                    id=ids.EXPLAIN_PLOT_TYPE_DROPDOWN,
                    options=[
                        "predsplot",
                        "linear contribution",
                        "split contribution",
                        "global contribution",
                    ],
                    placeholder="Plot type",
                ),
            ],
            width=3,
        ),
        dbc.Col(
            html.Div(id=ids.EXPLAIN_OUTPUT_CONTAINER),
            width=9,
        ),
        html.Div("This is a work in progress, and doesn't work yet."),
    ]
)
