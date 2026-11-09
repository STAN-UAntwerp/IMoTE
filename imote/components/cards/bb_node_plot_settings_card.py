import dash_bootstrap_components as dbc
from dash import html, dcc

from imote import ids
from imote.config import DEFAULT_DISPLAY_TYPE, DEFAULT_FIG_H, DEFAULT_FIG_W, DEFAULT_NMAX

LABEL_CLASS = "small fw-bold text-muted mb-1"

def make_settings_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Plot Settings", className="mb-2 fw-bold text-primary white-space"),
                    dbc.Row(
                        [
                            dbc.Col(dbc.Label("Figure size (inches)", className=f"{LABEL_CLASS} mb-0"), width="auto"),
                            dbc.Col(
                                dbc.InputGroup(
                                    [
                                        dbc.Input(id=ids.INPUT_FIG_W, type="number", value=DEFAULT_FIG_W, placeholder="W"),
                                        dbc.InputGroupText("×"),
                                        dbc.Input(id=ids.INPUT_FIG_H, type="number", value=DEFAULT_FIG_H, placeholder="H"),
                                    ],
                                    size="sm",
                                ),
                                md=5,
                                sm=8,
                            ),
                        ],
                        className="g-2 mb-2 align-items-center",
                    ),
                    html.Fieldset(
                        id=ids.PREDSPLOT_SETTINGS,
                        className="border-top pt-2 mb-3",
                        children=[
                            html.Div("Prediction plot (leaf nodes)", className=LABEL_CLASS),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Display type", className=LABEL_CLASS),
                                            dbc.Select(
                                                id=ids.INPUT_DISPLAY_TYPE,
                                                options=[
                                                    {"label": "Histogram", "value": "histogram"},
                                                    {"label": "Density", "value": "density"},
                                                ],
                                                value=DEFAULT_DISPLAY_TYPE,
                                                size="sm",
                                            ),
                                        ],
                                        md=4,
                                        sm=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("# Variables", className=LABEL_CLASS),
                                            dbc.Input(id=ids.INPUT_NMAX, type="number", value=DEFAULT_NMAX, min=1, size="sm"),
                                        ],
                                        md=3,
                                        sm=6,
                                    ),
                                    dbc.Col(
                                        dbc.Checklist(
                                            id=ids.PREDSPLOT_TYPE,
                                            options=[{"label": "Global predsplot", "value": "type2"}],
                                            value=[],
                                            switch=True,
                                        ),
                                        md=5,
                                        sm=12,
                                        className="d-flex align-items-end",
                                    ),
                                ],
                                className="g-2 mb-2",
                            ),
                            dbc.Checklist(
                                id=ids.PREDSPLOT_OPTIONS,
                                options=[
                                    {"label": "Use intercept", "value": "intercept"},
                                    {"label": "Truncate total pred", "value": "truncate"},
                                    {"label": "Staircase (highlight only)", "value": "staircase"},
                                ],
                                value=[],
                                inline=True,
                                switch=True,
                            ),
                        ],
                    ),
                    dcc.Download(id=ids.DOWNLOAD_PLOT),
                    html.Div([
                        dbc.Stack([
                            dbc.Button(
                                "Refit Plot",
                                id=ids.BTN_REFIT_PLOTS,
                                color="primary",
                                size="sm",
                                className="px-4 fw-bold flex-fill",
                            ),
                            dbc.Button(
                                "Download Plot",
                                id=ids.BTN_DOWNLOAD_PLOT,
                                color="secondary",
                                size="sm",
                                className="px-4 fw-bold flex-fill",
                            ),
                        ], direction="horizontal", gap=2)],
                        className="d-flex",
                    ),

                    # Pin the node plot on top of the graph, made with the settings above
                    html.Div(
                        className="border-top pt-2 mt-3",
                        children=[
                            html.Div("Plots on the graph", className=LABEL_CLASS),
                            html.Small(
                                "Pin this plot next to its node in the graph, you can move and resize it.",
                                className="text-muted d-block mb-2",
                            ),
                            dbc.Stack([
                                dbc.Button(
                                    [html.I(className="bi bi-pin-angle me-1"), "Pin plot"],
                                    id=ids.BTN_ADD_GRAPH_PLOT,
                                    color="primary",
                                    size="sm",
                                    className="fw-bold flex-fill text-nowrap",
                                    title="Pin the plot of the selected node on the graph",
                                ),
                                dbc.Button(
                                    [html.I(className="bi bi-x-circle me-1"), "Unpin all"],
                                    id=ids.BTN_CLEAR_GRAPH_PLOTS,
                                    color="danger",
                                    outline=True,
                                    size="sm",
                                    className="fw-bold flex-fill text-nowrap",
                                    title="Remove all pinned plots from the graph",
                                ),
                            ], direction="horizontal", gap=2),
                        ],
                    ),
                ]
            ),
        ],
        className="mb-3 shadow-sm",
    )
