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
                            html.Div("Predsplot (leaf nodes)", className=LABEL_CLASS),
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
                                "Download Plot",
                                id=ids.BTN_DOWNLOAD_PLOT,
                                color="secondary",
                                size="sm",
                                className="px-4 fw-bold flex-fill",
                            ),
                            dbc.Button(
                                "Refit Plot",
                                id=ids.BTN_REFIT_PLOTS,
                                color="primary",
                                size="sm",
                                className="px-4 fw-bold flex-fill",
                            ),
                        ], direction="horizontal", gap=2)],
                        className="d-flex justify-content-end",
                    ),
                ]
            ),
        ],
        className="mb-3 shadow-sm",
    )
