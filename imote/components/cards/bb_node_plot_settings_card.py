import dash_bootstrap_components as dbc
from dash import html, dcc

from imote import ids
from imote.config import DEFAULT_DISPLAY_TYPE, DEFAULT_FIG_H, DEFAULT_FIG_W, DEFAULT_NMAX

def make_settings_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6("Plot Settings", className="mb-0 fw-bold text-primary white-space"),
                    # Row 1: Main Controls Grid
                    dbc.Row(
                        [
                            # Display Type Dropdown
                            dbc.Col(
                                [
                                    dbc.Label("Display Type", className="small fw-bold text-muted mb-1"),
                                    dbc.Select(
                                        id=ids.INPUT_DISPLAY_TYPE,
                                        options=[
                                            {"label": "Histogram", "value": "histogram"},
                                            {"label": "Density", "value": "density"},
                                        ],
                                        value=DEFAULT_DISPLAY_TYPE,
                                    ),
                                ],
                                md=4,
                                sm=12,
                            ),
                            # Number of Variables Input
                            dbc.Col(
                                [
                                    dbc.Label("# Variables", className="small fw-bold text-muted mb-1"),
                                    dbc.Input(
                                        id=ids.INPUT_NMAX,
                                        type="number",
                                        value=DEFAULT_NMAX,
                                        min=1,
                                    ),
                                ],
                                md=3,
                                sm=6,
                            ),
                            # Figure Size Inputs
                            dbc.Col(
                                [
                                    dbc.Label("Figure Size (Inches)", className="small fw-bold text-muted mb-1"),
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id=ids.INPUT_FIG_W,
                                                type="number",
                                                value=DEFAULT_FIG_W,
                                                placeholder="W",
                                            ),
                                            dbc.InputGroupText("×"),
                                            dbc.Input(
                                                id=ids.INPUT_FIG_H,
                                                type="number",
                                                value=DEFAULT_FIG_H,
                                                placeholder="H",
                                            ),
                                        ]
                                    ),
                                ],
                                md=5,
                                sm=6,
                            ),
                        ],
                        className="g-3 mb-3",
                    ),

                    # Row 2: Checkboxes
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Options", className="small fw-bold text-muted d-block mb-1"),
                                    dbc.Checklist(
                                        id=ids.PREDSPLOT_OPTIONS,
                                        options=[
                                            {"label": "Use intercept", "value": "intercept"},
                                            {"label": "Truncate total pred", "value": "truncate"},
                                            {"label": "Staircase (highlight only)", "value": "staircase"},
                                        ],
                                        value=[],
                                        inline=True,
                                        switch=True,  # Optional: set to False if you want square checkboxes
                                    ),
                                ],
                                md=8,
                                sm=12,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Plot Type", className="small fw-bold text-muted d-block mb-1"),
                                    dbc.Checklist(
                                        id=ids.PREDSPLOT_TYPE,
                                        options=[{"label": "Global predsplot", "value": "type2"}],
                                        value=[],
                                        inline=True,
                                        switch=True,
                                    ),
                                ],
                                md=4,
                                sm=12,
                            ),
                        ],
                        className="g-3 mb-3 pt-2",
                    ),

                    # Action Buttons
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
