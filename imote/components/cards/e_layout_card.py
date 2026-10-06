import dash_bootstrap_components as dbc
from dash import dcc, html

from imote import ids
from imote.config import DEFAULT_NODE_SEP, DEFAULT_RANK_SEP

def make_layout_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Button(
                    "Layout",
                    id=ids.LAYOUT_TOGGLE_BUTTON,
                    color="link",
                    class_name="p-0 text-decoration-none",
                )
            ),
            dbc.Collapse(
                dbc.CardBody(
                    [
                        dbc.Button("Refit layout", id=ids.BTN_LAYOUT, size="sm", class_name="mb-3"),
                        html.Div(
                            [
                                "Display direction:",
                                dcc.RadioItems(
                                    id=ids.DISPLAY_DIRECTION,
                                    options=["Top Bottom", "Left Right"],
                                    value="Top Bottom",
                                    inline=True,
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                "Layout type:",
                                dcc.RadioItems(
                                    id=ids.NODE_RANKER,
                                    options=["Compact", "Align leaves to bottom"],
                                    value="Compact",
                                    inline=True,
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                "Nodes vertical gap",
                                dcc.Slider(id=ids.RANK_SEP, min=0, max=150, step=1, value=DEFAULT_RANK_SEP, marks=None),
                            ],
                            className="mb-2",
                        ),
                        html.Div(
                            [
                                "Nodes horizontal gap",
                                dcc.Slider(id=ids.NODE_SEP, min=0, max=150, step=1, value=DEFAULT_NODE_SEP, marks=None),
                            ]
                        ),
                    ]
                ),
                id=ids.LAYOUT_COLLAPSE,
                is_open=False,
            ),
        ],
        class_name="mb-3",
    )
