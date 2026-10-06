import dash_bootstrap_components as dbc
from dash import dcc, html

from imote import ids
from imote.config import DEFAULT_COLLAPSE_LEVEL

def make_edit_tree_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Button(
                    "Edit Tree",
                    id=ids.EDIT_TREE_TOGGLE_BUTTON,
                    color="link",
                    class_name="p-0 text-decoration-none",
                )
            ),
            dbc.Collapse(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div("Tree structure", className="fw-bold mb-2"),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        dbc.Button("Collapse/expand selected node", id=ids.BTN_COLLAPSE_EXPAND, size="sm"),
                                        dbc.Button("Subtree from selected node", id=ids.BTN_SUBTREE, size="sm"),
                                        dbc.Button("Prune from selected node", id=ids.BTN_PRUNE, size="sm"),
                                        dbc.Button("Expand all nodes", id=ids.BTN_EXPAND_ALL, size="sm"),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                                    children=[
                                        dbc.Button("Collapse nodes to level:", id=ids.BTN_COLLAPSE_LEVEL, size="sm"),
                                        dcc.Input(
                                            id=ids.INPUT_COLLAPSE_LEVEL, type="number", value=DEFAULT_COLLAPSE_LEVEL,
                                            style={"width": "80px"}
                                        ),
                                        dcc.Checklist(
                                            id=ids.COLLAPSE_LEVEL_OPTIONS,
                                            options=[{"label": "include linear nodes", "value": "include_lin"}],
                                            value=["include_lin"],
                                            inline=True,
                                        ),
                                    ],
                                ),
                            ],
                            className="mb-3",
                        ),

                        html.Div(
                            [
                                html.Div("Display switches", className="fw-bold mb-2"),
                                html.Div(
                                    [
                                        html.Span("Linear nodes: "),
                                        dbc.RadioItems(
                                            id=ids.SWITCH_COMBINE_LIN,
                                            options=[
                                                {"label": "Show all", "value": 0},
                                                {"label": "Combine into node", "value": 1},
                                                {"label": "Combine into edge", "value": 2},
                                            ],
                                            value=0,
                                            inline=True,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.Div(
                                    [
                                        html.Span("Node size: "),
                                        dbc.RadioItems(
                                            id=ids.SWITCH_MINIMAL,
                                            options=[
                                                {"label": "Normal", "value": 0},
                                                {"label": "Minimal", "value": 1},
                                                {"label": "Tiny", "value": 2},
                                            ],
                                            value=0,
                                            inline=True,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        dbc.Switch(
                                            id=ids.SWITCH_NODE_PLOTS,
                                            label="Show all node plots",
                                            value=False,
                                        ),
                                        dbc.Switch(
                                            id=ids.SWITCH_COLOR_FEATURES,
                                            label="Features colors",
                                            value=False,
                                        ),
                                        dbc.Switch(
                                            id=ids.SWITCH_RSS,
                                            label="Show RSS",
                                            value=False,
                                        ),
                                    ],
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Confirmation"),
                                dbc.ModalBody([
                                    html.P([
                                        "Are you sure you want to show all node plots?",
                                        html.Br(),
                                        "This may take a while for large trees."
                                    ]),
                                    dbc.Checkbox(id=ids.MODAL_CHECK_DONT_ASK, label="Don't ask me again."),
                                ]),
                                dbc.ModalFooter([
                                    dbc.Button("No, cancel.", id=ids.MODAL_BTN_CANCEL),
                                    dbc.Button("Yes, show all node plots", id=ids.MODAL_BTN_CONFIRM)
                                ])
                            ],
                            id=ids.MODAL_NODE_PLOTS,
                            is_open=False,
                            backdrop=False
                        ),
                        html.Div(
                            [
                                html.Div("Show data flow", className="fw-bold mb-2"),
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        dbc.Switch(
                                            id=ids.SWITCH_DATA_EDGE_WIDTH,
                                            label="With edge width",
                                            value=False,
                                        ),
                                        dbc.Switch(
                                            id=ids.SWITCH_DATA_NODE_SIZE,
                                            label="With node size",
                                            value=False,
                                        ),
                                    ],
                                ),
                            ],
                            className="mb-3",
                        ),
                    ]
                ),
                id=ids.EDIT_TREE_COLLAPSE,
                is_open=False,
            ),
        ],
        class_name="mb-3",
    )
