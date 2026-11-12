import dash_bootstrap_components as dbc
from dash import html

from imote import ids
from imote.config import (
    SHOWN_METHODS,
    DEFAULT_DATASET_NAME,
    DEFAULT_METHOD_NAME,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_MODEL_DEPTH,
    DEFAULT_MIN_SAMPLE_LEAF,
    DEFAULT_MIN_SAMPLE_SPLIT,
    ADAPTER_OPTIONS,
    DEFAULT_ADAPTER_NAME,
)
from imote.components.dataset_modal import make_dataset_modal
from imote.dataset.dataset_registry import dropdown_options

NUMBER_INPUT_STYLE = {"width": "120px"}

def make_new_tree_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Button(
                    "New Tree",
                    id=ids.NEW_TREE_TOGGLE_BUTTON,
                    color="link",
                    class_name="p-0 text-decoration-none",
                )
            ),
            dbc.Collapse(
                dbc.Spinner(
                    dbc.CardBody(
                        [
                            html.Div(
                                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                children=[
                                    html.Div(
                                        [
                                            html.Div("Dataset", style={"marginBottom": "4px"}),
                                            dbc.InputGroup(
                                                [
                                                    dbc.Select(
                                                        id=ids.INPUT_DATASET,
                                                        options=dropdown_options(),
                                                        value=DEFAULT_DATASET_NAME,
                                                    ),
                                                    dbc.Button(
                                                        html.I(className="bi bi-pencil"),
                                                        id=ids.BTN_DATASET_EDIT,
                                                        color="secondary",
                                                        outline=True,
                                                        title="Adjust the settings of this dataset",
                                                    ),
                                                    dbc.Button(
                                                        [html.I(className="bi bi-plus-lg me-1"), "Add"],
                                                        id=ids.BTN_DATASET_ADD,
                                                        color="secondary",
                                                        outline=True,
                                                        title="Add a new dataset from PMLB, the UCI repository or a file",
                                                    ),
                                                ],
                                                size="sm",
                                                style={"width": "100%", "maxWidth": "460px", "flexWrap": "nowrap"},
                                            ),
                                        ],
                                        style={"flexBasis": "100%"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Method", style={"marginBottom": "4px"}),
                                            dbc.Select(
                                                id=ids.INPUT_METHOD,
                                                options=[{"label": m["label"], "value": value} for value, m in SHOWN_METHODS.items()],
                                                value=DEFAULT_METHOD_NAME,
                                                style={"width": "220px"},
                                                size="sm",
                                            ),
                                            html.Small(id=ids.METHOD_NOTE, className="text-muted d-block",
                                                       style={"maxWidth": "220px"}),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Max depth", style={"marginBottom": "4px"}),
                                            dbc.Input(id=ids.INPUT_MAX_DEPTH, type="number", value=DEFAULT_MAX_DEPTH, size="sm",
                                                      style=NUMBER_INPUT_STYLE),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Max model depth", style={"marginBottom": "4px"}),
                                            dbc.Input(
                                                id=ids.INPUT_MAX_MODEL_DEPTH, type="number", value=DEFAULT_MAX_MODEL_DEPTH, size="sm",
                                                style=NUMBER_INPUT_STYLE,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Min sample split", style={"marginBottom": "4px"}),
                                            dbc.Input(
                                                id=ids.INPUT_MIN_SAMPLE_SPLIT, type="number", value=DEFAULT_MIN_SAMPLE_SPLIT, size="sm",
                                                style=NUMBER_INPUT_STYLE,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Min sample leaf", style={"marginBottom": "4px"}),
                                            dbc.Input(
                                                id=ids.INPUT_MIN_SAMPLE_LEAF, type="number", value=DEFAULT_MIN_SAMPLE_LEAF, size="sm",
                                                style=NUMBER_INPUT_STYLE,
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                children=[
                                    dbc.Button("Fit Tree", id=ids.BTN_FIT_NEW_TREE, size="sm"),
                                    dbc.Button("Reload tree", id=ids.BTN_RELOAD_TREE, size="sm"),
                                    dbc.Button("Save shown tree", id=ids.BTN_SAVE_TREE, size="sm", color='secondary'),
                                    dbc.Button("Download shown tree", id=ids.BTN_SAVE_TREE_SVG, size="sm", color='secondary'),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                children=[
                                    dbc.Input(
                                        id=ids.INPUT_LOAD_TREE,
                                        type="text",
                                        placeholder="path to a saved tree (.pkl), empty = most recent",
                                        size="sm",
                                    ),
                                    dbc.Button("Load saved viz_tree", id=ids.BTN_LOAD_TREE, size="sm", class_name="text-nowrap"),
                                ],
                            ),
                            html.Hr(style={"margin": "12px 0"}),
                            dbc.Button(
                                "Load a tree with an adapter",
                                id=ids.ADAPTER_TOGGLE_BUTTON,
                                color="link",
                                size="sm",
                                class_name="p-0 mb-2 text-decoration-none",
                            ),
                            dbc.Collapse(
                                html.Div(
                                    [
                                        html.Div(
                                            style={"display": "flex", "gap": "10px", "flexWrap": "nowrap", "marginBottom": "10px"},
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div("Adapter", style={"marginBottom": "4px"}),
                                                        dbc.Select(
                                                            id=ids.INPUT_ADAPTER,
                                                            options=ADAPTER_OPTIONS,
                                                            value=DEFAULT_ADAPTER_NAME,
                                                            style={"width": "220px"},
                                                            size="sm",
                                                        ),
                                                    ],
                                                    style={"flex": "1"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div("Dataset for Adapter", style={"marginBottom": "4px"}),
                                                        html.Div(
                                                            "Uses the dataset selected above (this can be a subset of "
                                                            "the training data)." ,
                                                            style={"marginBottom": "0px", "fontSize": "12px", "color": "#666"},
                                                        ),
                                                    ],
                                                    style={"flex": "1"},
                                                ),
                                            ],
                                        ),
                                        html.Div("Adapter model", style={"marginBottom": "4px"}),
                                        html.Div(
                                            style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                            children=[
                                                dbc.Input(
                                                    id=ids.INPUT_LOAD_TREE_ADAPTER,
                                                    type="text",
                                                    placeholder="path to the exported model, e.g. ~/models/tree.json",
                                                    size="sm",
                                                ),
                                                dbc.Button("Load with Adapter", id=ids.BTN_LOAD_TREE_ADAPTER, size="sm", class_name="text-nowrap"),
                                            ],
                                        ),
                                    ]
                                ),
                                id=ids.ADAPTER_COLLAPSE,
                                is_open=False,
                            ),
                            html.Div(id=ids.TRIGGER_FOR_SPINNER, style={"display": "none"}),
                        ],
                    ),
                    delay_show=300,
                ),
                id=ids.NEW_TREE_COLLAPSE,
                is_open=False,
            ),
            make_dataset_modal(),
        ],
        class_name="mb-3",
    )
