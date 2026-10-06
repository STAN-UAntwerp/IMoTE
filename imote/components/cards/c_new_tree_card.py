import dash_bootstrap_components as dbc
from dash import dcc, html

from imote import ids
from imote.config import (
    DEFAULT_DATASET_NAME,
    METHOD_OPTIONS,
    DEFAULT_METHOD_NAME,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_MODEL_DEPTH,
    DEFAULT_MIN_SAMPLE_LEAF,
    DEFAULT_MIN_SAMPLE_SPLIT,
    NO_FILE_SELECTED_PLACEHOLDER,
    ADAPTER_OPTIONS,
    DEFAULT_ADAPTER_NAME,
    DIR_BASE,
)
from imote.dataset.dataset_registry import build_dropdown_options

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
                                            dcc.Dropdown(
                                                id=ids.INPUT_DATASET,
                                                options=build_dropdown_options(),
                                                value=DEFAULT_DATASET_NAME,
                                                clearable=False,
                                                style={"width": "220px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Method", style={"marginBottom": "4px"}),
                                            dcc.Dropdown(
                                                id=ids.INPUT_METHOD,
                                                options=METHOD_OPTIONS,
                                                value=DEFAULT_METHOD_NAME,
                                                clearable=False,
                                                style={"width": "220px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Max depth", style={"marginBottom": "4px"}),
                                            dcc.Input(id=ids.INPUT_MAX_DEPTH, type="number", value=DEFAULT_MAX_DEPTH),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Max model depth", style={"marginBottom": "4px"}),
                                            dcc.Input(
                                                id=ids.INPUT_MAX_MODEL_DEPTH, type="number", value=DEFAULT_MAX_MODEL_DEPTH
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Min sample split", style={"marginBottom": "4px"}),
                                            dcc.Input(
                                                id=ids.INPUT_MIN_SAMPLE_SPLIT, type="number", value=DEFAULT_MIN_SAMPLE_SPLIT
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Min sample leaf", style={"marginBottom": "4px"}),
                                            dcc.Input(
                                                id=ids.INPUT_MIN_SAMPLE_LEAF, type="number", value=DEFAULT_MIN_SAMPLE_LEAF
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                children=[
                                    dbc.Button("Fit Tree", id=ids.BTN_FIT_NEW_TREE, size="sm"),
                                    dbc.Button("Save shown tree", id=ids.BTN_SAVE_TREE, size="sm"),
                                    dbc.Button("Download shown tree", id=ids.BTN_SAVE_TREE_SVG, size="sm"),
                                    dbc.Button("Reload tree", id=ids.BTN_RELOAD_TREE, size="sm"),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px"},
                                children=[
                                    dbc.Input(
                                        id=ids.INPUT_LOAD_TREE,
                                        type="text",
                                        placeholder="path to saved_viz_trees/...",
                                    ),
                                    dbc.Button("Load saved viz_tree", id=ids.BTN_LOAD_TREE, size="sm"),
                                ],
                            ),
                            html.Hr(style={"margin": "12px 0"}),
                            html.Div(
                                style={"display": "flex", "gap": "10px", "flexWrap": "nowrap", "marginBottom": "10px"},
                                children=[
                                    html.Div(
                                        [
                                            html.Div("Adapter", style={"marginBottom": "4px"}),
                                            dcc.Dropdown(
                                                id=ids.INPUT_ADAPTER,
                                                options=ADAPTER_OPTIONS,
                                                value=DEFAULT_ADAPTER_NAME,
                                                clearable=False,
                                                style={"width": "220px"},
                                            ),
                                        ],
                                        style={"flex": "1"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Dataset for Adapter", style={"marginBottom": "4px"}),
                                            html.Div(
                                                "The Training dataset selected above is also the one used when loading via the "
                                                "adapter. (It can be a subset from the total dataset used for training.)",
                                                style={"marginBottom": "8px", "fontSize": "12px", "color": "#666"},
                                            ),
                                        ],
                                        style={"flex": "1"},
                                    ),
                                ],
                            ),
                            html.Div("Adapter model", style={"marginBottom": "4px"}),
                            html.Div(
                                style={"display": "flex", "gap": "8px"},
                                children=[
                                    dbc.Input(
                                        id=ids.INPUT_LOAD_TREE_ADAPTER,
                                        type="text",
                                        placeholder="path to model_adapter/...",
                                        value=str(DIR_BASE / "output" / "r_partykit_trees" / "tree.json"),
                                    ),
                                    dbc.Button("Load with Adapter", id=ids.BTN_LOAD_TREE_ADAPTER, size="sm"),
                                ],
                            ),
                            html.Div(id=ids.TRIGGER_FOR_SPINNER, style={"display": "none"}),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader("Upload new dataset as CSV file"),
                                    dbc.ModalBody(
                                        [
                                            dcc.Upload(
                                                id=ids.MODAL_CSV_UPLOAD,
                                                children=html.Div("Drag and drop or click to select a CSV file"),
                                                style={
                                                    "width": "100%",
                                                    "padding": "10px",
                                                    "border": "1px dashed gray",
                                                    "textAlign": "center",
                                                },
                                                multiple=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "File uploaded:",
                                                        style={"margin": 0, "marginRight": "8px",
                                                               "whiteSpace": "nowrap"},
                                                    ),
                                                    html.Span(
                                                        id=ids.MODAL_CSV_FILE_NAME,
                                                        children=NO_FILE_SELECTED_PLACEHOLDER,
                                                        style={"color": "#6c757d", "overflowWrap": "anywhere"},
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "marginTop": "12px",
                                                    "marginBottom": "4px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Target column name",
                                                               style={"marginTop": "12px", "marginBottom": "4px"}),
                                                    dbc.Input(
                                                        id=ids.MODAL_CSV_INPUT_TARGET_COL,
                                                        placeholder="e.g. price, count, time, 0, 7, -1, ...",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id=ids.MODAL_CSV_FEEDBACK,
                                                style={"marginTop": "8px", "color": "red"},
                                            ),
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        [
                                            dbc.Button(
                                                "Cancel",
                                                id=ids.MODAL_CSV_BTN_CANCEL,
                                                color="secondary",
                                            ),
                                            dbc.Button(
                                                "Confirm",
                                                id=ids.MODAL_CSV_BTN_CONFIRM,
                                                color="primary",
                                            ),
                                        ]
                                    ),
                                ],
                                id=ids.MODAL_CSV,
                                is_open=False,
                                backdrop=False,
                            ),
                        ],
                    ),
                    delay_show=300,
                ),
                id=ids.NEW_TREE_COLLAPSE,
                is_open=False,
            ),
        ],
        class_name="mb-3",
    )
