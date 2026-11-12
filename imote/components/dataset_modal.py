"""Dialog to add a dataset (search PMLB or UCI, or upload a file) and to change the settings of a dataset."""
import dash_bootstrap_components as dbc
from dash import dcc, html

from imote import ids
from imote.dataset.readers import SUPPORTED_FORMATS_TEXT, SUPPORTED_SUFFIXES

LABEL_STYLE = {"marginBottom": "4px", "fontWeight": 500}
RESULTS_STYLE = {"maxHeight": "340px", "overflowY": "auto", "marginTop": "8px"}


def _field(label: str, component, hint: str | None = None) -> html.Div:
    children = [html.Div(label, style=LABEL_STYLE), component]
    if hint:
        children.append(html.Small(hint, className="text-muted"))
    return html.Div(children, className="mb-3")


def _search_row(input_id: str, placeholder: str, extra=None) -> html.Div:
    children = [dbc.Input(id=input_id, type="search", placeholder=placeholder, size="sm", debounce=0.3)]
    if extra is not None:
        children.append(extra)
    return html.Div(children, className="d-flex gap-2 mt-3")


def _source_stage() -> html.Div:
    return html.Div(
        id=ids.DS_SOURCE_STAGE,
        children=dbc.Tabs(
            id=ids.DS_TABS,
            active_tab="pmlb",
            children=[
                dbc.Tab(
                    tab_id="pmlb",
                    label="PMLB",
                    children=[
                        _search_row(
                            ids.DS_PMLB_SEARCH,
                            "Search (by name)",
                            dbc.Select(
                                id=ids.DS_PMLB_TASK,
                                options=[
                                    {"label": "All tasks", "value": "all"},
                                    {"label": "Classification", "value": "classification"},
                                    {"label": "Regression", "value": "regression"},
                                ],
                                value="all",
                                size="sm",
                                style={"width": "170px"},
                            ),
                        ),
                        html.Div(id=ids.DS_PMLB_RESULTS, style=RESULTS_STYLE),
                    ],
                ),
                dbc.Tab(
                    tab_id="uci",
                    label="UCI repository",
                    children=[
                        _search_row(ids.DS_UCI_SEARCH, "Search (by name or id)"),
                        html.Div(id=ids.DS_UCI_RESULTS, style=RESULTS_STYLE),
                    ],
                ),
                dbc.Tab(
                    tab_id="upload",
                    label="Upload a file",
                    children=html.Div(
                        [
                            dcc.Upload(
                                id=ids.DS_UPLOAD,
                                children=html.Div(["Drag and drop or ", html.A("click to select a file", href="#")]),
                                accept=",".join(sorted(SUPPORTED_SUFFIXES | {".gz"})),
                                multiple=False,
                                style={
                                    "width": "100%",
                                    "padding": "24px",
                                    "border": "1px dashed gray",
                                    "borderRadius": "6px",
                                    "textAlign": "center",
                                },
                            ),
                            html.Small(f"Supported formats: {SUPPORTED_FORMATS_TEXT}.", className="text-muted"),
                        ],
                        className="mt-3",
                    ),
                ),
            ],
        ),
    )


def _config_stage() -> html.Div:
    return html.Div(
        id=ids.DS_CONFIG_STAGE,
        style={"display": "none"},
        children=[
            _field("Name", dbc.Input(id=ids.DS_CFG_NAME, size="sm")),
            html.Div(
                [
                    html.Div(
                        _field("Response", dcc.Dropdown(id=ids.DS_CFG_RESPONSE, clearable=False)),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        _field(
                            "Task",
                            dbc.RadioItems(
                                id=ids.DS_CFG_TASK,
                                options=[
                                    {"label": "Regression", "value": "regression"},
                                    {"label": "Classification", "value": "classification"},
                                ],
                                inline=True,
                            ),
                        ),
                        style={"flex": "1"},
                    ),
                ],
                className="d-flex gap-3",
            ),
            _field(
                "Categorical features",
                dcc.Dropdown(id=ids.DS_CFG_CATEGORICAL, multi=True, placeholder="None"),
                "Text columns are always categorical.",
            ),
            _field(
                "Columns to leave out",
                dcc.Dropdown(id=ids.DS_CFG_IGNORE, multi=True, placeholder="None, use all columns as features"),
            ),
            dbc.Alert(id=ids.DS_CFG_SUMMARY, color="light", className="py-2 mb-2"),
            html.Small(id=ids.DS_CFG_FILE, className="text-muted"),
        ],
    )


def make_dataset_modal() -> html.Div:
    return html.Div(
        [
            dcc.Store(id=ids.DS_STORE),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Add a dataset", id=ids.DS_MODAL_TITLE), close_button=False),
                    dbc.ModalBody(
                        dcc.Loading(
                            [
                                _source_stage(),
                                _config_stage(),
                                html.Div(id=ids.DS_FEEDBACK, className="text-danger mt-2"),
                            ],
                            type="circle",
                            delay_show=300,
                            overlay_style={"visibility": "visible", "opacity": 0.5},
                        )
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Remove dataset", id=ids.DS_BTN_REMOVE, color="danger", outline=True,
                                       size="sm", class_name="me-auto", style={"display": "none"}),
                            dbc.Button("Cancel", id=ids.DS_BTN_CANCEL, color="secondary", size="sm"),
                            dbc.Button("Save", id=ids.DS_BTN_SAVE, color="primary", size="sm",
                                       style={"display": "none"}),
                        ]
                    ),
                ],
                id=ids.DS_MODAL,
                is_open=False,
                size="lg",
                backdrop="static",
                keyboard=False,
                scrollable=True,
            ),
        ]
    )
