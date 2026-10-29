import dash_bootstrap_components as dbc
from dash import html, dcc
from imote.components.cards.bb_node_plot_settings_card import make_settings_card
from imote import ids
from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY
from imote.config import DEFAULT_NODE_METRICS, DEFAULT_GRAPH_PLOT_SIZE

def make_node_info_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardBody([
                html.Div(

                    className="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
                    children=[
                        html.H5("Node Information", className="mb-0 text-primary fw-bold"),
                        html.Span(
                            "Standby",
                            id=ids.NODE_INFO_TYPE,
                            className="badge fs-6",
                            style={"backgroundColor": "#6c757d", "color": "#ffffff"}
                        )
                    ]
                ),
                html.Div(
                    id=ids.NODE_INFO_DESCRIPTION,
                    children="Click a node in the graph to see its details.",
                    className="fs-6 mb-3",
                ),

                # Metrics
                dbc.Row([
                    dbc.Col([
                        html.Small("Metrics", className="text-muted d-block fw-bold"),
                        dcc.Dropdown(
                            list(NODE_METRICS_REGISTRY.keys()),
                            DEFAULT_NODE_METRICS,
                            id=ids.NODE_INFO_METRICS_DROPDOWN,
                            multi=True,
                            closeOnSelect=False,
                            searchable=False,
                            debounce=True,
                            maxHeight=300,
                            search_order="original",
                            placeholder="Select metrics",
                            className="mb-2",
                        ),
                        html.Div(
                            id=ids.NODE_INFO_METRICS,
                            children="Select a node to see its metrics."
                        )
                    ], width=12),
                ], className="mb-3"),

                # Pin the node plot on top of the graph, uses the plot settings below
                html.Small("Plot on graph", className="text-muted d-block fw-bold"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Width"),
                        dbc.Input(id=ids.INPUT_GRAPH_PLOT_SIZE, type="number", value=DEFAULT_GRAPH_PLOT_SIZE,
                                  min=50, step=50),
                        dbc.Button([html.I(className="bi bi-pin-angle me-1"), "Add to graph"],
                                   id=ids.BTN_ADD_GRAPH_PLOT, color="primary", className="fw-bold"),
                        dbc.Button(html.I(className="bi bi-trash"), id=ids.BTN_CLEAR_GRAPH_PLOTS,
                                   color="secondary", title="Remove all plots from the graph"),
                    ],
                    size="sm",
                    className="mb-3",
                ),
                dbc.Switch(
                    id=ids.NODE_INFO_PLOT_SWITCH,
                    label="Show node plot",
                    value=False,
                ),
                dbc.Collapse(
                    id=ids.NODE_INFO_PLOT_COLLAPSE,
                    is_open=False,
                    children=[
                        dbc.Spinner(
                            html.Div(
                                id=ids.NODE_INFO_PLOT_CONTAINER,
                                children="Select a node in the graph to see its plot.",
                                style={"marginBottom": "8px"},
                            ),
                            delay_show=300,
                        ),
                        make_settings_card()
                    ]
                ),
            ]),
        ],
        class_name="mb-3",
    )
