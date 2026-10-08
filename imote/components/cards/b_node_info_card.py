import dash_bootstrap_components as dbc
from dash import html, dcc
from imote.components.cards.bb_node_plot_settings_card import make_settings_card
from imote import ids
from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY
from imote.config import DEFAULT_NODE_METRICS

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
                # Label
                dbc.Row([
                    dbc.Col([
                        html.Small("Label", className="text-muted d-block fw-bold"),
                        html.Span(id = ids.NODE_INFO_LABEL, children="Click a node in the graph to see its details.", className="fs-6")
                    ], width=12),
                ], className="mb-3"),

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
