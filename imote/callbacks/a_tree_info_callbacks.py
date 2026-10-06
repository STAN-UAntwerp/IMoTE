import dash_bootstrap_components as dbc
from dash import html, Output, Input
from imote import ids

def format_tree_info(
        dataset_name: str,
        method_name: str,
        max_depth: int,
        max_model_depth: int,
        min_sample_split: int,
        min_sample_leaf: int,
        training_time: float,
        n_internal_nodes: int,
        n_leafs: int,
        depth: int,
        n_samples: int,
        n_features: int,
        subtree_node_id: int = -1,
        pruned: bool = False,
        collapsed_nodes_count: int = 0,
        highlight_x = None
):
    badges = []
    if subtree_node_id == -1 and not pruned:
        badges.append(dbc.Badge("Full tree", color="info", className="me-2"))
    elif subtree_node_id == -1 and pruned:
        badges.append(dbc.Badge("Pruned tree", color="info", className="me-2"))
    elif subtree_node_id != -1 and not pruned:
        badges.append(dbc.Badge(f"Subtree at {subtree_node_id}", color="info", className="me-2"))
    else:
        badges.append(dbc.Badge(f"Subtree (Pruned) at {subtree_node_id}", color="info", className="me-2"))
    if collapsed_nodes_count == 0:
        badges.append(dbc.Badge("No Collapsed Nodes", color="warning", className="me-2"))
    else:
        badges.append(dbc.Badge(f"{collapsed_nodes_count} Collapsed Nodes", color="warning", className="me-2"))
    if highlight_x is not None:
        badges.append(dbc.Badge(f"Highlighted point", color="danger", className="me-2"))
    return dbc.CardBody([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
            children=[
                html.H5("Tree Information", className="mb-0 text-primary fw-bold"),
                html.Div(badges)
            ]
        ),

        dbc.Row([
            dbc.Col([
                html.Small("Dataset name", className="text-muted d-block fw-bold"),
                html.Span(children=dataset_name, className="fs-5")
            ], width=6),
            dbc.Col([
                html.Small("Method ⓘ", className="text-muted d-block fw-bold"),
                html.Span(children=method_name, className="fs-5")
            ], width=6, id="method-tooltip"),

            dbc.Tooltip(
                [
                    html.Strong("Parameters"),
                    html.Br(),
                    f"Max depth: {max_depth}",
                    html.Br(),
                    f"Max model depth: {max_model_depth}",
                    html.Br(),
                    f"Min sample split: {min_sample_split}",
                    html.Br(),
                    f"Min sample leaf: {min_sample_leaf}",
                    html.Br(),
                    f"Training time: {training_time}",
                ],
                target="method-tooltip",
                placement="bottom",
            )
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Small("Dataset size", className="text-muted d-block fw-bold"),
                html.Span(f"{n_samples} x {n_features}", className="fs-6")
            ], width=3),
            dbc.Col([
                html.Small("Nodes ⓘ", className="text-muted d-block fw-bold"),
                html.Span(f"{n_internal_nodes + n_leafs}", className="fs-6")
            ], width=3, id="nodes-tooltip"),
            dbc.Tooltip(
                [
                    f"Internal nodes: {n_internal_nodes}",
                    html.Br(),
                    f"Leaf nodes: {n_leafs}",
                ],
                target="nodes-tooltip",
                placement="bottom",
            ),
            dbc.Col([
                html.Small("Depth", className="text-muted d-block fw-bold"),
                html.Span(str(depth), className="fs-6")
            ], width=3),
        ])
    ])

def register_callbacks(app):
    # --- UPDATE TREE PARAMS ---
    @app.callback(
        Output(ids.TREE_INFO_TEXT, "children"),

        Input(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True
    )
    def update_tree_info_card(tree_params):
        return format_tree_info(**tree_params)
