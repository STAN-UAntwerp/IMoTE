import logging
import warnings
from pathlib import Path

from imote import ids, status
from dash import Input, Output, State, html, dcc, no_update
from dash.exceptions import PreventUpdate
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import quote
import dash_bootstrap_components as dbc

from imote.callbacks.d_edit_tree_callbacks import find_node_by_cytoscape_id
from imote.config import DIR_LIVE_OUTPUT, DIR_PACKAGE, NODE_TYPE_COLORS
from imote.nodes.internal_node import InternalNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import LinearNodeModel
from imote.plots.predsplot import predsplot
from imote.plots.predsplot2 import predsplot2
from imote.plots.regplot import make_regression_plot
from imote.viz_tree.viz_tree import VizTree
from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY

logger = logging.getLogger(__name__)

NO_NODE_SELECTED_LABEL = "Click a node in the graph to see its details."

def _plot_note(text: str, level: str = "muted", icon: str = "bi-info-circle"):
    return html.Small([html.I(className=f"bi {icon} me-1"), text], className=f"text-{level} d-block")

def register_callbacks(app):
    @app.callback(
        Output(ids.NODE_INFO_TYPE, "children"),
        Output(ids.NODE_INFO_TYPE, "style"),
        Output(ids.NODE_INFO_LABEL, "children"),
        Output(ids.NODE_INFO_METRICS, "children"),

        Input(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        Input(ids.NODE_INFO_METRICS_DROPDOWN, "value"),
        State(ids.STORE_VIZ_TREE, "data"),
        prevent_initial_call=True
    )
    def update_node_info(selected_node, metric_names, viz_tree_dict):
        node = selected_node[0] if selected_node else {}

        node_type = node.get("node_type", "—")
        badge_label = f"{node_type[:-4]} Node" if selected_node else "Standby"
        badge_color = NODE_TYPE_COLORS.get(node_type, "#6c757d")
        badge_style = {"backgroundColor": badge_color, "color": "#ffffff"}

        label = node.get("label", NO_NODE_SELECTED_LABEL)

        if not node:
            metrics_components = "Select a node to see its metrics."
        elif not metric_names:
            metrics_components = "Choose metrics in the dropdown above."
        else:
            metric_cols = []
            viz_tree = VizTree.from_dict(viz_tree_dict)
            viz_tree_node = find_node_by_cytoscape_id(viz_tree, node["id"])
            for metric_name in metric_names:
                metric_cls = NODE_METRICS_REGISTRY[metric_name]
                metric_value = metric_cls.run(viz_tree_node, viz_tree.X_train, viz_tree.y_train, viz_tree.y_hat)
                metric_cols.append(
                    dbc.Col([
                        html.Small(metric_name, className="text-muted d-block fw-bold"),
                        html.Span(children=metric_value, className="fs-6")
                    ], xs=6, sm=4, md=3)
                )
            metrics_components = dbc.Row(metric_cols, className="gy-2")

        return badge_label, badge_style, label, metrics_components


    @app.callback(
        Output(ids.NODE_INFO_PLOT_COLLAPSE, "is_open"),
        Input(ids.NODE_INFO_PLOT_SWITCH, "value"),
        prevent_initial_call=True
    )
    def toggle_node_plot(switch_on):
        return bool(switch_on)

    @app.callback(
        Output(ids.DOWNLOAD_PLOT, "data"),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_DOWNLOAD_PLOT, "n_clicks"),
        State(ids.STORE_LAST_PLOT_PATH, "data"),
        prevent_initial_call=True,
    )
    def download_node_plot(n_clicks, plot_path):
        if n_clicks is None:
            raise PreventUpdate
        if not plot_path:
            return no_update, status.warning("There is no node plot to download, select a node that has a plot first.")
        return dcc.send_file(plot_path), status.info(f"Downloading the node plot as {Path(plot_path).name}.")

    @app.callback(
        Output(ids.NODE_INFO_PLOT_CONTAINER, "children"),
        Output(ids.STORE_LAST_PLOT_PATH, "data"),

        Input(ids.NODE_INFO_PLOT_SWITCH, "value"),
        Input(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        Input(ids.BTN_REFIT_PLOTS, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),

        State(ids.INPUT_DISPLAY_TYPE, "value"),
        State(ids.INPUT_NMAX, "value"),
        State(ids.INPUT_FIG_W, "value"),
        State(ids.INPUT_FIG_H, "value"),
        State(ids.PREDSPLOT_OPTIONS, "value"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        State(ids.PREDSPLOT_TYPE, "value"),

        prevent_initial_call=True,
    )
    def render_node_plot(switch_on, selected_node, _n_clicks, viz_tree_dict,
                         display_type,
                         nmax,
                         figw,
                         figh,
                         predsplot_options,
                         highlight_x,
                         predsplot_type,
                         ):
        if not switch_on or not selected_node:
            return _plot_note("Select a node in the graph to see its plot."), None

        use_intercept = "intercept" in predsplot_options
        truncate_total_pred = "truncate" in predsplot_options
        staircase = "staircase" in predsplot_options
        type2 = "type2" in predsplot_type
        highlight_x_arr = None if highlight_x is None or not selected_node[0]['highlight'] else np.array(highlight_x)

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])
        n_features = viz_tree.X_train.shape[1]
        cmap = plt.colormaps['tab20'].resampled(n_features)
        feature_colors = [cmap(i) for i in range(n_features)]

        try:
            if isinstance(node, LeafNode):
                if not isinstance(node.node_model, LinearNodeModel):
                    message = "No plot available for this type of leaf node."
                    return _plot_note(message), None
                if type2:
                    file_dir = DIR_LIVE_OUTPUT / "predsplots" / f"predsplot2_node{node.id}_{viz_tree.tree_id}.svg"
                    predsplot2(viz_tree=viz_tree,
                               leaf_node=node,
                               y_hat=viz_tree.y_hat,
                               n_max=nmax,
                               fig_size=(figw, figh),
                               truncate_total_pred=truncate_total_pred,
                               variable_tick_width=True,
                               display_type=display_type,
                               file_directory=str(file_dir),
                               highlight_x=highlight_x_arr,
                               staircase=staircase,
                               feature_names=None,
                               all_feature_colors=feature_colors,
                               )
                else:
                    if not np.any(np.array(node.node_model.coefficients) != 0):
                        message =  ("This leaf predicts a constant, so there is no prediction plot. "
                                "Turn on Global predsplot in the plot settings to see how the tree gets to this value.")
                        return _plot_note(message), None
                    file_dir = DIR_LIVE_OUTPUT / "predsplots" / f"predsplot_node{node.id}_{viz_tree.tree_id}.svg"
                    node_X = viz_tree.X_train[node.indices, :]
                    if use_intercept:
                        intercept = node.node_model.intercept
                    else:
                        intercept = None
                    predsplot(node_X,
                              np.array(node.node_model.coefficients),
                              y_hat=node.node_model.predict(node_X),
                              n_max=nmax,
                              intercept=intercept,
                              fig_size=(figw, figh),
                              feature_names=None,
                              all_feature_colors=feature_colors,
                              display_type=display_type,
                              truncate_total_pred=truncate_total_pred,
                              variable_tick_width=True,
                              file_directory=str(file_dir),
                              highlight_x=highlight_x_arr,
                              staircase=staircase)
            elif isinstance(node, InternalNode):
                file_dir = DIR_LIVE_OUTPUT / "regplots" / f"regplot_node{node.id}_{viz_tree.tree_id}.svg"
                make_regression_plot(
                    node,
                    viz_tree.X_train,
                    str(file_dir),
                    (figw, figh),
                    feature_colors,
                    None,
                    highlight_x_arr
                )
            else:
                message =  f"No plot available for a {node.__class__.__name__}."
                return _plot_note(message), None
        except Exception as e:
            logger.exception("Could not make the plot of node %s", node.id)
            return _plot_note(f"Could not make the plot: {e}", "danger", "bi-x-circle-fill"), None

        svg_data = file_dir.read_text(encoding="utf-8")
        encoded_svg = quote(svg_data)

        src_url =  f"data:image/svg+xml;utf8,{encoded_svg}"
        return html.Img(src=src_url, style={"maxWidth": "100%"}), str(file_dir)
