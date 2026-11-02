import logging
import warnings
from pathlib import Path

from imote import ids, status
from dash import Input, Output, State, html, dcc, no_update, ctx
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

NO_NODE_SELECTED_DESCRIPTION = "Click a node in the graph to see its details."

def _plot_note(text: str, level: str = "muted", icon: str = "bi-info-circle"):
    return html.Small([html.I(className=f"bi {icon} me-1"), text], className=f"text-{level} d-block")

def make_node_plot(viz_tree, node, node_highlight, display_type, nmax, figw, figh,
                   predsplot_options, highlight_x, predsplot_type):
    """Makes the plot of a node with the plot settings card values.

    Returns the path of the svg file, or ``None`` and a note that says why there is no plot.
    """
    use_intercept = "intercept" in predsplot_options
    truncate_total_pred = "truncate" in predsplot_options
    staircase = "staircase" in predsplot_options
    type2 = "type2" in predsplot_type
    highlight_x_arr = None if highlight_x is None or not node_highlight else np.array(highlight_x)

    n_features = viz_tree.X_train.shape[1]
    cmap = plt.colormaps['tab20'].resampled(n_features)
    feature_colors = [cmap(i) for i in range(n_features)]

    try:
        if isinstance(node, LeafNode):
            if not isinstance(node.node_model, LinearNodeModel):
                message = "No plot available for this type of leaf node."
                return None, _plot_note(message)
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
                           feature_names=viz_tree.feature_names,
                           all_feature_colors=feature_colors,
                           )
            else:
                if not np.any(np.array(node.node_model.coefficients) != 0):
                    message =  ("This leaf predicts a constant, so there is no prediction plot. "
                            "Turn on Global predsplot in the plot settings to see how the tree gets to this value.")
                    return None, _plot_note(message)
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
                          feature_names=viz_tree.feature_names,
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
                viz_tree.feature_names,
                highlight_x_arr,
                viz_tree.target_name,
            )
        else:
            message =  f"No plot available for a {node.__class__.__name__}."
            return None, _plot_note(message)
    except Exception as e:
        logger.exception("Could not make the plot of node %s", node.id)
        return None, _plot_note(f"Could not make the plot: {e}", "danger", "bi-x-circle-fill")
    return file_dir, None


def svg_data_url(file_dir) -> str:
    return f"data:image/svg+xml;utf8,{quote(file_dir.read_text(encoding='utf-8'))}"


def register_callbacks(app):
    @app.callback(
        Output(ids.NODE_INFO_TYPE, "children"),
        Output(ids.NODE_INFO_TYPE, "style"),
        Output(ids.NODE_INFO_DESCRIPTION, "children"),
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

        description = node.get("description", NO_NODE_SELECTED_DESCRIPTION)

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
                metric_value = metric_cls.run(viz_tree_node, viz_tree.y_train)
                metric_cols.append(
                    dbc.Col([
                        html.Small(metric_name, className="text-muted d-block fw-bold"),
                        html.Span(children=metric_value, className="fs-6")
                    ], xs=6, sm=4, md=3)
                )
            metrics_components = dbc.Row(metric_cols, className="gy-2")

        return badge_label, badge_style, description, metrics_components


    @app.callback(
        Output(ids.NODE_INFO_PLOT_COLLAPSE, "is_open"),
        Input(ids.NODE_INFO_PLOT_SWITCH, "value"),
        prevent_initial_call=True
    )
    def toggle_node_plot(switch_on):
        return bool(switch_on)

    @app.callback(
        Output(ids.PREDSPLOT_SETTINGS, "disabled"),
        Input(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        prevent_initial_call=True,
    )
    def toggle_predsplot_settings(selected_node):
        return bool(selected_node) and selected_node[0]["node_type"] != "LeafNode"

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
        if not selected_node:
            return _plot_note("Select a node in the graph to see its plot."), None
        if selected_node[0]["node_type"] == "CombinedLinNode":
            return _plot_note("No plot available for combined linear nodes, show all linear nodes to see their plots."), None

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])
        file_dir, note = make_node_plot(viz_tree, node, selected_node[0]["highlight"], display_type, nmax, figw, figh,
                                        predsplot_options, highlight_x, predsplot_type)
        if file_dir is None:
            return note, None

        return html.Img(src=svg_data_url(file_dir), style={"maxWidth": "100%"}), str(file_dir)

    @app.callback(
        Output(ids.STORE_GRAPH_PLOTS, "data"),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_ADD_GRAPH_PLOT, "n_clicks"),
        Input(ids.BTN_CLEAR_GRAPH_PLOTS, "n_clicks"),
        Input(ids.NEW_TREE_TRIGGER, "data"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_GRAPH_PLOTS, "data"),
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
    def update_graph_plots(_n_add, _n_clear, _new_tree, selected_node, graph_plots, viz_tree_dict,
                           display_type, nmax, figw, figh, predsplot_options, highlight_x, predsplot_type):
        if _n_add is None and _n_clear is None:
            raise PreventUpdate
        if ctx.triggered_id == ids.NEW_TREE_TRIGGER:
            return [], no_update
        if ctx.triggered_id == ids.BTN_CLEAR_GRAPH_PLOTS:
            return [], status.info("Unpinned all plots from the graph.")

        if not selected_node:
            return no_update, status.warning("Select a node in the graph to pin its plot.")
        node_data = selected_node[0]
        if node_data["node_type"] == "CombinedLinNode":
            return no_update, status.warning("No plot available for combined linear nodes, show all linear nodes to see their plots.")

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, node_data["id"])
        file_dir, note = make_node_plot(viz_tree, node, node_data["highlight"], display_type, nmax, figw, figh,
                                        predsplot_options, highlight_x, predsplot_type)
        if file_dir is None:
            return no_update, status.warning(f"No plot pinned: {note.children[-1]}")

        # Adding a plot to a node that already has one replaces it, the overlay keeps its position and size.
        # assets/graph_plots.js picks the size of a new plot, based on the view (normal, minimal or tiny).
        graph_plot = {"id": node_data["id"], "src": svg_data_url(file_dir), "aspect": figw / figh}
        for old_plot in graph_plots or []:
            if old_plot["id"] == node_data["id"] and "width" in old_plot:
                graph_plot["width"] = old_plot["width"]
        graph_plots = [plot for plot in graph_plots or [] if plot["id"] != node_data["id"]]
        graph_plots.append(graph_plot)
        return graph_plots, status.success(f"Pinned the plot of node {node.id} to the graph, drag it to move it or its corners to resize it.")

    # The plots are drawn by assets/graph_plots.js as an overlay on the graph, so they don't change the tree layout.
    # The store lives outside the page and the graph inside it, so a second callback redraws the plots when the
    # Tree View page (and with it the graph) is shown again.
    render_graph_plots = """
        function(graphPlots) {
            window.imoteGraphPlots.render(%r, %r, graphPlots || []);
        }
        """ % (ids.CYTOSCAPE_GRAPH, ids.STORE_GRAPH_PLOTS)
    app.clientside_callback(render_graph_plots, Input(ids.STORE_GRAPH_PLOTS, "data"))
    app.clientside_callback(
        render_graph_plots.replace("function(graphPlots)", "function(_graphId, graphPlots)"),
        Input(ids.CYTOSCAPE_GRAPH, "id"),
        State(ids.STORE_GRAPH_PLOTS, "data"),
    )
