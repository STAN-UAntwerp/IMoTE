import time
import numpy as np
from dash import Input, Output, State, html, no_update, ctx
from dash.exceptions import PreventUpdate

from imote import ids, status
from imote.config import DIR_LIVE_OUTPUT
from imote.viz_tree.viz_tree import VizTree
from imote.viz_tree.viz_tree_cytoscape import viz_tree_to_cytoscape_elements

def register_callbacks(app):
    @app.callback(
        Output(ids.SWITCH_COMBINE_LIN, "value"),
        Output(ids.SWITCH_NODE_PLOTS, "value"),
        Output(ids.SWITCH_RSS, "value"),
        Output(ids.SWITCH_MINIMAL, "value", allow_duplicate=True),
        Output(ids.SWITCH_COLOR_FEATURES, "value", allow_duplicate=True),
        Output(ids.SWITCH_DATA_EDGE_WIDTH, "value"),
        Output(ids.SWITCH_DATA_NODE_SIZE, "value"),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.SWITCH_COMBINE_LIN, "value"),
        Input(ids.SWITCH_NODE_PLOTS, "value"),
        Input(ids.SWITCH_RSS, "value"),
        Input(ids.SWITCH_MINIMAL, "value"),
        Input(ids.SWITCH_COLOR_FEATURES, "value"),
        Input(ids.SWITCH_DATA_EDGE_WIDTH, "value"),
        Input(ids.SWITCH_DATA_NODE_SIZE, "value"),

        prevent_initial_call=True,
    )
    def sync_switches(combine, node_plots, rss, minimal, color, data_egde, data_node):
        trig = ctx.triggered_id
        out = [no_update] * 7

        if trig == ids.SWITCH_NODE_PLOTS and node_plots:
            out[0] = 0  # combine_lin
            out[2] = False  # rss
            out[3] = 0  # minimal
            out[4] = True  # color_features
            out[5] = False # data_edge
            out[6] = False # data_node

        elif trig == ids.SWITCH_COMBINE_LIN and combine:
            out[1] = False  # node_plots

        elif trig == ids.SWITCH_RSS and rss:
            out[1] = False  # node_plots
            out[3] = 0  # minimal

        elif trig == ids.SWITCH_MINIMAL and minimal > 0:
            out[1] = False  # node_plots
            out[2] = False  # rss

        elif trig == ids.SWITCH_DATA_NODE_SIZE and data_node:
            out[1] = False # node_plots
            out[2] = False  # rss
            out[3] = 1  # minimal

        out.append(time.time())
        return out

    @app.callback(
        Output(ids.MODAL_NODE_PLOTS, "is_open"),

        Input(ids.ELEMENTS_TRIGGER, "data"),
        State(ids.SWITCH_NODE_PLOTS, "value"),
        State(ids.STORE_MODAL_DONT_ASK, "data"),
        prevent_initial_call=True,
    )
    def maybe_open_modal(_, show_node_plots, dont_ask):
        if show_node_plots and not dont_ask:
            return True
        raise PreventUpdate

    @app.callback(
        Output(ids.MODAL_NODE_PLOTS, "is_open", allow_duplicate=True),
        Output(ids.SWITCH_NODE_PLOTS, "value", allow_duplicate=True),

        Input(ids.MODAL_BTN_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def cancel_modal(_):
        return False, False

    @app.callback(
        Output(ids.STORE_MODAL_DONT_ASK, "data"),
        Output(ids.MODAL_NODE_PLOTS, "is_open", allow_duplicate=True),

        Input(ids.MODAL_BTN_CONFIRM, "n_clicks"),
        State(ids.MODAL_CHECK_DONT_ASK, "value"),
        prevent_initial_call=True,
    )
    def confirm_modal(_, dont_ask):
        return bool(dont_ask), False

    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "elements", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.ELEMENTS_TRIGGER, "data"),
        Input(ids.MODAL_BTN_CONFIRM, "n_clicks"),

        State(ids.MODAL_CHECK_DONT_ASK, "value"),
        State(ids.SWITCH_COMBINE_LIN, "value"),
        State(ids.SWITCH_NODE_PLOTS, "value"),
        State(ids.SWITCH_MINIMAL, "value"),
        State(ids.SWITCH_COLOR_FEATURES, "value"),
        State(ids.SWITCH_RSS, "value"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.INPUT_DISPLAY_TYPE, "value"),
        State(ids.INPUT_NMAX, "value"),
        State(ids.INPUT_FIG_W, "value"),
        State(ids.INPUT_FIG_H, "value"),
        State(ids.PREDSPLOT_OPTIONS, "value"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        State(ids.HIGHLIGHT_OPTIONS, "value"),
        State(ids.PREDSPLOT_TYPE, "value"),
        State(ids.SWITCH_DATA_EDGE_WIDTH, "value"),
        State(ids.SWITCH_DATA_NODE_SIZE, "value"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def update_elements(
            trigger,
            _,
            modal_dont_ask,
            combine_lin,
            show_node_plots,
            use_minimal,
            use_color_features,
            show_rss,
            viz_tree_dict,
            display_type,
            nmax,
            figw,
            figh,
            predsplot_options,
            highlight_x,
            highlight_options,
            predsplot_type,
            use_edge_width,
            use_node_size,
            tree_params,
    ):
        if show_node_plots and not modal_dont_ask and ctx.triggered_id != ids.MODAL_BTN_CONFIRM:
            raise PreventUpdate

        if viz_tree_dict is None:
            raise PreventUpdate

        use_intercept = "intercept" in predsplot_options
        truncate_total_pred = "truncate" in predsplot_options
        staircase = "staircase" in predsplot_options
        type2 = "type2" in predsplot_type
        only_show_highlight = "only_show_highlight" in highlight_options

        viz_tree = VizTree.from_dict(viz_tree_dict)
        highlight_x_arr = None if highlight_x is None else np.array(highlight_x)
        show_all_labels = (tree_params["subtree_node_id"] !=-1 or
                           tree_params["collapsed_nodes_count"] != 0 or
                           tree_params["pruned"])

        try:
            elements = viz_tree_to_cytoscape_elements(
                viz_tree,
                str(DIR_LIVE_OUTPUT),
                combine_lin=combine_lin,
                use_color_features = use_color_features,
                show_rss=show_rss,
                show_node_plots=show_node_plots,
                fig_size=(figw, figh),
                predsplot_n_max=nmax,
                predsplot_use_intercept=use_intercept,
                predsplot_display_type=display_type,
                predsplot_truncate_total_pred=truncate_total_pred,
                predsplot_staircase=staircase,
                predsplot_type2=type2,
                highlight_x=highlight_x_arr,
                only_show_highlight=only_show_highlight,
                use_edge_width = use_edge_width,
                use_node_size = use_node_size,
                show_all_labels = show_all_labels,
            )
        except Exception as e:
            return no_update, status.error(f"Could not draw the tree: {e}")

        if use_minimal > 0:
            if use_minimal == 1:
                class_name = "minimal"
            else: #use_minimal == 2:
                class_name = "tiny"
            for el in elements:
                if "data" in el and "id" in el["data"]:  # node, not edge
                    classes = el.get("classes", "").split()
                    classes.append(class_name)
                    el["classes"] = " ".join(classes)

        return elements, no_update
