import time
from dash import Input, Output, State

from imote import ids
from imote.config import NODE_PLOTS_EXTRA_SEP, LAYOUT_SEP_BY_NODE_SIZE

_DISPLAY_DIRECTION_TO_DAGRE = {"Top Bottom": "TB", "Left Right": "LR"}
_NODE_RANKER_TO_DAGRE = {"Compact": "network-simplex", "Align leaves to bottom": "longest-path"}

def register_callbacks(app):
    @app.callback(
        Output(ids.LAYOUT_COLLAPSE, "is_open"),
        Input(ids.LAYOUT_TOGGLE_BUTTON, "n_clicks"),
        State(ids.LAYOUT_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_card(n_clicks, is_open):
        return not is_open

    # Smaller nodes get smaller gaps, the new gaps trigger update_layout below
    @app.callback(
        Output(ids.RANK_SEP, "value"),
        Output(ids.RANK_SEP, "max"),
        Output(ids.RANK_SEP, "step"),
        Output(ids.NODE_SEP, "value"),
        Output(ids.NODE_SEP, "max"),
        Output(ids.NODE_SEP, "step"),
        Input(ids.SWITCH_MINIMAL, "value"),
        prevent_initial_call=True,
    )
    def set_gaps_for_node_size(node_size):
        sep, slider_max, slider_step = LAYOUT_SEP_BY_NODE_SIZE[node_size]
        return sep, slider_max, slider_step, sep, slider_max, slider_step

    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "layout", allow_duplicate=True),

        Input(ids.BTN_LAYOUT, "n_clicks"),
        Input(ids.DISPLAY_DIRECTION, "value"),
        Input(ids.NODE_RANKER, "value"),
        Input(ids.RANK_SEP, "value"),
        Input(ids.NODE_SEP, "value"),
        Input(ids.SWITCH_NODE_PLOTS, "value"),
        prevent_initial_call=True,
    )
    def update_layout(n_clicks, rank_dir, node_rank, rank_sep, node_sep, show_node_plots):
        if show_node_plots:
            rank_sep += NODE_PLOTS_EXTRA_SEP
            node_sep += NODE_PLOTS_EXTRA_SEP
        return {
            "name": "dagre",
            "rankDir": _DISPLAY_DIRECTION_TO_DAGRE[rank_dir],
            "ranker": _NODE_RANKER_TO_DAGRE[node_rank],
            "rankSep": rank_sep,
            "nodeSep": node_sep,
            "edgeSep": 0,
            "animate": True,
            "fit": True,
            "_time": time.time(),  # dummy changing field, forces cytoscape to re-layout
        }
