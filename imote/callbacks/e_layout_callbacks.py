import time
from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from imote import ids
from imote.config import NODE_PLOTS_EXTRA_SEP, LAYOUT_SEP_BY_NODE_SIZE
from imote.viz_tree.tree_layout import layout_elements

DISPLAY_DIRECTIONS = {"Top Bottom": "TB", "Left Right": "LR"}


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

    # The graph gets its elements and their layout together. dash-cytoscape writes the elements back without their
    # classes (which give the size of the nodes), so the elements come from a store. Dash runs this after both the
    # store and the gaps change, e.g. when the node size changes. If the nodes keep their positions, e.g. a path got
    # highlighted, the layout isn't run again, so the zoom and dragged nodes stay (unless the layout is refitted).
    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "elements", allow_duplicate=True),
        Output(ids.CYTOSCAPE_GRAPH, "layout", allow_duplicate=True),

        Input(ids.STORE_ELEMENTS, "data"),
        Input(ids.BTN_LAYOUT, "n_clicks"),
        Input(ids.DISPLAY_DIRECTION, "value"),
        Input(ids.NODE_RANKER, "value"),
        Input(ids.RANK_SEP, "value"),
        Input(ids.NODE_SEP, "value"),
        State(ids.SWITCH_NODE_PLOTS, "value"),  # changes the elements too, which triggers this
        State(ids.CYTOSCAPE_GRAPH, "layout"),
        prevent_initial_call=True,
    )
    def update_layout(elements, n_clicks, rank_dir, node_rank, rank_sep, node_sep, show_node_plots,
                      old_layout):
        if not elements:
            raise PreventUpdate
        if show_node_plots:
            rank_sep += NODE_PLOTS_EXTRA_SEP
            node_sep += NODE_PLOTS_EXTRA_SEP
        # Only a change of the tree keeps the nodes in place, a layout setting or Refit lays out the whole tree
        tree_changed = set(ctx.triggered_prop_ids) == {f"{ids.STORE_ELEMENTS}.data"}
        elements, layout = layout_elements(
            elements, direction=DISPLAY_DIRECTIONS[rank_dir], align_leaves=node_rank == "Align leaves to bottom",
            leaf_columns=node_rank == "Leaves in columns", rank_sep=rank_sep,
            node_sep=node_sep, previous=old_layout if tree_changed else None)
        layout["_time"] = time.time()  # changes every time, so cytoscape runs the layout even if nothing else changed
        old_positions = (old_layout or {}).get("positions") or {}
        if ctx.triggered_id != ids.BTN_LAYOUT and old_positions == layout["positions"]:
            return elements, no_update
        # The nodes move to their new positions with an animation. New nodes start at their closest ancestor that was
        # already there, so they grow out of it.
        parents = {el["data"]["target"]: el["data"]["source"] for el in elements if "source" in el["data"]}
        for el in elements:
            node_id = el["data"]["id"]
            if "source" in el["data"] or node_id in old_positions:
                continue
            ancestor = node_id
            while ancestor in parents and ancestor not in old_positions:
                ancestor = parents[ancestor]
            el["position"] = old_positions.get(ancestor, layout["positions"][node_id])
        return elements, layout
