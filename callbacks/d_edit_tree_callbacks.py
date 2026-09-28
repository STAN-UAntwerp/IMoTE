import time
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

import ids
from nodes.collapsed_node import CollapsedNode
from nodes.leaf_node import LeafNode
from nodes.internal_node import LinearNode
from viz_tree.viz_tree import VizTree

def find_node_by_cytoscape_id(viz_tree: VizTree, cytoscape_id: str):
    for node in viz_tree.nodes:
        if f"node{node.id}" == cytoscape_id:
            return node
    raise ValueError(f"No node found for cytoscape id {cytoscape_id!r}")

def register_callbacks(app):
    @app.callback(
        Output(ids.EDIT_TREE_COLLAPSE, "is_open"),
        Input(ids.EDIT_TREE_TOGGLE_BUTTON, "n_clicks"),
        State(ids.EDIT_TREE_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_card(n_clicks, is_open):
        return not is_open

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_COLLAPSE_EXPAND, "n_clicks"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def toggle_collapse_expand_node(n_clicks, selected_node, viz_tree_dict, tree_params):
        if not selected_node or "leaf" in selected_node[0]["node_type"]:
            raise PreventUpdate

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])

        n_collapsed_nodes = 0
        if isinstance(node, CollapsedNode):
            viz_tree.expand(node)
            message = f"Expanded node {node.id}"
        else:
            children = node.get_children()
            if isinstance(children[0], CollapsedNode):
                viz_tree.expand(children[0])
                message = f"Expanded node {children[0].id}"
            else:
                if isinstance(node, LinearNode) and isinstance(children[0], LeafNode):
                    raise PreventUpdate
                n_collapsed_nodes = viz_tree.collapse(node)
                message = f"Collapsed below node {node.id}"

        new_tree_params = tree_params.copy()
        new_tree_params["collapsed_nodes_count"] = n_collapsed_nodes
        return viz_tree.to_dict(), new_tree_params, message, time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_COLLAPSE_LEVEL, "n_clicks"),
        State(ids.INPUT_COLLAPSE_LEVEL, "value"),
        State(ids.COLLAPSE_LEVEL_OPTIONS, "value"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def collapse_to_level(n_clicks, collapse_level, collapse_level_options, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        collapse_level = int(collapse_level)
        include_lin = "include_lin" in collapse_level_options
        viz_tree = VizTree.from_dict(viz_tree_dict)
        viz_tree.expand_all_nodes()

        nodes_to_collapse = [viz_tree.root_node]
        for _ in range(collapse_level - 1):
            nodes_to_collapse = sum([node.get_children() for node in nodes_to_collapse], [])

            if not include_lin: #TODO: include lin doesn't seem to be correctly set up
                new_nodes_to_collapse = nodes_to_collapse.copy()
                for node in nodes_to_collapse:
                    if isinstance(node, LinearNode):
                        new_nodes_to_collapse.remove(node)
                        while isinstance(node, LinearNode):
                            node = node.get_children()[0]
                        new_nodes_to_collapse.append(node)
                nodes_to_collapse = new_nodes_to_collapse

        n_collapsed_nodes = 0
        for node in nodes_to_collapse:
            if isinstance(node, LeafNode) or (isinstance(node, LinearNode) and isinstance(node.get_children()[0], LeafNode)):
                continue
            n_collapsed_nodes += viz_tree.collapse(node)
        
        new_tree_params = tree_params.copy()
        new_tree_params["collapsed_nodes_count"] = n_collapsed_nodes
        return viz_tree.to_dict(), new_tree_params, f"Collapsed tree to level {collapse_level}", time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_EXPAND_ALL, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def expand_all_nodes(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        viz_tree = VizTree.from_dict(viz_tree_dict)
        viz_tree.expand_all_nodes()

        new_tree_params = tree_params.copy()
        new_tree_params["collapsed_nodes_count"] = 0
        return viz_tree.to_dict(), new_tree_params, "Expanded all nodes.", time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_SUBTREE, "n_clicks"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def subtree_from_node(n_clicks, selected_node, viz_tree_dict, tree_params):
        if not selected_node:
            raise PreventUpdate

        viz_tree = VizTree.from_dict(viz_tree_dict)
        root = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])

        viz_tree.root_node = root
        viz_tree.nodes = viz_tree.collect_nodes()
        viz_tree.edges = viz_tree.collect_edges()

        new_tree_params = tree_params.copy()
        new_tree_params["subtree_node_id"] = selected_node[0]["id"]

        return viz_tree.to_dict(), new_tree_params, f"Subtree from {selected_node[0]['id']}.", time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_PRUNE, "n_clicks"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def prune_from_node(n_clicks, selected_node, viz_tree_dict, tree_params):
        if not selected_node or tree_params['method_name'] != "Pilot":
            raise PreventUpdate

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])
        if isinstance(node, LeafNode):
            raise PreventUpdate

        viz_tree.prune(node)
        depth = viz_tree.get_depth()
        n_leafs = viz_tree.get_n_leafs()
        n_internal_nodes = len(viz_tree.nodes) - n_leafs

        new_tree_params = tree_params.copy()
        new_tree_params["pruned"] = True
        new_tree_params["depth"] = depth
        new_tree_params["n_leafs"] = n_leafs
        new_tree_params["n_internal_nodes"] = n_internal_nodes

        return viz_tree.to_dict(), new_tree_params, f"Pruned tree at node {selected_node[0]['id']}.", time.time()
