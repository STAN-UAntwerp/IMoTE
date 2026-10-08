import time
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from imote import ids, status
from imote.nodes.collapsed_node import CollapsedNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.internal_node import LinearNode
from imote.viz_tree.viz_tree import VizTree

NO_NODE_SELECTED = "Select a node in the graph first (click on it), then click this button again."

def find_node_by_cytoscape_id(viz_tree: VizTree, cytoscape_id: str):
    for node in viz_tree.nodes:
        if f"node{node.id}" == cytoscape_id:
            return node
    raise ValueError(f"No node found for cytoscape id {cytoscape_id!r}")

def _count_collapsed_nodes(viz_tree: VizTree) -> int:
    return sum(node.n_nodes for node in viz_tree.nodes if isinstance(node, CollapsedNode))

def _warn(text: str) -> tuple:
    return no_update, no_update, status.warning(text), no_update

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
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_COLLAPSE_EXPAND, "n_clicks"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def toggle_collapse_expand_node(n_clicks, selected_node, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        if not selected_node:
            return _warn(NO_NODE_SELECTED)
        if "Leaf" in selected_node[0]["node_type"]:
            return _warn("A leaf node has no children to collapse, select an internal or collapsed node.")

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])

        if isinstance(node, CollapsedNode):
            viz_tree.expand(node)
            message = status.success(f"Expanded the {node.n_nodes} collapsed nodes below node {node.parent_id}.")
        else:
            children = node.get_children()
            if isinstance(children[0], CollapsedNode):
                viz_tree.expand(children[0])
                message = status.success(f"Expanded the {children[0].n_nodes} collapsed nodes below node {node.id}.")
            else:
                if isinstance(node, LinearNode) and isinstance(children[0], LeafNode):
                    return _warn(f"Nothing to collapse below node {node.id}, its only child is a leaf.")
                n_collapsed = viz_tree.collapse(node)
                message = status.success(f"Collapsed the {n_collapsed} nodes below node {node.id}.")

        new_tree_params = tree_params.copy()
        new_tree_params["collapsed_nodes_count"] = _count_collapsed_nodes(viz_tree)
        return viz_tree.to_dict(), new_tree_params, message, time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
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
        if collapse_level is None:
            return _warn("Enter the level to collapse the tree to, next to the button.")
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

        if n_collapsed_nodes:
            message = status.success(f"Collapsed the tree to level {collapse_level}, "
                                     f"{n_collapsed_nodes} nodes are hidden.")
        else:
            message = status.info(f"The tree has no nodes below level {collapse_level}, all nodes are shown.")
        new_tree_params = tree_params.copy()
        new_tree_params["collapsed_nodes_count"] = n_collapsed_nodes
        return viz_tree.to_dict(), new_tree_params, message, time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
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
        n_collapsed_nodes = _count_collapsed_nodes(viz_tree)
        if n_collapsed_nodes == 0:
            return no_update, no_update, status.info("There are no collapsed nodes to expand."), no_update
        viz_tree.expand_all_nodes()

        new_tree_params = tree_params.copy()
        new_tree_params["collapsed_nodes_count"] = 0
        message = status.success(f"Expanded all nodes.")
        return viz_tree.to_dict(), new_tree_params, message, time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_SUBTREE, "n_clicks"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def subtree_from_node(n_clicks, selected_node, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        if not selected_node:
            return _warn(NO_NODE_SELECTED)

        viz_tree = VizTree.from_dict(viz_tree_dict)
        root = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])
        if isinstance(root, CollapsedNode):
            return _warn("Expand the collapsed node first, a subtree can't start at a collapsed node.")
        if isinstance(root, LeafNode):
            return _warn("A leaf node has no subtree, select an internal node.")
        if root is viz_tree.root_node:
            return no_update, no_update, status.info(f"Node {root.id} is already the root of the shown tree."), no_update

        viz_tree.root_node = root
        viz_tree.nodes = viz_tree.collect_nodes()
        viz_tree.edges = viz_tree.collect_edges()

        new_tree_params = tree_params.copy()
        new_tree_params["subtree_node_id"] = root.id
        new_tree_params["collapsed_nodes_count"] = _count_collapsed_nodes(viz_tree)

        message = status.success(f"Showing the subtree from node {root.id}. "
                                 f"Use Reload tree in the New Tree card to see the full tree again.")
        return viz_tree.to_dict(), new_tree_params, message, time.time()

    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_PRUNE, "n_clicks"),
        State(ids.CYTOSCAPE_GRAPH, "selectedNodeData"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def prune_from_node(n_clicks, selected_node, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        if tree_params['method_name'] != "Pilot":
            return _warn(f"Pruning is only available for PILOT trees, this tree was made with {tree_params['method_name']}.")
        if not selected_node:
            return _warn(NO_NODE_SELECTED)

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, selected_node[0]["id"])
        if isinstance(node, (LeafNode, CollapsedNode)):
            return _warn("Select an internal node to prune, leaves and collapsed nodes can't be pruned.")
        if node is viz_tree.root_node:
            return _warn("The root node can't be pruned, select a node below it.")

        viz_tree.prune(node)
        depth = viz_tree.get_depth()
        n_leafs = viz_tree.get_n_leafs()
        n_internal_nodes = len(viz_tree.nodes) - n_leafs

        new_tree_params = tree_params.copy()
        new_tree_params["pruned"] = True
        new_tree_params["depth"] = depth
        new_tree_params["n_leafs"] = n_leafs
        new_tree_params["n_internal_nodes"] = n_internal_nodes
        new_tree_params["collapsed_nodes_count"] = 0

        message = status.success(f"Pruned the tree at node {node.id}, it is now a leaf.")
        return viz_tree.to_dict(), new_tree_params, message, time.time()
