import time
import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from imote import ids, status
from imote.nodes.collapsed_node import CollapsedNode
from imote.viz_tree.viz_tree import VizTree


def get_path_info(viz_tree: VizTree, x: np.ndarray) -> dict:
    """Where a data point ends up in the tree and what the tree predicts for it."""
    path = viz_tree.get_decision_path(x)
    leaf = path[-1]
    shown_ids = {node.id for node in viz_tree.nodes if not isinstance(node, CollapsedNode)}
    x_train_row = np.flatnonzero(np.all(np.isclose(viz_tree.X_train, x), axis=1))
    return {
        "leaf_id": leaf.id,
        "prediction": float(leaf.node_model.predict(x.reshape(1, -1))[0]),
        "leaf_hidden": leaf.id not in shown_ids,
        "path_ids": [node.id for node in path],
        "training_row": int(x_train_row[0]) if len(x_train_row) else None,
        "actual": float(viz_tree.y_train[x_train_row[0]]) if len(x_train_row) else None,
    }


def _value(label: str, value: str) -> dbc.Col:
    return dbc.Col([
        html.Small(label, className="text-muted d-block fw-bold"),
        html.Span(value, className="fs-6"),
    ], width=4)

def register_callbacks(app):
    @app.callback(
        Output(ids.HIGHLIGHT_COLLAPSE, "is_open"),
        Input(ids.HIGHLIGHT_TOGGLE_BUTTON, "n_clicks"),
        State(ids.HIGHLIGHT_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_card(n_clicks, is_open):
        return not is_open

    @app.callback(
        Output(ids.INPUT_HIGHLIGHT, "value"),
        Input(ids.BTN_RANDOM_POINT, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        prevent_initial_call=True,
    )
    def set_random_point(n_clicks, viz_tree_dict):
        if n_clicks is None:
            raise PreventUpdate
        rows = viz_tree_dict["X_train"]
        return ", ".join(map(str, rows[np.random.randint(len(rows))]))

    @app.callback(
        Output(ids.STORE_HIGHLIGHT_X, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),
        Output(ids.HIGHLIGHT_RESULT, "children", allow_duplicate=True),

        Input(ids.BTN_HIGHLIGHT, "n_clicks"),
        State(ids.INPUT_HIGHLIGHT, "value"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_VIZ_TREE_BASE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def highlight_path(n_clicks, input_highlight_x, viz_tree_dict, base_viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        if not input_highlight_x or not input_highlight_x.strip():
            message = status.warning("Enter the feature values of a data point first, or click Random point.")
            return no_update, no_update, message, no_update, no_update
        n_features = len(viz_tree_dict["X_train"][0])
        x = []
        for value in input_highlight_x.split(","):
            try:
                x.append(float(value.strip()))
            except ValueError:
                problem = f"Couldn't read '{value.strip()}' as a number. Enter comma-separated numbers, e.g. 1.2, 0.5, 3.1."
                return no_update, no_update, status.warning(problem), no_update, no_update
        if len(x) != n_features:
            problem = f"The point has {len(x)} values, but the tree's dataset has {n_features} features."
            return no_update, no_update, status.warning(problem), no_update, no_update
        highlight_x = np.array(x)

        viz_tree = VizTree.from_dict(viz_tree_dict)
        path_info = get_path_info(viz_tree, highlight_x)
        text = "Highlighted the path of the point."
        note = None
        if path_info["leaf_hidden"]:
            note = "The leaf is hidden in a collapsed part of the tree, expand it to see the full path."

        subtree_root_id = tree_params["subtree_node_id"]
        if subtree_root_id != -1:
            full_tree_path_info = get_path_info(VizTree.from_dict(base_viz_tree_dict), highlight_x)
            if subtree_root_id not in full_tree_path_info["path_ids"]:
                note = (f"This point doesn't reach node {subtree_root_id}, the root of this subtree. The path shown "
                        f"is the one it would follow inside the subtree. Reload the tree to see the true path.")
                path_info = full_tree_path_info
        message = status.warning(f"{text} {note}") if note else status.success(text)

        new_tree_params = tree_params.copy()
        new_tree_params["highlight_x"] = input_highlight_x

        actual = "—" if path_info["actual"] is None else f"{path_info['actual']:.4g}"
        children = [dbc.Row([
            _value("Ends in leaf", f"Node {path_info['leaf_id']}"),
            _value("Prediction", f"{path_info['prediction']:.4g}"),
            _value("Actual value", actual),
        ])]
        if path_info["training_row"] is not None:
            children.append(html.Small(f"The point is row {path_info['training_row']} of the training data.",
                                       className="text-muted d-block mt-1"))
        if note:
            children.append(html.Small([html.I(className="bi bi-exclamation-triangle-fill me-1"), note],
                                       className="text-warning-emphasis d-block mt-1"))
        formated_highlight_result = html.Div(children, className="border rounded p-2 mt-2")

        return highlight_x.tolist(), new_tree_params, message, time.time(), formated_highlight_result

    @app.callback(
        Output(ids.STORE_HIGHLIGHT_X, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_CLEAR_HIGHLIGHT, "n_clicks"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def clear_highlight(n_clicks, highlight_x, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        if highlight_x is None:
            return no_update, no_update, status.info("There is no highlighted point to clear."), no_update

        new_tree_params = tree_params.copy()
        new_tree_params["highlight_x"] = None
        return None, new_tree_params, status.success("Highlight cleared."), time.time()

    @app.callback(
        Output(ids.HIGHLIGHT_RESULT, "children", allow_duplicate=True),
        Input(ids.STORE_HIGHLIGHT_X, "data"),
        prevent_initial_call=True,
    )
    def clear_highlight_result(highlight_x):
        if highlight_x is None:
            return None
        raise PreventUpdate

    @app.callback(
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),
        Input(ids.HIGHLIGHT_OPTIONS, "value"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        prevent_initial_call=True,
    )
    def toggle_only_show_highlight(options, highlight_x):
        if highlight_x is not None:
            return time.time(), status.success("Only showing the highlighted path.")
        if "only_show_highlight" in options:
            return no_update, status.info("Only the path will be shown once you highlight a point.")
        raise PreventUpdate
