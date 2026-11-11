from m5py import M5Prime
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import logging
import os
import pickle
import time
from dash import Input, Output, State, no_update, set_props
from dash.exceptions import PreventUpdate
from pathlib import Path
import numpy as np

from imote import ids, status
from imote.config import DIR_SAVED_VIZ_TREES, METHODS
from imote.viz_tree.viz_tree import VizTree
from imote.adapters.base_adapter import ADAPTERS_REGISTRY

from imote.dataset.dataset_registry import display_name, dropdown_options, load_dataset
from imote.dataset.metadata import task_kind

logger = logging.getLogger(__name__)


def _resolve_saved_tree_path(load_path) -> tuple[str | None, str | None]:
    if not load_path or not load_path.strip():
        saved_trees = list(DIR_SAVED_VIZ_TREES.glob("*.pkl"))
        if not saved_trees:
            return None, f"No saved trees found in {DIR_SAVED_VIZ_TREES}."
        return str(max(saved_trees, key=os.path.getmtime)), None

    candidate = Path(load_path.strip()).expanduser()
    path = next((p for p in (candidate, DIR_SAVED_VIZ_TREES / candidate) if p.exists()), candidate)
    if path.suffix != ".pkl":
        return None, f"Can't load {load_path}: not a .pkl file."
    if not path.exists():
        return None, f"Can't load {load_path}: file doesn't exist."
    if not path.is_file():
        return None, f"Can't load {load_path}: not a regular file."
    return str(path), None

def _make_tree_params(viz_tree: VizTree, dataset_name, method_name, max_depth=-1, max_model_depth=-1,
                      min_sample_split=-1, min_sample_leaf=-1, training_time=-1) -> dict:
    n_leafs = viz_tree.get_n_leafs()
    return {
        "dataset_name": dataset_name,
        "method_name": method_name,
        "max_depth": max_depth,
        "max_model_depth": max_model_depth,
        "min_sample_split": min_sample_split,
        "min_sample_leaf": min_sample_leaf,
        "training_time": training_time,
        "subtree_node_id": -1,
        "collapsed_nodes_count": 0,
        "highlight_x": None,
        "n_internal_nodes": len(viz_tree.nodes) - n_leafs,
        "n_leafs": n_leafs,
        "depth": viz_tree.get_depth(),
        "n_samples": viz_tree.X_train.shape[0],
        "n_features": viz_tree.X_train.shape[1],
        "pruned": False,
        "task": viz_tree.task,
    }

def _dataset_label(input_dataset: str) -> str:
    return display_name(input_dataset or "")

def _tree_file_name(viz_tree_dict, tree_params) -> str:
    dataset_name = Path(tree_params['dataset_name']).name
    return f"tree_{viz_tree_dict['tree_id']}__{dataset_name}-{tree_params['method_name']}-{tree_params['max_depth']}-{tree_params['max_model_depth']}-{tree_params['min_sample_split']}-{tree_params['min_sample_leaf']}"

def fit_new_tree(input_dataset, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf) -> tuple[VizTree, str]:
    logger.info("Fitting %s on %s (max_depth=%s, max_model_depth=%s, min_sample_split=%s, min_sample_leaf=%s)",
                method_name, input_dataset, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
    dataset = load_dataset(input_dataset)
    X, y, cat_ids = dataset.X, dataset.y, dataset.cat_ids

    adapter = ADAPTERS_REGISTRY[method_name]
    kind = task_kind(dataset.task, dataset.classes or list(np.unique(y)))
    method = METHODS[method_name]
    if kind not in method["tasks"]:
        raise ValueError(f"{method['label']} doesn't work on {kind} datasets, only on "
                         f"{' or '.join(sorted(method['tasks']))}. Change the task in the dataset settings if needed.")
    if kind == "binary":
        y = (y == np.unique(y)[-1]).astype(float)

    if method_name == "Pilot":
        from pilot import PILOT
        start_time = time.perf_counter()
        model = PILOT(max_depth=max_depth,
                      max_model_depth=max_model_depth,
                      min_sample_fit=min_sample_split,
                      min_sample_leaf=min_sample_leaf,
                      )
        categorical = np.zeros(X.shape[1])
        if not np.array_equal(cat_ids, np.array([-1])):
            categorical[cat_ids] = 1
        model.train(X, y, categorical)
        elapsed_time = time.perf_counter() - start_time
    elif method_name == "M5":
        start_time = time.perf_counter()
        model = M5Prime(
            use_pruning=True,
            use_smoothing=True,
            min_samples_leaf=min_sample_leaf,
            min_samples_split=min_sample_split,
            random_state=42,
            max_depth=max_depth,
        )
        model.fit(X, y)
        elapsed_time = time.perf_counter() - start_time
    elif method_name == "BcPilot":
        from imote.bcpilot.bcPilot import Pilot as BcPilot
        start_time = time.perf_counter()
        model = BcPilot(max_depth=max_depth,
                        max_model_depth=max_model_depth,
                        min_samples_split=min_sample_split,
                        min_samples_leaf=min_sample_leaf,
                        )
        bc_cat_ids = None if np.array_equal(cat_ids, np.array([-1])) else list(cat_ids)
        model.fit(X.astype(float), y, cat_ids=bc_cat_ids)
        elapsed_time = time.perf_counter() - start_time
    elif method_name in ("DecisionTreeRegressor", "DecisionTreeClassifier"):
        model_cls = DecisionTreeRegressor if method_name == "DecisionTreeRegressor" else DecisionTreeClassifier
        start_time = time.perf_counter()
        model = model_cls(max_depth=max_depth,
                          min_samples_split=min_sample_split,
                          min_samples_leaf=min_sample_leaf,
                          random_state=42,
                          )
        model.fit(X, y)
        elapsed_time = time.perf_counter() - start_time
    else:
        raise ValueError(f'Method name {method_name} not recognized.')
    viz_tree = VizTree.from_model(adapter, X, y, model, dataset.feature_names, dataset.target_name)
    if elapsed_time < 1:
        time_string = "less than 1 sec"
    elif elapsed_time < 60:
        time_string = f"{int(elapsed_time)} sec"
    else:
        time_string = f"{int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec"
    return viz_tree, time_string

def register_callbacks(app):
    @app.callback(
        Output(ids.NEW_TREE_COLLAPSE, "is_open"),
        Input(ids.NEW_TREE_TOGGLE_BUTTON, "n_clicks"),
        State(ids.NEW_TREE_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_card(_, is_open):
        return not is_open

    @app.callback(
        Output(ids.ADAPTER_COLLAPSE, "is_open"),
        Input(ids.ADAPTER_TOGGLE_BUTTON, "n_clicks"),
        State(ids.ADAPTER_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_adapter(_, is_open):
        return not is_open

    # --- LOAD TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_LOAD_TREE, "n_clicks"),
        State(ids.INPUT_LOAD_TREE, "value"),
        prevent_initial_call=True,
    )
    def load_tree(n_clicks, load_path):
        if n_clicks is None:
            raise PreventUpdate
        resolved_path, error = _resolve_saved_tree_path(load_path)
        if error:
            return no_update, no_update, no_update, no_update, no_update, status.warning(error)
        try:
            with open(resolved_path, "rb") as f:
                input_dict = pickle.load(f)
            viz_tree_dict = input_dict["viz_tree_dict"]
            tree_params = {**input_dict["tree_params"], "highlight_x": None,
                           "task": viz_tree_dict.get("task", "regression")}
        except Exception as e:
            message = status.error(f"Could not load {resolved_path}, is it a tree saved by IMoTE? ({e})")
            return no_update, no_update, no_update, no_update, no_update, message

        most_recent = " (the most recently saved tree)" if not load_path or not load_path.strip() else ""
        message = status.success(f"Loaded {Path(resolved_path).name}{most_recent}.")
        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), message

    # --- RELOAD BASE TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_RELOAD_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE_BASE, "data"),
        State(ids.STORE_TREE_PARAMS_BASE, "data"),
        prevent_initial_call=True,
    )
    def reload_tree(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        message = status.success("Reloaded the tree as it was right after fitting or loading it.")
        return viz_tree_dict, tree_params, time.time(), message

    # --- FIT NEW TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.SWITCH_COLOR_FEATURES, "value", allow_duplicate=True),
        Output(ids.TRIGGER_FOR_SPINNER, "children"),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_FIT_NEW_TREE, "n_clicks"),
        State(ids.INPUT_DATASET, "value"),
        State(ids.INPUT_METHOD, "value"),
        State(ids.INPUT_MAX_DEPTH, "value"),
        State(ids.INPUT_MAX_MODEL_DEPTH, "value"),
        State(ids.INPUT_MIN_SAMPLE_SPLIT, "value"),
        State(ids.INPUT_MIN_SAMPLE_LEAF, "value"),
        State(ids.SWITCH_COLOR_FEATURES, "value"),
        prevent_initial_call=True,
    )
    def fit_tree(n_clicks, dataset_name, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf, feature_color):
        if n_clicks is None:
            raise PreventUpdate
        if not dataset_name:
            message = status.warning("Select or add a dataset first.")
            return no_update, no_update, no_update, no_update, no_update, no_update, "", message
        if None in (max_depth, max_model_depth, min_sample_split, min_sample_leaf):
            message = status.warning("Fill in all tree parameters (max depth, max model depth, min sample split "
                                     "and min sample leaf) before fitting.")
            return no_update, no_update, no_update, no_update, no_update, no_update, "", message
        try:
            viz_tree, training_time = fit_new_tree(dataset_name, method_name, max_depth, max_model_depth,
                                                   min_sample_split, min_sample_leaf)
        except Exception as e:
            message = status.error(f"Could not fit {method_name} on {_dataset_label(dataset_name)}: {e}")
            return no_update, no_update, no_update, no_update, no_update, no_update, "", message
        viz_tree_dict = viz_tree.to_dict()
        tree_params = _make_tree_params(viz_tree, dataset_name, method_name, max_depth, max_model_depth,
                                        min_sample_split, min_sample_leaf, training_time)
        set_props(ids.INPUT_DATASET, {"options": dropdown_options()})  # a suggested dataset is now downloaded

        color_switch = no_update if feature_color or method_name in ("Pilot", "BcPilot") else True
        message = status.success(f"Fitted {method_name} on {_dataset_label(dataset_name)} in {training_time}.")
        if viz_tree.task == "classification" and len(np.unique(viz_tree.y_train)) > 2:
            message = status.warning(f"{message['text']} This is a multiclass tree: the node metrics and plots "
                                     f"still treat it as two classes, multiclass views are not implemented yet.")
        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), color_switch, "", message

    # --- LOAD NEW TREE WITH ADAPTER ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_LOAD_TREE_ADAPTER, "n_clicks"),
        State(ids.INPUT_DATASET, "value"),
        State(ids.INPUT_ADAPTER, "value"),
        State(ids.INPUT_LOAD_TREE_ADAPTER, "value"),
        prevent_initial_call=True,
    )
    def load_tree_with_adapter(n_clicks, input_dataset, adapter_name, model_path):
        if n_clicks is None:
            raise PreventUpdate

        if not model_path or not model_path.strip():
            message = status.warning("Enter the path to a model file first.")
            return no_update, no_update, no_update, no_update, no_update, message

        adapter = ADAPTERS_REGISTRY[adapter_name]
        try:
            model = adapter.load_model(str(Path(model_path.strip()).expanduser()))
        except (NotImplementedError, OSError, ValueError) as e:
            message = status.error(f"Could not load the model with the {adapter_name} adapter: {e}")
            return no_update, no_update, no_update, no_update, no_update, message
        try:
            dataset = load_dataset(input_dataset)
            viz_tree = VizTree.from_model(adapter, dataset.X, dataset.y, model, dataset.feature_names, dataset.target_name)
        except Exception as e:
            message = status.error(f"Could not build the tree from {Path(model_path).name} on "
                                   f"{_dataset_label(input_dataset)}, does the dataset match the model? ({e})")
            return no_update, no_update, no_update, no_update, no_update, message

        viz_tree_dict = viz_tree.to_dict()
        tree_params = _make_tree_params(viz_tree, input_dataset, adapter_name)

        message = status.success(f"Loaded {Path(model_path).name} with the {adapter_name} adapter on {_dataset_label(input_dataset)}.")
        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), message

    # --- NEW TREE TRIGGER ---
    @app.callback(
        Output(ids.SWITCH_MINIMAL, "value", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),
        Output(ids.STORE_HIGHLIGHT_X, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.NEW_TREE_TRIGGER, "data"),
        State(ids.SWITCH_MINIMAL, "value"),
        State(ids.SWITCH_DATA_NODE_SIZE, "value"),
        State(ids.STORE_TREE_PARAMS, "data"),
        State(ids.STORE_STATUS, "data"),
        prevent_initial_call=True,
    )
    def new_tree_trigger(_, minimal, data_node, tree_params, message):
        n_nodes = tree_params["n_leafs"] + tree_params["n_internal_nodes"]
        if data_node:
            new_minimal = minimal
        elif n_nodes >= 40:
            new_minimal = 2
        elif n_nodes > 11:
            new_minimal = 1
        else:
            new_minimal = 0

        if new_minimal == minimal:
            return no_update, time.time(), None, no_update
        else:
            size_name = {0: "Normal", 1: "Minimal", 2: "Tiny"}[new_minimal]
            note = status.with_note(message, f"Node size set to {size_name} to fit tree with {n_nodes} nodes.")
            return new_minimal, no_update, None, note

    # --- SAVE TREE ---
    @app.callback(
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_SAVE_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def save_tree(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        output_path = str(DIR_SAVED_VIZ_TREES / f"{_tree_file_name(viz_tree_dict, tree_params)}.pkl")
        output_dict = {"tree_params": tree_params,
                       "viz_tree_dict": viz_tree_dict}
        try:
            with open(output_path, "wb") as handle:
                # noinspection PyTypeChecker
                pickle.dump(output_dict, handle)
        except OSError as e:
            return status.error(f"Could not save the tree: {e}")
        return status.success(f"Tree saved to {output_path}")

    # --- DOWNLOAD SVG ---
    # assets/graph_plots.js makes the svg, so the plots pinned on the graph are in it too
    @app.callback(
        Output(ids.STORE_TREE_DOWNLOAD, "data"),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_SAVE_TREE_SVG, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def download_tree_svg(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        file_name = _tree_file_name(viz_tree_dict, tree_params)
        message = status.info(f"Downloading the shown tree as {file_name}.svg.")
        return {"filename": f"{file_name}.svg", "time": time.time()}, message

    app.clientside_callback(
        """
        function(download) {
            if (download) window.imoteGraphPlots.downloadSvg(%r, download.filename);
        }
        """ % ids.CYTOSCAPE_GRAPH,
        Input(ids.STORE_TREE_DOWNLOAD, "data"),
    )
