from m5py import M5Prime
import logging
import os
import pickle
import time
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import base64
from pathlib import Path
import numpy as np

from imote import ids
from imote.config import DIR_SAVED_VIZ_TREES, NO_FILE_SELECTED_PLACEHOLDER, DEFAULT_DATASET_NAME
from imote.viz_tree.viz_tree import VizTree
from imote.adapters.base_adapter import ADAPTERS_REGISTRY

from imote.dataset.dataset_registry import (
    NEW_CSV_OPTION,
    DIR_DATASET_UPLOAD,
    build_dropdown_options,
    get_target_col,
)
from imote.dataset.dataset import Dataset

logger = logging.getLogger(__name__)


def get_dataset_X_y(input_dataset):
    if input_dataset.endswith(".csv"):
        csv_path = Path(input_dataset)
        dataset_name = input_dataset[:-4]
        dataset = Dataset.from_csv(input_dataset, target_col=get_target_col(csv_path), name=dataset_name)
    elif input_dataset.endswith(".pmlb"):
        dataset_name = input_dataset[:-5]
        dataset = Dataset.from_pmlb(dataset_name)
    else:
        raise ValueError("Unknown dataset type")

    return dataset.X, dataset.y, dataset.cat_ids

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
    }

def _tree_file_name(viz_tree_dict, tree_params) -> str:
    dataset_name = Path(tree_params['dataset_name']).name
    return f"tree_{viz_tree_dict['tree_id']}__{dataset_name}-{tree_params['method_name']}-{tree_params['max_depth']}-{tree_params['max_model_depth']}-{tree_params['min_sample_split']}-{tree_params['min_sample_leaf']}"

def fit_new_tree(input_dataset, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf) -> tuple[VizTree, str]:
    logger.info("Fitting %s on %s (max_depth=%s, max_model_depth=%s, min_sample_split=%s, min_sample_leaf=%s)",
                method_name, input_dataset, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
    X, y, cat_ids = get_dataset_X_y(input_dataset)

    adapter = ADAPTERS_REGISTRY[method_name]
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
    else:
        raise ValueError(f'Method name {method_name} not recognized.')
    viz_tree = VizTree.from_model(adapter, X, y, model)
    time_string = f"{int(elapsed_time // 60)}min {int(elapsed_time % 60)}sec"
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

    # --- OPEN CSV MODAL ---
    @app.callback(
        Output(ids.MODAL_CSV, "is_open"),
        Input(ids.INPUT_DATASET, "value"),
        prevent_initial_call=True,
    )
    def maybe_open_csv_modal(value):
        if value == NEW_CSV_OPTION:
            return True
        raise PreventUpdate

    # --- CANCEL CSV MODAL ---
    @app.callback(
        Output(ids.MODAL_CSV_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV, "is_open", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "value", allow_duplicate=True),
        Output(ids.MODAL_CSV_FILE_NAME, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV_INPUT_TARGET_COL, "value", allow_duplicate=True),

        Input(ids.MODAL_CSV_BTN_CANCEL, "n_clicks"),
        State(ids.MODAL_CSV_FILE_NAME, "children"),
        prevent_initial_call=True,
    )
    def cancel_csv_modal(_, file_name):
        file_path = str(DIR_DATASET_UPLOAD / file_name)
        if file_name != NO_FILE_SELECTED_PLACEHOLDER:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        return None, False, DEFAULT_DATASET_NAME, NO_FILE_SELECTED_PLACEHOLDER, None

    # --- CONFIRM CSV MODAL ---
    @app.callback(
        Output(ids.MODAL_CSV_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV, "is_open", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "options", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "value", allow_duplicate=True),
        Output(ids.MODAL_CSV_FILE_NAME, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV_INPUT_TARGET_COL, "value", allow_duplicate=True),

        Input(ids.MODAL_CSV_BTN_CONFIRM, "n_clicks"),
        State(ids.MODAL_CSV_FILE_NAME, "children"),
        State(ids.MODAL_CSV_INPUT_TARGET_COL, "value"),
        prevent_initial_call=True,
    )
    def confirm_csv_modal(_, file_name, target_col):
        file_path = str(DIR_DATASET_UPLOAD / file_name)
        if file_name == NO_FILE_SELECTED_PLACEHOLDER:
            return "No file was uploaded yet.", no_update, no_update, no_update, no_update, no_update

        if not target_col:
            return "No target column was given.", no_update, no_update, no_update, no_update, no_update

        try:
            Dataset.from_csv(file_path, target_col=target_col)
        except (ValueError, KeyError, TypeError) as e:
            return f"Could not load CSV: {e}", no_update, no_update, no_update, no_update, no_update

        path = Path(file_path)
        path.with_suffix(".target.txt").write_text(target_col)
        return None, False, build_dropdown_options(), str(path), NO_FILE_SELECTED_PLACEHOLDER, None

    # --- CSV UPLOAD ---
    @app.callback(
        Output(ids.MODAL_CSV_FILE_NAME, "children"),
        Output(ids.MODAL_CSV_FEEDBACK, "children"),

        Input(ids.MODAL_CSV_UPLOAD, "contents"),
        State(ids.MODAL_CSV_UPLOAD, "filename"),
        State(ids.MODAL_CSV_FILE_NAME, "children"),
        prevent_initial_call=True,
    )
    def csv_upload(contents, filename, old_file_name):
        DIR_DATASET_UPLOAD.mkdir(parents=True, exist_ok=True)

        old_file_path = str(DIR_DATASET_UPLOAD / old_file_name)
        if old_file_name != NO_FILE_SELECTED_PLACEHOLDER:
            path = Path(old_file_path)
            if path.exists():
                path.unlink()

        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        filename = Path(filename).name
        path = DIR_DATASET_UPLOAD / filename

        if path.exists():
            return NO_FILE_SELECTED_PLACEHOLDER, f"There already exists a dataset with the name {filename}."

        path.write_bytes(decoded)
        return filename, no_update

    # --- LOAD TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),

        Input(ids.BTN_LOAD_TREE, "n_clicks"),
        State(ids.INPUT_LOAD_TREE, "value"),
        prevent_initial_call=True,
    )
    def load_tree(n_clicks, load_path):
        if n_clicks is None:
            raise PreventUpdate
        resolved_path, error = _resolve_saved_tree_path(load_path)
        if error:
            return no_update, no_update, no_update, no_update, no_update, error
        with open(resolved_path, "rb") as f:
            input_dict = pickle.load(f)
        
        tree_params = input_dict["tree_params"]
        viz_tree_dict = input_dict["viz_tree_dict"]
        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), f"Loaded tree from {resolved_path}"

    # --- RELOAD BASE TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_RELOAD_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE_BASE, "data"),
        State(ids.STORE_TREE_PARAMS_BASE, "data"),
        prevent_initial_call=True,
    )
    def reload_tree(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        return  viz_tree_dict, tree_params, time.time()

    # --- FIT NEW TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.SWITCH_COLOR_FEATURES, "value", allow_duplicate=True),
        Output(ids.TRIGGER_FOR_SPINNER, "children"),

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
        viz_tree, training_time = fit_new_tree(dataset_name, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
        viz_tree_dict = viz_tree.to_dict()
        tree_params = _make_tree_params(viz_tree, dataset_name, method_name, max_depth, max_model_depth,
                                        min_sample_split, min_sample_leaf, training_time)

        color_switch = no_update if feature_color or method_name == "Pilot" else True
        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), color_switch, ""

    # --- LOAD NEW TREE WITH ADAPTER ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.NEW_TREE_TRIGGER, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),

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
            return no_update, no_update, no_update, no_update, no_update, "Enter the path to a model file first."

        adapter = ADAPTERS_REGISTRY[adapter_name]
        try:
            model = adapter.load_model(str(Path(model_path.strip()).expanduser()))
        except (NotImplementedError, OSError, ValueError) as e:
            return no_update, no_update, no_update, no_update, no_update, f"Could not load the model: {e}"
        X, y, cat_ids = get_dataset_X_y(input_dataset)

        viz_tree = VizTree.from_model(adapter, X, y, model)

        viz_tree_dict = viz_tree.to_dict()
        tree_params = _make_tree_params(viz_tree, input_dataset, adapter_name)

        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), f"Loaded {model_path} with the {adapter_name} adapter."

    # --- NEW TREE TRIGGER ---
    @app.callback(
        Output(ids.SWITCH_MINIMAL, "value", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.NEW_TREE_TRIGGER, "data"),
        State(ids.SWITCH_MINIMAL, "value"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def new_tree_trigger(_, minimal, tree_params):
        if minimal == 0 and 8 < tree_params["n_leafs"] < 18:
            return 1, no_update
        elif tree_params["n_leafs"] >= 18:
            return 2, no_update
        else:
            return no_update, time.time()

    # --- SAVE TREE ---
    @app.callback(
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),

        Input(ids.BTN_SAVE_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def save_tree(n_clicks, viz_tree_dict, tree_params):
        output_path = str(DIR_SAVED_VIZ_TREES / f"{_tree_file_name(viz_tree_dict, tree_params)}.pkl")
        output_dict = {"tree_params": tree_params, 
                       "viz_tree_dict": viz_tree_dict}
        with open(output_path, "wb") as handle:
            # noinspection PyTypeChecker
            pickle.dump(output_dict, handle)
        return f"Tree saved to {output_path}"

    # --- DOWNLOAD SVG ---
    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "generateImage"),

        Input(ids.BTN_SAVE_TREE_SVG, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def download_tree_svg(n_clicks, viz_tree_dict, tree_params):
        file_name = _tree_file_name(viz_tree_dict, tree_params)
        return {"type": "svg", "action": "download", "filename": file_name}
