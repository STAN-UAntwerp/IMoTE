"""Callbacks of the dataset dialog and of the dataset/method selection in the New Tree card.

The dialog has two stages: choosing a source (search PMLB or UCI, or upload a file) and the settings of the
dataset (response, task, categorical features). ``ids.DS_STORE`` holds the state of the dialog::

    {"mode": "add" | "edit", "stage": "source" | "config", "id": dataset id, "doc": metadata (TOML text)}

In "add" mode the data is already on disk but the metadata file isn't, so the dataset isn't listed until Save.
"""
from __future__ import annotations

import base64
import logging

import dash_bootstrap_components as dbc
import tomlkit
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from imote import ids, status
from imote.config import DEFAULT_DATASET_NAME, ENABLE_CLASSIFICATION, SHOWN_METHODS
from imote.dataset import dataset_registry as registry
from imote.dataset import sources
from imote.dataset.metadata import (
    complete_meta, guess_task, new_categorical_table, prepare, set_value, task_kind, write_meta,
)
from imote.dataset.readers import MissingDependencyError

logger = logging.getLogger(__name__)

HIDDEN = {"display": "none"}
SHOWN = {}
REMOVE_LABEL = "Remove dataset"
REMOVE_CONFIRM_LABEL = "Click again to remove it"
NO_CLASSIFICATION_NOTE = ("No methods available yet: fitting classification trees isn't supported yet, "
                          "only regression trees.")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _store(mode: str, dataset_id: str, doc) -> dict:
    return {"mode": mode, "stage": "config", "id": dataset_id, "doc": tomlkit.dumps(doc)}


def _result_list(results: list[sources.SearchResult], empty_text: str):
    if not results:
        return html.Div(empty_text, className="text-muted small mt-2")
    items = []
    on_list = registry.source_index()
    for result in results:
        existing = on_list.get((result.source, result.key))
        badge = dbc.Badge("on your list", color="success", className="ms-2") if existing else None
        items.append(
            dbc.ListGroupItem(
                html.Div(
                    [
                        html.Div([html.Span(result.name, className="fw-semibold"), badge]),
                        html.Small(result.details, className="text-muted"),
                    ]
                ),
                id={"type": ids.DS_PICK, "source": result.source, "key": result.key},
                action=True,
                n_clicks=0,
            )
        )
    footer = []
    if len(results) >= sources.MAX_RESULTS:
        footer = [html.Small(f"Showing the first {sources.MAX_RESULTS} matches, refine the search to see others.",
                             className="text-muted")]
    return [dbc.ListGroup(items, flush=True), *footer]


def apply_form(doc, df, name, response, task, categorical, ignore) -> None:
    """Write the values of the settings form into the metadata. Values that didn't change keep their comment
    (so automatically detected values stay marked), changed values are stored as chosen by the user."""
    if name and name != doc.get("name"):
        set_value(doc, "name", name)
    target_changed = False
    if response and response != doc.get("response"):
        set_value(doc, "response", response)
        target_changed = True
    if task and task != doc.get("task"):
        set_value(doc, "task", task)
        target_changed = True
    if target_changed and "classes" in doc:
        del doc["classes"]

    ignore = [col for col in (ignore or []) if col != response]
    if set(ignore) != {str(c) for c in doc.get("ignore", [])}:
        set_value(doc, "ignore", ignore)

    if "categorical" not in doc:
        doc["categorical"] = new_categorical_table()
    table = doc["categorical"]
    selected = [col for col in (categorical or []) if col != response and col not in ignore]
    for col in list(table.keys()):
        if col in df.columns and col not in selected:
            del table[col]
    for col in selected:
        if col not in table:
            set_value(table, col, [])  # the categories are filled in by complete_meta
    complete_meta(doc, df)


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

def register_callbacks(app):
    # --- OPEN THE DIALOG ---
    @app.callback(
        Output(ids.DS_STORE, "data", allow_duplicate=True),
        Output(ids.DS_MODAL, "is_open", allow_duplicate=True),
        Output(ids.DS_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.BTN_DATASET_ADD, "n_clicks"),
        Input(ids.BTN_DATASET_EDIT, "n_clicks"),
        State(ids.INPUT_DATASET, "value"),
        prevent_initial_call=True,
    )
    def open_dialog(_, __, dataset_id):
        # Dash also calls this when the page layout is inserted (prevent_initial_call doesn't stop that), only a
        # real click may open the dialog
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            raise PreventUpdate
        if ctx.triggered_id == ids.BTN_DATASET_ADD:
            return {"mode": "add", "stage": "source"}, True, None, no_update
        if not dataset_id:
            return no_update, no_update, no_update, status.warning("Select a dataset first.")
        try:
            doc, _ = registry.load_completed(dataset_id)
        except Exception as e:
            logger.exception("Could not open dataset %s", dataset_id)
            return no_update, no_update, no_update, status.error(f"Could not load dataset {dataset_id}: {e}")
        return _store("edit", dataset_id, doc), True, None, no_update

    # --- SHOW THE STAGE OF THE DIALOG ---
    @app.callback(
        Output(ids.DS_MODAL_TITLE, "children"),
        Output(ids.DS_SOURCE_STAGE, "style"),
        Output(ids.DS_CONFIG_STAGE, "style"),
        Output(ids.DS_BTN_SAVE, "style"),
        Output(ids.DS_BTN_REMOVE, "style"),
        Output(ids.DS_BTN_REMOVE, "children", allow_duplicate=True),
        Output(ids.DS_CFG_FILE, "children"),
        Output(ids.DS_CFG_NAME, "value"),
        Output(ids.DS_CFG_RESPONSE, "options"),
        Output(ids.DS_CFG_RESPONSE, "value"),
        Output(ids.DS_CFG_TASK, "value", allow_duplicate=True),
        Output(ids.DS_CFG_CATEGORICAL, "options"),
        Output(ids.DS_CFG_CATEGORICAL, "value"),
        Output(ids.DS_CFG_IGNORE, "options"),
        Output(ids.DS_CFG_IGNORE, "value"),

        Input(ids.DS_STORE, "data"),
        prevent_initial_call=True,
    )
    def show_stage(store):
        if not store:
            raise PreventUpdate
        if store.get("stage") != "config":
            return ("Add a dataset", SHOWN, HIDDEN, HIDDEN, HIDDEN, REMOVE_LABEL, None,
                    *([no_update] * 8))
        doc = tomlkit.parse(store["doc"])
        columns = list(registry.read_data(store["id"], doc).columns)
        options = [{"label": col, "value": col} for col in columns]
        editing = store["mode"] == "edit"
        title = f"Dataset settings: {doc.get('name', store['id'])}" if editing else f"Add {doc.get('name', '')}"
        file_hint = [
            "All settings are saved in ",
            html.Code(str(registry.meta_path(store["id"]))),
            ", edit that file for more control, e.g. the order of the categories or of the classes.",
        ]
        return (title, HIDDEN, SHOWN, SHOWN, SHOWN if editing else HIDDEN, REMOVE_LABEL, file_hint,
                str(doc.get("name", "")),
                options, str(doc.get("response", "")),
                doc.get("task"),
                options, [str(c) for c in (doc.get("categorical") or {}) if str(c) in columns],
                options, [str(c) for c in doc.get("ignore", []) if str(c) in columns])

    # --- SEARCH PMLB ---
    @app.callback(
        Output(ids.DS_PMLB_RESULTS, "children"),

        Input(ids.DS_PMLB_SEARCH, "value"),
        Input(ids.DS_PMLB_TASK, "value"),
        Input(ids.DS_MODAL, "is_open"),
        prevent_initial_call=True,
    )
    def search_pmlb(query, task, is_open):
        if not is_open:
            raise PreventUpdate
        task = None if task == "all" else task
        results = sources.search_pmlb(query or "", task, registry.cache_dir())
        return _result_list(results, "No PMLB datasets match this search.")

    # --- SEARCH UCI ---
    @app.callback(
        Output(ids.DS_UCI_RESULTS, "children"),

        Input(ids.DS_UCI_SEARCH, "value"),
        Input(ids.DS_TABS, "active_tab"),
        prevent_initial_call=True,
    )
    def search_uci(query, active_tab):
        if active_tab != "uci":
            raise PreventUpdate
        try:
            results = sources.search_uci(query or "", registry.cache_dir())
        except ConnectionError as e:
            return html.Div(str(e), className="text-danger small mt-2")
        return _result_list(results, "No UCI datasets match this search.")

    # --- PICK A SEARCH RESULT: DOWNLOAD IT ---
    @app.callback(
        Output(ids.DS_STORE, "data", allow_duplicate=True),
        Output(ids.DS_FEEDBACK, "children", allow_duplicate=True),

        Input({"type": ids.DS_PICK, "source": ALL, "key": ALL}, "n_clicks"),
        State(ids.DS_STORE, "data"),
        prevent_initial_call=True,
    )
    def pick_result(n_clicks, store):
        if not ctx.triggered_id or not ctx.triggered[0]["value"] or (store or {}).get("stage") != "source":
            raise PreventUpdate
        source, key = ctx.triggered_id["source"], ctx.triggered_id["key"]
        existing = registry.source_index().get((source, key))
        if existing:
            return no_update, f"This dataset is already on your list as {existing}."
        try:
            dataset_id, doc, _ = registry.add_from_source(source, key)
        except Exception as e:
            logger.exception("Could not add %s dataset %s", source, key)
            return no_update, f"Could not add this dataset: {e}"
        return _store("add", dataset_id, doc), None

    # --- UPLOAD A FILE ---
    @app.callback(
        Output(ids.DS_STORE, "data", allow_duplicate=True),
        Output(ids.DS_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.DS_UPLOAD, "contents"),

        Input(ids.DS_UPLOAD, "contents"),
        State(ids.DS_UPLOAD, "filename"),
        prevent_initial_call=True,
    )
    def upload_file(contents, filename):
        if contents is None:
            raise PreventUpdate
        _, content_string = contents.split(",", 1)
        try:
            dataset_id, doc, _ = registry.add_upload(filename, base64.b64decode(content_string))
        except MissingDependencyError as e:
            return no_update, str(e), None
        except Exception as e:
            logger.exception("Could not read uploaded file %s", filename)
            return no_update, f"Could not read {filename}: {e}", None
        return _store("add", dataset_id, doc), None, None

    # --- A NEW RESPONSE: GUESS ITS TASK ---
    @app.callback(
        Output(ids.DS_CFG_TASK, "value", allow_duplicate=True),

        Input(ids.DS_CFG_RESPONSE, "value"),
        State(ids.DS_STORE, "data"),
        prevent_initial_call=True,
    )
    def guess_task_of_response(response, store):
        if not response or not store or store.get("stage") != "config":
            raise PreventUpdate
        doc = tomlkit.parse(store["doc"])
        if response == doc.get("response"):
            return doc.get("task")
        df = registry.read_data(store["id"], doc)
        return guess_task(df[response]) if response in df.columns else no_update

    # --- PREVIEW THE RESULT OF THE SETTINGS ---
    @app.callback(
        Output(ids.DS_CFG_SUMMARY, "children"),
        Output(ids.DS_CFG_SUMMARY, "color"),

        Input(ids.DS_CFG_RESPONSE, "value"),
        Input(ids.DS_CFG_TASK, "value"),
        Input(ids.DS_CFG_CATEGORICAL, "value"),
        Input(ids.DS_CFG_IGNORE, "value"),
        State(ids.DS_CFG_NAME, "value"),
        State(ids.DS_STORE, "data"),
        prevent_initial_call=True,
    )
    def preview(response, task, categorical, ignore, name, store):
        if not store or store.get("stage") != "config" or not response or not task:
            raise PreventUpdate
        doc = tomlkit.parse(store["doc"])
        df = registry.read_data(store["id"], doc)
        try:
            apply_form(doc, df, name, response, task, categorical, ignore)
            prepared = prepare(df, doc)
        except Exception as e:
            return f"These settings don't work yet: {e}", "warning"
        kind = task_kind(prepared.task, prepared.classes)
        methods = [method["label"] for method in SHOWN_METHODS.values() if kind in method["tasks"]]
        methods_text = f"Methods: {', '.join(methods)}." if methods else NO_CLASSIFICATION_NOTE
        return [prepared.summary(), html.Br(), methods_text], "light"

    # --- SAVE ---
    @app.callback(
        Output(ids.DS_MODAL, "is_open", allow_duplicate=True),
        Output(ids.DS_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "options", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "value", allow_duplicate=True),
        Output(ids.DS_STORE, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.DS_BTN_SAVE, "n_clicks"),
        State(ids.DS_STORE, "data"),
        State(ids.DS_CFG_NAME, "value"),
        State(ids.DS_CFG_RESPONSE, "value"),
        State(ids.DS_CFG_TASK, "value"),
        State(ids.DS_CFG_CATEGORICAL, "value"),
        State(ids.DS_CFG_IGNORE, "value"),
        prevent_initial_call=True,
    )
    def save(n_clicks, store, name, response, task, categorical, ignore):
        if not n_clicks or not store or store.get("stage") != "config":
            raise PreventUpdate
        dataset_id = store["id"]
        # when editing, start from the file on disk, it may have been changed by hand since the dialog opened
        doc = registry.load_meta(dataset_id) if store["mode"] == "edit" else tomlkit.parse(store["doc"])
        df = registry.read_data(dataset_id, doc)
        try:
            apply_form(doc, df, name, response, task, categorical, ignore)
            prepared = prepare(df, doc)
        except Exception as e:
            return no_update, f"Can't save these settings: {e}", no_update, no_update, no_update, no_update
        write_meta(registry.meta_path(dataset_id), doc)
        verb = "Saved the settings of" if store["mode"] == "edit" else "Added"
        message = status.success(f"{verb} dataset {doc.get('name')}: {prepared.summary()} Click Fit Tree to fit a "
                                 f"tree on it.")
        return False, None, registry.dropdown_options(), dataset_id, None, message

    # --- CANCEL ---
    @app.callback(
        Output(ids.DS_MODAL, "is_open", allow_duplicate=True),
        Output(ids.DS_STORE, "data", allow_duplicate=True),
        Output(ids.DS_FEEDBACK, "children", allow_duplicate=True),

        Input(ids.DS_BTN_CANCEL, "n_clicks"),
        State(ids.DS_STORE, "data"),
        prevent_initial_call=True,
    )
    def cancel(n_clicks, store):
        if not n_clicks:
            raise PreventUpdate
        if store and store.get("mode") == "add" and store.get("id"):
            registry.discard_pending(store["id"])
        return False, None, None

    # --- REMOVE (click twice) ---
    @app.callback(
        Output(ids.DS_BTN_REMOVE, "children", allow_duplicate=True),
        Output(ids.DS_MODAL, "is_open", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "options", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "value", allow_duplicate=True),
        Output(ids.DS_STORE, "data", allow_duplicate=True),
        Output(ids.STORE_STATUS, "data", allow_duplicate=True),

        Input(ids.DS_BTN_REMOVE, "n_clicks"),
        State(ids.DS_BTN_REMOVE, "children"),
        State(ids.DS_STORE, "data"),
        prevent_initial_call=True,
    )
    def remove(n_clicks, label, store):
        if not n_clicks or not store or store.get("mode") != "edit":
            raise PreventUpdate
        if label != REMOVE_CONFIRM_LABEL:
            return REMOVE_CONFIRM_LABEL, no_update, no_update, no_update, no_update, no_update
        dataset_id = store["id"]
        name = registry.display_name(dataset_id)
        registry.remove(dataset_id)
        still_suggested = " It stays on the list as a suggestion." if registry.suggested_name(dataset_id) else ""
        message = status.success(f"Removed dataset {name} and its files.{still_suggested}")
        return REMOVE_LABEL, False, registry.dropdown_options(), DEFAULT_DATASET_NAME, None, message

    # --- METHODS THAT WORK ON THE SELECTED DATASET ---
    @app.callback(
        Output(ids.INPUT_METHOD, "options"),
        Output(ids.INPUT_METHOD, "value"),
        Output(ids.METHOD_NOTE, "children"),

        Input(ids.INPUT_DATASET, "value"),
        Input(ids.INPUT_DATASET, "options"),
        State(ids.INPUT_METHOD, "value"),
    )
    def filter_methods(dataset_id, _, method_name):
        kind = registry.dataset_kind(dataset_id) if dataset_id else None
        options = [{"label": method["label"], "value": value, "disabled": kind is not None and kind not in method["tasks"]}
                   for value, method in SHOWN_METHODS.items()]
        allowed = [o["value"] for o in options if not o["disabled"]]
        value = method_name if method_name in allowed else (allowed[0] if allowed else None)
        if not ENABLE_CLASSIFICATION and kind in ("binary", "multiclass"):
            note = NO_CLASSIFICATION_NOTE
        else:
            note = "Multiclass dataset: only the decision tree works for now." if kind == "multiclass" else None
        return options, value, note
