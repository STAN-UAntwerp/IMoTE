from dash import ALL, Input, Output, State, ctx

from imote import ids
from imote.components.status_bar import render_status_history, render_status_message

MAX_HISTORY = 15


def register_callbacks(app):
    @app.callback(
        Output({"type": ids.STATUS_MESSAGE, "index": ALL}, "children"),
        Output({"type": ids.STATUS_MESSAGE, "index": ALL}, "className"),
        Output(ids.STORE_STATUS_HISTORY, "data"),
        Output({"type": ids.STATUS_HISTORY_LIST, "index": ALL}, "children"),

        Input(ids.STORE_STATUS, "data"),
        State(ids.STORE_STATUS_HISTORY, "data"),
    )
    def show_status(message, history):
        if message["time"] and (not history or history[-1]["id"] != message["id"]):
            history = (history + [message])[-MAX_HISTORY:]
        children, class_name = render_status_message(message)
        n_bars = len(ctx.outputs_list[0])
        return [children] * n_bars, [class_name] * n_bars, history, [render_status_history(history)] * n_bars
