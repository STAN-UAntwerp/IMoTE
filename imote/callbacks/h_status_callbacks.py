from dash import Input, Output, State

from imote import ids
from imote.components.status_bar import render_status_history, render_status_message

MAX_HISTORY = 50


def register_callbacks(app):
    @app.callback(
        Output(ids.STATUS_MESSAGE, "children"),
        Output(ids.STATUS_MESSAGE, "className"),
        Output(ids.STORE_STATUS_HISTORY, "data"),
        Output(ids.STATUS_HISTORY_LIST, "children"),

        Input(ids.STORE_STATUS, "data"),
        State(ids.STORE_STATUS_HISTORY, "data"),
    )
    def show_status(message, history):
        if message["time"] and (not history or history[-1]["id"] != message["id"]):
            history = (history + [message])[-MAX_HISTORY:]
        children, class_name = render_status_message(message)
        return children, class_name, history, render_status_history(history)
