import logging

import matplotlib
matplotlib.use('agg')

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, html, set_props
from flask import send_from_directory

from imote import ids, status
from imote.callbacks import register_all_callbacks
from imote.components.stores import make_global_stores
from imote.config import DIR_LIVE_OUTPUT, DIR_SAVED_VIZ_TREES

cyto.load_extra_layouts()

for plot_dir in (DIR_LIVE_OUTPUT / "regplots", DIR_LIVE_OUTPUT / "predsplots"):
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("*.svg"):
        old_plot.unlink(missing_ok=True)
DIR_SAVED_VIZ_TREES.mkdir(parents=True, exist_ok=True)

def _report_callback_error(err: Exception):
    set_props(ids.STORE_STATUS, {"data": status.error(
        f"Something went wrong: {err.__class__.__name__}: {err}. See the terminal for details."
    )})


app: Dash = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    on_error=_report_callback_error,
)
app.title = "IMoTE: Interactive Model Tree Explorer"

@app.server.route("/internal_regplots/<path:filename>")
def serve_regplots(filename):
    return send_from_directory(str(DIR_LIVE_OUTPUT / "regplots"), filename)


@app.server.route("/internal_predsplots/<path:filename>")
def serve_predsplots(filename):
    return send_from_directory(str(DIR_LIVE_OUTPUT / "predsplots"), filename)


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavLink(page["name"], href=page["path"], active="exact")
        for page in dash.page_registry.values()
    ],
    brand="IMoTE: Interactive Model Tree Explorer",
    color="#1f77b4",
    dark=True,
)

# Stores live OUTSIDE dash.page_container so their data survives navigating
# between the Tree View and Explain pages.
app.layout = html.Div(
    [
        navbar,
        make_global_stores(),
        dbc.Container(dash.page_container, fluid=True, class_name="pt-3"),
    ]
)

register_all_callbacks(app)


def _setup_logging(level: int):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("imote")
    logger.addHandler(handler)
    logger.setLevel(level)


def main():
    _setup_logging(logging.INFO)
    app.run()


if __name__ == "__main__":
    _setup_logging(logging.DEBUG)
    app.run(debug=True)
