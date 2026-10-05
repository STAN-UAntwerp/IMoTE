import matplotlib
matplotlib.use('agg')

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, html
from flask import send_from_directory

from callbacks import register_all_callbacks
from components.stores import make_global_stores
from config import DIR_LIVE_OUTPUT, DIR_SAVED_VIZ_TREES

cyto.load_extra_layouts()

(DIR_LIVE_OUTPUT / "regplots").mkdir(parents=True, exist_ok=True)
(DIR_LIVE_OUTPUT / "predsplots").mkdir(parents=True, exist_ok=True)
DIR_SAVED_VIZ_TREES.mkdir(parents=True, exist_ok=True)

app: Dash = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Visualisation of linear model trees"

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
    brand="Visualisation of linear model trees",
    color="dark",
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

if __name__ == "__main__":
    app.run(debug=True)
