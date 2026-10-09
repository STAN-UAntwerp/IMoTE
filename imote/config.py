import os
from pathlib import Path
from functools import lru_cache
import pickle

# --- Filesystem ---------------------------------------------------------
# Read-only files shipped inside the package.
DIR_PACKAGE = Path(__file__).resolve().parent
INITIAL_TREE_PATH = DIR_PACKAGE / "assets" / "initial_tree.pkl"

# Files the app writes at runtime go to a folder. Override with environment variable IMOTE_HOME.
DIR_USER_DATA = Path(os.environ.get("IMOTE_HOME", Path.home() / ".imote")).expanduser()
DIR_LIVE_OUTPUT = DIR_USER_DATA / "live"
DIR_SAVED_VIZ_TREES = DIR_USER_DATA / "saved_trees"
DIR_DATASET_UPLOAD = DIR_USER_DATA / "uploaded_datasets"

# --- New Tree card defaults / options -----------------------------------

NEW_CSV_OPTION = "__new_csv__"
NO_FILE_SELECTED_PLACEHOLDER = "No file uploaded."

PMLB_OPTIONS = [
    {"label": "PMLB: 547_no2", "value": "547_no2.pmlb"},
    {"label": "PMLB: 294_satellite_image", "value": "294_satellite_image.pmlb"},
    {"label": "PMLB: 1199_BNG_echoMonths", "value": "1199_BNG_echoMonths.pmlb"},
    {"label": "PMLB: 537_houses", "value": "537_houses.pmlb"},
    {"label": "PMLB: 658_fri_c3_250_25", "value": "658_fri_c3_250_25.pmlb"},
    {"label": "PMLB: 505_tecator", "value": "505_tecator.pmlb"},
    {"label": "PMLB: 560_bodyfat", "value": "560_bodyfat.pmlb"},
    {"label": "PMLB: 485_analcatdata_vehicle", "value": "485_analcatdata_vehicle.pmlb"},
    {"label": "PMLB: 210_cloud", "value": "210_cloud.pmlb"},
    {"label": "PMLB: 1028_SWD", "value": "1028_SWD.pmlb"},
    {"label": "PMLB: 197_cpu_act", "value": "197_cpu_act.pmlb"},
]

METHOD_OPTIONS = [
    {"label": "Pilot", "value": "Pilot"},
    {"label": "M5", "value": "M5"},
]

ADAPTER_OPTIONS = [
    {"label": "Partykit", "value": "Partykit"},
    {"label": "Pilot", "value": "Pilot"},
    {"label": "M5", "value": "M5"},
]
DEFAULT_ADAPTER_NAME = "Partykit"

NODE_TYPE_COLORS = {
    "LeafNode": "#2ca02c",
    "LinearNode": "#9467bd",
    "BlinNode": "#1f77b4",
    "PconNode": "#d62728",
    "SplitNode": "#d62728",
    "SplitCNode": "#8c564b",
    "PlinNode": "#ff7f0e",
    "PconcNode": "#8c564b",
    "CombinedLinNode": "#9467bd",
    "CollapsedNode": "#808080",
}

DEFAULT_METHOD_NAME = "Pilot"
DEFAULT_DATASET_NAME = "547_no2.pmlb"
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_MODEL_DEPTH = 15
DEFAULT_MIN_SAMPLE_SPLIT = 100
DEFAULT_MIN_SAMPLE_LEAF = 25

# --- Layout card defaults ------------------------------------------------
DEFAULT_RANK_SEP = 10
DEFAULT_NODE_SEP = 10

# --- Settings node plot defaults -----------------------------------------
DEFAULT_COLLAPSE_LEVEL = 5
DEFAULT_DISPLAY_TYPE = "histogram"
DEFAULT_NMAX = 7
DEFAULT_FIG_W = 6
DEFAULT_FIG_H = 3.4

MIN_EDGE_WIDTH = 0.8
MAX_EDGE_WIDTH = 8

MIN_NODE_HEIGHT = 0.1
MAX_NODE_HEIGHT = 50

SPINNER_COLOR = "primary"

DEFAULT_NODE_METRICS = ["ID", "# Samples", "RSS", 'MAE']

@lru_cache(maxsize=1)
def get_initial_graph_info():
    with open(INITIAL_TREE_PATH, "rb") as f:
        input_dict = pickle.load(f)
    return input_dict["viz_tree_dict"], input_dict["tree_params"]

# ── Stylesheet ────────────────────────────────────────────────────────────
font_family = "Arial, sans-serif"
font_size = 12

NODE_PLOTS_FIG_SIZE = (4.4, 2.6)
NODE_PLOTS_N_MAX = 6
NODE_PLOTS_EXTRA_SEP = 40
plots_on_font_size = 28
selected_color = "#000000"
CYTOSCAPE_STYLESHEET = [
    # ── Default node ──────────────────────────────────────────────────────
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "text-wrap": "wrap",
            "text-max-width": "140px",
            "font-family": font_family,
            "font-size": f"{font_size}px",
            "color": "#ffffff",
            "background-color": "data(color)",
            "shape": "roundrectangle",
            "width": "100px",
            "height": "40px",
            "padding": "6px",
            "border-width": "1.5px",
            "border-color": "#000000",
        },
    },
    # ── Specific nodes ────────────────────────────────────────────────────
    {
        "selector": "node.LeafNode",
        "style": {
            "shape": "ellipse",
            "width": "160px",
            "height": "60px",
        },
    },
    {
        "selector": "node.LinearNode, node.CombinedLinNode",
        "style": {
            "shape": "ellipse",
        },
    },
    {
        "selector": "node.CollapsedNode",
        "style": {
            "shape": "rectangle",
        },
    },
    # ── Show all node plots ───────────────────────────────────────────────
    {
        "selector": "node.plots_on",
        "style": {
            "font-size": f"{plots_on_font_size}px",
            "text-max-width": "300px",
            "width": "220px",
            "height": "90px",
            "border-width": "3px",
        },
    },
    {
        "selector": "node.LeafNode.plots_on",
        "style": {
            "width": "340px",
            "height": "130px",
        },
    },
    {
        "selector": "node.regplot, node.predsplot",
        "style": {
            "label": "",
            "shape": "rectangle",
            "width": f"{NODE_PLOTS_FIG_SIZE[0] * 100}px",
            "height": f"{NODE_PLOTS_FIG_SIZE[1] * 100}px",
            "background-fit": "contain",
            "background-opacity": 0,
            "border-width": 0,
        },
    },
    {
        "selector": "node.regplot",
        "style": {
            "background-image": 'data(dir_regplot)',
        },
    },
    {
        "selector": "node.predsplot",
        "style": {
            "background-image": 'data(dir_predsplot)',
        },
    },
    {
        "selector": "node.minimal",
        "style": {
            "label": "data(label_minimal)",
            "width": "30px",
            "height": "15px",
        },
    },
    {
        "selector": "node.tiny",
        "style": {
            "label": "",
            "width": "0.2px",
            "height": "0.1px",
        },
    },
    {
        "selector": "node.data_size",
        "style": {
            "label": "data(label_minimal)",
            "width": "data(width)",
            "height": "data(height)",
        },
    },

    # ── Default edge ──────────────────────────────────────────────────────
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#000000",
            "line-color": "#000000",
            "width": 1.5,
            "label": "data(label)",
            "font-family": font_family,
            "font-size": f"{font_size}px",
            "color": "#000000",
        },
    },
    # ── Specific edges ────────────────────────────────────────────────────
    {
        "selector": "edge.combine_lin",
        "style": {
            "mid-target-arrow-shape": "circle",
            "mid-target-arrow-color": "#9467bd",
        }
    },
    {
        "selector": "edge.label",
        "style": {
            "label": "data(label)",
            "text-background-color": "white",
            "text-background-opacity": 1,
            "text-background-padding": "5px",
            "font-size": "16px",
        },
    },
    {
        "selector": "edge.plots_on",
        "style": {
            "width": 4,
            "arrow-scale": 2,
            "font-size": f"{plots_on_font_size}px",
            "text-background-padding": "8px",
        },
    },
    {
        "selector": "edge.data_width",
        "style": {
            "width": "data(width)"
        },
    },

    # ── Selected highlight ────────────────────────────────────────────────
    {
        "selector": "node:selected",
        "style": {
            "border-width": "3px",
            "border-color": "#f0e442",
        },
    },
    # ── Highlighted path  ────────────────────────────────────────────────
    {
        "selector": "node.highlight",
        "style": {
            "border-width": "5px",
            "border-color": selected_color,
            "border-opacity": 1,
        },
    },
    {
        "selector": "edge.highlight",
        "style": {
            "line-color": selected_color,
            "target-arrow-color": selected_color,
            "width": 4,
        },
    },
    {
        "selector": "edge.highlight.plots_on",
        "style": {
            "width": 10,
        },
    },
]
