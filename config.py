from pathlib import Path
from functools import lru_cache
import pickle

# --- Filesystem ---------------------------------------------------------
DIR_BASE = Path(__file__).resolve().parent
DIR_LIVE_OUTPUT = DIR_BASE / "output" / "live"
DIR_SAVED_VIZ_TREES = DIR_BASE / "output" / "saved_viz_trees"
DIR_DATASET_UPLOAD = DIR_BASE / "dataset" / "uploaded"

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
    {"label": "Pilot in C++", "value": "PilotC"},
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
DEFAULT_MAX_DEPTH = 12
DEFAULT_MAX_MODEL_DEPTH = 30
DEFAULT_MIN_SAMPLE_SPLIT = 10
DEFAULT_MIN_SAMPLE_LEAF = 5

# --- Layout card defaults ------------------------------------------------
DEFAULT_RANK_SEP = 10
DEFAULT_NODE_SEP = 10

# --- Settings node plot defaults -----------------------------------------
DEFAULT_COLLAPSE_LEVEL = 5
DEFAULT_DISPLAY_TYPE = "histogram"
DEFAULT_NMAX = 5
DEFAULT_FIG_W = 5
DEFAULT_FIG_H = 3

MIN_EDGE_WIDTH = 0.8
MAX_EDGE_WIDTH = 8

MIN_NODE_HEIGHT = 0.1
MAX_NODE_HEIGHT = 50

SPINNER_COLOR = "primary"

DEFAULT_NODE_METRICS = ["ID", "# Samples", "RSS", 'MAE']

@lru_cache(maxsize=1)
def get_initial_graph_info():
    with open(
            DIR_SAVED_VIZ_TREES / "tree_14-08-26_12-45-55__1199_BNG_echoMonths.pmlb-Pilot-12-30-10-5.pkl",
            "rb") as f:
        input_dict = pickle.load(f)
    return input_dict["viz_tree_dict"], input_dict["tree_params"]

# ── Stylesheet ────────────────────────────────────────────────────────────
font_family = "Arial, sans-serif"
font_size = 12
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
    {
        "selector": "node.regplot",
        "style": {
            "label": "",
            "background-image": 'data(dir_regplot)',
            "shape": "rectangle",
            "width": "500px",
            "height": "300px",
        },
    },
    {
        "selector": "node.predsplot",
        "style": {
            "label": "",
            "background-image": 'data(dir_predsplot)',
            "shape": "rectangle",
            "width": "500px",
            "height": "300px",
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
]
