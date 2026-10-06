import dash_bootstrap_components as dbc
from imote.config import get_initial_graph_info

from imote import ids
from imote.callbacks.a_tree_info_callbacks import format_tree_info

def make_tree_info_card() -> dbc.Card:
    _, tree_params = get_initial_graph_info()

    return dbc.Card(
        id=ids.TREE_INFO_TEXT,
        children=format_tree_info(**tree_params),
        class_name="mb-3"
    )
