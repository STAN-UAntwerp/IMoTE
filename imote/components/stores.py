from dash import dcc, html
from imote.config import get_initial_graph_info
from imote import ids, status

def make_global_stores() -> html.Div:
    viz_tree_dict, tree_params = get_initial_graph_info()

    return html.Div(
        [
            dcc.Store(id=ids.ELEMENTS_TRIGGER, data=0),
            dcc.Store(id=ids.NEW_TREE_TRIGGER, data=0),
            dcc.Store(id=ids.STORE_VIZ_TREE, data=viz_tree_dict),
            dcc.Store(id=ids.STORE_VIZ_TREE_BASE, data=viz_tree_dict),
            dcc.Store(id=ids.STORE_HIGHLIGHT_X, data=None),
            dcc.Store(id=ids.STORE_TREE_PARAMS,data=tree_params),
            dcc.Store(id=ids.STORE_TREE_PARAMS_BASE, data=tree_params),
            dcc.Store(id=ids.STORE_MODAL_DONT_ASK, data=False),
            dcc.Store(id=ids.STORE_LAST_PLOT_PATH),
            dcc.Store(id=ids.STORE_STATUS, data=status.WELCOME),
            dcc.Store(id=ids.STORE_STATUS_HISTORY, data=[]),
        ]
    )
