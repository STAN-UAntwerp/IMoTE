import dash
import dash_bootstrap_components as dbc

from imote.components.cytoscape_graph import make_cytoscape_graph
from imote.components.status_bar import make_status_bar
from imote.components.cards.a_tree_info_card import make_tree_info_card
from imote.components.cards.b_node_info_card import make_node_info_card
from imote.components.cards.c_new_tree_card import make_new_tree_card
from imote.components.cards.d_edit_tree_card import make_edit_tree_card
from imote.components.cards.e_layout_card import make_layout_card
from imote.components.cards.f_highlight_card import make_highlight_card

dash.register_page(__name__, path="/", name="Tree View")

# The status bar sits right above the graph: that's where the user looks after clicking a button, and it stays
# visible however far the cards on the right are scrolled (the old debug text was below the last card).
layout = dbc.Row(
    [
        dbc.Col(
            [make_status_bar(), make_cytoscape_graph()],
            width=8,
            style={"height": "85vh", "display": "flex", "flexDirection": "column"},
        ),
        dbc.Col(
            [
                make_tree_info_card(),
                make_node_info_card(),
                make_new_tree_card(),
                make_edit_tree_card(),
                make_layout_card(),
                make_highlight_card(),
            ],
            width=4,
            style={"maxHeight": "85vh", "overflowY": "auto"},
        ),
    ]
)
