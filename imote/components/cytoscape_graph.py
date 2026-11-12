import dash_cytoscape as cyto
from imote.viz_tree.viz_tree import VizTree

from imote.viz_tree.viz_tree_cytoscape import viz_tree_to_cytoscape_elements
from imote.viz_tree.tree_layout import layout_elements
from imote import ids
from imote.config import (DIR_LIVE_OUTPUT, get_initial_graph_info,
                    CYTOSCAPE_STYLESHEET, DEFAULT_RANK_SEP, DEFAULT_NODE_SEP)


viz_tree_dict, _ = get_initial_graph_info()
initial_viz_tree      = VizTree.from_dict(viz_tree_dict)
initial_base_elements = viz_tree_to_cytoscape_elements(initial_viz_tree, str(DIR_LIVE_OUTPUT))
initial_elements, initial_layout = layout_elements(initial_base_elements, rank_sep=DEFAULT_RANK_SEP,
                                                   node_sep=DEFAULT_NODE_SEP)
initial_stylesheet    = CYTOSCAPE_STYLESHEET

def make_cytoscape_graph() -> cyto.Cytoscape:
    return cyto.Cytoscape(
        id=ids.CYTOSCAPE_GRAPH,
        elements=initial_elements,
        stylesheet=initial_stylesheet,
        layout=initial_layout,
        autoRefreshLayout=False,
        style={
            "width": "100%",
            "height": "100%",
            "border": "1px solid #ddd",
            "borderRadius": "6px",
        },
        minZoom=0.1,
        maxZoom=10,
        autounselectify=False,
        boxSelectionEnabled=False,
    )
