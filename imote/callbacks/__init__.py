from imote.callbacks.a_tree_info_callbacks import register_callbacks as register_tree_info_callbacks
from imote.callbacks.b_node_callbacks import register_callbacks as register_node_callbacks
from imote.callbacks.c_new_tree_callbacks import register_callbacks as register_new_tree_callbacks
from imote.callbacks.d_edit_tree_callbacks import register_callbacks as register_edit_tree_callbacks
from imote.callbacks.e_layout_callbacks import register_callbacks as register_layout_callbacks
from imote.callbacks.f_highlight_callbacks import register_callbacks as register_highlight_callbacks
from imote.callbacks.g_elements_callbacks import register_callbacks as register_elements_callbacks
from imote.callbacks.h_status_callbacks import register_callbacks as register_status_callbacks

def register_all_callbacks(app):
    register_tree_info_callbacks(app)
    register_node_callbacks(app)
    register_new_tree_callbacks(app)
    register_edit_tree_callbacks(app)
    register_layout_callbacks(app)
    register_highlight_callbacks(app)
    register_elements_callbacks(app)
    register_status_callbacks(app)
