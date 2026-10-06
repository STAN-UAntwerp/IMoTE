import pytest

from imote.config import MAX_EDGE_WIDTH, MAX_NODE_HEIGHT, MIN_EDGE_WIDTH, MIN_NODE_HEIGHT, NODE_TYPE_COLORS
from imote.nodes.internal_node import LinearNode
from imote.nodes.node_model import ConstantNodeModel, SimpleLinearNodeModel
from imote.nodes.split_node import PconNode
from tests.helpers import X, leaf, mask, y
from imote.viz_tree.viz_tree import VizTree
from imote.viz_tree.viz_tree_cytoscape import viz_tree_to_cytoscape_elements


def nodes_of(els):
    return {e["data"]["id"]: e for e in els if "id" in e["data"]}


def edges_of(els):
    return {(e["data"]["source"], e["data"]["target"]): e for e in els if "source" in e["data"]}


def elements(tree, tmp_path, **kwargs):
    return viz_tree_to_cytoscape_elements(tree, str(tmp_path), **kwargs)


@pytest.fixture
def live_dir(tmp_path):
    (tmp_path / "regplots").mkdir()
    (tmp_path / "predsplots").mkdir()
    return tmp_path


@pytest.fixture
def chain_tree():
    """Two chained LinearNodes under a split:
        node0 PconNode(x0 <= 3.5)
        ├── node1 LinearNode(x2)          rows 0-3
        │   └── node3 LinearNode(x0)
        │       └── node4 Leaf
        └── node2 Leaf                    rows 4-7
    """
    lin2 = LinearNode(mask(range(4)), y[:4], 0.3, 0, SimpleLinearNodeModel(0, 1.0, 0.0), leaf(range(4), [1, 0, 2], 1.0))
    lin1 = LinearNode(mask(range(4)), y[:4], 0.4, 2, SimpleLinearNodeModel(2, 2.0, 0.0), lin2)
    root = PconNode(mask(range(8)), y, 1.0, 0, 3.5, lin1, leaf(range(4, 8), [0, 0, 1], -1.0),
                    ConstantNodeModel(1.0), ConstantNodeModel(-1.0))
    return VizTree(root, X, y, None)


# --- Default output ---
def test_default_elements(pilot_tree, tmp_path):
    els = elements(pilot_tree, tmp_path)
    nodes, edges = nodes_of(els), edges_of(els)
    assert sorted(nodes) == [f"node{i}" for i in range(6)]
    assert len(edges) == 5
    assert all(s in nodes and t in nodes for s, t in edges)
    for el in nodes.values():
        assert el["data"]["node_type"] in el["classes"]
        assert {"node_type", "color", "label", "label_minimal", "n_samples", "rss", "highlight"} <= el["data"].keys()
    assert nodes["node0"]["data"]["n_samples"] == 8


# --- Combine linear nodes ---
def test_combine_lin_to_node(chain_tree, tmp_path):
    els = elements(chain_tree, tmp_path, combine_lin=1)
    nodes, edges = nodes_of(els), edges_of(els)
    assert sorted(el["data"]["node_type"] for el in nodes.values()) == ["CombinedLinNode", "LeafNode", "LeafNode", "PconNode"]
    assert all(s in nodes and t in nodes for s, t in edges)


def test_combine_lin_to_edge(chain_tree, tmp_path):
    els = elements(chain_tree, tmp_path, combine_lin=2)
    nodes, edges = nodes_of(els), edges_of(els)
    assert sorted(nodes) == ["node0", "node2", "node4"]
    assert "combine_lin" in edges[("node0", "node4")]["classes"]


# --- Highlight ---
def test_highlight_path(pilot_tree, tmp_path):
    els = elements(pilot_tree, tmp_path, highlight_x=X[1])
    assert {k for k, e in nodes_of(els).items() if e["data"]["highlight"]} == {"node0", "node1", "node2", "node5"}
    assert {k for k, e in edges_of(els).items() if "highlight" in e["classes"]} == \
           {("node0", "node1"), ("node1", "node2"), ("node2", "node5")}


# --- Visual options ---
def test_color_features(pilot_tree, tmp_path):
    nodes = nodes_of(elements(pilot_tree, tmp_path, use_color_features=True))
    # node0 (LinearNode) and node2 (BlinNode) both use x2
    assert nodes["node0"]["data"]["color"] == nodes["node2"]["data"]["color"]
    assert nodes["node0"]["data"]["color"] != nodes["node1"]["data"]["color"]
    assert all(nodes[f"node{i}"]["data"]["color"] == NODE_TYPE_COLORS["LeafNode"] for i in (3, 4, 5))


# --- Node plots ---
@pytest.mark.parametrize("predsplot_type2, prefix", [(False, "predsplot_"), (True, "predsplot2_")])
def test_node_plots(pilot_tree, live_dir, predsplot_type2, prefix):
    nodes = nodes_of(elements(pilot_tree, live_dir, show_node_plots=True, predsplot_type2=predsplot_type2))
    assert len(list((live_dir / "regplots").glob("regplot_node*.svg"))) == 3
    assert len(list((live_dir / "predsplots").glob(f"{prefix}node*.svg"))) == 3
    for i in (0, 1, 2):
        assert "regplot" in nodes[f"node{i}"]["classes"] and "dir_regplot" in nodes[f"node{i}"]["data"]
    for i in (3, 4, 5):
        assert "predsplot" in nodes[f"node{i}"]["classes"] and "dir_predsplot" in nodes[f"node{i}"]["data"]


def test_no_predsplot_for_zero_coefficient_leaf(split_tree, live_dir):
    nodes = nodes_of(elements(split_tree, live_dir, show_node_plots=True))
    assert "dir_predsplot" not in nodes["node4"]["data"]
    assert "dir_predsplot" in nodes["node3"]["data"]
