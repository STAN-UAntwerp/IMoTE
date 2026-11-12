import itertools
import math

import pytest

from imote.config import CYTOSCAPE_STYLESHEET
from imote.viz_tree.tree_layout import ROUTED_EDGE_CLASS, layout_elements, node_size, segment_data
from imote.viz_tree.viz_tree import VizTree
from imote.viz_tree.viz_tree_cytoscape import viz_tree_to_cytoscape_elements
from tests.helpers import make_pilot_tree, make_single_leaf_tree, make_split_tree


def node(node_id, classes="SplitNode", **data):
    return {"data": {"id": node_id, "label": "", **data}, "classes": classes}


def edge(source, target):
    return {"data": {"source": source, "target": target}, "classes": ""}


def positions(layout):
    return {node_id: (p["x"], p["y"]) for node_id, p in layout["positions"].items()}


def boxes(elements, layout, direction="TB"):
    pos = positions(layout)
    result = {}
    for el in elements:
        if "source" not in el["data"]:
            w, h = node_size(el)
            x, y = pos[el["data"]["id"]]
            result[el["data"]["id"]] = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    return result


def overlapping(elements, layout):
    return [(a, b) for (a, ba), (b, bb) in itertools.combinations(boxes(elements, layout).items(), 2)
            if ba[0] < bb[2] and bb[0] < ba[2] and ba[1] < bb[3] and bb[1] < ba[3]]


def full_binary(depth, classes="SplitNode"):
    """Elements of a full binary tree, node ids by heap index, edges left child first."""
    elements = []
    for i in range(2 ** (depth + 1) - 1):
        is_leaf = i >= 2 ** depth - 1
        elements.append(node(f"n{i}", "LeafNode" if is_leaf else classes, label=f"node {i}"))
        if i:
            elements.append(edge(f"n{(i - 1) // 2}", f"n{i}"))
    return elements


def tree_elements(viz_tree, tmp_path, view=None, **kwargs):
    elements = viz_tree_to_cytoscape_elements(viz_tree, str(tmp_path), **kwargs)
    if view:
        for el in elements:
            el["classes"] = f"{el.get('classes', '')} {view}".strip()
    return elements


# ── Node sizes ──────────────────────────────────────────────────────────────────
def test_node_size_follows_the_stylesheet():
    # 100 x 40 plus 6px padding on both sides plus the 1.5px border
    assert node_size(node("a", "SplitNode")) == (113.5, 53.5)
    assert node_size(node("a", "LeafNode")) == (173.5, 73.5)
    assert node_size(node("a", "LeafNode minimal")) == (43.5, 28.5)
    assert node_size(node("a", "LeafNode tiny")) == (4.5, 4.5)


def test_node_size_with_data_mapper():
    assert node_size(node("a", "SplitNode data_size", width=20, height=10, label_minimal="")) == (33.5, 23.5)


def test_node_size_grows_with_a_long_label():
    narrow = node_size(node("a", "SplitNode minimal", label_minimal="x"))
    wide = node_size(node("a", "SplitNode minimal", label_minimal="a_very_long_feature_name"))
    assert wide[0] > narrow[0]


def test_state_selectors_dont_change_the_size():
    stylesheet = CYTOSCAPE_STYLESHEET + [{"selector": "node:selected", "style": {"width": "500px"}}]
    assert node_size(node("a"), stylesheet) == node_size(node("a"))


# ── Tidy tree ───────────────────────────────────────────────────────────────────
def test_children_keep_their_order():
    elements = [node("p"), node("b"), node("a"), edge("p", "b"), edge("p", "a")]
    _, layout = layout_elements(elements)
    pos = positions(layout)
    assert pos["b"][0] < pos["p"][0] < pos["a"][0]
    assert pos["a"][1] == pos["b"][1] > pos["p"][1]


def test_parent_centred_above_children():
    _, layout = layout_elements(full_binary(3))
    pos = positions(layout)
    for i in range(7):
        assert pos[f"n{i}"][0] == pytest.approx((pos[f"n{2 * i + 1}"][0] + pos[f"n{2 * i + 2}"][0]) / 2)


def test_single_child_straight_below():
    elements = [node("a", "LinearNode"), node("b", "LinearNode"), node("c", "LeafNode"), edge("a", "b"), edge("b", "c")]
    _, layout = layout_elements(elements)
    assert {x for x, _ in positions(layout).values()} == {0}


def test_no_overlaps_and_left_to_right_order_on_every_level():
    elements = full_binary(5)
    _, layout = layout_elements(elements, node_sep=7)
    assert not overlapping(elements, layout)
    pos = positions(layout)
    for depth in range(6):
        level = [f"n{i}" for i in range(2 ** depth - 1, 2 ** (depth + 1) - 1)]
        xs = [pos[n][0] for n in level]
        assert xs == sorted(xs)
        assert len({pos[n][1] for n in level}) == 1


def test_node_sep_is_the_gap_between_neighbours():
    elements = [node("p"), node("a"), node("b"), edge("p", "a"), edge("p", "b")]
    _, layout = layout_elements(elements, node_sep=25)
    pos = positions(layout)
    assert pos["b"][0] - pos["a"][0] == pytest.approx(node_size(node("a"))[0] + 25)


def test_subtrees_are_pushed_together():
    """The right subtree tucks in under the wide left one: a small leaf next to a deep subtree is placed right next to
    that subtree's root, not past its widest level."""
    elements = full_binary(3) + [node("r"), node("leaf", "LeafNode"), edge("r", "n0"), edge("r", "leaf")]
    _, layout = layout_elements(elements)
    pos = positions(layout)
    widest_left = max(pos[f"n{i}"][0] for i in range(7, 15))
    assert pos["leaf"][0] < widest_left


def test_flat_edges_dont_go_through_nodes():
    """A leaf next to a node with children far apart stays clear of the edges to those children."""
    elements = full_binary(4) + [node("r"), node("leaf", "LeafNode"), node("q"), edge("r", "leaf"), edge("r", "q"),
                                 edge("q", "n0")]
    _, layout = layout_elements(elements, node_sep=1)
    pos = positions(layout)
    box = boxes(elements, layout)
    for el in elements:
        if "source" in el["data"]:
            (x1, y1), (x2, y2) = pos[el["data"]["source"]], pos[el["data"]["target"]]
            for other, (bx1, by1, bx2, by2) in box.items():
                if other in (el["data"]["source"], el["data"]["target"]):
                    continue
                for t in (i / 50 for i in range(1, 50)):
                    x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                    assert not (bx1 < x < bx2 and by1 < y < by2), (el["data"], other)


def test_forest_roots_side_by_side():
    elements = [node("a"), node("b")]
    _, layout = layout_elements(elements)
    pos = positions(layout)
    assert pos["a"][1] == pos["b"][1] and pos["a"][0] < pos["b"][0]
    assert not overlapping(elements, layout)


def test_empty_and_single_node():
    assert layout_elements([])[1]["positions"] == {}
    assert layout_elements([node("a")])[1]["positions"] == {"a": {"x": 0, "y": 0}}


# ── Options ─────────────────────────────────────────────────────────────────────
def test_left_right_swaps_the_axes():
    elements = full_binary(2)
    _, top_bottom = layout_elements(elements)
    _, left_right = layout_elements(elements, direction="LR")
    tb, lr = positions(top_bottom), positions(left_right)
    assert lr["n1"][1] < lr["n0"][1] < lr["n2"][1]  # first child above
    assert lr["n0"][0] < lr["n1"][0] == lr["n2"][0]   # children to the right
    assert tb != lr
    assert not overlapping(elements, left_right)


def test_align_leaves_puts_leaves_on_the_lowest_level_with_bent_edges():
    elements = [node("r"), node("leaf", "LeafNode"), node("s"), node("l1", "LeafNode"), node("l2", "LeafNode"),
                edge("r", "leaf"), edge("r", "s"), edge("s", "l1"), edge("s", "l2")]
    new_elements, layout = layout_elements(elements, align_leaves=True)
    pos = positions(layout)
    assert pos["leaf"][1] == pos["l1"][1] == pos["l2"][1]
    routed = [el for el in new_elements if ROUTED_EDGE_CLASS in el.get("classes", "")]
    assert [(el["data"]["source"], el["data"]["target"]) for el in routed] == [("r", "leaf")]
    # the edge bends on the level the leaf would be on without aligning (where its sibling is), right above it
    weight, distance = (float(v) for v in (routed[0]["data"]["seg_w"], routed[0]["data"]["seg_d"]))
    (sx, sy), (tx, ty) = pos["r"], pos["leaf"]
    length = math.hypot(tx - sx, ty - sy)
    bend = (sx + weight * (tx - sx) - distance * (ty - sy) / length,
            sy + weight * (ty - sy) + distance * (tx - sx) / length)
    assert bend == pytest.approx((tx, pos["s"][1]), abs=0.05)


def test_routing_is_removed_again():
    elements = [node("r"), node("leaf", "LeafNode"), node("s"), node("l1", "LeafNode"),
                edge("r", "leaf"), edge("r", "s"), edge("s", "l1")]
    aligned, _ = layout_elements(elements, align_leaves=True)
    compact, _ = layout_elements(aligned)
    assert all(ROUTED_EDGE_CLASS not in el.get("classes", "") and "seg_w" not in el["data"] for el in compact)


def test_gaps_can_be_zero():
    elements = [node("r"), node("a"), node("b"), edge("r", "a"), edge("r", "b")]
    width, height = node_size(elements[0])
    pos = positions(layout_elements(elements, rank_sep=0, node_sep=0)[1])
    assert pos["b"][0] - pos["a"][0] == pytest.approx(width)
    assert pos["a"][1] - pos["r"][1] == pytest.approx(height)


def test_segment_data_round_trip():
    source, target, point = (0, 0), (100, 50), (20, 70)
    weights, distances = segment_data(source, target, [point])
    w, d = float(weights), float(distances)
    length = math.hypot(100, 50)
    assert (w * 100 - d * 50 / length, w * 50 + d * 100 / length) == pytest.approx(point, abs=0.01)


# ── Real trees ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("make_tree", [make_pilot_tree, make_split_tree, make_single_leaf_tree])
@pytest.mark.parametrize("view", [None, "minimal", "tiny"])
@pytest.mark.parametrize("direction", ["TB", "LR"])
def test_real_trees(make_tree, view, direction, tmp_path):
    elements = tree_elements(make_tree(), tmp_path, view)
    new_elements, layout = layout_elements(elements, direction=direction)
    assert set(layout["positions"]) == {el["data"]["id"] for el in elements if "source" not in el["data"]}
    assert not overlapping(new_elements, layout)
    assert layout["name"] == "preset"


def test_left_child_stays_left_after_collapse(tmp_path):
    """Collapsing a subtree used to flip the children of nodes with dagre."""
    viz_tree = make_pilot_tree()
    pos = positions(layout_elements(tree_elements(viz_tree, tmp_path))[1])
    assert pos["node2"][0] < pos["node3"][0]
    viz_tree.collapse(viz_tree.nodes[2])
    viz_tree = VizTree.from_dict(viz_tree.to_dict())
    elements = tree_elements(viz_tree, tmp_path)
    pos = positions(layout_elements(elements)[1])
    pcon = next(el["data"]["id"] for el in elements if el["data"].get("node_type") == "PconNode")
    left, right = [el["data"]["target"] for el in elements if el["data"].get("source") == pcon]
    assert pos[left][0] < pos[right][0]
