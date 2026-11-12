"""Layout of the tree in the cytoscape graph.

The positions of the nodes are computed here and given to cytoscape as a "preset" layout. A general graph layout
(dagre) doesn't know the tree: it may swap the left and right child of a split, which also changes when nodes are
collapsed or pruned, and places leaves and chains of linear nodes far from their parent. This is a tidy tree layout
(Reingold & Tilford): every node is centred above its children, the left child is always on the left, a single child
(a linear node's) is right below its parent, and subtrees are pushed together as close as their contours allow.

The layout works on the cytoscape elements, the child order is the order of the edges, as viz_tree_to_cytoscape_elements
adds them (left child first). The size of each node comes from the stylesheet and its classes, so the layout fits
every view (normal, minimal, tiny, node plots, data sized nodes).

Everything is computed top to bottom (x along a level, y down the levels), the "Left Right" direction swaps the axes.
"""
from __future__ import annotations

import math
import re
from typing import Any

from imote.config import CYTOSCAPE_STYLESHEET

CHAR_WIDTH = 0.6        # average width of a label character, in font sizes
LINE_HEIGHT = 1.2       # in font sizes
ANIMATION_DURATION = 400  # ms
FIT_PADDING = 30
ROUTED_EDGE_CLASS = "routed"

_DATA_MAPPER = re.compile(r"data\((\w+)\)")


# ── Node sizes from the stylesheet ─────────────────────────────────────────────
def _value(value, data: dict, default: float = 0.0) -> float:
    """Number of pixels in a style value: 12, "100px" or "data(width)"."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    mapper = _DATA_MAPPER.fullmatch(value.strip())
    if mapper:
        return float(data.get(mapper.group(1)) or default)
    return float(value.strip().removesuffix("px"))


def _selects(selector: str, classes: set[str]) -> bool:
    """Whether a node with these classes matches the selector, e.g. "node.LeafNode.plots_on, node.minimal". Selectors
    with a state (node:selected) don't change the layout, so they never match."""
    for part in selector.split(","):
        name, *needed = part.strip().split(".")
        if name == "node" and ":" not in part and classes.issuperset(needed):
            return True
    return False


def node_size(element: dict, stylesheet: list[dict] = CYTOSCAPE_STYLESHEET) -> tuple[float, float]:
    """Width and height a node takes in the graph: its shape with padding and border, or its label if that's larger.
    Like cytoscape, later rules in the stylesheet override earlier ones."""
    data = element["data"]
    classes = set(element.get("classes", "").split()) - {"highlight"}  # its thicker border shouldn't move nodes
    style: dict[str, Any] = {}
    for rule in stylesheet:
        if _selects(rule["selector"], classes):
            style.update(rule["style"])

    extra = 2 * _value(style.get("padding"), data) + _value(style.get("border-width"), data)
    width = _value(style.get("width"), data, 30) + extra
    height = _value(style.get("height"), data, 30) + extra

    label = style.get("label", "")
    mapper = _DATA_MAPPER.fullmatch(label)
    text = str(data.get(mapper.group(1)) or "") if mapper else label
    if text:
        font = _value(style.get("font-size"), data, 16)
        max_width = _value(style.get("text-max-width"), data, math.inf)
        n_lines = 0
        text_width = 0.0
        for line in text.split("\n"):
            line_width = len(line) * font * CHAR_WIDTH
            n_lines += max(1, math.ceil(line_width / max_width))
            text_width = max(text_width, min(line_width, max_width))
        width = max(width, text_width)
        height = max(height, n_lines * font * LINE_HEIGHT)
    return width, height


# ── Tidy tree ─────────────────────────────────────────────────────────────────
class TreeStructure:
    """Nodes and their ordered children, taken from the cytoscape elements."""

    def __init__(self, elements: list[dict]):
        node_ids = [el["data"]["id"] for el in elements if "source" not in el["data"]]
        self.children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        has_parent = set()
        for el in elements:
            source, target = el["data"].get("source"), el["data"].get("target")
            if source in self.children and target in self.children and target not in has_parent:
                self.children[source].append(target)
                has_parent.add(target)
        self.roots = [node_id for node_id in node_ids if node_id not in has_parent]

        # Depth first order, parents before their children
        self.order: list[str] = []
        self.depth: dict[str, int] = {}
        stack = [(root, 0) for root in reversed(self.roots)]
        while stack:
            node_id, depth = stack.pop()
            self.order.append(node_id)
            self.depth[node_id] = depth
            stack.extend((child, depth + 1) for child in reversed(self.children[node_id]))


class Contour:
    """Left and right edge of a subtree on each level below its root, relative to the x of the root."""

    def __init__(self, left: list[float], right: list[float]):
        self.left = left
        self.right = right

    def shifted(self, dx: float) -> "Contour":
        return Contour([x + dx for x in self.left], [x + dx for x in self.right])

    def gap_to(self, other: "Contour", gap: float) -> float:
        """How far right other must go to be at least gap away from self on every level."""
        return max(self.right[i] - other.left[i] + gap for i in range(min(len(self.left), len(other.left))))

    def merged(self, other: "Contour") -> "Contour":
        n = max(len(self.left), len(other.left))
        pick = lambda a, b, f: [f(a[i], b[i]) if i < len(a) and i < len(b) else (a[i] if i < len(a) else b[i])
                                for i in range(n)]
        return Contour(pick(self.left, other.left, min), pick(self.right, other.right, max))


def _side_by_side(contours: list[Contour], gap: float) -> tuple[list[float], Contour]:
    """x of each subtree, left to right as close as they fit, centred around 0. Returns the xs and the joint contour."""
    xs = [0.0]
    joint = contours[0]
    for contour in contours[1:]:
        x = joint.gap_to(contour, gap)
        xs.append(x)
        joint = joint.merged(contour.shifted(x))
    middle = (xs[0] + xs[-1]) / 2
    return [x - middle for x in xs], joint.shifted(-middle)


def tidy_tree(tree: TreeStructure, sizes: dict[str, tuple[float, float]], node_gap: float, level_gap: float,
              align_leaves: bool = False, leaf_columns: bool = False
              ) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str], list[tuple[float, float]]]]:
    """Positions of the nodes (top to bottom) and the bends of the edges that need them.

    Args:
        tree: The nodes and their children.
        sizes: Width and height of each node.
        node_gap: Horizontal gap between nodes on the same level.
        level_gap: Vertical gap between two levels.
        align_leaves: Put all nodes without children on the lowest level. Their edge goes to where the leaf would be
            otherwise and then straight down, the layout keeps that vertical part free.
        leaf_columns: Leaves stay right below their parent, but nothing goes below them: every leaf has a column to
            itself down to the lowest level, so the tree is as wide as with align_leaves.
    """
    max_depth = max(tree.depth.values(), default=0)
    level = {node_id: max_depth if align_leaves and not tree.children[node_id] else tree.depth[node_id]
             for node_id in tree.order}

    level_heights = [0.0] * (max_depth + 1)
    for node_id in tree.order:
        level_heights[level[node_id]] = max(level_heights[level[node_id]], sizes[node_id][1])
    level_y = [0.0]
    for i in range(1, max_depth + 1):
        level_y.append(level_y[-1] + level_heights[i - 1] / 2 + level_gap + level_heights[i] / 2)

    contours: dict[str, Contour] = {}
    offset: dict[str, float] = {}  # x of a node relative to its parent
    for node_id in reversed(tree.order):
        half_width = sizes[node_id][0] / 2
        children = tree.children[node_id]
        if not children and leaf_columns:
            n_levels = max_depth - level[node_id] + 1
            contours[node_id] = Contour([-half_width] * n_levels, [half_width] * n_levels)
            continue
        if not children:
            n_above = level[node_id] - tree.depth[node_id]  # levels the vertical part of the edge goes through
            room = node_gap / 2
            contours[node_id] = Contour([-room] * n_above + [-half_width], [room] * n_above + [half_width])
            continue
        xs, joint = _side_by_side([contours[child] for child in children], node_gap)
        offset.update(zip(children, xs))
        # A flat edge to a child far away would go through the nodes next to this one, so they keep clear of where
        # the edges leave this level
        d = level[node_id]
        edge_spread = level_heights[d] / 2 / max(level_y[d + 1] - level_y[d], 1e-9)
        contours[node_id] = Contour([min(-half_width, edge_spread * xs[0])] + joint.left,
                                    [max(half_width, edge_spread * xs[-1])] + joint.right)

    x: dict[str, float] = {}
    if tree.roots:
        x.update(zip(tree.roots, _side_by_side([contours[root] for root in tree.roots], node_gap)[0]))
    for node_id in tree.order:
        for child in tree.children[node_id]:
            x[child] = x[node_id] + offset[child]

    positions = {node_id: (x[node_id], level_y[level[node_id]]) for node_id in tree.order}
    bends = {(node_id, child): [(x[child], level_y[tree.depth[child]])]
             for node_id in tree.order for child in tree.children[node_id] if level[child] > tree.depth[child]}
    return positions, bends


# ── Cytoscape elements and layout ───────────────────────────────────────────────
def segment_data(source: tuple[float, float], target: tuple[float, float],
                 points: list[tuple[float, float]]) -> tuple[str, str]:
    """Cytoscape "segments" edge through points: their weight along the line from source to target and their distance
    from that line (positive on the right of the line, looking from the source)."""
    dx, dy = target[0] - source[0], target[1] - source[1]
    length_sq = dx * dx + dy * dy or 1e-9
    length = math.sqrt(length_sq)
    weights, distances = [], []
    for px, py in points:
        rx, ry = px - source[0], py - source[1]
        weights.append((rx * dx + ry * dy) / length_sq)
        distances.append((dx * ry - dy * rx) / length)
    return " ".join(f"{w:.4f}" for w in weights), " ".join(f"{d:.2f}" for d in distances)


def _n_samples(element: dict) -> float | None:
    n_samples = element["data"].get("n_samples")
    return n_samples.item() if hasattr(n_samples, "item") else n_samples  # numpy numbers to plain ones for JSON


def _keep_positions(tree: TreeStructure, sizes: dict[str, tuple[float, float]], positions: dict, bends: dict,
                    nodes: dict[str, list], previous: dict, swap: bool) -> tuple[dict, dict] | None:
    """Positions that keep the nodes of the previous layout in place, None if that isn't possible.

    A node stays where it was if it's the same node: the same parent, place among its siblings, number of samples, size
    and being a leaf or not. A new node takes the place of the sibling it replaces (a pruned subtree becomes a leaf),
    otherwise it goes where the tidy tree puts it relative to its parent (the node below a collapsed node, the children
    of an expanded node). If nodes would overlap or children would swap, None: the whole tree is laid out again.
    """
    old_nodes = previous.get("_nodes") or {}
    old = {node_id: (p["y"], p["x"]) if swap else (p["x"], p["y"])
           for node_id, p in (previous.get("positions") or {}).items()}
    kept = {node_id for node_id in tree.order if node_id in old and old_nodes.get(node_id) == nodes[node_id]}
    if not kept:
        return None
    old_children: dict[str, dict[int, str]] = {}
    for node_id, (parent, index, *_) in old_nodes.items():
        old_children.setdefault(parent, {})[index] = node_id

    new: dict[str, tuple[float, float]] = {}
    for node_id in tree.order:  # parents first
        parent, index = nodes[node_id][:2]
        if node_id in kept:
            new[node_id] = old[node_id]
            continue
        if parent is None:
            return None  # a new root
        dy = positions[node_id][1] - positions[parent][1]
        replaced = old_children.get(parent, {}).get(index)
        if len(old_children.get(parent, {})) == len(tree.children[parent]) and replaced in old:
            new[node_id] = (old[replaced][0], new[parent][1] + dy)
        else:
            new[node_id] = (new[parent][0] + positions[node_id][0] - positions[parent][0], new[parent][1] + dy)

    for node_id in tree.order:
        xs = [new[child][0] for child in tree.children[node_id]]
        if xs != sorted(xs) or any(new[child][1] <= new[node_id][1] for child in tree.children[node_id]):
            return None
    boxes = sorted((x - sizes[n][0] / 2, x + sizes[n][0] / 2, y - sizes[n][1] / 2, y + sizes[n][1] / 2)
                   for n, (x, y) in new.items())
    for i, (left, right, top, bottom) in enumerate(boxes):
        for other in boxes[i + 1:]:
            if other[0] >= right - 0.01:
                break
            if other[2] < bottom - 0.01 and top < other[3] - 0.01:
                return None

    new_bends = {(parent, child): [(new[child][0], new[parent][1] + y - positions[parent][1]) for _, y in points]
                 for (parent, child), points in bends.items()}
    return new, new_bends


def layout_elements(elements: list[dict], direction: str = "TB", align_leaves: bool = False, leaf_columns: bool = False,
                    rank_sep: float = 10, node_sep: float = 10, previous: dict | None = None,
                    stylesheet: list[dict] = CYTOSCAPE_STYLESHEET) -> tuple[list[dict], dict]:
    """Lays out the tree in the elements.

    Args:
        elements: Cytoscape elements of a tree (or a forest), edges in child order.
        direction: "TB" (top to bottom) or "LR" (left to right).
        align_leaves: Put all leaves on the lowest level.
        leaf_columns: Leaves right below their parent, each in a column of its own (as wide as align_leaves).
        rank_sep: Gap between the levels.
        node_sep: Gap between nodes next to each other.
        previous: The previous layout, its nodes stay where they are if the tree only changed a little (collapsed,
            pruned), see _keep_positions. None lays out the whole tree.
        stylesheet: Gives the size of the nodes.

    Returns:
        The elements, edges with a bend get the class ROUTED_EDGE_CLASS and their segments in their data, and a
        cytoscape "preset" layout with the positions.
    """
    tree = TreeStructure(elements)
    node_elements = {el["data"]["id"]: el for el in elements if "source" not in el["data"]}
    sizes = {node_id: node_size(el, stylesheet) for node_id, el in node_elements.items()}
    if direction == "LR":
        sizes = {node_id: (h, w) for node_id, (w, h) in sizes.items()}

    positions, bends = tidy_tree(tree, sizes, node_sep, rank_sep, align_leaves, leaf_columns)
    parents = {child: node_id for node_id in tree.order for child in tree.children[node_id]}
    nodes = {node_id: [parents.get(node_id), tree.children[parents[node_id]].index(node_id) if node_id in parents else 0,
                       _n_samples(node_elements[node_id]), not tree.children[node_id],
                       round(sizes[node_id][0], 1), round(sizes[node_id][1], 1)] for node_id in tree.order}
    kept = _keep_positions(tree, sizes, positions, bends, nodes, previous, direction == "LR") if previous else None
    if kept:
        positions, bends = kept
    if direction == "LR":
        positions = {node_id: (y, x) for node_id, (x, y) in positions.items()}
        bends = {edge: [(y, x) for x, y in points] for edge, points in bends.items()}

    new_elements = []
    for el in elements:
        data = el["data"]
        if "source" in data:
            classes = [c for c in el.get("classes", "").split() if c != ROUTED_EDGE_CLASS]
            # A fixed id, otherwise the graph replaces all edges on every update, which runs the layout again. The graph
            # adds the new data to the old data of an edge, so an edge without a label gets an empty one.
            data = {"id": f"{data['source']}->{data['target']}", "label": "",
                    **{k: v for k, v in data.items() if k not in ("seg_w", "seg_d")}}
            points = bends.get((data["source"], data["target"]))
            if points:
                data["seg_w"], data["seg_d"] = segment_data(positions[data["source"]], positions[data["target"]],
                                                            points)
                classes.append(ROUTED_EDGE_CLASS)
            el = {**el, "data": data, "classes": " ".join(classes)}
        new_elements.append(el)

    layout = {
        "name": "preset",
        "positions": {node_id: {"x": round(x, 2), "y": round(y, 2)} for node_id, (x, y) in positions.items()},
        "fit": not kept,  # kept nodes keep the zoom as well
        "padding": FIT_PADDING,
        "animate": True,
        "animationDuration": ANIMATION_DURATION,
        "_nodes": nodes,  # cytoscape ignores this, the next layout uses it to recognise the nodes
    }
    return new_elements, layout
