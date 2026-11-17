# Reference

This section documents the main internal components of the application that are relevant for extending it, in particular for implementing new adapters or node types.

The documentation focuses on the parts needed to understand and work with the internal `VizTree` representation. It does not provide complete documentation of the entire codebase.

## Contents

* [VizTree](viz_tree.md) — Internal representation of a linear model tree.
* **Adapters**
    * [BaseAdapter](adapters/base_adapter.md) — Interface for implementing new adapters.
    * [PartyKitAdapter](adapters/partykit_adapter.md) — Example of an adapter implementation.
* **Nodes**
    * [BaseNode](nodes/base_node.md) — Base class for all tree nodes.
    * [NodeModel](nodes/node_model.md) — Models used within nodes.
    * [InternalNode](nodes/internal_node.md) — Base class for internal nodes.
    * [SplitNode](nodes/split_node.md) — Nodes that split observations.
    * [LeafNode](nodes/leaf_node.md) — Terminal nodes containing a model.
    * [NoneNode](nodes/none_node.md) — Node used when no specific node type is required.
