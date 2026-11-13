# Example: the partykit adapter

`partykit` is an R package for fitting model-based recursive partitioning trees [[1]](#1). Its `lmtree` function
implements the MOB model tree algorithm [[2]](#2): it recursively splits the data on the variable for which the
parameters of a linear model are most unstable, and fits a linear regression in every leaf. This page walks through
the adapter that loads these trees into IMoTE. It shows the complete path for a model fitted outside Python:

1. fit the tree in R and export it to a file IMoTE can read,
2. write an adapter that reads the file and converts it into a `VizTree`,
3. load the tree in the application.

See the [adapter overview](adapter_overview.md) for the general interface and the `VizTree` structure that the adapter
builds.

## 1. Fit and export the model in R

`partykit` has no export format that is easy to read from Python, so the relevant information of every node is
collected in a list and written to a JSON file. The example below fits a tree on the `547_no2` dataset from PMLB,
using all variables both as regressors in the leaf models and as candidate split variables (`target ~ . | .`).

```r
library(partykit)
library(jsonlite)

# Load PMLB no2 dataset
library(pmlbr)
no2 <- fetch_data("547_no2")

# Fit linear model tree to all the data
tree <- lmtree(target ~ . | ., data = no2)

# Collect the information of every node
nodes <- lapply(nodeids(tree), function(id) {
  nd <- tree[id]                       # subtree rooted at node id
  node_obj <- node_party(nd)
  is_terminal <- is.terminal(node_obj)

  node_info <- list(
    id = id - 1,                       # zero-based, to index the node list from Python
    is_terminal = is_terminal,
    n = info_node(node_obj)$nobs
  )

  if (is_terminal) {
    # coefficients of the linear model in this leaf
    node_info$coefficients <- as.list(coef(tree, node = id))
  } else {
    split <- split_node(node_obj)
    varname <- names(data_party(tree))[split$varid]
    node_info$split_var <- varname
    node_info$breaks <- split$breaks   # threshold of a numerical split
    node_info$index <- split$index     # child (1 or 2) of every level for a categorical split
    node_info$levels <- levels(tree$data[[varname]])
    # ids in the subtree start at 1 for node id, convert them to zero-based ids in the full tree
    node_info$kids <- id - 1 + c(kids_node(node_obj)[[1]]$id - 1, kids_node(node_obj)[[2]]$id - 1)
  }
  node_info
})

tree_json <- list(
  names = names(tree$data),
  nodes = nodes
)

write_json(tree_json, "path/to/tree_547_no2.json", auto_unbox = TRUE, null = "null")
```

The resulting JSON looks like this:

```json
{
  "names": ["target", "x1", "x2", "..."],
  "nodes": [
    {
      "id": 0,
      "is_terminal": false,
      "n": 500,
      "split_var": "x1",
      "breaks": 0.5,
      "kids": [1, 2]
    },
    {
      "id": 1,
      "is_terminal": true,
      "n": 230,
      "coefficients": {"(Intercept)": 1.2, "x1": 0.1, "x2": 0.3}
    },
    "..."
  ]
}
```

- `names` contains the column names of the data, with the response first, followed by the features in the same
  column order as the dataset used in IMoTE.
- `nodes` is a flat list in which node `i` is stored at position `i`, with the root at position `0`. Every node is
  either:
    - an **internal node**, with the split variable `split_var`, either `breaks` (the threshold of a numerical split)
      or `index` and `levels` (the child each category is sent to, for a categorical split), and the positions of its
      two children in `kids`,
    - a **leaf** (`is_terminal: true`), with the `coefficients` of its linear model, including the `(Intercept)`.

## 2. Define the adapter

The adapter is registered under the name `Partykit`. `load_model` simply parses the JSON file:

```python
@register_adapter('Partykit')
class PartyKitAdapter(BaseAdapter):

    @staticmethod
    def load_model(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No such file path to load model: {model_path}")
        with open(model_path) as f:
            return json.load(f)
```

`build_root_node` starts at the root node (position `0` in the node list) with all training samples, and drops the
response from `names` to get the feature names in column order:

```python
    @staticmethod
    def build_root_node(X_train, y_train, model):
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        node_list = model["nodes"]
        feature_names = model["names"][1:]
        return PartyKitAdapter._build_recursive(
            node_list, feature_names, node_list[0], X_train, y_train, root_indices
        )
```

The recursive function converts one node at a time. Since `lmtree` only fits models in the leaves, the `y_res` of
every node are simply the `y` values of the samples that reach it.

**Leaves** become a `LeafNode` with a `LinearNodeModel`. The coefficients are looked up by feature name and put in
column order, with `0.0` for features that are not in the leaf model:

```python
        current_y_res = y_train[current_indices]

        if node["is_terminal"]:
            coefficients = np.array([node["coefficients"].get(name, 0.0) for name in feature_names])
            node_model = LinearNodeModel(coefficients, node["coefficients"]["(Intercept)"])
            return LeafNode(indices=current_indices, y_res=current_y_res, node_model=node_model)
```

**Internal nodes** become a `SplitNode` for a numerical split or a `SplitCNode` for a categorical split. The split
variable is converted to its column index `pivot_idx`, the samples are divided over the two children with the
[helpers](adapter_overview.md#helpers), and both children are built recursively. `partykit` sends observations with
`x <= breaks` to the first child, which matches the convention of `SplitNode`. For a categorical split, the categories
whose `index` is `1` go to the left child. No model is applied on either side of the split, so both side models are a
`NoneNodeModel`:

```python
        pivot_idx = feature_names.index(node["split_var"])
        left_node = node_list[node["kids"][0]]
        right_node = node_list[node["kids"][1]]

        if "breaks" in node:  # numerical split
            pivot_value = node["breaks"]
            left_indices, right_indices = split_by_threshold(X_train, current_indices, pivot_idx, pivot_value)
            node_cls = SplitNode
        else:                 # categorical split
            pivot_value = [lvl for lvl, idx in zip(node["levels"], node["index"]) if idx == 1]
            left_indices, right_indices = split_by_categories(X_train, current_indices, pivot_idx, pivot_value)
            node_cls = SplitCNode

        return node_cls(
            indices=current_indices,
            y_res=current_y_res,
            pivot_idx=pivot_idx,
            pivot_value=pivot_value,
            left_child=PartyKitAdapter._build_recursive(
                node_list, feature_names, left_node, X_train, y_train, left_indices),
            right_child=PartyKitAdapter._build_recursive(
                node_list, feature_names, right_node, X_train, y_train, right_indices),
            left_model=NoneNodeModel(),
            right_model=NoneNodeModel(),
        )
```

This is a condensed version of the code, see [`PartyKitAdapter`](../reference/adapters/partykit_adapter.md) for the
full implementation.

The adapter doesn't override `predict`: the R model is not available from Python, so `VizTree` computes the
predictions by routing every observation down the converted tree to the linear model of its leaf.

## 3. Load the tree in IMoTE

In the [New Tree](../user-guide/c-new-tree.md) card, open **Load a tree with an adapter**, select the `Partykit`
adapter and the `547_no2` dataset, enter the path to `tree_547_no2.json` and click **Load with Adapter**. The dataset
must be the one the tree was fitted on, with the features in the same order as in `names`. The tree can then be
explored, highlighted, collapsed and saved like any tree fitted in the application.

The same tree can also be loaded in Python:

```python
from imote import PartyKitAdapter, VizTree

model = PartyKitAdapter.load_model("path/to/tree_547_no2.json")
viz_tree = VizTree.from_model(PartyKitAdapter, X_train, y_train, model)
```

## References
<a id="1">[1]</a>
Hothorn, T., & Zeileis, A. (2015). partykit: A modular toolkit for recursive partytioning in R. The Journal of Machine Learning Research, 16(1), 3905-3909.

<a id="2">[2]</a>
Zeileis, A., Hothorn, T., & Hornik, K. (2008). Model-based recursive partitioning. Journal of Computational and Graphical Statistics, 17(2), 492-514.
