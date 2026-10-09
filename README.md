# IMoTE: Interactive MOdel Tree Explorer

[![PyPI](https://img.shields.io/pypi/v/imote)](https://pypi.org/project/imote/)
[![Python](https://img.shields.io/pypi/pyversions/imote)](https://pypi.org/project/imote/)
[![Documentation](https://img.shields.io/badge/docs-stan--uantwerp.github.io%2FIMoTE-1f77b4)](https://stan-uantwerp.github.io/IMoTE/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/STAN-UAntwerp/IMoTE/blob/main/LICENSE)

**IMoTE** is an interactive application to fit, visualise and explore **linear model trees** for regression on
tabular data. It shows a fitted tree as an interactive graph, lets you look at the linear model and the data behind
every node, and traces a single data point through the tree to explain its prediction.

![IMoTE: the tree graph with the interaction cards on the right](https://raw.githubusercontent.com/STAN-UAntwerp/IMoTE/main/docs/assets/screenshots/app-overview.png)

## Why IMoTE?

A linear model tree combines the rule based structure of a decision tree with linear models in its leaves. That usually makes it more accurate than a classic regression tree of the same
depth. In practice, these trees are often hard to read from code or a text dump. IMoTE
makes them easy to explore, explain and present.

## Features

**Fit or load a tree**

- Fit **PILOT** [[1]](#references) or **M5** [[2]](#references) trees on built-in
  [PMLB](https://epistasislab.github.io/pmlb/) datasets or on your own CSV file.
- Save the tree you are looking as an SVG image or to load it again later.
- Load trees fitted with other tools through an **adapter**. An adapter for trees from the R package
  [partykit](https://cran.r-project.org/package=partykit) is included as an example.

**Edit and explore the tree**

- Pan, zoom and click through the tree as an interactive graph.
- Collapse and expand branches, collapse the whole tree to a given level, show a single branch as a subtree, or prune the tree (PILOT only).
- Switch between normal, minimal and tiny nodes or color nodes by the feature they use.
- Show the data flow: edge width or node size scale with the number of training samples that pass through them.

**Inspect nodes and predictions**

- Click a node to see its type, split and metrics (number of samples, RSS, MAE, ...).
- See the plot of any node: the data and the fit of a split, or the prediction plot of a leaf that shows how each
  feature contributes to its predictions.
- Turn on **Show all node plots** to replace every node of the graph by its plot. 
- Highlight the path of a data point through the tree, with its prediction.


## Installation

IMoTE needs Python 3.10 or newer.

```bash
pip install imote
```

Start the application with:
```bash
imote
```

Then open the address shown in the terminal (by default <http://127.0.0.1:8050>). The app starts with an example
PILOT tree. Click a node to inspect it, or open the **New Tree** card on the right to fit a tree on another dataset.

IMoTE stores saved trees, uploaded datasets and generated plots in `~/.imote`. Set the `IMOTE_HOME` environment
variable to use another folder.

## Documentation

The full documentation is at **<https://stan-uantwerp.github.io/IMoTE/>**, it includes:

- the [User Guide](https://stan-uantwerp.github.io/IMoTE/user-guide/interface-overview/), with a page for every part of
  the interface,
- the [Gallery](https://stan-uantwerp.github.io/IMoTE/gallery/), with screenshots of what IMoTE can show,
- a guide on how to [write your own adapter](https://stan-uantwerp.github.io/IMoTE/adapter/adapter_overview/) for another type
  of linear model tree,
- an [API reference](https://stan-uantwerp.github.io/IMoTE/reference/reference_overview/) of the tree and node classes used for visualisation.


<!--
## Citing IMoTE

If you use IMoTE in your research, please cite the paper that introduces it:

> *The paper describing IMoTE is in preparation. Its reference will be added here once it is published.*
Replace the note above with the reference once the paper is published, e.g.:

> Debois, F., Servotte, T., Raymaekers, J., & Verdonck, T. (YEAR). TITLE. *JOURNAL*. https://doi.org/...

```bibtex
@article{imote,
  title   = {TITLE},
  author  = {Debois, Flor and Servotte, Thomas and Raymaekers, Jakob and Verdonck, Tim},
  journal = {JOURNAL},
  year    = {YEAR},
  doi     = {DOI}
}
```
-->

## Development

```bash
git clone https://github.com/STAN-UAntwerp/IMoTE
cd IMoTE
pip install -e ".[dev]"   # IMoTE in editable mode, plus pytest, build and twine
python -m imote.app       # start the app in Dash debug mode, it reloads on code changes
```

To work on the documentation: `pip install -e ".[docs]"`, then `mkdocs serve`.

Questions and ideas are welcome in [Discussions](https://github.com/STAN-UAntwerp/IMoTE/discussions), bugs can be
reported as an [issue](https://github.com/STAN-UAntwerp/IMoTE/issues).

### References

<sub>[1] Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT.
*Machine Learning*, 113(9), 6561-6610. <https://doi.org/10.1007/s10994-024-06590-3> (implementation:
[fast-model-trees](https://github.com/STAN-UAntwerp/fast-model-trees))</sub>

<sub>[2] Quinlan, J. R. (1992). Learning with continuous classes. In *5th Australian Joint Conference on Artificial
Intelligence* (Vol. 92, pp. 343-348). (implementation: [m5py](https://pypi.org/project/m5py/))</sub>
