# IMoTE: Interactive MOdel Tree Explorer

**IMoTE** is an interactive app to fit, visualise and explore **linear model trees** for regression.
Inspect the tree as an interactive graph, see the linear model and data behind every node, and trace a single
data point through the tree to understand its prediction.

![IMoTE: the tree graph with the interaction cards on the right](https://flordebois.github.io/model-tree-cytoscape/assets/screenshots/app-overview.png)

## Overview

IMoTE supports two linear model tree algorithms out of the box:

- **PILOT** (*PIecewise Linear Organic Tree*): a fast linear model tree algorithm, using the
  [fast-model-trees](https://github.com/STAN-UAntwerp/fast-model-trees) implementation.
- **M5**: the classic model tree algorithm, using [m5py](https://pypi.org/project/m5py/).

Trees fitted with other tools can be loaded by writing an **adapter**. A partykit (R) adapter is included as an example.

## Installation

```bash
pip install imote
```

IMoTE needs Python 3.10 or newer.

## Quick start

### Start the app

```bash
imote
```

Then open the address shown in the terminal (by default <http://127.0.0.1:8050>). From the **New Tree** card you can
fit PILOT or M5 on a built-in dataset or on your own CSV file.

IMoTE stores saved trees, uploaded datasets and generated plots in `~/.imote`. Set the `IMOTE_HOME` environment
variable to use another folder.

## Documentation

The full documentation, with a guide to every part of the interface and how to write your own adapter, is at
<https://flordebois.github.io/model-tree-cytoscape/>.

## Papers

- **PILOT**: Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT.
  *Machine Learning*, 113(9), 6561-6610. <https://doi.org/10.1007/s10994-024-06590-3>
- **M5**: Quinlan, J. R. (1992). Learning with continuous classes. In *5th Australian Joint Conference on Artificial
  Intelligence* (Vol. 92, pp. 343-348).

## Development

```bash
git clone https://github.com/flordebois/model-tree-cytoscape
cd model-tree-cytoscape
pip install -e ".[dev]"   # IMoTE in editable mode, plus pytest, build and twine
python -m imote.app       # start the app in Dash debug mode, it reloads on code changes
```

To work on the documentation: `pip install -e ".[docs]"`, then `mkdocs serve`.

## License

MIT, see [LICENSE](https://github.com/flordebois/model-tree-cytoscape/blob/main/LICENSE).
