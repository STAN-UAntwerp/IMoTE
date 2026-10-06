# IMoTE: Interactive MOdel Tree Explorer

**IMoTE** (*Interactive MOdel Tree Explorer*) is a Python application for visualizing and exploring **linear model trees** with dash cytoscape. The application provides an interactive interface to inspect tree structures, understand individual node models, and analyze how predictions are made.

The application currently supports two linear model tree algorithms:

* **PILOT** (*PIecewise Linear Organic Tree*) — a fast and interpretable linear model tree algorithm for regression, using the [fast-model-trees](https://github.com/STAN-UAntwerp/fast-model-trees) implementation.
* **M5** — a classic model tree algorithm that combines decision tree splits with linear regression models in the leaves.

The main goal of this project is to make linear model trees easier to understand and explain through interactive visualization.

## Documentation

Full documentation, including explanations of the interface and visualization options, is available [here](https://flordebois.github.io/model-tree-cytoscape/home/).

## Installation

Clone the repository:

```bash
git clone https://github.com/flordebois/model-tree-cytoscape
cd model-tree-cytoscape
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install IMoTE and its dependencies (`-e` installs it in editable mode, so code changes take effect immediately):

```bash
pip install -e .
```

## Running the application

Start the application with:

```bash
imote
```

For development, `python -m imote.app` starts it in Dash debug mode, which reloads on code changes.

The application will start a local web server. Open the provided URL in your browser to access the visualization interface.


