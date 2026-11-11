"""Searching and downloading datasets from PMLB and the UCI Machine Learning Repository."""
from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from imote.benchmark_info import PMLB_DATASETS_CAT_IDS
from imote.dataset.metadata import Hints

logger = logging.getLogger(__name__)

PMLB_SUMMARY_URL = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"
UCI_LIST_URL = "https://archive.ics.uci.edu/api/datasets/list"
CACHE_MAX_AGE = 7 * 24 * 3600  # seconds before the cached lists of datasets are downloaded again
MAX_RESULTS = 50


@dataclass
class SearchResult:
    source: str  # "pmlb" or "uci"
    key: str  # PMLB name or UCI id, as a string
    name: str
    details: str = ""


# ----------------------------------------------------------------------
# PMLB
# ----------------------------------------------------------------------

def pmlb_info(name: str, summary: pd.DataFrame | None) -> tuple[str | None, int | None]:
    """Task and number of classes of a PMLB dataset (None when unknown). The summary stats are used when available,
    the lists in the pmlb package have a few datasets in the wrong list (e.g. titanic is listed as regression)."""
    if summary is not None and name in summary.index and summary.loc[name, "task"] in ("classification",
                                                                                        "regression"):
        row = summary.loc[name]
        return str(row["task"]), int(row["n_classes"]) if row["task"] == "classification" else None
    import pmlb
    if name in pmlb.classification_dataset_names:
        return "classification", None
    if name in pmlb.regression_dataset_names:
        return "regression", None
    return None, None


_summary_memory: dict[str, tuple[float, pd.DataFrame]] = {}


def pmlb_summary(cache_dir: Path) -> pd.DataFrame | None:
    """Number of rows, features and classes and the task of every PMLB dataset."""
    path = Path(cache_dir) / "pmlb_summary_stats.tsv"
    stale = not path.exists() or time.time() - path.stat().st_mtime > CACHE_MAX_AGE
    if stale:
        try:
            df = pd.read_csv(PMLB_SUMMARY_URL, sep="\t")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, sep="\t", index=False)
        except Exception as e:
            raise ConnectionError(f"Could not reach the PMLB repository ({e}).") from e
    mtime = path.stat().st_mtime
    cached = _summary_memory.get(str(path))
    if cached is None or cached[0] != mtime:
        cached = (mtime, pd.read_csv(path, sep="\t").set_index("dataset"))
        _summary_memory[str(path)] = cached
    return cached[1]


def search_pmlb(query: str, task: str | None, cache_dir: Path) -> list[SearchResult]:
    import pmlb
    summary = pmlb_summary(cache_dir)
    words = (query or "").lower().split()
    results = []
    for name in sorted(pmlb.dataset_names):
        if not all(word in name.lower() for word in words):
            continue
        name_task, n_classes = pmlb_info(name, summary)
        if task and name_task != task:
            continue
        details = name_task or ""
        if summary is not None and name in summary.index:
            row = summary.loc[name]
            details = f"{details}, {int(row['n_instances'])} rows, {int(row['n_features'])} features"
            if n_classes:
                details += f", {n_classes} classes"
        results.append(SearchResult("pmlb", name, name, details))
        if len(results) >= MAX_RESULTS:
            break
    return results


def download_pmlb(name: str, cache_dir: Path | None = None) -> tuple[pd.DataFrame, Hints]:
    from pmlb import fetch_data
    try:
        df = fetch_data(name)
    except (OSError, ValueError) as e:
        raise ConnectionError(f"Could not download PMLB dataset {name}: ({e})") from e
    return df, pmlb_hints(name, list(df.columns), cache_dir)


def pmlb_hints(name: str, columns: list[str] | None = None, cache_dir: Path | None = None) -> Hints:
    task = pmlb_info(name, pmlb_summary(cache_dir) if cache_dir is not None else None)[0]
    hints = Hints(response="target", task=task, source="PMLB")
    cat_ids = PMLB_DATASETS_CAT_IDS.get(name)
    if cat_ids is not None and columns is not None:
        features = [col for col in columns if col != "target"]
        hints.categorical = [features[i] for i in cat_ids if 0 <= i < len(features)]
        hints.categorical_source = "PMLB list in benchmark_info.py"
    return hints


# ----------------------------------------------------------------------
# UCI
# ----------------------------------------------------------------------

def _uci_list(cache_dir: Path) -> list[dict]:
    """All UCI datasets that can be downloaded with ucimlrepo, as dicts with an id and a name."""
    path = cache_dir / "uci_datasets.json"
    if not path.exists() or time.time() - path.stat().st_mtime > CACHE_MAX_AGE:
        url = UCI_LIST_URL + "?" + urllib.parse.urlencode({"filter": "python"})
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        try:
            with urllib.request.urlopen(url, context=context, timeout=20) as response:
                data = json.load(response)["data"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        except Exception as e:
            raise ConnectionError(f"Could not reach the UCI repository ({e}).") from e
    return json.loads(path.read_text())


def search_uci(query: str, cache_dir: Path) -> list[SearchResult]:
    words = (query or "").lower().split()
    results = []
    for item in sorted(_uci_list(cache_dir), key=lambda d: d["name"].lower()):
        if not (all(word in item["name"].lower() for word in words) or query.strip() == str(item["id"])):
            continue
        results.append(SearchResult("uci", str(item["id"]), item["name"], f"id {item['id']}"))
        if len(results) >= MAX_RESULTS:
            break
    return results


def download_uci(uci_id: int | str) -> tuple[pd.DataFrame, Hints, str, dict]:
    """Returns the data, hints from the UCI metadata, the dataset name and the metadata to save as JSON
    (read back with uci_hints_from_file)."""
    from ucimlrepo import fetch_ucirepo
    try:
        repo = fetch_ucirepo(id=int(uci_id))
    except Exception as e:
        raise ConnectionError(f"Could not download UCI dataset {uci_id}: ({e})") from e
    df = repo.data.original
    if df is None:
        df = pd.concat([d for d in (repo.data.ids, repo.data.features, repo.data.targets) if d is not None], axis=1)
    df.columns = [str(c) for c in df.columns]
    variables = repo.variables if repo.variables is not None else pd.DataFrame()
    keep = [c for c in ("name", "role", "type") if c in variables.columns]
    info = {
        "target_col": [str(t) for t in (repo.metadata.get("target_col") or [])],
        "tasks": [str(t) for t in (repo.metadata.get("tasks") or [])],
        "variables": variables[keep].astype(str).to_dict("records") if keep else [],
    }
    name = repo.metadata.get("name") or f"uci_{uci_id}"
    return df, uci_hints(info, pd.DataFrame(info["variables"]), list(df.columns)), name, info


def uci_hints(metadata: dict, variables: pd.DataFrame, columns: list[str]) -> Hints:
    hints = Hints(source="UCI metadata")
    targets = [str(t) for t in (metadata.get("target_col") or []) if str(t) in columns]
    if targets:
        hints.response = targets[0]
    tasks = [str(t).lower() for t in (metadata.get("tasks") or [])]
    if "regression" in tasks and "classification" not in tasks:
        hints.task = "regression"
    elif "classification" in tasks and "regression" not in tasks:
        hints.task = "classification"
    if not variables.empty and {"name", "role", "type"} <= set(variables.columns):
        roles = dict(zip(variables["name"].astype(str), variables["role"].astype(str)))
        types = dict(zip(variables["name"].astype(str), variables["type"].astype(str)))
        hints.ignore = [col for col in columns if roles.get(col) == "ID"]
        hints.ignore += [t for t in targets[1:] if t not in hints.ignore]  # other targets would leak the response
        hints.categorical = [col for col in columns if types.get(col) in ("Categorical", "Binary")]
    return hints


def uci_hints_from_file(path: Path, columns: list[str]) -> Hints:
    try:
        info = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return Hints(source="UCI metadata")
    return uci_hints(info, pd.DataFrame(info.get("variables", [])), columns)
