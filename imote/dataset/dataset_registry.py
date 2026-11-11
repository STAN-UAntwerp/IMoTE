# The datasets on disk: every dataset is a folder in DIR_DATASETS with its data file and a ``dataset.toml``.
from __future__ import annotations

import functools
import json
import logging
import re
import shutil
from pathlib import Path

import pandas as pd
from tomlkit.toml_document import TOMLDocument

from imote import config
from imote.dataset import sources
from imote.dataset.dataset import Dataset
from imote.dataset.metadata import (
    META_FILE_NAME, complete_meta, new_meta, prepare, read_meta, task_kind, write_meta,
)
from imote.dataset.readers import guess_read_options, read_table

logger = logging.getLogger(__name__)

SOURCE_LABELS = {"pmlb": "PMLB", "uci": "UCI", "upload": "Uploaded"}
DOWNLOADED_FILE = "data.tsv.gz"
DOWNLOADED_READ_OPTIONS = {"sep": "\t", "header": True}
UCI_INFO_FILE = "uci_metadata.json"


def root() -> Path:
    return Path(config.DIR_DATASETS)


def cache_dir() -> Path:
    return root() / ".cache"


def dataset_dir(dataset_id: str) -> Path:
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or dataset_id.startswith("."):
        raise ValueError(f"Invalid dataset id {dataset_id!r}.")
    return root() / dataset_id


def meta_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / META_FILE_NAME


def exists(dataset_id: str) -> bool:
    """Whether the dataset is on disk."""
    try:
        return meta_path(dataset_id).exists()
    except ValueError:
        return False


def new_dataset_id(prefix: str, name: str) -> str:
    base = f"{prefix}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('._-') or 'dataset'}"
    candidate, i = base, 2
    while (root() / candidate).exists():
        candidate, i = f"{base}-{i}", i + 1
    return candidate


def suggested_name(dataset_id: str) -> str | None:
    """The PMLB name if this is the id of a suggested dataset."""
    name = dataset_id.removeprefix("pmlb-")
    return name if name in config.SUGGESTED_PMLB_DATASETS and dataset_id.startswith("pmlb-") else None


def display_name(value: str) -> str:
    if exists(value):
        return str(read_meta(meta_path(value)).get("name", value))
    return Path(value.removeprefix("pmlb-")).stem


def data_file(dataset_id: str, meta) -> Path:
    return dataset_dir(dataset_id) / str(meta.get("file", DOWNLOADED_FILE))


# ----------------------------------------------------------------------
# Listing
# ----------------------------------------------------------------------

def list_datasets() -> list[tuple[str, dict]]:
    """The datasets on disk as (id, metadata), sorted by name."""
    datasets = []
    for path in sorted(root().glob(f"*/{META_FILE_NAME}")):
        try:
            datasets.append((path.parent.name, read_meta(path).unwrap()))
        except Exception as e:
            logger.warning("Can't read %s, the dataset is left out: %s", path, e)
    return sorted(datasets, key=lambda d: str(d[1].get("name", d[0])).lower())


def dataset_kind(dataset_id: str) -> str | None:
    """"regression", "binary" or "multiclass", None if it isn't known."""
    if exists(dataset_id):
        meta = read_meta(meta_path(dataset_id))
        return task_kind(meta.get("task"), meta.get("classes"))
    if suggested_name(dataset_id):
        task, n_classes = sources.pmlb_info(suggested_name(dataset_id), sources.pmlb_summary(cache_dir()))
        return task_kind(task, list(range(n_classes or 0)))
    return None


def dropdown_options() -> list[dict]:
    """The datasets on disk, then the suggested datasets that aren't downloaded yet."""
    options = []
    for dataset_id, meta in list_datasets():
        kind = task_kind(meta.get("task"), meta.get("classes"))
        if kind == "multiclass":
            kind = f"multiclass, {len(meta['classes'])} classes"
        parts = [str(meta.get("name", dataset_id)), SOURCE_LABELS.get(meta.get("source"), "?"), kind]
        options.append({"label": " · ".join(p for p in parts if p), "value": dataset_id})
    for name in config.SUGGESTED_PMLB_DATASETS:
        dataset_id = f"pmlb-{name}"
        if not exists(dataset_id):
            label = " · ".join(p for p in (name, "PMLB", dataset_kind(dataset_id)) if p)
            options.append({"label": f"{label} (suggested, downloads when used)", "value": dataset_id})
    return options


def source_index() -> dict[tuple[str, str], str]:
    """(source, source id) -> dataset id, for the PMLB and UCI datasets on disk."""
    return {(str(meta.get("source")), str(meta.get("source_id"))): dataset_id for dataset_id, meta in list_datasets()
            if meta.get("source") in ("pmlb", "uci")}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _read_cached(path: str, mtime: float, options: tuple) -> pd.DataFrame:
    return read_table(path, dict(options))


def _read(path: Path, options: dict) -> pd.DataFrame:
    return _read_cached(str(path), path.stat().st_mtime, tuple(sorted(options.items())))


def read_data(dataset_id: str, meta) -> pd.DataFrame:
    """The raw data of a dataset on disk (also of one that is being added)."""
    return _read(data_file(dataset_id, meta), dict(meta.get("read") or {}))


def load_meta(dataset_id: str) -> TOMLDocument:
    return read_meta(meta_path(dataset_id))


def load_completed(dataset_id: str) -> tuple[TOMLDocument, pd.DataFrame]:
    """The metadata (with missing values filled in and saved) and the raw data of a dataset.
    """
    if not exists(dataset_id):
        if not suggested_name(dataset_id):
            raise ValueError(f"Dataset {dataset_id!r} isn't on the list of datasets.")
        discard_pending(dataset_id)  # a download that was never saved would get another id
        _, doc, _ = add_from_source("pmlb", suggested_name(dataset_id))
        write_meta(meta_path(dataset_id), doc)
    doc = load_meta(dataset_id)
    df = read_data(dataset_id, doc)
    if doc.get("source") == "pmlb":
        hints = sources.pmlb_hints(str(doc.get("source_id")), list(df.columns), cache_dir())
    elif doc.get("source") == "uci":
        hints = sources.uci_hints_from_file(dataset_dir(dataset_id) / UCI_INFO_FILE, list(df.columns))
    else:
        hints = None
    if complete_meta(doc, df, hints):
        write_meta(meta_path(dataset_id), doc)
    return doc, df


def load_dataset(dataset_id: str) -> Dataset:
    doc, df = load_completed(dataset_id)
    prepared = prepare(df, doc)
    return Dataset(
        X=prepared.X,
        y=prepared.y,
        feature_names=prepared.feature_names,
        target_name=prepared.target_name,
        cat_ids=prepared.cat_ids,
        name=str(doc.get("name", dataset_id)),
        task=prepared.task,
        classes=prepared.classes,
    )


# ----------------------------------------------------------------------
# Adding and removing
# ----------------------------------------------------------------------

def add_from_source(source: str, key: str) -> tuple[str, TOMLDocument, pd.DataFrame]:
    """Download a PMLB or UCI dataset into a new folder. Returns its id, the proposed metadata and the data.

    The metadata isn't saved yet: write it to meta_path(id) to add the dataset to the list, or call
    ``discard_pending`` to drop it.
    """
    info = None
    if source == "pmlb":
        df, hints = sources.download_pmlb(key, cache_dir())
        name = key
        dataset_id = new_dataset_id("pmlb", key)
    elif source == "uci":
        df, hints, name, info = sources.download_uci(key)
        dataset_id = new_dataset_id(f"uci-{key}", name)
    else:
        raise ValueError(f"Unknown source {source!r}.")
    folder = dataset_dir(dataset_id)
    folder.mkdir(parents=True)
    try:
        if info is not None:
            (folder / UCI_INFO_FILE).write_text(json.dumps(info, indent=1))
        df.to_csv(folder / DOWNLOADED_FILE, sep="\t", index=False)
        doc = new_meta(name, source, DOWNLOADED_FILE, source_id=int(key) if source == "uci" else key,
                       read_options=DOWNLOADED_READ_OPTIONS)
        df = _read(folder / DOWNLOADED_FILE, DOWNLOADED_READ_OPTIONS)
        complete_meta(doc, df, hints)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    logger.info("Downloaded %s dataset %s to %s", source, key, folder)
    return dataset_id, doc, df


def add_upload(filename: str, content: bytes) -> tuple[str, TOMLDocument, pd.DataFrame]:
    """Save an uploaded file into a new folder. Returns its id, the proposed metadata and the data."""
    filename = Path(filename).name
    stem = filename.split(".")[0]
    dataset_id = new_dataset_id("upload", stem)
    folder = dataset_dir(dataset_id)
    folder.mkdir(parents=True)
    path = folder / filename
    try:
        path.write_bytes(content)
        read_options = guess_read_options(path)
        df = _read(path, read_options)
        doc = new_meta(stem, "upload", filename, read_options=read_options)
        complete_meta(doc, df)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return dataset_id, doc, df


def discard_pending(dataset_id: str) -> None:
    """Remove the folder of a dataset that was downloaded or uploaded but never saved."""
    folder = dataset_dir(dataset_id)
    if folder.exists() and not (folder / META_FILE_NAME).exists():
        shutil.rmtree(folder, ignore_errors=True)


def remove(dataset_id: str) -> None:
    """Delete the folder of a dataset (a suggested dataset stays on the list as a suggestion)."""
    folder = dataset_dir(dataset_id)
    if folder.exists():
        shutil.rmtree(folder)
