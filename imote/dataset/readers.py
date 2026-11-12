"""Read tabular data files (CSV/TSV/TXT/.data, Excel, Parquet, Feather, ARFF, JSON) into a pandas DataFrame.

The read options are stored in the ``[read]`` table of a dataset's metadata file.
``guess_read_options`` fills them in for a new file.
"""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pandas as pd

# Values that mean "missing" in delimited text files, on top of the pandas defaults ("?" is common in UCI files)
EXTRA_NA_VALUES = ["?"]

DELIMITED_SUFFIXES = {".csv", ".tsv", ".txt", ".data", ".dat", ".tab"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
ARROW_SUFFIXES = {".parquet", ".pq", ".feather", ".ftr"}
OTHER_SUFFIXES = {".arff", ".json"}
SUPPORTED_SUFFIXES = DELIMITED_SUFFIXES | EXCEL_SUFFIXES | ARROW_SUFFIXES | OTHER_SUFFIXES

# Shown in the upload dialog
SUPPORTED_FORMATS_TEXT = "CSV, TSV, TXT, .data, Excel (.xlsx/.xls), Parquet, JSON, ..."


class MissingDependencyError(ImportError):
    """An optional package is needed to read this file format."""


def _format_suffix(path: Path) -> str:
    # data.csv.gz -> .csv
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] in (".gz", ".bz2", ".zip", ".xz"):
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def _require(module: str, extra: str | None, what: str):
    try:
        __import__(module)
    except ImportError as e:
        install = f"pip install imote[{extra}]" if extra else f"pip install {module}"
        raise MissingDependencyError(f"Install {module} to load {what} files ({install}).") from e


def guess_read_options(path: Path | str) -> dict:
    """Read options for a new file: separator and header row for delimited text, nothing for other formats."""
    path = Path(path)
    suffix = _format_suffix(path)
    if suffix not in DELIMITED_SUFFIXES:
        return {}
    with (gzip.open if path.suffix.lower() == ".gz" else open)(path, "rb") as f:
        sample = f.read(64_000).decode("utf-8", errors="replace")
    lines = [line for line in sample.splitlines() if line.strip()][:50]
    sample = "\n".join(lines)
    sniffer = csv.Sniffer()
    try:
        sep = sniffer.sniff(sample, delimiters=",;\t| ").delimiter
    except csv.Error:
        sep = "\t" if suffix == ".tsv" else ","
    if sep == " ":
        sep = r"\s+"
    try:
        header = sniffer.has_header(sample)
    except csv.Error:
        header = suffix != ".data"
    return {"sep": sep, "header": bool(header)}


def read_table(path: Path | str, options: dict | None = None) -> pd.DataFrame:
    """Read a data file into a DataFrame with string column names.

    Raises:
        MissingDependencyError: An optional package for this format isn't installed.
        ValueError: The format isn't supported or the file can't be parsed.
    """
    path = Path(path)
    options = dict(options or {})
    suffix = _format_suffix(path)

    if suffix in DELIMITED_SUFFIXES:
        if "sep" not in options:
            options = {**guess_read_options(path), **options}
        sep = options.get("sep", ",")
        header = 0 if options.get("header", True) else None
        kwargs = {"sep": sep, "header": header, "na_values": EXTRA_NA_VALUES, "skipinitialspace": True}
        if sep == r"\s+" or len(sep) > 1:
            kwargs["engine"] = "python"
        else:
            kwargs["float_precision"] = "round_trip"
        df = pd.read_csv(path, **kwargs)
        if header is None:
            df.columns = [f"col{i}" for i in range(df.shape[1])]
    elif suffix in EXCEL_SUFFIXES:
        if suffix == ".xls":
            _require("xlrd", "xls", "old Excel (.xls)")
        else:
            _require("openpyxl", None, "Excel")
        df = pd.read_excel(path, sheet_name=options.get("sheet", 0), na_values=EXTRA_NA_VALUES)
    elif suffix in (".parquet", ".pq"):
        _require("pyarrow", "parquet", "Parquet/Feather")
        df = pd.read_parquet(path)
    elif suffix in (".feather", ".ftr"):
        _require("pyarrow", "parquet", "Parquet/Feather")
        df = pd.read_feather(path)
    elif suffix == ".arff":
        from scipy.io import arff
        with (gzip.open if path.suffix.lower() == ".gz" else open)(path, "rt", errors="replace") as f:
            df = pd.DataFrame(arff.loadarff(io.StringIO(f.read()))[0])
        for col in df.columns:
            if df[col].dtype == object:  # nominal attributes come back as bytes, "?" means missing
                df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v).replace("?", pd.NA)
    elif suffix == ".json":
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file type {path.name!r}, supported are: {SUPPORTED_FORMATS_TEXT}.")

    if df.shape[1] < 2:
        raise ValueError(f"{path.name} has {df.shape[1]} column(s), a dataset needs a response and at least one "
                         f"feature. Is the file imported correctly?")
    df.columns = [str(c).strip() for c in df.columns]
    return df
