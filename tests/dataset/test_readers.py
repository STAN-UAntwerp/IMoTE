import builtins

import numpy as np
import pandas as pd
import pytest

from imote.dataset.readers import MissingDependencyError, guess_read_options, read_table

DF = pd.DataFrame({"a": [1.5, 2.0, 3.25], "color": ["red", "blue", "red"], "target": [1, 0, 1]})


def _assert_same(df):
    assert list(df.columns) == ["a", "color", "target"]
    np.testing.assert_allclose(df["a"], DF["a"])
    assert df["color"].tolist() == DF["color"].tolist()


@pytest.mark.parametrize("name, sep", [("d.csv", ","), ("d.tsv", "\t"), ("d.txt", ";"), ("d.csv.gz", ",")])
def test_delimited_with_header(tmp_path, name, sep):
    path = tmp_path / name
    DF.to_csv(path, sep=sep, index=False)
    assert guess_read_options(path) == {"sep": sep, "header": True}
    _assert_same(read_table(path))


def test_headerless_data_file(tmp_path):
    path = tmp_path / "d.data"
    path.write_text("1.5,red,1\n2.0,blue,0\n3.25,red,?\n")
    df = read_table(path, guess_read_options(path))
    assert list(df.columns) == ["col0", "col1", "col2"]
    assert df["col2"].isna().tolist() == [False, False, True]  # "?" means missing


@pytest.mark.parametrize("name, write", [
    ("d.xlsx", lambda path: DF.to_excel(path, index=False)),
    ("d.json", lambda path: DF.to_json(path, orient="records")),
    ("d.parquet", lambda path: DF.to_parquet(path)),
    ("d.feather", lambda path: DF.to_feather(path)),
])
def test_other_formats(tmp_path, name, write):
    if name.endswith(("parquet", "feather")):
        pytest.importorskip("pyarrow")
    write(tmp_path / name)
    _assert_same(read_table(tmp_path / name))


def test_parquet_without_pyarrow_says_what_to_install(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    with pytest.raises(MissingDependencyError, match=r"pip install imote\[parquet\]"):
        read_table(tmp_path / "d.parquet")


def test_arff(tmp_path):
    path = tmp_path / "d.arff"
    path.write_text("@relation test\n@attribute a numeric\n@attribute color {red,blue}\n@attribute target numeric\n"
                    "@data\n1.5,red,1\n2.0,blue,0\n3.25,red,1\n")
    _assert_same(read_table(path))


def test_unsupported_or_single_column_raises(tmp_path):
    (tmp_path / "d.docx").write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        read_table(tmp_path / "d.docx")
    (tmp_path / "one.csv").write_text("a\n1\n2\n")
    with pytest.raises(ValueError, match="at least one feature"):
        read_table(tmp_path / "one.csv")


def test_downloaded_table_keeps_exact_values(tmp_path):
    df = pd.DataFrame({"a": [0.1 + 0.2, 1 / 3, 1e-300], "target": [2 / 7, -1e10 / 3, 5.0]})
    df.to_csv(tmp_path / "d.tsv.gz", sep="\t", index=False)  # as downloaded datasets are saved
    np.testing.assert_array_equal(read_table(tmp_path / "d.tsv.gz", {"sep": "\t"}).to_numpy(), df.to_numpy())
