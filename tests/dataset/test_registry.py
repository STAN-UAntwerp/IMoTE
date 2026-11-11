import numpy as np
import pandas as pd
import pytest
import tomlkit

from imote import config
from imote.dataset import dataset_registry as registry
from imote.dataset import sources
from imote.dataset.metadata import write_meta

PMLB_DF = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 4.0, 5.0, 6.0], "c": [0.0, 1.0, 0.0, 1.0],
                        "target": [5.0, 6.0, 7.5, 8.25]})


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """An empty datasets folder, PMLB downloads return PMLB_DF and are counted."""
    monkeypatch.setattr(config, "DIR_DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(config, "SUGGESTED_PMLB_DATASETS", ["547_no2", "1199_BNG_echoMonths"])
    monkeypatch.setattr(sources, "pmlb_summary", lambda cache_dir: None)  # no network
    names = []

    def fake_fetch_data(name):
        names.append(name)
        return PMLB_DF.copy()

    monkeypatch.setattr("pmlb.fetch_data", fake_fetch_data)
    return names


def test_suggestions_are_listed_without_writing_anything(downloads):
    options = registry.dropdown_options()
    assert [o["value"] for o in options] == ["pmlb-547_no2", "pmlb-1199_BNG_echoMonths"]
    assert all("suggested" in o["label"] and "regression" in o["label"] for o in options)
    assert registry.dataset_kind("pmlb-547_no2") == "regression"
    assert downloads == [] and not registry.root().exists()


def test_suggestion_is_downloaded_once_when_used(downloads):
    ds = registry.load_dataset("pmlb-1199_BNG_echoMonths")
    registry.load_dataset("pmlb-1199_BNG_echoMonths")
    assert downloads == ["1199_BNG_echoMonths"]
    np.testing.assert_array_equal(ds.y, PMLB_DF["target"])
    assert ds.feature_names == ["a", "b", "c"] and ds.task == "regression"
    # the categorical columns from benchmark_info (indices 0 and 2)
    np.testing.assert_array_equal(ds.cat_ids, [0, 2])
    doc = registry.load_meta("pmlb-1199_BNG_echoMonths")
    assert doc["categorical"]["a"] == [1, 2, 3, 4]
    assert doc["categorical"]["a"].trivia.comment == "# automatically detected (PMLB list in benchmark_info.py)"
    label = next(o["label"] for o in registry.dropdown_options() if o["value"] == "pmlb-1199_BNG_echoMonths")
    assert "suggested" not in label
    assert registry.display_name("pmlb-1199_BNG_echoMonths") == "1199_BNG_echoMonths"
    assert registry.display_name("547_no2.pmlb") == "547_no2"  # trees saved by older versions


def test_removed_suggestion_stays_suggested(downloads):
    registry.load_dataset("pmlb-547_no2")
    registry.remove("pmlb-547_no2")
    assert not registry.exists("pmlb-547_no2")
    assert "pmlb-547_no2" in [o["value"] for o in registry.dropdown_options()]


def test_unknown_dataset_raises(downloads):
    with pytest.raises(ValueError, match="isn't on the list"):
        registry.load_dataset("pmlb-not_suggested")


def test_upload_add_and_discard(downloads):
    content = pd.DataFrame({"x": [1, 2, 3, 4], "kind": ["a", "b", "a", "b"], "y": [0.5, 1.5, 2.5, 3.5]}).to_csv(
        index=False).encode()
    dataset_id, doc, df = registry.add_upload("my data.csv", content)
    assert dataset_id == "upload-my_data"
    assert not registry.exists(dataset_id)  # not on the list before saving
    registry.discard_pending(dataset_id)
    assert not registry.dataset_dir(dataset_id).exists()

    dataset_id, doc, df = registry.add_upload("my data.csv", content)
    write_meta(registry.meta_path(dataset_id), doc)
    ds = registry.load_dataset(dataset_id)
    assert ds.target_name == "y" and ds.task == "regression"
    assert dataset_id in [o["value"] for o in registry.dropdown_options()]
    assert registry.add_upload("my data.csv", content)[0] == "upload-my_data-2"


def test_upload_unsupported_format_raises_and_leaves_nothing(downloads):
    with pytest.raises(ValueError, match="Unsupported"):
        registry.add_upload("notes.docx", b"x")
    assert list(registry.root().iterdir()) == []


def test_add_from_pmlb(downloads):
    dataset_id, doc, _ = registry.add_from_source("pmlb", "titanic")
    assert dataset_id == "pmlb-titanic" and doc["response"] == "target"
    write_meta(registry.meta_path(dataset_id), doc)
    assert registry.source_index()[("pmlb", "titanic")] == dataset_id


def test_manual_edits_survive_loading(downloads):
    registry.load_dataset("pmlb-547_no2")
    path = registry.meta_path("pmlb-547_no2")
    doc = tomlkit.parse(path.read_text() + "# my note\n")
    doc["ignore"] = ["b"]
    path.write_text(tomlkit.dumps(doc))
    ds = registry.load_dataset("pmlb-547_no2")
    assert ds.feature_names == ["a", "c"]
    assert "# my note" in path.read_text()
