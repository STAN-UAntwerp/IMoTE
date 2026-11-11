import numpy as np
import pandas as pd
import pytest
import tomlkit

from imote.callbacks.cc_dataset_callbacks import apply_form
from imote.dataset.metadata import AUTO, Hints, complete_meta, guess_task, new_meta, prepare, task_kind

DF = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "x": [0.1, 0.5, 1.2, 3.3, 2.2, 0.9, 1.8, 2.7, 3.1, 0.4],
    "color": ["red", "blue", "red", "green", "blue", "red", None, "green", "red", "blue"],
    "size": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "label": ["no", "yes", "no", "yes", "maybe", "no", "yes", "no", "yes", "no"],
})


def is_auto(container, key):
    return AUTO in container[key].trivia.comment


def _doc():
    return new_meta("test", "upload", "d.csv")


def test_complete_fills_in_and_marks_everything():
    doc = _doc()
    assert complete_meta(doc, DF)
    assert doc["response"] == "label" and is_auto(doc, "response")
    assert doc["task"] == "classification" and is_auto(doc, "task")
    assert doc["classes"] == ["maybe", "no", "yes"]
    assert set(doc["categorical"]) == {"color", "size"}
    assert doc["categorical"]["color"] == ["blue", "green", "red"]
    assert is_auto(doc["categorical"], "color")
    assert not complete_meta(doc, DF)  # nothing left to fill in


def test_complete_keeps_what_the_user_wrote():
    text = (
        'name = "test"\n'
        'file = "d.csv"\n'
        'response = "x"  # my response\n'
        'task = "regression"\n'
        "\n[categorical]\n"
        "# my comment\n"
        'color = ["red", "blue"]\n'
        "size = []\n"
    )
    doc = tomlkit.parse(text)
    complete_meta(doc, DF)
    out = tomlkit.dumps(doc)
    assert doc["response"] == "x" and "# my response" in out and "# my comment" in out
    assert doc["categorical"]["color"] == ["red", "blue"] and not is_auto(doc["categorical"], "color")
    assert doc["categorical"]["size"] == [1, 2] and is_auto(doc["categorical"], "size")  # empty list filled in
    assert "label" in doc["categorical"]  # text columns are always categorical
    assert "classes" not in doc


def test_hints_are_used_and_named():
    doc = _doc()
    complete_meta(doc, DF, Hints(response="x", task="regression", categorical=["size"], ignore=["id"],
                                 source="UCI metadata"))
    assert doc["response"] == "x" and doc["ignore"] == ["id"]
    assert "automatically detected (UCI metadata)" in tomlkit.dumps(doc)
    assert set(doc["categorical"]) == {"size", "color", "label"}


def test_prepare_codes_text_and_drops_missing_and_unlisted_rows():
    doc = _doc()
    complete_meta(doc, DF)
    doc["categorical"]["color"] = ["red", "blue"]  # green rows are left out
    doc["ignore"] = ["id"]
    prepared = prepare(DF, doc)
    assert prepared.feature_names == ["x", "color", "size"]
    np.testing.assert_array_equal(prepared.cat_ids, [1, 2])
    assert prepared.n_dropped == 3  # one missing color, two green
    np.testing.assert_array_equal(prepared.X[:, 1], [0, 1, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(prepared.y, [1, 2, 1, 0, 1, 2, 1])  # classes maybe, no, yes
    assert "3 classes (multiclass)" in prepared.summary()


def test_binary_classes_follow_the_order_in_the_file():
    doc = _doc()
    df = DF[(DF["label"] != "maybe") & DF["color"].notna()]
    complete_meta(doc, df)
    doc["classes"] = ["yes", "no"]  # "no" is the positive class
    np.testing.assert_array_equal(prepare(df, doc).y, (df["label"] == "no").astype(float))


@pytest.mark.parametrize("change, message", [
    ({"response": "nope"}, "isn't in the data"),
    ({"task": "clustering"}, "task must be"),
    ({"response": "color", "task": "regression"}, "isn't numeric"),
])
def test_prepare_errors(change, message):
    doc = _doc()
    complete_meta(doc, DF)
    for key, value in change.items():
        doc[key] = value
    complete_meta(doc, DF)  # as when the dataset is loaded after editing the file
    with pytest.raises(ValueError, match=message):
        prepare(DF, doc)


@pytest.mark.parametrize("values, task", [
    ([1.5, 2.5, 3.5], "regression"),
    (list(range(50)), "regression"),
    ([0, 1, 1, 0], "classification"),
    (["a", "b"], "classification"),
])
def test_guess_task(values, task):
    assert guess_task(pd.Series(values)) == task


def test_form_changes_are_stored_as_manual_and_unchanged_values_stay_automatic():
    doc = _doc()
    complete_meta(doc, DF)
    apply_form(doc, DF, "test", "label", "classification", ["color", "x"], ["id"])
    assert is_auto(doc, "response") and is_auto(doc, "task")  # unchanged
    assert doc["ignore"] == ["id"] and not is_auto(doc, "ignore")
    assert set(doc["categorical"]) == {"color", "x"}  # the response isn't a feature
    assert is_auto(doc["categorical"], "x")  # categories of a newly chosen column are filled in

    apply_form(doc, DF, "renamed", "x", "regression", ["color"], ["id"])
    assert doc["name"] == "renamed" and not is_auto(doc, "response") and not is_auto(doc, "task")
    assert "classes" not in doc
    assert "label" in doc["categorical"]  # text column, it's no longer the response
    prepare(DF, doc)


@pytest.mark.parametrize("task, classes, kind", [
    ("regression", None, "regression"),
    ("classification", ["a", "b"], "binary"),
    ("classification", [1, 2, 3], "multiclass"),
    (None, None, None),
])
def test_task_kind(task, classes, kind):
    assert task_kind(task, classes) == kind
