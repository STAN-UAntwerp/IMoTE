import pytest

from tests.helpers import make_pilot_tree, make_split_tree


# Fresh trees per test: collapse/prune mutate them in place.
@pytest.fixture
def pilot_tree():
    return make_pilot_tree()


@pytest.fixture
def split_tree():
    return make_split_tree()
