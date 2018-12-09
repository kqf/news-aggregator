import pytest
from model.model import read_dataset


@pytest.fixture()
def data():
    # TODO: Add size parameter
    return read_dataset(size=0.1)
