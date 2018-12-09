from model.model import build_model


def test_splits_data(data):
    X_tr, X_te, y_tr, y_te = data
    assert "CATEGORY" not in X_tr.columns
    assert "CATEGORY" not in X_te.columns


# Dummy check for the pipeline integrity
def test_calculates_the_model():
    build_model()
