import pandas as pd

from molscout.benchmark import TASKS, _features_and_labels


def test_task_metric_configuration():
    assert TASKS["hERG"] == {"kind": "classification", "metric": "AUROC"}
    assert TASKS["Solubility_AqSolDB"] == {"kind": "regression", "metric": "MAE"}


def test_features_skip_invalid_molecules():
    frame = pd.DataFrame({"Drug": ["CCO", "bad smiles"], "Y": [1, 0]})
    features, labels = _features_and_labels(frame)
    assert features.shape == (1, 2048)
    assert labels.tolist() == [1.0]
