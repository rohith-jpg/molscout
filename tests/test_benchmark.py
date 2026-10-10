import pandas as pd
import pytest

from molscout.benchmark import TASKS, _aggregate_metrics, _features_and_labels


def test_task_metric_configuration():
    assert TASKS["hERG"] == {"kind": "classification", "metric": "AUROC"}
    assert TASKS["Solubility_AqSolDB"] == {"kind": "regression", "metric": "MAE"}


def test_features_fail_on_invalid_molecule_instead_of_changing_test_population():
    frame = pd.DataFrame({"Drug": ["CCO", "bad smiles"], "Y": [1, 0]})
    with pytest.raises(ValueError, match="refusing to silently drop"):
        _features_and_labels(frame)


def test_aggregate_reports_mean_population_std_and_run_count():
    rows = [
        {"dataset": "hERG", "metric": "AUROC", "value": 0.7, "seed": 1},
        {"dataset": "hERG", "metric": "AUROC", "value": 0.9, "seed": 2},
    ]
    result = _aggregate_metrics(rows)[0]
    assert result["mean"] == pytest.approx(0.8)
    assert result["std"] == pytest.approx(0.1)
    assert result["n_runs"] == 2
    assert result["seeds"] == [1, 2]
