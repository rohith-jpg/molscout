from pathlib import Path

import pytest

from molscout.prediction import ENDPOINTS, predict_admet


def test_registry_has_the_four_intended_endpoints():
    assert set(ENDPOINTS) == {"hERG", "AMES", "BBB_Martins", "Solubility_AqSolDB"}
    assert ENDPOINTS["Solubility_AqSolDB"]["metric"] == "MAE"


def test_prediction_requires_an_explicitly_supported_endpoint():
    with pytest.raises(ValueError, match="Unknown endpoint"):
        predict_admet("CCO", "made_up", model_dir=Path("missing"))


def test_prediction_never_returns_a_fabricated_value_without_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the TDC baseline benchmark first"):
        predict_admet("CCO", "hERG", model_dir=tmp_path)
