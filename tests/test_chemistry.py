import pytest

from molscout.chemistry import compute_properties, morgan_fingerprint, validate_smiles


def test_ethanol_is_valid_and_canonicalizes():
    result = validate_smiles("CCO")
    assert result["valid"] is True
    assert result["canonical_smiles"] == "CCO"


def test_properties_have_expected_shape_and_types():
    result = compute_properties("CCO")
    assert result.canonical_smiles == "CCO"
    assert result.molecular_weight > 0
    assert isinstance(result.h_bond_donors, int)
    assert isinstance(result.h_bond_acceptors, int)


@pytest.mark.parametrize("smiles", ["", "not a molecule", "C1CC"])
def test_invalid_smiles_raises(smiles):
    with pytest.raises(ValueError):
        validate_smiles(smiles)


def test_morgan_fingerprint_is_fixed_length_binary_vector():
    fingerprint = morgan_fingerprint("CCO", n_bits=256)
    assert fingerprint.shape == (256,)
    assert set(fingerprint.tolist()) <= {0, 1}


def test_fingerprint_rejects_invalid_size():
    with pytest.raises(ValueError):
        morgan_fingerprint("CCO", n_bits=0)
