"""Small, provenance-preserving PubChem PUG REST lookup by SMILES."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from molscout.chemistry import validate_smiles

_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _get_json(url: str, *, timeout: float) -> dict[str, object]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "MolScout/0.1 (portfolio demo)"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return {}
        if exc.code == 503:
            raise RuntimeError("PubChem is busy; retry the lookup later.") from exc
        raise RuntimeError(f"PubChem lookup failed with HTTP {exc.code}.") from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise RuntimeError("PubChem could not be reached; check network access and retry.") from exc
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PubChem returned a response that was not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("PubChem returned an unexpected response shape.")
    return decoded


def pubchem_lookup(smiles: str, *, timeout: float = 10.0) -> dict[str, object]:
    """Look up a canonical SMILES in PubChem and return identifiers and properties.

    The response records the exact PubChem PUG REST URLs used so downstream
    reports can cite the source. A 404/no-match is represented with found=False.
    """
    if timeout <= 0 or timeout > 60:
        raise ValueError("timeout must be greater than 0 and at most 60 seconds")
    canonical = str(validate_smiles(smiles)["canonical_smiles"])
    encoded = quote(canonical, safe="")
    cids_url = f"{_BASE_URL}/compound/smiles/{encoded}/cids/JSON"
    cid_payload = _get_json(cids_url, timeout=timeout)
    identifier_list = cid_payload.get("IdentifierList")
    cids = identifier_list.get("CID") if isinstance(identifier_list, dict) else None
    if not isinstance(cids, list) or not cids:
        return {
            "found": False,
            "query_smiles": canonical,
            "source": cids_url,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }

    cid = int(cids[0])
    properties_url = (
        f"{_BASE_URL}/compound/cid/{cid}/property/"
        "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChIKey,IUPACName/JSON"
    )
    property_payload = _get_json(properties_url, timeout=timeout)
    table = property_payload.get("PropertyTable")
    rows = table.get("Properties") if isinstance(table, dict) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise RuntimeError(f"PubChem returned CID {cid} without the requested properties.")
    return {
        "found": True,
        "query_smiles": canonical,
        "cid": cid,
        "properties": rows[0],
        "source": properties_url,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }
