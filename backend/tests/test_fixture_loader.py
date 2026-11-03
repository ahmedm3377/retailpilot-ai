from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from retail_domain.models import FixtureRoot
from tests.conftest import PUBLIC_CASES


def test_all_public_cases_load_in_order(public_fixtures):
    root, _ = public_fixtures
    raw_ids = [c["case_id"] for c in json.loads(PUBLIC_CASES.read_bytes())["cases"]]
    assert root.contract_version == 1
    assert len(root.cases) == 6
    assert [c.case_id for c in root.cases] == raw_ids


def test_sha256_is_of_exact_input_bytes(public_fixtures):
    _, sha = public_fixtures
    assert sha == hashlib.sha256(PUBLIC_CASES.read_bytes()).hexdigest()


def test_unknown_fields_are_preserved():
    raw = json.loads(PUBLIC_CASES.read_bytes())
    raw["cases"][0]["initial_state"]["products"][0]["color"] = "teal"
    raw["cases"][0]["future_field"] = {"x": 1}
    dumped = FixtureRoot.model_validate(raw).model_dump(mode="json", exclude_none=True)
    assert dumped["cases"][0]["initial_state"]["products"][0]["color"] == "teal"
    assert dumped["cases"][0]["future_field"] == {"x": 1}


@pytest.mark.parametrize("path, value", [
    (("products", 0, "price_cents"), 40.5),
    (("orders", 0, "lines", 0, "paid_cents"), "3000"),
    (("products", 0, "stock"), 1.0),
])
def test_money_and_stock_must_be_integers(path, value):
    raw = json.loads(PUBLIC_CASES.read_bytes())
    target = raw["cases"][0]["initial_state"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        FixtureRoot.model_validate(raw)


def test_wrong_contract_version_is_rejected():
    raw = json.loads(PUBLIC_CASES.read_bytes())
    raw["contract_version"] = 2
    with pytest.raises(ValidationError):
        FixtureRoot.model_validate(raw)


