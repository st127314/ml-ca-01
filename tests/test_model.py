import numpy as np
import pandas as pd
import pytest

from model_utils import OWNER_OPTIONS, category_options, model

REQUIRED_COLUMNS = [
    "brand",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
]


def make_row(**overrides):
    row = {column: None for column in REQUIRED_COLUMNS}
    row.update(
        {
            "brand": category_options["brand"][0],
            "year": 2015,
            "km_driven": 50000,
            "fuel": category_options["fuel"][0],
            "seller_type": category_options["seller_type"][0],
            "transmission": category_options["transmission"][0],
            "owner": 1,
            "mileage": 20.0,
            "engine": 1200.0,
            "max_power": 82.0,
            "seats": 5.0,
        }
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_model_pipeline_has_expected_steps():
    assert "preprocessor" in model.named_steps


def test_category_options_are_populated():
    for column in ("brand", "fuel", "seller_type", "transmission"):
        assert category_options[column], f"no categories learned for {column}"


def test_owner_options_are_ordered_first_to_fourth():
    assert [option["value"] for option in OWNER_OPTIONS] == [1, 2, 3, 4]


def test_predict_returns_a_plausible_price():
    prediction = float(np.exp(model.predict(make_row())[0]))
    # log-scale target, so anything outside this range means the pipeline is broken
    assert 10_000 < prediction < 100_000_000


def test_predict_tolerates_missing_optional_fields():
    sparse = make_row(
        brand=None,
        km_driven=None,
        fuel=None,
        seller_type=None,
        transmission=None,
        owner=None,
        mileage=None,
        engine=None,
        seats=None,
    )
    prediction = float(np.exp(model.predict(sparse)[0]))
    assert np.isfinite(prediction)


def test_newer_car_is_worth_more_than_older_one():
    old = float(np.exp(model.predict(make_row(year=2005))[0]))
    new = float(np.exp(model.predict(make_row(year=2020))[0]))
    assert new > old


def test_more_power_is_worth_more():
    weak = float(np.exp(model.predict(make_row(max_power=60.0))[0]))
    strong = float(np.exp(model.predict(make_row(max_power=150.0))[0]))
    assert strong > weak


@pytest.mark.parametrize("year", [1990, 2000, 2010, 2020])
def test_prediction_is_finite_across_year_range(year):
    assert np.isfinite(model.predict(make_row(year=year))[0])
