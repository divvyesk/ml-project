import pandas as pd
import pytest

from src.artifacts import load_bundle
from src.config import BUNDLE_PATH, FEATURE_COLUMNS
from src.predict import prepare_input, predict_price

VEHICLE = {
    "name": "Maruti Swift Dzire VDI",
    "year": 2016,
    "km_driven": 50000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
}

FAKE_BUNDLE = {"reference_year": 2020, "frequency_map": {"Maruti Swift Dzire VDI": 69}}


def test_prepare_input_builds_exactly_the_model_columns():
    frame = prepare_input(VEHICLE, FAKE_BUNDLE)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 1


def test_prepare_input_uses_the_stored_frequency_and_reference_year():
    frame = prepare_input(VEHICLE, FAKE_BUNDLE)
    assert frame.loc[0, "name_frequency"] == 69
    assert frame.loc[0, "car_age"] == 4


def test_prepare_input_rejects_incomplete_records():
    incomplete = dict(VEHICLE)
    del incomplete["transmission"]
    with pytest.raises(ValueError, match="Missing input fields"):
        prepare_input(incomplete, FAKE_BUNDLE)


def test_prepare_input_normalises_untidy_names():
    messy = dict(VEHICLE, name="  Maruti   Swift  Dzire VDI ")
    frame = prepare_input(messy, FAKE_BUNDLE)
    assert frame.loc[0, "name"] == "Maruti Swift Dzire VDI"
    assert frame.loc[0, "name_frequency"] == 69


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="run scripts/run_all.py first")
def test_bundle_carries_everything_needed_to_predict():
    bundle = load_bundle()
    for key in ("model", "model_name", "reference_year", "frequency_map",
                "feature_columns", "metrics"):
        assert key in bundle
    assert len(bundle["frequency_map"]) > 0
    assert bundle["feature_columns"] == FEATURE_COLUMNS


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="run scripts/run_all.py first")
def test_end_to_end_prediction_is_a_plausible_price():
    price = float(predict_price([VEHICLE])[0])
    assert 20000 < price < 9000000


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="run scripts/run_all.py first")
def test_an_older_car_is_predicted_cheaper_than_a_newer_one():
    newer = dict(VEHICLE, year=2019)
    older = dict(VEHICLE, year=2010)
    prices = predict_price([newer, older])
    assert prices[0] > prices[1]


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="run scripts/run_all.py first")
def test_an_unseen_car_name_still_predicts():
    unknown = dict(VEHICLE, name="Nonexistent Model ZZZ")
    price = float(predict_price([unknown])[0])
    assert price > 0


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="run scripts/run_all.py first")
def test_batch_and_single_predictions_agree():
    batch = predict_price([VEHICLE, dict(VEHICLE, year=2014)])
    single = predict_price([VEHICLE])
    assert batch[0] == pytest.approx(single[0])
