import pandas as pd

from src.config import FEATURE_COLUMNS
from src.features import add_features, extract_brand, extract_model_family


def sample():
    return pd.DataFrame(
        [
            {"name": "Maruti Swift Dzire VDI", "year": 2016, "km_driven": 50000,
             "fuel": "Diesel", "seller_type": "Individual",
             "transmission": "Manual", "owner": "First Owner"},
            {"name": "Maruti Swift Dzire VDI", "year": 2020, "km_driven": 10000,
             "fuel": "Diesel", "seller_type": "Dealer",
             "transmission": "Manual", "owner": "First Owner"},
            {"name": "Hyundai Creta SX", "year": 2018, "km_driven": 30000,
             "fuel": "Diesel", "seller_type": "Dealer",
             "transmission": "Automatic", "owner": "First Owner"},
        ]
    )


def test_brand_and_model_family():
    assert extract_brand("Maruti Swift Dzire VDI") == "Maruti"
    assert extract_model_family("Maruti Swift Dzire VDI") == "Maruti Swift"
    assert extract_brand(None) == "Unknown"
    assert extract_model_family(float("nan")) == "Unknown"


def test_car_age_uses_the_supplied_reference_year():
    frame, reference_year, _ = add_features(sample(), reference_year=2020)
    assert reference_year == 2020
    assert frame["car_age"].tolist() == [4, 0, 2]


def test_car_age_is_never_negative():
    frame, _, _ = add_features(sample(), reference_year=2015)
    assert (frame["car_age"] >= 0).all()


def test_km_per_year_does_not_divide_by_zero():
    frame, _, _ = add_features(sample(), reference_year=2020)
    assert frame["km_per_year"].notna().all()
    # Age 0 is treated as one year of use.
    assert frame.loc[1, "km_per_year"] == 10000


def test_supplied_frequency_map_is_reused_not_recomputed():
    """A single row must not be told its own name is unique.

    This is the bug the earlier prototype had: the app recomputed the map on
    one row, so name_frequency was always 1 while training values ranged up
    to 69.
    """
    fitted_map = {"Maruti Swift Dzire VDI": 69, "Hyundai Creta SX": 3}
    one_row = sample().head(1)

    frame, _, returned_map = add_features(
        one_row, reference_year=2020, frequency_map=fitted_map
    )

    assert frame.loc[0, "name_frequency"] == 69
    assert returned_map is fitted_map


def test_unseen_name_gets_zero_frequency():
    row = sample().head(1).copy()
    row.loc[row.index[0], "name"] = "Never Seen Before"
    frame, _, _ = add_features(row, reference_year=2020, frequency_map={"x": 5})
    assert frame.loc[frame.index[0], "name_frequency"] == 0


def test_all_model_input_columns_are_produced():
    frame, _, _ = add_features(sample())
    for column in FEATURE_COLUMNS:
        assert column in frame.columns
