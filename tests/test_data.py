import pandas as pd
import pytest

from src.data import clean_data, clean_text_columns, invalid_value_report, load_csv
from src.config import BASE_COLUMNS, CLEAN_PATH, RAW_PATH


def make_frame(rows):
    return pd.DataFrame(rows, columns=BASE_COLUMNS)


def test_load_csv_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["a"], "year": [2015]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(path)


def test_clean_text_collapses_internal_whitespace():
    frame = make_frame(
        [["  Maruti   Swift  Dzire ", 2015, 500000, 40000, " Diesel ",
          "Individual", "Manual", "First Owner"]]
    )
    cleaned = clean_text_columns(frame)
    assert cleaned.loc[0, "name"] == "Maruti Swift Dzire"
    assert cleaned.loc[0, "fuel"] == "Diesel"


def test_clean_data_counts_each_removal_stage_separately():
    good = ["Maruti Swift", 2015, 500000, 40000, "Diesel", "Individual",
            "Manual", "First Owner"]
    frame = make_frame(
        [
            good,
            list(good),                       # exact duplicate
            ["Bad Price", 2015, 0, 40000, "Petrol", "Dealer", "Manual",
             "First Owner"],                  # non-positive price
            ["Bad Km", 2015, 400000, -5, "Petrol", "Dealer", "Manual",
             "First Owner"],                  # negative km
            ["Bad Year", 1800, 400000, 40000, "Petrol", "Dealer", "Manual",
             "First Owner"],                  # impossible year
        ]
    )

    cleaned, stages = clean_data(frame)

    assert stages["original_records"] == 5
    assert stages["removed_invalid_values"] == 3
    assert stages["removed_exact_duplicates"] == 1
    assert stages["final_records"] == 1
    assert len(cleaned) == 1


def test_invalid_value_report_counts_before_dropping():
    frame = make_frame(
        [
            ["A", 2015, 500000, -1, "Diesel", "Individual", "Manual", "First Owner"],
            ["B", 1800, 0, 10, "Petrol", "Dealer", "Manual", "First Owner"],
        ]
    )
    report = invalid_value_report(frame)
    assert report["negative_km_driven"] == 1
    assert report["non_positive_selling_price"] == 1
    assert report["invalid_manufacturing_years"] == 1


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw dataset not present")
def test_project_dataset_matches_documented_counts():
    raw = load_csv(RAW_PATH)
    cleaned, stages = clean_data(raw)

    assert stages["original_records"] == 4340
    assert stages["removed_exact_duplicates"] == 763
    assert stages["final_records"] == 3577
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run week06_cleaning.py first")
def test_cleaned_file_on_disk_matches_a_fresh_clean():
    on_disk = load_csv(CLEAN_PATH)
    fresh, _ = clean_data(load_csv(RAW_PATH))
    pd.testing.assert_frame_equal(
        on_disk.reset_index(drop=True), fresh.reset_index(drop=True)
    )
