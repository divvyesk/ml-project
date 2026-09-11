"""Week 13 - Working prototype, verified from the command line.

Runs the same inference path the Streamlit app uses, so the prototype can be
demonstrated (and checked in CI) without a browser.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.artifacts import load_bundle
from src.config import REPORT_DIR, ensure_directories
from src.predict import explain_inputs, predict_price
from src.reporting import rupees

SAMPLE_VEHICLES = [
    {
        "name": "Maruti Swift Dzire VDI",
        "year": 2016,
        "km_driven": 50000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
    },
    {
        "name": "Maruti Swift Dzire VDI",
        "year": 2012,
        "km_driven": 120000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "Second Owner",
    },
    {
        "name": "Hyundai Creta 1.6 CRDi SX Option",
        "year": 2018,
        "km_driven": 30000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Automatic",
        "owner": "First Owner",
    },
    {
        "name": "Totally Unknown Model XYZ",
        "year": 2015,
        "km_driven": 60000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
    },
]


def main():
    ensure_directories()

    bundle = load_bundle()
    predictions = predict_price(SAMPLE_VEHICLES, bundle)
    derived = explain_inputs(SAMPLE_VEHICLES, bundle)

    table = pd.DataFrame(SAMPLE_VEHICLES)
    table["car_age"] = derived["car_age"].values
    table["name_frequency"] = derived["name_frequency"].values
    table["predicted_price"] = predictions
    table["predicted_price_formatted"] = [rupees(v) for v in predictions]

    table.to_csv(REPORT_DIR / "week13_sample_predictions.csv", index=False)

    print("Week 13 prototype check complete.")
    print("Model: {0}   Reference year: {1}".format(
        bundle["model_name"], bundle["reference_year"]
    ))
    print()
    print(
        table[
            ["name", "year", "km_driven", "transmission", "car_age",
             "name_frequency", "predicted_price_formatted"]
        ].to_string(index=False)
    )
    print()
    print(
        "The last row uses a car name that does not appear in the training "
        "data; name_frequency falls back to 0 and the prediction still "
        "succeeds."
    )


if __name__ == "__main__":
    main()
