"""Week 13/15 - Streamlit prototype.

The app never engineers features itself. It loads the model bundle and calls
`src.predict`, which is the same code path the command-line scripts use, so
the demo cannot drift away from the trained model.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.artifacts import load_bundle
from src.config import BUNDLE_PATH
from src.data import load_csv
from src.config import CLEAN_PATH
from src.predict import explain_inputs, predict_price
from src.reporting import rupees

ROOT = Path(__file__).resolve().parent

FUEL_OPTIONS = ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
SELLER_OPTIONS = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSION_OPTIONS = ["Manual", "Automatic"]
OWNER_OPTIONS = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Used Car Price Predictor")
st.caption("Foundation of Machine Learning (SEML3211) — CarDekho dataset")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_known_names():
    if not CLEAN_PATH.exists():
        return []
    return sorted(load_csv(CLEAN_PATH)["name"].unique().tolist())


if not BUNDLE_PATH.exists():
    st.error(
        "No trained model found at `{0}`.\n\n"
        "Run `python scripts/run_all.py` first.".format(BUNDLE_PATH)
    )
    st.stop()

try:
    bundle = get_bundle()
except (ValueError, FileNotFoundError) as error:
    st.error(str(error))
    st.stop()

metrics = bundle.get("metrics", {})
with st.sidebar:
    st.subheader("Model")
    st.write("**{0}**".format(bundle["model_name"]))
    st.write("Reference year: {0}".format(bundle["reference_year"]))
    st.write("Trained: {0}".format(bundle.get("created_at", "unknown")))
    if metrics:
        st.subheader("Held-out test performance")
        st.metric("R²", "{0:.4f}".format(metrics.get("R2", float("nan"))))
        st.metric("MAE", rupees(metrics.get("MAE", 0)))
        st.metric("RMSE", rupees(metrics.get("RMSE", 0)))
        st.caption(
            "Measured on the test split, not a target figure. See "
            "`reports/final_report.md` section 14.6."
        )

st.subheader("Vehicle details")

known_names = get_known_names()
if known_names:
    default_index = (
        known_names.index("Maruti Swift Dzire VDI")
        if "Maruti Swift Dzire VDI" in known_names
        else 0
    )
    pick_from_list = st.checkbox("Pick a car name from the dataset", value=True)
    if pick_from_list:
        name = st.selectbox("Car name", known_names, index=default_index)
    else:
        name = st.text_input("Car name", "Maruti Swift Dzire VDI")
else:
    name = st.text_input("Car name", "Maruti Swift Dzire VDI")

left, right = st.columns(2)
with left:
    year = st.number_input(
        "Manufacturing year",
        min_value=1992,
        max_value=int(bundle["reference_year"]),
        value=min(2016, int(bundle["reference_year"])),
        step=1,
        help="The model was trained on vehicles up to {0}.".format(
            bundle["reference_year"]
        ),
    )
    km_driven = st.number_input(
        "Kilometres driven", min_value=0, max_value=1000000, value=50000, step=1000
    )
    fuel = st.selectbox("Fuel type", FUEL_OPTIONS)
with right:
    seller_type = st.selectbox("Seller type", SELLER_OPTIONS)
    transmission = st.selectbox("Transmission", TRANSMISSION_OPTIONS)
    owner = st.selectbox("Owner", OWNER_OPTIONS)

if st.button("Predict price", type="primary"):
    vehicle = {
        "name": name,
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }

    price = float(predict_price([vehicle], bundle)[0])
    st.success("Estimated selling price: {0}".format(rupees(price)))

    mae = metrics.get("MAE")
    if mae:
        st.write(
            "Typical error on the test set is {0}, so treat this as a range of "
            "roughly {1} to {2}.".format(
                rupees(mae), rupees(max(price - mae, 0)), rupees(price + mae)
            )
        )

    derived = explain_inputs([vehicle], bundle)
    st.subheader("Features the model actually saw")
    st.dataframe(
        pd.DataFrame(
            {
                "feature": ["car_age", "km_per_year", "name_frequency", "brand",
                            "model_family"],
                "value": [
                    int(derived["car_age"].iloc[0]),
                    round(float(derived["km_per_year"].iloc[0]), 1),
                    int(derived["name_frequency"].iloc[0]),
                    derived["brand"].iloc[0],
                    derived["model_family"].iloc[0],
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    if int(derived["name_frequency"].iloc[0]) == 0:
        st.warning(
            "This exact car name does not appear in the training data. The "
            "prediction falls back on brand, model family and the numeric "
            "features, and is less reliable."
        )

    st.info(
        "This is a machine-learning estimate from historical CarDekho "
        "listings. It does not account for condition, service history, "
        "accident record or city, and it is not a valuation."
    )
