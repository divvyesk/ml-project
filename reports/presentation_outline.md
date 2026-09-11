# Week 15 - Final Presentation Outline

Used Car Price Prediction Using Machine Learning (SEML3211)

Twelve slides, roughly one minute each, then the live demo.

## Slide 1 - Title

- Used Car Price Prediction Using Machine Learning
- Vaibhaw Ojha (24SE02ML287), Yaahvi Parekh (24SE02ML315)
- Branch MLAI5D - Foundation of Machine Learning (SEML3211)

## Slide 2 - Problem

- Manual used-car valuation varies 15-30% between dealers for the same vehicle.
- Buyers cannot tell whether a listed price is fair; sellers have no consistent way to set one.
- Goal: a data-driven price estimate from vehicle specifications alone.

## Slide 3 - Dataset

- CarDekho vehicle dataset (Kaggle), 4,340 raw records, 8 columns.
- 763 exact duplicates removed, leaving 3,577 unique records.
- Target: selling_price, continuous, so this is a regression problem.

## Slide 4 - EDA findings

- Selling price is strongly right-skewed; the log transform makes it close to symmetric.
- Year correlates positively with price; kilometres driven correlates negatively.
- Diesel and automatic vehicles command higher average prices.

## Slide 5 - Feature engineering

- car_age, brand, model_family, km_per_year, name_frequency.
- name_frequency is fitted on the training split only, so no test information leaks into training.
- Reference year 2020 is stored with the model, not recomputed at prediction time.

## Slide 6 - Method

- Split first, then engineer features, then fit inside an sklearn Pipeline.
- All estimators fit on log1p(price) and invert on prediction.
- Hyper-parameters chosen by 5-fold cross-validation on the training split.

## Slide 7 - Model comparison

- Five algorithms compared: Linear Regression, Ridge, Decision Tree, Random Forest, XGBoost.
- Plus a stacked ensemble of the tuned models.
- Show reports/plots/model_comparison.png.

## Slide 8 - Results

- Final model: Stacked Ensemble.
- Test R2 0.7229, MAE Rs 99,280, RMSE Rs 2,98,743.
- R2 on the log scale 0.8881; R2 excluding the top 1% of prices 0.8590.

## Slide 9 - Honest objective check

- Documented objective was R2 above 0.90. Status: NOT MET.
- The 90-95% figures in the literature use Present_Price, a column the supplied eight-column CSV does not contain.
- We report the measured score rather than tuning against the test set until the number looks right.

## Slide 10 - Feature importance

- Top drivers: brand, model_family, transmission, year, car_age.
- Matches the domain expectation that age and model line dominate resale value.
- Show reports/feature_importance.csv.

## Slide 11 - Live demo

- streamlit run app.py
- Enter a vehicle, show the predicted price and the derived features behind it.
- Fallback if Streamlit is unavailable: python scripts/week13_prototype_check.py

## Slide 12 - Limitations and future work

- No condition, service history, accident record or city.
- Luxury segment is thinly represented and drives most of the residual error.
- Next: richer schema (engine, power, mileage), then re-evaluate the 0.90 objective.

## Demo script

1. Start the app: `streamlit run app.py`.
2. Enter a common vehicle (Maruti Swift Dzire VDI, 2016, 50,000 km).
3. Point out the derived features shown beside the prediction.
4. Change the year to 2012 and show the price fall with age.
5. Change transmission to Automatic and show the price rise.
6. Enter a name the model has never seen and show that name_frequency becomes 0 and the prediction still works.
