# Car Price Prediction

Chaky Company sells used cars and was pricing them by gut feeling, so similar cars kept ending up
with very different price tags. This project builds a model that gives the sales team a starting
number before they negotiate.

It comes in two parts:

1. `notebooks/01_car_price_prediction.ipynb` does the EDA, preprocessing, model selection and tuning,
   and saves the final pipeline.
2. `app/` is a small Plotly Dash site that loads that pipeline and predicts a price from whatever
   the user types in.

## Screenshots

Landing page, explaining the problem and how the model was chosen:

![Home page](figures/app/home.png)

Prediction form. Year and max power are required, everything else can be left blank and is imputed:

![Predict page](figures/app/predict.png)

## Results

The dataset is 8,128 used-car listings. Preprocessing drops 1,202 duplicates, the 5 Test Drive Cars
and the CNG/LPG rows, converts mileage/max_power zeros into proper missing values, removes 3 cars
with impossible odometer readings, and folds 13 one-off brands into `Other`. That leaves 6,805 cars.
The target is modelled as `log(selling_price)`. There is a section below on why each of those
corrections is there.

Model comparison, 5-fold cross-validation on train+validation (MAE on the log target, lower is
better):

| Model | CV log MAE | Std |
| --- | --- | --- |
| Random Forest | 0.165 | 0.004 |
| Ridge | 0.198 | 0.005 |
| Baseline (median) | 0.595 | 0.006 |

A 24-combination grid search tuned the Random Forest to 200 trees, `max_depth=20`,
`min_samples_leaf=1`, `max_features='sqrt'` (CV log MAE 0.164). Scored once on the held-out test set:

| Metric | Value |
| --- | --- |
| Log RMSE | 0.223 |
| MAE | 75,747 |
| RMSE | 251,921 |
| R² (price scale) | 0.749 |
| Cars priced within 20% of actual | 73.5% |

`year` and `max_power` dominate the prediction on both built-in and permutation importance, then
`engine`, `fuel` and `brand`. R² is measured on the price scale, where a thin tail of luxury cars
carries most of the squared error, so it looks worse than the log-scale numbers the model was
actually fitted on. Section 10 of the notebook goes into this.

## Data cleaning, and why each step is there

The raw file has problems that do not show up in `df.isna().sum()`, so each of these came out of
actually looking at the values.

**Zeros that mean "missing".** 15 cars have `mileage == 0` and 3 have `max_power == 0`. Their other
specs are populated, so the rows are not blank: a petrol Santro was simply recorded as doing 0 kmpl.
This matters because `SimpleImputer` only replaces `NaN`. A `0.0` is a valid float, so it passes
straight through imputation and the model learns it as a real measurement. Converting them to `NaN`
first hands them to the imputer the pipeline already runs.

**A flag for imputed values.** The four spec columns are missing on the same 198 rows, and those
rows are not a random sample: median price 180,000 against 422,000 for the rest, median year 2008
against 2014. Filling them with the median tells the model they are average cars and throws that
signal away, so `SimpleImputer(add_indicator=True)` keeps a "was missing" column. It earns its place
in the final feature importances.

**Impossible odometer readings.** The 99.9th percentile of `km_driven` is about 428,000 km; the
largest value in the file is 2,360,457 km, roughly 59 times around the Earth. Three rows sit above
500,000 km and are dropped.

**Brand parsing.** Taking the first word of `name` turns "Land Rover Freelander" into the brand
`Land` and "Ashok Leyland Stile" into `Ashok`, which invents makes that do not exist. Those two are
the only two-word brands here and are now handled explicitly.

**Rare brands.** 13 brands have fewer than 10 cars, and Lexus, Opel and Peugeot appear once each.
Each became a one-hot column fitted to a single example, so they are grouped into `Other`.
`handle_unknown="ignore"` on the encoder already covered brands never seen in training; this covers
brands seen too rarely to learn from.

Honest note on what this bought: these corrections move cross-validated log MAE by about 0.004 on
Random Forest and 0.001 on Ridge, both close to fold-to-fold noise. Trees are robust to outliers by
construction, the log transform on the target already compresses the extremes, and 18 bad rows are
0.26% of the data. The reason to do them is correctness, not the metric: a model that has learned
"0 kmpl" as a real value will produce a nonsense quote for the one customer who owns that car.

## Run the notebook

Python 3.13, with pandas, numpy, matplotlib, seaborn, scikit-learn and jupyter installed in `.venv`.
Select that virtual environment as the kernel and open:

```text
notebooks/01_car_price_prediction.ipynb
```

Running it end to end refits the pipeline (imputation, one-hot encoding, grid-searched Random
Forest) and saves it to `app/models/car_price_model.joblib`.

The model is saved with `joblib.dump(..., compress=("zlib", 6))` rather than a plain pickle. The
tuned forest is about 82 MB as a raw pickle, which is under GitHub's 100 MB hard limit but over the
50 MB point where it starts warning, and it bloats every clone. Compressed it comes to 18 MB and the
trees load back byte for byte identical. An earlier 400-tree version was 152 MB, over the limit
outright, which is what prompted the change.

## Run the Dash app locally

```bash
./.venv/bin/pip install -r app/requirements.txt
./.venv/bin/python app/code/app.py
```

Then open `http://127.0.0.1:8050`. There are two pages:

- `/` explains the problem, the preprocessing, and why Random Forest was picked.
- `/predict` is the form. Year and max power are required because they carry most of the prediction.
  Anything else can be left blank and the pipeline imputes it.

## Run with Docker

```bash
docker compose -f app/docker-compose.yaml up --build
```

Same address, `http://127.0.0.1:8050`.

## Tests

```bash
./.venv/bin/pip install -r app/requirements.txt pytest
./.venv/bin/python -m pytest
```

`tests/test_model.py` checks that the saved pipeline loads, still exposes the categories the
dropdowns are built from, and returns sensible prices, including when the optional fields are blank
and the imputer has to fill them in.

`tests/test_app.py` checks that the Dash app builds, both pages register, and the predict callback
refuses to guess when year or max power is missing.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request, in three stages:

1. Install `app/requirements.txt` plus pytest on Python 3.13 and run the tests.
2. Build the image from `app/.Dockerfile`, start the container, and check that `/`, `/predict` and
   the Dash callback registry all respond.
3. On pushes to `main`, log in to Docker Hub and push the image that just passed those checks, as
   `<user>/car-price-predictor:latest` and `:<sha>`. Pull requests build and test the image but
   never push it.

GitHub Actions does the build, not Docker Hub, so the image is only published after the tests pass.
Docker Hub is just the registry.

Before the first push to `main`, add these under Settings, Secrets and variables, Actions:

| Name | Kind | Value |
| --- | --- | --- |
| `DOCKERHUB_USERNAME` | secret | your Docker Hub username |
| `DOCKERHUB_TOKEN` | secret | a Docker Hub access token with Read & Write permission |
| `DOCKERHUB_REPOSITORY` | variable, optional | `user/repo`, if you want a name other than `<user>/car-price-predictor` |
