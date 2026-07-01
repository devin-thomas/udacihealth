# UdaciHealth

Deep learning project 1 for the Udacity Deep Learning Nanodegree. The project trains and evaluates a PyTorch multi-layer perceptron that screens for diabetes risk from CDC health indicators, then extends the baseline with threshold tuning, feature engineering, and class-weighted training on the original imbalanced population.

## Highlights

- Built a full tabular deep-learning workflow in `starter-kit/diabetes_prediction_mlp.ipynb`
- Completed all rubric TODOs and executed the notebook end to end
- Tuned the screening threshold to lift balanced-test recall from `0.8070` to `0.8974`
- Re-ran the model on the original `253,680`-row CDC dataset and used class weighting to improve default-threshold recall from `0.1884` to `0.8014`
- Added a GitHub Pages landing site plus a static HTML export of the final notebook

## Results

Balanced test set:

- Baseline: accuracy `0.7483`, precision `0.7222`, recall `0.8070`, F1 `0.7623`, ROC-AUC `0.8249`
- Best screening configuration: threshold-tuned deep architecture at `0.35`, precision `0.6720`, recall `0.8974`, F1 `0.7685`, ROC-AUC `0.8258`

Original imbalanced CDC set:

- Unweighted model at `0.50`: accuracy `0.8514`, recall `0.1884`
- Class-weighted model at `0.50`: accuracy `0.7116`, recall `0.8014`

## Repository Layout

```text
.
├── docs/
│   ├── index.html
│   └── notebook.html
├── scripts/
│   └── update_notebook.py
├── starter-kit/
│   ├── data/
│   │   ├── data_dictionary.md
│   │   ├── diabetes_data.csv
│   │   └── diabetes_012_health_indicators_BRFSS2015.csv
│   └── diabetes_prediction_mlp.ipynb
├── .github/workflows/pages.yml
├── .gitignore
└── requirements.txt
```

## Setup

Use Python `3.12` or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the notebook locally:

```bash
jupyter notebook starter-kit/diabetes_prediction_mlp.ipynb
```

Rebuild the notebook source programmatically if needed:

```bash
python scripts/update_notebook.py
```

Export a fresh HTML artifact:

```bash
jupyter nbconvert --to html --output notebook.html --output-dir docs starter-kit/diabetes_prediction_mlp.ipynb
```

## Data Sources

- Balanced instructional dataset: `starter-kit/data/diabetes_data.csv`
- Original CDC dataset: `starter-kit/data/diabetes_012_health_indicators_BRFSS2015.csv`
- Source reference: [CDC Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Public mirror used for the imbalanced CSV: [JavaConterry/Data_Analysis](https://github.com/JavaConterry/Data_Analysis/blob/main/diabetes_012_health_indicators_BRFSS2015.csv)

## GitHub Pages

The repository includes a static Pages landing site in `docs/` and an Actions workflow in `.github/workflows/pages.yml`. After deployment, the site should be available at:

`https://devin-thomas.github.io/udacihealth/`
