# Machine Learning Tasks

**Author:** Islambek Ziyash  

## Overview

Two machine learning assignments covering the full ML pipeline: data preprocessing, model training, hyperparameter tuning, evaluation, and generating predictions on held-out evaluation sets.

## Project Structure

```

01 hm/
  ├── homework_01.ipynb   # Binary classification (Titanic survival)
  ├── data.csv            # Training data
  ├── evaluation.csv      # Unlabelled evaluation data
  └── results.csv         # Predicted labels (submission)
02 hm/
  ├── homework_02.ipynb   # Regression (Life expectancy prediction)
  ├── data.csv            # Training data
  ├── evaluation.csv      # Unlabelled evaluation data
  └── results.csv         # Predicted values (submission)
```

---

## Homework 1 — Binary Classification

**Task:** Predict passenger survival on the Titanic (`survived`: 0 / 1).

### Preprocessing
- Dropped high-cardinality / irrelevant columns: `ID`, `cabin`, `home.dest`, `ticket`, `name`
- Median imputation for `age` and `fare`
- `StandardScaler` normalization for numerical features
- `LabelEncoder` for `sex`; one-hot encoding for `embarked`
- Stratified 70 / 15 / 15 train / validation / test split

### Models & Tuning
| Model | Hyperparameter search |
|---|---|
| Decision Tree | `GridSearchCV` over `max_depth` (1–29), `min_samples_leaf` (1, 5, 10) |
| K-Nearest Neighbours | `GridSearchCV` over `n_neighbors` (1–29), `weights` (uniform / distance) |

Both tuned with 5-fold cross-validation, optimising **F1 score**.

### Evaluation
- F1 score and ROC-AUC reported on validation set
- ROC curves plotted for both models
- Best model used to generate `results.csv` on evaluation data

---

## Homework 2 — Regression

**Task:** Predict life expectancy (continuous value) from WHO country health indicators.

### Preprocessing
- KNN imputation (`n_neighbors=5`) for missing numerical values
- `Status` column encoded as binary: `Developed → 1`, `Developing → 0`
- Dropped non-predictive columns: `Country`, `Year`
- `StandardScaler` normalization
- 70 / 15 / 15 train / validation / test split

### Models
| Model | Notes |
|---|---|
| **CustomRandomForest** | Implemented from scratch using `DecisionTreeRegressor` as base estimators with bootstrap sampling |
| Linear Regression | Baseline sklearn implementation |
| Ridge Regression | L2 regularization (`alpha=1.0`) |
| KNN Regressor | sklearn implementation |

### CustomRandomForest
A custom Random Forest implementation built on top of sklearn's `DecisionTreeRegressor`:
- Configurable `n_estimators`, `max_samples` (bootstrap fraction), `max_depth`
- Bootstrap sampling via `sklearn.utils.resample`
- Prediction by averaging base estimator outputs

### Evaluation
- Metrics: **RMSE** and **MAE** on validation set
- All four models compared; `CustomRandomForest` selected as the final model
- Predictions saved to `results.csv`

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running

Open each notebook in Jupyter and run all cells in order. Both notebooks expect `data.csv` and `evaluation.csv` to be present in the same directory as the notebook.

```bash
jupyter notebook "01 hm/homework_01.ipynb"
jupyter notebook "02 hm/homework_02.ipynb"
```
