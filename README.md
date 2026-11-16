# Cyber 2: Hyperparameter Tuning for PhishGuard

This project tunes the machine learning model behind the PhishGuard phishing URL detector (Assignment 1) and compares the tuned model with the original.

## What it does

1. Builds the same 23 URL-only features the app uses (`features.py`).
2. Trains the **baseline**: `HistGradientBoostingClassifier` with default settings and one-hot encoding of the TLD.
3. Runs **RandomizedSearchCV** (30 random combinations, 3-fold cross-validation on the training set only, scored on F1 of the phishing class).
4. Evaluates baseline and tuned models on the same untouched test set.
5. Saves the tuned model, a results table and a comparison chart.

## Files

| File | Purpose |
|---|---|
| `cyber2_tuning.py` | Main script (can also be run cell by cell in VS Code or Spyder via the `# %%` markers) |
| `cyber2_hyperparameter_tuning.ipynb` | Same code as a Jupyter / Colab notebook |
| `features.py` | URL feature extraction (shared with the app) |
| `PhiUSIIL_Phishing_URL_Dataset.csv` | Dataset (you add this; not included) |

## Setup

```bash
pip install scikit-learn==1.9.1 pandas numpy scipy matplotlib joblib
```

Keep `scikit-learn==1.9.1`. It matches `requirements.txt`, and a model saved with a different version may not load in the app.

Put `features.py` and `PhiUSIIL_Phishing_URL_Dataset.csv` in the same folder as the script.

## Run

```bash
python cyber2_tuning.py
```

- Feature building takes about a minute.
- The search can take 10 to 20 minutes. Lower `N_ITER` (default 30) to 10 if your machine is slow.

## Outputs

| File | Contents |
|---|---|
| `url_model_tuned.pkl` | Tuned pipeline, a drop-in replacement for `url_model.pkl` |
| `cyber2_results.csv` | Baseline vs tuned metrics table |
| `cyber2_comparison.png` | Metric bars, error counts and ROC curve |

## Tuned hyperparameters

| Parameter | Search range |
|---|---|
| `learning_rate` | 0.02 to 0.3 (log scale) |
| `max_iter` | 200, 400, 600, 800 |
| `max_leaf_nodes` | 15, 31, 63, 127 |
| `max_depth` | None, 6, 10, 15 |
| `min_samples_leaf` | 10, 20, 50, 100 |
| `l2_regularization` | 0.001 to 10 (log scale) |
| TLD `min_frequency` | 5, 20, 50, 200 |

## Metrics reported

Phishing is treated as the positive class (`label = 0` in PhiUSIIL; `label = 1` is legitimate, matching `app.py`).

- **Accuracy, precision, recall, F1** for the phishing class
- **ROC-AUC and PR-AUC**
- **Missed phishing (FN):** phishing URLs marked safe (the dangerous error)
- **False alarms (FP):** legitimate URLs marked phishing
- **Train F1 vs test F1:** a large gap suggests overfitting
- **Fit time**

## Comparing with Assignment 1

The script re-trains the default model on the same split, so the comparison is fair. If your Assignment 1 numbers came from a different split, paste them into the `PREV` variable in the script and they will appear as an extra row.

## Using the tuned model in the app

1. Copy `url_model_tuned.pkl` into the app folder.
2. Either rename it to `url_model.pkl` or change the filename in `load_model()` in `app.py`.
3. No other changes are needed, because the input features and label encoding are the same.

## Notes and limitations

- PhiUSIIL URL features are already very predictive, so the improvement from tuning may be small.
- The dataset may contain duplicate URLs, which can slightly inflate test scores. Dropping duplicates before splitting is a possible next step.
- The model sees only URL text. The app's live checks (DNS, WHOIS, SSL, page content) cover what the URL alone cannot show.
