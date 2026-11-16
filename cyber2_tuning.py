
import time, warnings
import numpy as np, pandas as pd, joblib, sklearn
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, make_scorer)

from features import extract_features      

warnings.filterwarnings("ignore")
print("scikit-learn:", sklearn.__version__)


DATA_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
print(df.shape, "\n", df["label"].value_counts())

t0 = time.time()
X = pd.DataFrame([extract_features(u) for u in df["URL"]])
y = df["label"].astype(int).reset_index(drop=True)
print(f"features built in {time.time()-t0:.0f}s ->", X.shape)
print(X.head())


RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(X_train.shape, X_test.shape)


def build_pipeline(min_freq=20, **hgb_params):
    prep = ColumnTransformer(
        [("tld", OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq,
                               sparse_output=False), ["TLD"])],
        remainder="passthrough")
    return Pipeline([("prep", prep),
                     ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **hgb_params))])

def evaluate(name, pipe, fit_seconds):
    pred_te = pipe.predict(X_test)
    proba_te = pipe.predict_proba(X_test)[:, 1]            
    cm = confusion_matrix(y_test, pred_te, labels=[0, 1])  
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred_te),
        "precision_phish": precision_score(y_test, pred_te, pos_label=0),
        "recall_phish": recall_score(y_test, pred_te, pos_label=0),
        "f1_phish": f1_score(y_test, pred_te, pos_label=0),
        "roc_auc": roc_auc_score(y_test, proba_te),
        "pr_auc_phish": average_precision_score(1 - y_test, 1 - proba_te),
        "missed_phishing (FN)": int(cm[0, 1]),
        "false_alarms (FP)": int(cm[1, 0]),
        "train_f1": f1_score(y_train, pipe.predict(X_train), pos_label=0),
        "fit_seconds": round(fit_seconds, 1),
    }, cm, proba_te


baseline = build_pipeline()
t0 = time.time(); baseline.fit(X_train, y_train); t_base = time.time() - t0
base_res, base_cm, base_proba = evaluate("Baseline (defaults)", baseline, t_base)
print(pd.Series(base_res))


N_ITER = 30     
CV_FOLDS = 3

base_for_search = build_pipeline(early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)

param_dist = {
    "model__learning_rate":     loguniform(0.02, 0.3),
    "model__max_iter":          [200, 400, 600, 800],
    "model__max_leaf_nodes":    [15, 31, 63, 127],
    "model__max_depth":         [None, 6, 10, 15],
    "model__min_samples_leaf":  [10, 20, 50, 100],
    "model__l2_regularization": loguniform(1e-3, 10),
    "prep__tld__min_frequency": [5, 20, 50, 200],
}

search = RandomizedSearchCV(
    base_for_search, param_dist, n_iter=N_ITER,
    scoring=make_scorer(f1_score, pos_label=0),
    cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE, n_jobs=1, verbose=2, refit=True, return_train_score=True)

t0 = time.time(); search.fit(X_train, y_train); t_search = time.time() - t0
print(f"search finished in {t_search/60:.1f} min")
print("Best CV F1 (phishing):", round(search.best_score_, 5))
print("Best params:")
for k, v in search.best_params_.items(): print("  ", k, "=", v)


cv = pd.DataFrame(search.cv_results_)
cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time"] + \
       [c for c in cv.columns if c.startswith("param_")]
print(cv.sort_values("rank_test_score")[cols].head(10).round(5).to_string())


tuned = search.best_estimator_

t0 = time.time(); tuned.fit(X_train, y_train); t_tuned = time.time() - t0
tuned_res, tuned_cm, tuned_proba = evaluate("Tuned (RandomizedSearchCV)", tuned, t_tuned)

rows = [base_res, tuned_res]


PREV = None 
if PREV: rows.insert(0, PREV)

comparison = pd.DataFrame(rows).set_index("model")
print(comparison.T)


metric_cols = ["accuracy", "precision_phish", "recall_phish", "f1_phish", "roc_auc", "pr_auc_phish"]
delta = (comparison.loc["Tuned (RandomizedSearchCV)", metric_cols]
         - comparison.loc["Baseline (defaults)", metric_cols])
out = pd.DataFrame({"baseline": comparison.loc["Baseline (defaults)", metric_cols],
                    "tuned": comparison.loc["Tuned (RandomizedSearchCV)", metric_cols],
                    "change": delta})
print("Error counts on the test set:")
print(comparison[["missed_phishing (FN)", "false_alarms (FP)"]])
print(out.round(5))



# %%
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))

# (a) metric bars (zoomed, since all scores are near 1.0)
w = 0.38; idx = np.arange(len(metric_cols))
ax[0].bar(idx - w/2, out["baseline"], w, label="Baseline")
ax[0].bar(idx + w/2, out["tuned"], w, label="Tuned")
ax[0].set_xticks(idx); ax[0].set_xticklabels(metric_cols, rotation=30, ha="right")
lo = min(out["baseline"].min(), out["tuned"].min())
ax[0].set_ylim(max(lo - 0.01, 0), 1.0005); ax[0].set_title("Metrics (y-axis zoomed)"); ax[0].legend()

# (b) confusion-matrix error counts
labels = ["Missed phishing\n(FN)", "False alarms\n(FP)"]
b = [base_cm[0, 1], base_cm[1, 0]]; t = [tuned_cm[0, 1], tuned_cm[1, 0]]
ax[1].bar(np.arange(2) - w/2, b, w, label="Baseline"); ax[1].bar(np.arange(2) + w/2, t, w, label="Tuned")
for i, (bv, tv) in enumerate(zip(b, t)):
    ax[1].text(i - w/2, bv, bv, ha="center", va="bottom"); ax[1].text(i + w/2, tv, tv, ha="center", va="bottom")
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(labels); ax[1].set_title("Errors on test set (lower is better)"); ax[1].legend()

# (c) ROC curve
for name, p in [("Baseline", base_proba), ("Tuned", tuned_proba)]:
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[2].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.5f})")
ax[2].plot([0, 1], [0, 1], "k--", alpha=.3); ax[2].set_xlim(0, 0.05); ax[2].set_ylim(0.95, 1.0)
ax[2].set_xlabel("False positive rate"); ax[2].set_ylabel("True positive rate"); ax[2].set_title("ROC (zoomed to top-left)"); ax[2].legend()

import matplotlib
plt.tight_layout(); plt.savefig("cyber2_comparison.png", dpi=150); plt.show()



joblib.dump(tuned, "url_model_tuned.pkl")
comparison.to_csv("cyber2_results.csv")
print("saved: url_model_tuned.pkl, cyber2_results.csv, cyber2_comparison.png")


chk = joblib.load("url_model_tuned.pkl")
print(chk.predict_proba(pd.DataFrame([extract_features("http://paypal-login.secure-update.xyz/verify?id=123")]))[0])


