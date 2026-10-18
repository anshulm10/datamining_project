"""INFS7203 Track 1 final program, student number 49769959.

Run from the submission folder with:  python main.py

1. Reads data/development.csv and data/test.csv (the files are not modified).
2. Recomputes the development cross-validation metrics of the final, fixed pipeline.
3. Fits the same pipeline on all development rows and predicts every test row in file order.
4. Writes s4976995.infs7203: 6,000 "yes,"/"no," rows and one metric row, no header.

The pipeline and parameters were chosen in development.ipynb; nothing is searched here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# The eight-digit student number and the confirmed login username are different.
STUDENT_ID = "49769959"
STUDENT_USERNAME = "s4976995"

PROJECT_DIR = Path(__file__).resolve().parent
DEVELOPMENT_PATH = PROJECT_DIR / "data" / "development.csv"
TEST_PATH = PROJECT_DIR / "data" / "test.csv"
OUTPUT_PATH = PROJECT_DIR / f"{STUDENT_USERNAME}.infs7203"

TARGET = "defaulted_next_month"
POSITIVE_CLASS = "yes"
RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5  # predict "yes" when the forest's P(yes) >= THRESHOLD

# Predictor groups, in the same column order as development.ipynb (client_id is excluded).
NUMERIC_COLUMNS = [
    "credit_limit", "age",
    "bill_amount_mar", "bill_amount_feb", "bill_amount_jan",
    "bill_amount_dec", "bill_amount_nov", "bill_amount_oct",
    "payment_amount_mar", "payment_amount_feb", "payment_amount_jan",
    "payment_amount_dec", "payment_amount_nov", "payment_amount_oct",
]
DEMOGRAPHIC_COLUMNS = ["sex", "education", "marital_status"]
REPAYMENT_COLUMNS = [
    "repayment_status_mar", "repayment_status_feb", "repayment_status_jan",
    "repayment_status_dec", "repayment_status_nov", "repayment_status_oct",
]
PREDICTOR_COLUMNS = NUMERIC_COLUMNS + DEMOGRAPHIC_COLUMNS + REPAYMENT_COLUMNS

# Ordinal codes 0-10. Delays follow their natural order; the order of the first three labels is an assumption.
STATUS_ORDER = [
    "no_balance", "paid_on_time", "revolving_credit",
    "payment_delayed_1_month", "payment_delayed_2_months", "payment_delayed_3_months",
    "payment_delayed_4_months", "payment_delayed_5_months", "payment_delayed_6_months",
    "payment_delayed_7_months", "payment_delayed_8_months",
]

# Order of the five metrics in the last row of the output file.
OUTPUT_METRICS = ["accuracy", "precision_yes", "recall_yes", "f1_yes", "macro_f1"]


def build_pipeline():
    """The final preprocessing and model; every learned step is fitted only on the data passed to fit()."""
    # Numeric NaNs are retained: each tree learns their routing from its training data.
    demographic_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    # Encode known statuses in their stated order; preserve missing/unknown status as NaN.
    repayment_steps = OrdinalEncoder(
        categories=[STATUS_ORDER + [np.nan]] * len(REPAYMENT_COLUMNS),
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            ("demographic", demographic_steps, DEMOGRAPHIC_COLUMNS),
            ("repayment", repayment_steps, REPAYMENT_COLUMNS),
        ],
        sparse_threshold=0,
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth=10,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", forest)])


def predict_labels(model, features):
    """Predict "yes" when P(yes) >= THRESHOLD, otherwise "no" (same rule as the notebook)."""
    yes_column = list(model.classes_).index(POSITIVE_CLASS)
    yes_probability = model.predict_proba(features)[:, yes_column]
    return np.where(yes_probability >= THRESHOLD, POSITIVE_CLASS, "no")


def score_labels(true_labels, predicted_labels):
    """Accuracy, precision/recall/F1 for "yes", macro-F1 and balanced accuracy for one fold."""
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision_yes": precision_score(true_labels, predicted_labels, pos_label=POSITIVE_CLASS),
        "recall_yes": recall_score(true_labels, predicted_labels, pos_label=POSITIVE_CLASS),
        "f1_yes": f1_score(true_labels, predicted_labels, pos_label=POSITIVE_CLASS),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
    }


def cross_validated_scores(features, target):
    """Held-out scores for each of the 5 stratified folds (shuffled, seed 42)."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    for train_index, valid_index in folds.split(features, target):
        # A new, unfitted pipeline is fitted on the training fold only.
        model = build_pipeline().fit(features.iloc[train_index], target.iloc[train_index])
        predicted = predict_labels(model, features.iloc[valid_index])
        fold_rows.append(score_labels(target.iloc[valid_index].to_numpy(), predicted))
    return pd.DataFrame(fold_rows, index=pd.RangeIndex(1, N_SPLITS + 1, name="fold"))


def write_output(test_labels, metric_means):
    """Write the 6,001-row file: one "label," row per test client, then the five rounded metric means."""
    lines = [f"{label}," for label in test_labels]
    lines.append(",".join(f"{metric_means[metric]:.3f}" for metric in OUTPUT_METRICS) + ",")
    with open(OUTPUT_PATH, "w", newline="\n") as output_file:
        output_file.write("\n".join(lines) + "\n")
    return lines


def main():
    development = pd.read_csv(DEVELOPMENT_PATH)
    test = pd.read_csv(TEST_PATH)
    X_development = development[PREDICTOR_COLUMNS]
    y_development = development[TARGET]

    # 1. Development cross-validation of the fixed final pipeline.
    fold_scores = cross_validated_scores(X_development, y_development)
    metric_means = fold_scores.mean()
    print("Held-out score per fold:")
    print(fold_scores.round(3).to_string())
    print("\nMean and sample standard deviation (ddof=1) over the 5 folds:")
    print(pd.DataFrame({"mean": metric_means, "std": fold_scores.std()}).round(3).to_string())

    # 2. Fit on all development rows, then predict the test rows in their original order.
    final_model = build_pipeline().fit(X_development, y_development)
    test_labels = predict_labels(final_model, test[PREDICTOR_COLUMNS])

    # 3. Validate before replacing an existing output, then write the file.
    assert len(test_labels) == len(test) == 6000, "expected one prediction per test row"
    assert set(test_labels) <= {"yes", "no"}, "labels must be yes/no"
    assert np.isfinite(metric_means).all(), "validation metrics must be finite"
    lines = write_output(test_labels, metric_means)
    assert len(lines) == 6001, "output must have exactly 6,001 rows"
    print(f"\nWrote {OUTPUT_PATH.name}: {len(lines)} rows "
          f"({(test_labels == POSITIVE_CLASS).sum()} 'yes', {(test_labels == 'no').sum()} 'no')")
    print("Last row:", lines[-1])


if __name__ == "__main__":
    main()
