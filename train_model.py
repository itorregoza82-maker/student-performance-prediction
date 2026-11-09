"""
Student Performance Prediction System
Activity A: retrieve and inspect the UCI Mathematics dataset.
Activity B: preprocess, train, evaluate, and save a Random Forest model.

Run from this folder:
    python train_model.py
"""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"
METRICS_PATH = BASE_DIR / "model_evaluation.json"
REPORT_PATH = BASE_DIR / "classification_report.txt"
RANDOM_STATE = 42


def retrieve_dataset() -> pd.DataFrame:
    """Download UCI dataset 320 and retain the Mathematics records."""
    print("\nACTIVITY A — DATASET RETRIEVAL")
    print("Fetching UCI Student Performance dataset (ID 320)...")
    try:
        student = fetch_ucirepo(id=320)
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve the UCI dataset. Check your internet connection "
            "and try again. The ucimlrepo package must also be installed."
        ) from exc

    features = student.data.features
    targets = student.data.targets

    # The UCI repository provides separate Mathematics and Portuguese files.
    # fetch_ucirepo(id=320) exposes the Mathematics dataset as the primary data.
    df = pd.concat([features, targets], axis=1)

    required_grades = {"G1", "G2", "G3"}
    missing_grades = required_grades.difference(df.columns)
    if missing_grades:
        raise ValueError(
            "The downloaded dataset is missing required grade columns: "
            + ", ".join(sorted(missing_grades))
        )

    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset: {DATA_PATH.name}")
    print("Rows and columns:", df.shape)
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print("\nConfirmed grade columns:", ", ".join(["G1", "G2", "G3"]))
    return df


def build_and_train(df: pd.DataFrame) -> dict:
    """Clean the data, train a leakage-safe pipeline, and report metrics."""
    print("\nACTIVITY B — PREPROCESSING, TRAINING, AND EVALUATION")
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Duplicate rows removed: {before - len(df)}")
    if "G3" not in df.columns:
        raise ValueError("G3 is required to create the prediction target.")

    # 1 = AT RISK (G3 < 10); 0 = NOT AT RISK (G3 >= 10)
    y = (pd.to_numeric(df["G3"], errors="coerce") < 10).astype(int)
    valid_target = pd.to_numeric(df["G3"], errors="coerce").notna()
    df = df.loc[valid_target].copy()
    y = y.loc[valid_target]

    # G3 is deliberately excluded to prevent target leakage.
    X = df.drop(columns=["G3"])
    if "G3" in X.columns:
        raise AssertionError("Target leakage detected: G3 is in the input features.")

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    print("Numerical columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print("Input feature count:", X.shape[1])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Positive class is 1 = AT RISK, so precision/recall/F1 refer to that class.
    metrics = {
        "dataset_rows_after_duplicate_removal": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "input_feature_count": int(X.shape[1]),
        "target_definition": {"1": "AT RISK (G3 < 10)", "0": "NOT AT RISK (G3 >= 10)"},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_at_risk": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": ["NOT AT RISK", "AT RISK"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["NOT AT RISK", "AT RISK"],
            zero_division=0,
            output_dict=True,
        ),
    }

    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["NOT AT RISK", "AT RISK"],
        zero_division=0,
    )
    print("\nMODEL EVALUATION")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision (AT RISK): {metrics['precision_at_risk']:.3f}")
    print(f"Recall (AT RISK):    {metrics['recall_at_risk']:.3f}")
    print(f"F1-score (AT RISK):  {metrics['f1_at_risk']:.3f}")
    print("\nClassification report:\n", report_text)
    print("Confusion matrix (rows=true, columns=predicted; NOT AT RISK, AT RISK):")
    print(metrics["confusion_matrix"])

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved evaluation metrics: {METRICS_PATH.name}")
    print(f"Saved classification report: {REPORT_PATH.name}")
    return metrics


def main():
    try:
        data = retrieve_dataset()
        build_and_train(data)
        print("\nSUCCESS: Activity A and Activity B completed.")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
