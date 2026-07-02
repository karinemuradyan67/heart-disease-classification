"""
Heart Disease ML Trainer
========================
Trains multiple models, picks the best one, saves it to disk.
Run again with new data to retrain and improve.

Usage:
    python train.py                          # train on data/heart_disease.csv
    python train.py --data path/to/new.csv  # retrain with new data
    python train.py --predict               # predict using saved model (interactive)
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "heart_disease.csv"
DEFAULT_DATA_ARG = Path("data") / "heart_disease.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
HISTORY_PATH = MODEL_DIR / "training_history.json"

MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def resolve_data_path(csv_path: str | Path) -> Path:
    """Resolve dataset path relative to project root when needed."""
    path = Path(csv_path)
    return path if path.is_absolute() else BASE_DIR / path


def project_relative_path(path: str | Path) -> str:
    """Return a project-relative path string when possible for cleaner logs/history."""
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(p)

# ── Data helpers ─────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, clean, engineer features, return X and y."""
    df = pd.read_csv(csv_path)

    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Missing required column: {target_col}")

    # Coerce numeric fields and drop malformed rows (e.g., duplicated header rows).
    numeric_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", target_col,
    ]
    present_numeric_cols = [c for c in numeric_cols if c in df.columns]
    for col in present_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        print(f"  Note: detected {duplicate_count} duplicated rows; keeping them as requested.")

    before_clean = len(df)
    df.dropna(subset=[target_col], inplace=True)
    dropped_target = before_clean - len(df)
    if dropped_target:
        print(f"  Warning: dropped {dropped_target} rows with invalid target values.")

    # Fix columns stored as object due to '?' placeholders
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    before_dropna = len(df)
    df.dropna(inplace=True)
    dropped_other = before_dropna - len(df)
    if dropped_other:
        print(f"  Warning: dropped {dropped_other} rows with missing feature values after cleaning.")

    # Binarise target (0 = healthy, 1+ = disease)
    y = (df[target_col] > 0).astype(int)
    df.drop(columns=[target_col], inplace=True)

    # Feature engineering (same as notebook)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 70, 120],
        labels=["young", "middle", "senior", "elderly"],
    )
    df["high_bp"] = (df["trestbps"] > 140).astype(int)
    df["high_chol"] = (df["chol"] > 240).astype(int)
    df["stress_index"] = df["thalach"] / (df["trestbps"] + 1)

    df = pd.get_dummies(df, columns=["age_group"], drop_first=True)

    return df, y


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


# ── Models ───────────────────────────────────────────────────────────────────

CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNN (k=13)": KNeighborsClassifier(n_neighbors=13),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
}


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Train model and return metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cv = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
    )

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "cv_f1_mean": round(cv.mean(), 4),
        "cv_f1_std": round(cv.std(), 4),
    }


# ── Training ─────────────────────────────────────────────────────────────────

def train(csv_path: Path = DEFAULT_DATA_ARG):
    resolved_csv_path = resolve_data_path(csv_path)
    display_csv_path = project_relative_path(csv_path)
    print(f"\n{'='*60}")
    print(f"  Heart Disease ML Trainer")
    print(f"{'='*60}")
    print(f"  Data : {display_csv_path}")

    X, y = load_and_preprocess(resolved_csv_path)
    print(f"  Rows : {len(X)}   Features : {X.shape[1]}   Positive rate : {y.mean():.1%}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print(f"\n{'─'*60}")
    print(f"  {'Model':<25} {'Acc':>6} {'F1':>6} {'AUC':>6}  CV-F1")
    print(f"{'─'*60}")

    results = {}
    trained_models = {}
    for name, model in CANDIDATES.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        trained_models[name] = model
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "  —   "
        print(
            f"  {name:<25} {metrics['accuracy']:>6.4f} {metrics['f1']:>6.4f} "
            f"{auc_str:>6}  {metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}"
        )

    # Pick best by CV F1
    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\n  ✅ Best model : {best_name}")
    print(f"     Accuracy  : {best_metrics['accuracy']}")
    print(f"     F1 Score  : {best_metrics['f1']}")
    print(f"     ROC-AUC   : {best_metrics['roc_auc']}")
    print(f"     CV F1     : {best_metrics['cv_f1_mean']} ± {best_metrics['cv_f1_std']}")

    # Detailed report
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"\n{'─'*60}")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Disease"]))

    # Feature importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        feat_names = X.columns.tolist()
        importances = best_model.feature_importances_
        top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:5]
        print("  Top-5 important features:")
        for feat, imp in top:
            bar = "█" * int(imp * 40)
            print(f"    {feat:<20} {imp:.4f}  {bar}")

    # Save model + scaler
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Update training history
    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        for entry in history:
            if "data" in entry:
                entry["data"] = project_relative_path(entry["data"])

    history.append({
        "run": len(history) + 1,
        "data": display_csv_path,
        "samples": len(X),
        "best_model": best_name,
        **best_metrics,
    })
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Show improvement over previous runs
    if len(history) > 1:
        prev = history[-2]
        delta_f1 = best_metrics["f1"] - prev["f1"]
        delta_acc = best_metrics["accuracy"] - prev["accuracy"]
        arrow = "▲" if delta_f1 >= 0 else "▼"
        print(f"\n  📈 vs previous run — F1 {arrow}{abs(delta_f1):.4f}  Acc {arrow}{abs(delta_acc):.4f}")

    print(f"\n  Model saved → {project_relative_path(MODEL_PATH)}")
    print(f"  Scaler saved → {project_relative_path(SCALER_PATH)}")
    print(f"{'='*60}\n")

    return best_model, scaler, X.columns.tolist()


# ── Prediction ───────────────────────────────────────────────────────────────

FEATURE_PROMPTS = {
    "age": ("Age (years)", float),
    "sex": ("Sex (1=male, 0=female)", float),
    "cp": ("Chest pain type (1-4)", float),
    "trestbps": ("Resting blood pressure (mm Hg)", float),
    "chol": ("Cholesterol (mg/dl)", float),
    "fbs": ("Fasting blood sugar > 120 mg/dl (1=yes, 0=no)", float),
    "restecg": ("Resting ECG result (0-2)", float),
    "thalach": ("Max heart rate achieved", float),
    "exang": ("Exercise-induced angina (1=yes, 0=no)", float),
    "oldpeak": ("ST depression", float),
    "slope": ("Slope of peak exercise ST segment (1-3)", float),
    "ca": ("Number of major vessels (0-3)", float),
    "thal": ("Thalassemia (3=normal, 6=fixed defect, 7=reversable)", float),
}


def predict_interactive():
    """Run an interactive prediction session using saved model."""
    if not MODEL_PATH.exists():
        print("No saved model found. Run `python train.py` first.")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("\n  🔍 Heart Disease Prediction  (type 'q' to quit)\n")

    while True:
        values = {}
        for col, (prompt, dtype) in FEATURE_PROMPTS.items():
            while True:
                raw = input(f"  {prompt}: ").strip()
                if raw.lower() == "q":
                    return
                try:
                    values[col] = dtype(raw)
                    break
                except ValueError:
                    print("    ⚠ Please enter a valid number.")

        # Build DataFrame matching training features
        row = pd.DataFrame([values])
        row["age_group"] = pd.cut(
            row["age"],
            bins=[0, 40, 55, 70, 120],
            labels=["young", "middle", "senior", "elderly"],
        )
        row["high_bp"] = (row["trestbps"] > 140).astype(int)
        row["high_chol"] = (row["chol"] > 240).astype(int)
        row["stress_index"] = row["thalach"] / (row["trestbps"] + 1)
        row = pd.get_dummies(row, columns=["age_group"], drop_first=True)

        # Align columns with training set
        expected_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else row.columns
        for c in expected_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[expected_cols]

        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)[0]
        prob = model.predict_proba(row_scaled)[0][1] if hasattr(model, "predict_proba") else None

        label = "🔴 Heart Disease Detected" if pred == 1 else "🟢 No Heart Disease"
        print(f"\n  Result : {label}")
        if prob is not None:
            print(f"  Confidence : {max(prob, 1 - prob):.1%}")
        print()

        again = input("  Predict another patient? (y/n): ").strip().lower()
        if again != "y":
            break


# ── History ──────────────────────────────────────────────────────────────────

def show_history():
    if not HISTORY_PATH.exists():
        print("No training history yet.")
        return
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    print(f"\n{'─'*70}")
    print(f"  {'Run':>3}  {'Model':<25} {'Samples':>7} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print(f"{'─'*70}")
    for h in history:
        print(
            f"  {h['run']:>3}  {h['best_model']:<25} {h['samples']:>7} "
            f"{h['accuracy']:>6.4f} {h['f1']:>6.4f} {h.get('roc_auc', 0) or 0:>6.4f}"
        )
    print(f"{'─'*70}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Heart Disease ML Trainer")
    parser.add_argument("--data", default=str(DEFAULT_DATA_ARG), help="Path to CSV dataset")
    parser.add_argument("--predict", action="store_true", help="Run interactive prediction")
    parser.add_argument("--history", action="store_true", help="Show training history")
    args = parser.parse_args()

    if args.history:
        show_history()
    elif args.predict:
        predict_interactive()
    else:
        train(Path(args.data))


if __name__ == "__main__":
    main()
