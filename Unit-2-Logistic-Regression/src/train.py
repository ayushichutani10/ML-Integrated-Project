"""Train Logistic Regression (main model) and Random Forest (comparison)."""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split

from config import FIG_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE
from data_prep import clean, load_data, make_pipeline


def eda_plots(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ax = df[TARGET].value_counts().sort_index().rename({0: "Stayed", 1: "Churned"}).plot.bar(color=["#4c9be8", "#e8684c"])
    ax.set_title("Class distribution"); ax.set_ylabel("Customers")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        (df.groupby(col)[TARGET].mean() * 100).sort_values().plot.barh(ax=ax, color="#e8684c")
        ax.set_title(f"Churn rate (%) by {col}"); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_churn_by_category.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
        for label, name, c in [(0, "Stayed", "#4c9be8"), (1, "Churned", "#e8684c")]:
            ax.hist(df.loc[df[TARGET] == label, col], bins=30, alpha=0.6, label=name, color=c)
        ax.set_title(f"{col} by churn"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_numeric_distributions.png", dpi=130); plt.close()


def evaluate(name, pipe, X_test, y_test, log):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    log.append(f"=== {name} ===\n{classification_report(y_test, pred, target_names=['Stayed', 'Churned'])}")
    return row, pred


def main():
    df = clean(load_data())
    print(f"Dataset: {df.shape[0]} rows, churn rate {df[TARGET].mean():.1%}")
    eda_plots(df)

    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    log, rows, preds, pipes = [], [], {}, {}

    # Main model: Logistic Regression, tuned with 5-fold cross-validation
    lr = make_pipeline(X, LogisticRegression(class_weight="balanced", max_iter=2000))
    grid = GridSearchCV(lr, {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    log.append(f"Best Logistic Regression params: {grid.best_params_} (CV AUC={grid.best_score_:.3f})\n")
    pipes["Logistic Regression"] = grid.best_estimator_

    # Comparison model
    rf = make_pipeline(X, RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(X_train, y_train)
    pipes["Random Forest"] = rf

    for name, pipe in pipes.items():
        row, pred = evaluate(name, pipe, X_test, y_test, log)
        rows.append(row); preds[name] = pred

    results = pd.DataFrame(rows).set_index("Model").round(3)
    log.append("=== Model comparison ===\n" + results.to_string())
    text = "\n".join(log)
    print(text)
    METRICS_PATH.write_text(text)

    best = pipes["Logistic Regression"]
    ConfusionMatrixDisplay.from_predictions(y_test, preds["Logistic Regression"],
                                            display_labels=["Stayed", "Churned"], cmap="Blues")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_confusion_matrix.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in pipes.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4); ax.set_title("ROC curves")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_roc_curves.png", dpi=130); plt.close()

    # Interpret Logistic Regression coefficients
    names = best.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(best.named_steps["clf"].coef_[0], index=names).sort_values()
    top = pd.concat([coefs.head(10), coefs.tail(10)])
    top.plot.barh(figsize=(8, 7), color=["#4c9be8" if v < 0 else "#e8684c" for v in top])
    plt.title("Top drivers of churn (red = increases, blue = decreases)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_feature_coefficients.png", dpi=130); plt.close()
    print("\nTop churn-increasing features:\n", coefs.tail(5).round(3).to_string())
    print("\nTop churn-reducing features:\n", coefs.head(5).round(3).to_string())

    metrics = {
        "results": results.reset_index().to_dict("records"),
        "best_params": {k: float(v) for k, v in grid.best_params_.items()},
        "cv_auc": float(grid.best_score_),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "coefficients": {k: float(v) for k, v in coefs.round(4).items()},
        "confusion": confusion_matrix(y_test, preds["Logistic Regression"]).tolist(),
    }
    (METRICS_PATH.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best, "columns": X.columns.tolist()}, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}\nFigures saved -> {FIG_DIR}")


if __name__ == "__main__":
    main()
