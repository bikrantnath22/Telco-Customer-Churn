"""
predictive_model.py
────────────────────────────────────────────────────────────────────────────────
Churn Prediction — Best-Fit Model Selection Pipeline

Steps:
  1. Feature engineering & encoding
  2. Train/test split + class-imbalance handling (SMOTE via class_weight)
  3. Train 6 classifiers with cross-validation
  4. Auto-select best model by ROC-AUC
  5. Final evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
  6. Visualisations: Confusion Matrix, ROC Curve, Feature Importance
  7. Save the best model to models/best_model.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from config import CLEAN_DATA_PATH, FIGURES_DIR, ROOT_DIR

# ── Output dirs 
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# ── 1. Load & Feature Engineering 
def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Drop customer ID (not predictive)
    df.drop(columns=["customer_id"], inplace=True)

    # Target
    y = (df["churn"] == "Yes").astype(int)
    df.drop(columns=["churn"], inplace=True)

    # Encode binary columns
    binary_map = {"Yes": 1, "No": 0,
                  "Male": 1, "Female": 0,
                  "No phone service": 0, "No internet service": 0}
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(binary_map).fillna(df[col])

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[DATA]  Shape after feature engineering: {df.shape}")
    print(f"[DATA]  Churn rate: {y.mean()*100:.2f}% ({y.sum()} churned / {len(y)} total)\n")
    return df, y


# ── 2. Model Definitions
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
    }


# ── 3. Cross-Validated Comparison 
def compare_models(X_train, y_train) -> pd.DataFrame:
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("=" * 60)
    print("       CROSS-VALIDATED MODEL COMPARISON (5-Fold CV)")
    print("=" * 60)

    for name, model in models.items():
        # Scale features for distance-based & linear models
        if name in ("Logistic Regression", "K-Nearest Neighbors"):
            from sklearn.pipeline import Pipeline
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=-1)
        else:
            scores = cross_val_score(model, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=-1)

        results.append({
            "Model": name,
            "CV ROC-AUC Mean": scores.mean(),
            "CV ROC-AUC Std":  scores.std(),
        })
        print(f"  {name:<25}  AUC = {scores.mean():.4f}  ± {scores.std():.4f}")

    print("=" * 60 + "\n")
    return pd.DataFrame(results).sort_values("CV ROC-AUC Mean", ascending=False)


# ── 4. Final Evaluation
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     model_name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {model_name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}")
    return metrics, y_pred, y_proba


# ── 5. Visualisations 
def plot_cv_comparison(cv_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if i == 0 else "#3498db"
              for i in range(len(cv_results))]
    bars = ax.barh(cv_results["Model"][::-1],
                   cv_results["CV ROC-AUC Mean"][::-1],
                   xerr=cv_results["CV ROC-AUC Std"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.6)
    for bar, val in zip(bars, cv_results["CV ROC-AUC Mean"][::-1]):
        ax.text(bar.get_width() - 0.015, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="right", color="white",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("ROC-AUC Score (5-Fold CV)")
    ax.set_title("Model Comparison — Cross-Validated ROC-AUC\n(Green = Best Model)")
    ax.set_xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print("[SAVED] model_comparison.png")


def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("[SAVED] confusion_matrix.png")


def plot_roc_curve(y_test, y_proba, model_name: str, auc: float):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"{model_name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Best Model")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print("[SAVED] roc_curve.png")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Works for tree-based models and Logistic Regression."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = "Feature Coefficient Magnitude (Logistic Regression)"
    elif hasattr(model, "named_steps"):
        # Pipeline
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
            title = "Feature Coefficient Magnitude (Logistic Regression)"
        elif hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            title = "Feature Importance"
        else:
            print("[INFO] Feature importance not available for this model.")
            return
    else:
        print("[INFO] Feature importance not available for this model.")
        return

    # Top 20
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
    ax.barh([feature_names[i] for i in idx],
            importances[idx], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top 20 Features — {model_name}\n{title}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("[SAVED] feature_importance.png")


def plot_metrics_radar(metrics: dict):
    """Radar / bar chart of all final metrics."""
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    values = [metrics[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#3498db","#2ecc71","#e67e22","#9b59b6","#e74c3c"],
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Performance Summary — {metrics['Model']}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_performance_summary.png", dpi=150)
    plt.close()
    print("[SAVED] model_performance_summary.png")


# ── 6. Main Pipeline ──────────────────────────────────────────────────────────
def run():
    print("\n" + "="*60)
    print("     TELCO CHURN — PREDICTIVE MODEL PIPELINE")
    print("="*60 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────
    X, y = load_and_prepare(CLEAN_DATA_PATH)
    feature_names = X.columns.tolist()

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ── CV Comparison ──────────────────────────────────────────────────────
    cv_results = compare_models(X_train, y_train)
    plot_cv_comparison(cv_results)

    # ── Best model ─────────────────────────────────────────────────────────
    best_name = cv_results.iloc[0]["Model"]
    print(f"[BEST]  Selected: {best_name}  "
          f"(CV AUC = {cv_results.iloc[0]['CV ROC-AUC Mean']:.4f})\n")

    all_models = get_models()
    best_model_raw = all_models[best_name]

    # Wrap with scaler if needed
    if best_name in ("Logistic Regression", "K-Nearest Neighbors"):
        from sklearn.pipeline import Pipeline
        best_model = Pipeline([("scaler", StandardScaler()), ("clf", best_model_raw)])
    else:
        best_model = best_model_raw

    # Fit on full training set
    best_model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics, y_pred, y_proba = evaluate_model(best_model, X_test, y_test, best_name)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_roc_curve(y_test, y_proba, best_name, metrics["ROC-AUC"])
    plot_feature_importance(best_model, feature_names, best_name)
    plot_metrics_radar(metrics)

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump({"model": best_model, "features": feature_names,
                 "model_name": best_name, "metrics": metrics}, model_path)
    print(f"\n[SAVED] Best model saved to: {model_path}")

    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    summary = cv_results.copy()
    summary["CV ROC-AUC Mean"] = summary["CV ROC-AUC Mean"].round(4)
    summary["CV ROC-AUC Std"]  = summary["CV ROC-AUC Std"].round(4)
    print(summary.to_string(index=False))
    print("\n  Best Model:", best_name)
    print(f"  Test ROC-AUC : {metrics['ROC-AUC']:.4f}")
    print(f"  Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Test Recall  : {metrics['Recall']:.4f}  "
          f"(correctly identifies {metrics['Recall']*100:.1f}% of actual churners)")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()
