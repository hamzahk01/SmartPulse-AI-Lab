"""Train, tune, and evaluate the KNN Traffic Congestion Classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

TRAFFIC_CLASSES = ["low", "normal", "high", "heavy"]
FEATURE_COLUMNS = [
    "TimeSin",
    "TimeCos",
    "IsWeekend",
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount",
    "HeavyRatio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw Kaggle traffic logs into scaled-ready numerical features.

    Args:
        df: DataFrame loaded from TrafficTwoMonth.csv.

    Returns:
        Feature DataFrame matching FEATURE_COLUMNS.
    """
    parsed_time = pd.to_datetime(df["Time"], format="%I:%M:%S %p")
    minutes = parsed_time.dt.hour * 60 + parsed_time.dt.minute
    angle = 2.0 * np.pi * minutes / 1440.0

    total_vehicles = (
        df["CarCount"] + df["BikeCount"] + df["BusCount"] + df["TruckCount"]
    )
    heavy_ratio = (df["BusCount"] + df["TruckCount"]) / (total_vehicles + 1e-5)

    features = pd.DataFrame(
        {
            "TimeSin": np.sin(angle),
            "TimeCos": np.cos(angle),
            "IsWeekend": df["Day of the week"].isin(["Saturday", "Sunday"]).astype(int),
            "CarCount": df["CarCount"].astype(float),
            "BikeCount": df["BikeCount"].astype(float),
            "BusCount": df["BusCount"].astype(float),
            "TruckCount": df["TruckCount"].astype(float),
            "HeavyRatio": heavy_ratio.astype(float),
        }
    )
    return features[FEATURE_COLUMNS]


def plot_knn_evaluation(
    k_values: list[int],
    cv_scores: list[float],
    best_k: int,
    cm: np.ndarray,
    output_path: Path,
) -> Path:
    """Save a two-panel figure showing K tuning curve and test confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    # Panel 1: Hyperparameter tuning curve
    axes[0].plot(k_values, [s * 100 for s in cv_scores], marker="o", linewidth=2, color="#1f77b4")
    best_idx = k_values.index(best_k)
    axes[0].scatter(
        [best_k],
        [cv_scores[best_idx] * 100],
        color="#d62728",
        s=110,
        zorder=5,
        label=f"Optimal k = {best_k} ({cv_scores[best_idx]*100:.2f}%)",
    )
    axes[0].set_title("KNN Hyperparameter Tuning (5-Fold CV)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Neighbors (k)")
    axes[0].set_ylabel("Cross-Validation Accuracy (%)")
    axes[0].set_xticks(k_values)
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend(loc="lower right")

    # Panel 2: Confusion matrix on held-out test set
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TRAFFIC_CLASSES)
    disp.plot(ax=axes[1], cmap="Blues", colorbar=False, values_format="d")
    axes[1].set_title("Test Set Confusion Matrix (20% Holdout)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def train_and_evaluate_knn(
    csv_path: Path,
    model_path: Path,
    plot_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the KNN congestion classifier, tune k, evaluate, and persist artifacts.

    Args:
        csv_path: Path to TrafficTwoMonth.csv.
        model_path: Destination pickle path for trained model bundle.
        plot_path: Destination PNG path for KNN evaluation chart.
        random_state: Seed for reproducible stratified splitting.

    Returns:
        Dictionary containing trained bundle and evaluation metrics.
    """
    df = pd.read_csv(csv_path)
    X = engineer_features(df)
    y = df["Traffic Situation"].str.lower().str.strip()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_candidates = list(range(1, 22, 2))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = [
        float(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean"),
                X_train_scaled,
                y_train,
                cv=cv,
                scoring="accuracy",
            ).mean()
        )
        for k in k_candidates
    ]

    best_k = k_candidates[int(np.argmax(cv_scores))]
    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="euclidean")
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    test_acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=TRAFFIC_CLASSES)
    report_dict = classification_report(
        y_test, y_pred, labels=TRAFFIC_CLASSES, output_dict=True, zero_division=0
    )

    plot_knn_evaluation(k_candidates, cv_scores, best_k, cm, plot_path)

    class_counts = y.value_counts().reindex(TRAFFIC_CLASSES, fill_value=0).to_dict()
    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in class_counts.items()},
        "k_candidates": k_candidates,
        "cv_scores": [round(s, 4) for s in cv_scores],
        "best_k": int(best_k),
        "best_cv_accuracy": float(max(cv_scores)),
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    bundle = {
        "model": knn,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "classes": TRAFFIC_CLASSES,
        "metrics": metrics,
        "test_records": df.loc[idx_test].reset_index(drop=True),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(
        f"[KNN] Trained KNeighborsClassifier (best_k={best_k}, "
        f"CV={max(cv_scores)*100:.2f}%, Test Acc={test_acc*100:.2f}%, "
        f"Macro F1={macro_f1:.4f})"
    )
    return bundle


def load_knn_bundle(model_path: Path) -> dict[str, Any]:
    """Load a previously trained KNN model bundle from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)
