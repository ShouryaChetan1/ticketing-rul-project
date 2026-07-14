import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def plot_training_history(history_path: str, output_dir: str):
    with open(history_path) as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16)

    axes[0, 0].plot(epochs, df["train_loss"], "b-o", label="Train")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, df["val_fault_accuracy"], "g-s", label="Accuracy")
    axes[0, 1].plot(epochs, df["val_fault_f1"], "r-d", label="F1 Score")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, df["val_fault_precision"], "c-^", label="Precision")
    axes[1, 0].plot(epochs, df["val_fault_recall"], "m-v", label="Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    if "val_rul_mae" in df.columns:
        axes[1, 1].plot(epochs, df["val_rul_mae"], "o-", color="orange", label="RUL MAE (days)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("RUL MAE (days)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    name: str = "confusion_matrix",
):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fault", "Fault"],
        yticklabels=["No Fault", "Fault"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\nTP={tp} FP={fp} FN={fn} TN={tn}")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: str,
):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_rul_scatter(
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    output_dir: str,
):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rul_true, rul_pred, alpha=0.4, s=20)
    min_val = min(rul_true.min(), rul_pred.min())
    max_val = max(rul_true.max(), rul_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual RUL (days)")
    ax.set_ylabel("Predicted RUL (days)")
    ax.set_title("RUL Prediction: Actual vs Predicted")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "rul_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(
    feature_names: list,
    importance_scores: np.ndarray,
    output_dir: str,
    top_k: int = 20,
):
    indices = np.argsort(importance_scores)[::-1][:top_k]
    names = [feature_names[i] for i in indices]
    scores = importance_scores[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), scores, align="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_k} Feature Importance")
    ax.invert_yaxis()
    ax.grid(True, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_failure_mode_breakdown(
    labels_df: pd.DataFrame,
    output_dir: str,
):
    fault_data = labels_df[labels_df["fault_label"] == 1]
    mode_counts = fault_data["failure_mode"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(mode_counts)))
    bars = ax.bar(mode_counts.index, mode_counts.values, color=colors)
    ax.set_xlabel("Failure Mode")
    ax.set_ylabel("Count")
    ax.set_title("Failure Mode Distribution")
    ax.tick_params(axis="x", rotation=45)

    for bar, val in zip(bars, mode_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mode_counts.values) * 0.01,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "failure_mode_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def generate_report(
    metrics: dict,
    output_dir: str,
):
    lines = [
        "=" * 50,
        "PREDICTIVE MAINTENANCE MODEL REPORT",
        "=" * 50,
        "",
        "FAULT DETECTION METRICS",
        "-" * 30,
    ]
    for k in ["fault_accuracy", "fault_precision", "fault_recall", "fault_f1"]:
        if k in metrics:
            lines.append(f"  {k}: {metrics[k]:.4f}")

    lines.extend(["", "RUL PREDICTION METRICS", "-" * 30])
    for k in ["rul_mae", "rul_rmse"]:
        if k in metrics:
            label = k.replace("rul_", "RUL ").upper()
            lines.append(f"  {label}: {metrics[k]:.2f} days")

    lines.append("")
    lines.append("=" * 50)

    report = "\n".join(lines)
    path = os.path.join(output_dir, "report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(report)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="saved_models/training_history.json")
    parser.add_argument("--labels", default="data_output/labels.csv")
    parser.add_argument("--output-dir", default="evaluation_output")
    parser.add_argument("--metrics", type=json.loads, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.history):
        plot_training_history(args.history, args.output_dir)

    if args.metrics:
        generate_report(args.metrics, args.output_dir)

    if os.path.exists(args.labels):
        labels_df = pd.read_csv(args.labels)
        plot_failure_mode_breakdown(labels_df, args.output_dir)

    print("\nAll visualizations saved to:", args.output_dir)


if __name__ == "__main__":
    main()
