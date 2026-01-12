"""
evaluation.py
---------------------------------
Centralized evaluation utilities for
video-based activity classification.

This module is shared across ALL
machine learning approaches (SVM,
Random Forest, k-NN) to ensure:

• Fair comparison
• Consistent metrics
• Reproducible results
• Deployment-aware evaluation
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------------------------------
# Module-level logger
# --------------------------------------------------
LOGGER = logging.getLogger(__name__)


# ==================================================
# 🔹 MODEL EVALUATION (PER MODEL)
# ==================================================
def evaluate_classification(
    y_true,
    y_pred,
    y_scores=None,
    class_names=None,
    plot_roc: bool = True,
    show_tables: bool = True,
):
    """
    Evaluate a multi-class classification model
    using a strictly defined academic order.

    Metrics computed:
    1. Accuracy
    2. Precision (per-class + macro)
    3. Recall (per-class + macro)
    4. F1-Score (per-class + macro)
    5. Confusion Matrix
    6. ROC Curve & AUC (One-vs-Rest)

    Returns a dictionary used for:
    • Reporting
    • Model comparison
    • Efficiency analysis
    """

    print("Starting model evaluation")

    metrics = {}

    # ==================================================
    # 1️⃣ Accuracy
    # ==================================================
    accuracy = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = accuracy
    print(f"Accuracy: {accuracy:.4f}")

    # ==================================================
    # 2️⃣ Precision (Macro)
    # ==================================================
    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = precision_macro
    print(f"Macro Precision: {precision_macro:.4f}")

    # ==================================================
    # 3️⃣ Recall (Macro)
    # ==================================================
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_macro
    print(f"Macro Recall: {recall_macro:.4f}")

    # ==================================================
    # 4️⃣ F1-Score (Macro)
    # ==================================================
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_macro
    print(f"Macro F1-Score: {f1_macro:.4f}")

    # --------------------------------------------------
    # Per-class detailed metrics
    # --------------------------------------------------
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["classification_report"] = class_report
    metrics["classification_report_df"] = pd.DataFrame(class_report).T

    # ==================================================
    # 5️⃣ Confusion Matrix
    # ==================================================
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(cmap="Blues", xticks_rotation=45)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 6️⃣ ROC Curve & AUC (OvR)
    # ==================================================
    if y_scores is not None and plot_roc:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)

        roc_auc = {}
        plt.figure(figsize=(8, 6))

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i], y_scores[:, i]
            )
            roc_auc[cls] = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"{class_names[i]} (AUC={roc_auc[cls]:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (One-vs-Rest)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        metrics["roc_auc"] = roc_auc
        metrics["roc_auc_df"] = pd.DataFrame(
            {
                "Class": class_names,
                "ROC-AUC": [roc_auc[c] for c in classes],
            }
        )

    # ==================================================
    # 📋 Overall Metrics Table
    # ==================================================
    if show_tables:
        metrics["overall_metrics_df"] = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1-Score",
                ],
                "Value": [
                    accuracy,
                    precision_macro,
                    recall_macro,
                    f1_macro,
                ],
            }
        )

    print("Evaluation completed successfully !!")
    return metrics


# ==================================================
# 🔹 PERFORMANCE + EFFICIENCY COMPARISON
# ==================================================
def create_full_comparison_table(model_results: dict):
    """
    Create a SINGLE consolidated table containing:
    • Classification performance
    • Training time
    • Inference time per video

    This table is intended for:
    • Comparative analysis section
    • Final report tables
    • Viva justification
    """

    comparison_df = pd.DataFrame(
        {
            "Model": model_results.keys(),
            "Accuracy": [
                m["accuracy"] for m in model_results.values()
            ],
            "Macro Precision": [
                m["precision_macro"] for m in model_results.values()
            ],
            "Macro Recall": [
                m["recall_macro"] for m in model_results.values()
            ],
            "Macro F1-Score": [
                m["f1_macro"] for m in model_results.values()
            ],
            "Training Time (s)": [
                m["training_time"] for m in model_results.values()
            ],
            "Inference Time / Video (s)": [
                m["inference_time_per_video"]
                for m in model_results.values()
            ],
        }
    )

    print("\n📊 PERFORMANCE & COMPUTATIONAL EFFICIENCY COMPARISON")
    print(comparison_df.round(4))

    return comparison_df


# ==================================================
# 🔹 DYNAMIC OBSERVATIONS (AUTO-GENERATED)
# ==================================================
def generate_dynamic_observations(comparison_df: pd.DataFrame):
    """
    Generate dynamic, data-driven observations
    directly from the comparison table.
    """

    print("\n🧠 DYNAMIC OBSERVATIONS")

    best_accuracy = comparison_df.loc[
        comparison_df["Accuracy"].idxmax(), "Model"
    ]
    print(f"• {best_accuracy} achieves the highest classification accuracy.")

    best_f1 = comparison_df.loc[
        comparison_df["Macro F1-Score"].idxmax(), "Model"
    ]
    print(
        f"• {best_f1} provides the best balance between precision and recall."
    )

    fastest_training = comparison_df.loc[
        comparison_df["Training Time (s)"].idxmin(), "Model"
    ]
    print(
        f"• {fastest_training} has the lowest training time, "
        "indicating minimal model fitting overhead."
    )

    fastest_inference = comparison_df.loc[
        comparison_df["Inference Time / Video (s)"].idxmin(), "Model"
    ]
    print(
        f"• {fastest_inference} achieves the fastest inference per video, "
        "making it suitable for real-time deployment."
    )

    if "k-NN" in comparison_df["Model"].values:
        print(
            "• k-NN demonstrates very low training cost but higher inference "
            "latency due to distance-based computations during prediction."
        )
