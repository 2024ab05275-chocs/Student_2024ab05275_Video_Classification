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

# Create a module-level logger
LOGGER = logging.getLogger(__name__)


def evaluate_classification(
    y_true,
    y_pred,
    y_scores=None,
    class_names=None,
    plot_roc: bool = True,
    show_tables: bool = True,
):
    """
    Evaluate a multi-class classification model and report
    metrics in a strictly defined academic order.

    This function computes:
    1. Accuracy
    2. Precision (per-class + macro)
    3. Recall (per-class + macro)
    4. F1-Score (per-class + macro)
    5. Confusion Matrix
    6. ROC Curve and AUC (One-vs-Rest)

    Parameters
    ----------
    y_true : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels.
    y_scores : array-like, optional
        Class probability estimates.
        Required for ROC/AUC computation.
    class_names : list of str, optional
        Human-readable class names.
    plot_roc : bool, default=True
        If True, plots ROC curves.
    show_tables : bool, default=True
        If True, generates tabular summaries.

    Returns
    -------
    dict
        Dictionary containing:
        • Scalar metrics
        • Confusion matrix
        • ROC-AUC values
        • Tabular DataFrames
    """

    print("Starting model evaluation")

    # Dictionary to store all computed results
    metrics = {}

    # ==================================================
    # 1️⃣ Accuracy
    # ==================================================
    # Accuracy measures overall classification correctness
    accuracy = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = accuracy

    print("Accuracy: %.4f", accuracy)

    # ==================================================
    # 2️⃣ Precision (Per-class + Macro Average)
    # ==================================================
    # Precision answers:
    # "Of all predicted samples for a class, how many were correct?"
    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics["precision_macro"] = precision_macro

    print("Macro Precision: %.4f", precision_macro)

    # ==================================================
    # 3️⃣ Recall (Per-class + Macro Average)
    # ==================================================
    # Recall answers:
    # "Of all actual samples of a class, how many were correctly identified?"
    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics["recall_macro"] = recall_macro

    print("Macro Recall: %.4f", recall_macro)

    # ==================================================
    # 4️⃣ F1-Score (Per-class + Macro Average)
    # ==================================================
    # F1-score is the harmonic mean of precision and recall
    # It balances false positives and false negatives
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics["f1_macro"] = f1_macro

    print("Macro F1-Score: %.4f", f1_macro)

    # Generate detailed per-class metrics
    # Includes precision, recall, f1-score, and support
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    metrics["classification_report"] = class_report

    # Convert per-class metrics to a DataFrame
    class_report_df = pd.DataFrame(class_report).T
    metrics["classification_report_df"] = class_report_df

    # ==================================================
    # 5️⃣ Confusion Matrix
    # ==================================================
    # Confusion matrix shows true vs predicted labels
    # and helps identify misclassification patterns
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 6️⃣ ROC Curve and AUC (Multi-class One-vs-Rest)
    # ==================================================
    # ROC curves evaluate model discrimination ability
    # For multi-class problems, One-vs-Rest strategy is used
    if y_scores is not None and plot_roc:
        print("Computing ROC Curve and AUC (OvR)")

        classes = np.unique(y_true)

        # Convert true labels into binary format
        y_true_bin = label_binarize(y_true, classes=classes)

        roc_auc = {}
        plt.figure(figsize=(8, 6))

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i],
                y_scores[:, i],
            )
            roc_auc[cls] = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"{class_names[i]} (AUC={roc_auc[cls]:.2f})",
            )

        # Reference diagonal for random classifier
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (One-vs-Rest)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        metrics["roc_auc"] = roc_auc

        # Create a ROC-AUC table per class
        roc_auc_df = pd.DataFrame(
            {
                "Class": class_names,
                "ROC-AUC": [roc_auc[c] for c in classes],
            }
        )
        metrics["roc_auc_df"] = roc_auc_df

    # ==================================================
    # 📋 Overall Metrics Table (Report-Ready)
    # ==================================================
    if show_tables:
        overall_metrics_df = pd.DataFrame(
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
        metrics["overall_metrics_df"] = overall_metrics_df

    print("Evaluation completed successfully !!")
    return metrics
