"""
evaluation.py
---------------------------------
Common evaluation utilities for
video activity classification.

Used by ALL ML approaches.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay

LOGGER = logging.getLogger(__name__)


def evaluate_classification(
    y_true,
    y_pred,
    y_scores=None,
    class_names=None,
    plot_roc: bool = True,
):
    """
    Compute and report evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    y_scores : array-like, optional
        Prediction probabilities (required for ROC/AUC)
    class_names : list, optional
        Class label names
    plot_roc : bool
        Whether to plot ROC curves

    Returns
    -------
    dict
        Dictionary containing all metrics
    """

    LOGGER.info("Starting model evaluation")

    metrics = {}

    # ----------------------------
    # Accuracy
    # ----------------------------
    acc = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = acc
    LOGGER.info("Accuracy: %.4f", acc)

    # ----------------------------
    # Precision / Recall / F1
    # ----------------------------
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    LOGGER.info(
        "Macro Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )

    # Per-class metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["classification_report"] = report

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
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

    # ----------------------------
    # ROC Curve & AUC
    # ----------------------------
    if y_scores is not None and plot_roc:
        LOGGER.info("Computing ROC-AUC")

        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)

        roc_auc = {}

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[cls] = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc[cls]:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (One-vs-Rest)")
        plt.legend()
        plt.grid(True)
        plt.show()

        metrics["roc_auc"] = roc_auc

    LOGGER.info("Evaluation completed")
    return metrics
