"""
evaluation.py
==========================================================
Centralized evaluation utilities for video-based
activity classification.

This module is SHARED across ALL machine learning
approaches (SVM, Random Forest, k-NN, and future
Deep Learning models) to ensure:

✔ Fair comparison
✔ Consistent metrics
✔ Reproducible results
✔ Deployment-aware evaluation
✔ Publication-quality visualizations

All plots are:
• Displayed inline (for notebooks)
• Saved to disk (for reports & comparison)
==========================================================
"""

# ==========================================================
# 📦 IMPORTS
# ==========================================================
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
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
from IPython.display import display

# ==========================================================
# 📁 RESULT DIRECTORIES (AUTO-CREATED)
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

CONF_MATRIX_DIR = os.path.join(BASE_RESULTS_DIR, "confusion_matrices")
ROC_CURVE_DIR = os.path.join(BASE_RESULTS_DIR, "roc_curves")

os.makedirs(CONF_MATRIX_DIR, exist_ok=True)
os.makedirs(ROC_CURVE_DIR, exist_ok=True)

# ==========================================================
# 🪵 MODULE-LEVEL LOGGER
# ==========================================================
LOGGER = logging.getLogger(__name__)


# ==========================================================
# 🔹 MODEL EVALUATION FUNCTION
# ==========================================================
def evaluate_classification(
    y_true,
    y_pred,
    y_scores=None,
    class_names=None,
    model_name: str = "model",
    plot_roc: bool = True,
    show_tables: bool = True,
):
    """
    Evaluate a multi-class classification model
    using a STRICT academic evaluation order.

    --------------------------------------------------
    Metrics Computed:
    1️⃣ Accuracy
    2️⃣ Macro Precision
    3️⃣ Macro Recall
    4️⃣ Macro F1-Score
    5️⃣ Per-class metrics table
    6️⃣ Confusion Matrix (saved + shown)
    7️⃣ ROC Curve & AUC (OvR) (saved + shown)

    --------------------------------------------------
    Returns:
    Dictionary containing:
    • All scalar metrics
    • Pandas DataFrames
    • File paths of saved plots
    --------------------------------------------------
    """

    print(f"\n🚀 Starting evaluation for: {model_name}")

    metrics = {}

    # ==================================================
    # 1️⃣ ACCURACY
    # ==================================================
    accuracy = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = accuracy

    # ==================================================
    # 2️⃣ MACRO PRECISION
    # ==================================================
    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = precision_macro

    # ==================================================
    # 3️⃣ MACRO RECALL
    # ==================================================
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_macro

    # ==================================================
    # 4️⃣ MACRO F1-SCORE
    # ==================================================
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_macro

    # ==================================================
    # 📋 OVERALL METRICS TABLE (DISPLAY)
    # ==================================================
    overall_metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1-Score"
        ],
        "Value": [
            round(accuracy, 4),
            round(precision_macro, 4),
            round(recall_macro, 4),
            round(f1_macro, 4)
        ]
    })

    if show_tables:
        print("\n📊 Overall Performance Metrics")
        display(overall_metrics_df)

    metrics["overall_metrics_df"] = overall_metrics_df

    # ==================================================
    # 📊 PER-CLASS CLASSIFICATION REPORT
    # ==================================================
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    class_report_df = pd.DataFrame(class_report).T
    metrics["classification_report"] = class_report
    metrics["classification_report_df"] = class_report_df

    if show_tables:
        print("\n📊 Per-Class Classification Report")
        display(class_report_df)

    # ==================================================
    # 5️⃣ CONFUSION MATRIX (DISPLAY + SAVE)
    # ==================================================
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(cmap="Blues", xticks_rotation=45, ax=ax)

    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()

    cm_save_path = os.path.join(
        CONF_MATRIX_DIR, f"{model_name.lower()}_confusion_matrix.png"
    )
    plt.savefig(cm_save_path, dpi=300)
    plt.show()
    plt.close()

    metrics["confusion_matrix_path"] = cm_save_path
    LOGGER.info(f"Confusion matrix saved → {cm_save_path}")

    # ==================================================
    # 6️⃣ ROC CURVE & AUC (OvR) – DISPLAY + SAVE
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
        plt.title(f"ROC Curve (One-vs-Rest) – {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        roc_save_path = os.path.join(
            ROC_CURVE_DIR, f"{model_name.lower()}_roc_curve.png"
        )
        plt.savefig(roc_save_path, dpi=300)
        plt.show()
        plt.close()

        metrics["roc_auc"] = roc_auc
        metrics["roc_auc_path"] = roc_save_path
        metrics["roc_auc_df"] = pd.DataFrame(
            {
                "Class": class_names,
                "ROC-AUC": [roc_auc[c] for c in classes],
            }
        )

        LOGGER.info(f"ROC curve saved → {roc_save_path}")

    print("✅ Evaluation completed successfully\n")
    return metrics


# ==========================================================
# 🔹 PERFORMANCE & EFFICIENCY COMPARISON TABLE
# ==========================================================
def create_full_comparison_table(model_results: dict):
    """
    Create a SINGLE consolidated table containing:

    • Classification performance
    • Training time
    • Inference time per video

    Used for:
    ✔ Comparative analysis
    ✔ Final report tables
    ✔ Viva justification
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
                m.get("training_time", np.nan)
                for m in model_results.values()
            ],
            "Inference Time / Video (s)": [
                m.get("inference_time_per_video", np.nan)
                for m in model_results.values()
            ],
        }
    )

    print("\n📊 PERFORMANCE & COMPUTATIONAL EFFICIENCY COMPARISON")
    display(comparison_df.round(4))

    return comparison_df


# ==========================================================
# 🔹 DYNAMIC OBSERVATIONS (AUTO-GENERATED)
# ==========================================================
def generate_dynamic_observations(comparison_df: pd.DataFrame):
    """
    Automatically generate data-driven observations
    from the comparison table.
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
            "• k-NN shows minimal training cost but higher inference latency "
            "due to distance-based computations."
        )
