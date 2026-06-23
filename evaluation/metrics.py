import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    average_precision_score,
)
from utils.logger import get_logger

logger = get_logger("Metrics")


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    specificity = float(tn / (tn + fp + 1e-10))
    sensitivity = float(tp / (tp + fn + 1e-10))
    fpr = float(fp / (fp + tn + 1e-10))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_prediction_rate": fpr,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "dice": float(2 * tp / (2 * tp + fp + fn + 1e-10)),
        "confusion_matrix": cm.tolist(),
    }

    if y_proba is not None:
        prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, prob))
        except Exception as e:
            logger.warning(f"ROC/PR AUC computation failed: {e}")
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "") -> None:
    header = f"=== {model_name} Metrics ===" if model_name else "=== Metrics ==="
    print(header)
    skip = {"confusion_matrix"}
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:25s}: {v:.4f}")
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print(f"  {'Confusion Matrix':25s}:")
        for row in cm:
            print(f"    {row}")
    print()
