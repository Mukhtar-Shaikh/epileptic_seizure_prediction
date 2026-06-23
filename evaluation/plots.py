import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from utils.config import RESULTS_DIR
from utils.logger import get_logger

logger = get_logger("Plots")

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")
    return path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-Seizure", "Seizure"],
                yticklabels=["Non-Seizure", "Seizure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    return _save(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "") -> str:
    prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    return _save(fig, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "") -> str:
    prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    precision, recall, _ = precision_recall_curve(y_true, prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    return _save(fig, f"pr_curve_{model_name.lower().replace(' ', '_')}.png")


def plot_feature_importance(importances: np.ndarray, feature_names=None, model_name: str = "", top_n: int = 20) -> str:
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(idx)), importances[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})")
    ax.set_ylabel("Importance")
    return _save(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")


def plot_shap(shap_values: np.ndarray, X: np.ndarray, feature_names=None, model_name: str = "") -> str:
    try:
        import shap
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type="bar")
        path = os.path.join(PLOT_DIR, f"shap_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")
        return ""


def plot_training_history(train_losses: list, val_losses: list = None, model_name: str = "") -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss")
    if val_losses:
        ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training History — {model_name}")
    ax.legend()
    return _save(fig, f"training_history_{model_name.lower().replace(' ', '_')}.png")


def plot_eeg_signal(signal: np.ndarray, fs: float, title: str = "EEG Signal", n_channels: int = 5) -> str:
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    n_show = min(n_channels, signal.shape[0])
    t = np.arange(signal.shape[1]) / fs
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, signal[i], lw=0.7)
        ax.set_ylabel(f"Ch{i+1}")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    return _save(fig, f"eeg_{title.replace(' ', '_').lower()}.png")
