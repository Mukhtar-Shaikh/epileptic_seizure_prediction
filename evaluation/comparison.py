import numpy as np
import pandas as pd
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from evaluation.metrics import compute_all_metrics
from utils.config import RESULTS_DIR
from utils.logger import get_logger

logger = get_logger("Comparison")

RESULTS_PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_PLOT_DIR, exist_ok=True)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = (time.perf_counter() - t0) / len(X_test) * 1000

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    metrics["inference_time_ms"] = inference_time
    metrics["model_name"] = model_name
    logger.info(f"{model_name}: acc={metrics['accuracy']:.4f} roc_auc={metrics.get('roc_auc', float('nan')):.4f} inference={inference_time:.3f}ms")
    return metrics


def compare_models(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict]:
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    display_keys = ["accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "false_prediction_rate", "mcc", "dice", "inference_time_ms"]

    rows = []
    for name, m in results.items():
        row = {"Model": name}
        for k in display_keys:
            row[k] = round(m.get(k, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    df_sorted = df.sort_values("accuracy", ascending=False)

    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    df_sorted.to_csv(csv_path)
    logger.info(f"Comparison table saved to {csv_path}")

    _plot_comparison_bar(df_sorted)
    return df_sorted, results


def _plot_comparison_bar(df: pd.DataFrame) -> str:
    metrics_to_plot = ["accuracy", "sensitivity", "specificity", "f1", "roc_auc"]
    available = [m for m in metrics_to_plot if m in df.columns]
    subset = df[available].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subset.index))
    width = 0.8 / len(available)
    for i, col in enumerate(available):
        offset = (i - len(available) / 2) * width + width / 2
        ax.bar(x + offset, subset[col], width=width * 0.9, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(subset.index, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    path = os.path.join(RESULTS_PLOT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison bar chart saved: {path}")
    return path


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = {"accuracy": False, "f1": False, "roc_auc": False,
                 "inference_time_ms": True, "false_prediction_rate": True}
    rankings = pd.DataFrame(index=df.index)
    for col, ascending in rank_cols.items():
        if col in df.columns:
            rankings[f"rank_{col}"] = df[col].rank(ascending=ascending)
    rankings["overall_rank"] = rankings.mean(axis=1)
    return rankings.sort_values("overall_rank")
