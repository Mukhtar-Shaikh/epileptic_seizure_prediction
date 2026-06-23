import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from utils.logger import get_logger

logger = get_logger("PredictionPanel")

MAX_HISTORY = 200


class PredictionPanel(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Real-Time Prediction", **kwargs)
        self._proba_history = deque(maxlen=MAX_HISTORY)
        self._label_history = deque(maxlen=MAX_HISTORY)
        self._seizure_threshold = 0.5
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=4)

        self._prob_label = ttk.Label(top, text="Seizure Probability: —", font=("Helvetica", 14, "bold"))
        self._prob_label.pack(side="left", padx=8)

        self._state_label = ttk.Label(top, text="State: NORMAL", font=("Helvetica", 12),
                                       foreground="#44ff88")
        self._state_label.pack(side="left", padx=8)

        ttk.Label(top, text="Threshold:").pack(side="left", padx=(20, 0))
        self._thresh_var = tk.DoubleVar(value=self._seizure_threshold)
        ttk.Scale(top, from_=0.1, to=0.9, variable=self._thresh_var,
                  orient="horizontal", length=100,
                  command=lambda _: self._update_threshold()).pack(side="left")
        self._thresh_lbl = ttk.Label(top, text=f"{self._seizure_threshold:.2f}")
        self._thresh_lbl.pack(side="left")

        self._bar_frame = ttk.Frame(self)
        self._bar_frame.pack(fill="x", padx=8, pady=2)
        ttk.Label(self._bar_frame, text="Probability:").pack(side="left")
        self._bar_canvas = tk.Canvas(self._bar_frame, height=20, bg="#2a2a2a", relief="flat")
        self._bar_canvas.pack(side="left", fill="x", expand=True, padx=4)

        self._fig = Figure(figsize=(10, 2.5), facecolor="#1e1e1e")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1e1e1e")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        self._metrics_frame = ttk.Frame(self)
        self._metrics_frame.pack(fill="x", padx=8, pady=4)
        self._metric_vars = {}
        for col, label in enumerate(["Accuracy", "Sensitivity", "Specificity", "F1", "ROC-AUC"]):
            ttk.Label(self._metrics_frame, text=label + ":", font=("Helvetica", 9, "bold")).grid(
                row=0, column=col * 2, padx=4, sticky="e")
            var = tk.StringVar(value="—")
            self._metric_vars[label] = var
            ttk.Label(self._metrics_frame, textvariable=var, font=("Helvetica", 9)).grid(
                row=0, column=col * 2 + 1, padx=4, sticky="w")

    def _update_threshold(self):
        self._seizure_threshold = self._thresh_var.get()
        self._thresh_lbl.configure(text=f"{self._seizure_threshold:.2f}")

    def update_prediction(self, prob: float, true_label: int = None):
        self._proba_history.append(prob)
        if true_label is not None:
            self._label_history.append(true_label)

        self._prob_label.configure(text=f"Seizure Probability: {prob:.3f}")

        is_seizure = prob >= self._seizure_threshold
        is_preictal = 0.3 <= prob < self._seizure_threshold

        if is_seizure:
            self._state_label.configure(text="State: ⚠ SEIZURE", foreground="#ff4444")
        elif is_preictal:
            self._state_label.configure(text="State: PRE-ICTAL", foreground="#ffaa00")
        else:
            self._state_label.configure(text="State: NORMAL", foreground="#44ff88")

        self._update_bar(prob)
        self._update_plot()

    def _update_bar(self, prob: float):
        self._bar_canvas.update_idletasks()
        w = self._bar_canvas.winfo_width()
        h = 20
        self._bar_canvas.delete("all")
        fill_w = int(w * prob)
        color = "#ff4444" if prob >= self._seizure_threshold else ("#ffaa00" if prob >= 0.3 else "#44ff88")
        self._bar_canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        thresh_x = int(w * self._seizure_threshold)
        self._bar_canvas.create_line(thresh_x, 0, thresh_x, h, fill="white", width=2, dash=(4, 2))

    def _update_plot(self):
        self._ax.clear()
        self._ax.set_facecolor("#1e1e1e")
        probs = list(self._proba_history)
        if probs:
            x = range(len(probs))
            self._ax.fill_between(x, probs, alpha=0.4, color="#44aaff")
            self._ax.plot(x, probs, color="#44aaff", lw=1.2)
            self._ax.axhline(self._seizure_threshold, color="#ff4444", lw=1, ls="--", alpha=0.7)
            self._ax.axhline(0.3, color="#ffaa00", lw=1, ls="--", alpha=0.5)
        self._ax.set_ylim(0, 1)
        self._ax.set_xlim(0, MAX_HISTORY)
        self._ax.set_ylabel("Prob", color="#aaaaaa", fontsize=8)
        self._ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#555555")
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def update_metrics(self, metrics: dict):
        mapping = {"Accuracy": "accuracy", "Sensitivity": "sensitivity",
                   "Specificity": "specificity", "F1": "f1", "ROC-AUC": "roc_auc"}
        for label, key in mapping.items():
            val = metrics.get(key, float("nan"))
            if isinstance(val, float):
                self._metric_vars[label].set(f"{val:.4f}")
            else:
                self._metric_vars[label].set(str(val))
