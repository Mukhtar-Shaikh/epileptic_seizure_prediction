import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable
from utils.logger import get_logger

logger = get_logger("Alerts")


class AlertPanel(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="⚠ Alert Log", **kwargs)
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=2)
        ttk.Button(top, text="Clear", command=self.clear).pack(side="right")

        self._text = tk.Text(self, height=8, state="disabled", bg="#1e1e1e", fg="#f0f0f0",
                             font=("Consolas", 9), wrap="word", relief="flat", borderwidth=0)
        sb = ttk.Scrollbar(self, command=self._text.yview)
        self._text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._text.pack(fill="both", expand=True, padx=4, pady=2)

        self._text.tag_configure("seizure", foreground="#ff4444", font=("Consolas", 9, "bold"))
        self._text.tag_configure("preictal", foreground="#ffaa00", font=("Consolas", 9, "bold"))
        self._text.tag_configure("normal", foreground="#44ff88")
        self._text.tag_configure("info", foreground="#88ccff")

    def log(self, message: str, level: str = "info"):
        self._text.configure(state="normal")
        self._text.insert("end", message + "\n", level)
        self._text.see("end")
        self._text.configure(state="disabled")
        logger.info(f"[Alert:{level}] {message}")

    def clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    def seizure_alert(self, prob: float, timestamp: str = ""):
        msg = f"[{timestamp}] 🔴 SEIZURE DETECTED — probability={prob:.3f}"
        self.log(msg, "seizure")
        messagebox.showwarning("SEIZURE ALERT", f"Seizure detected!\nProbability: {prob:.3f}")

    def preictal_alert(self, prob: float, timestamp: str = ""):
        msg = f"[{timestamp}] 🟡 PRE-ICTAL STATE — probability={prob:.3f}"
        self.log(msg, "preictal")

    def normal_alert(self, prob: float, timestamp: str = ""):
        msg = f"[{timestamp}] 🟢 NORMAL — probability={prob:.3f}"
        self.log(msg, "normal")
