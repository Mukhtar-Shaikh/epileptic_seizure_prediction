import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from utils.logger import get_logger

logger = get_logger("EEGViewer")


class EEGViewer(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="EEG Signal Viewer", **kwargs)
        self._raw: np.ndarray = None
        self._filtered: np.ndarray = None
        self._fs: float = 256.0
        self._n_channels_show: int = 5
        self._offset_scale: float = 3.0
        self._build()

    def _build(self):
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=4, pady=2)

        ttk.Label(controls, text="Channels to show:").pack(side="left")
        self._ch_var = tk.IntVar(value=self._n_channels_show)
        ttk.Spinbox(controls, from_=1, to=24, textvariable=self._ch_var, width=4,
                    command=self._redraw).pack(side="left", padx=2)

        ttk.Label(controls, text="Spacing:").pack(side="left", padx=(8, 0))
        self._spacing_var = tk.DoubleVar(value=self._offset_scale)
        ttk.Scale(controls, from_=0.5, to=10.0, variable=self._spacing_var,
                  orient="horizontal", length=120, command=lambda _: self._redraw()).pack(side="left")

        self._mode_var = tk.StringVar(value="Raw")
        for m in ("Raw", "Filtered", "Both"):
            ttk.Radiobutton(controls, text=m, variable=self._mode_var, value=m,
                            command=self._redraw).pack(side="left", padx=2)

        self._fig = Figure(figsize=(10, 4), facecolor="#1e1e1e")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1e1e1e")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(self._canvas, toolbar_frame).update()

    def load_signal(self, raw: np.ndarray, fs: float, filtered: np.ndarray = None):
        self._raw = raw
        self._filtered = filtered
        self._fs = fs
        self._redraw()

    def _redraw(self):
        if self._raw is None:
            return
        n_show = min(self._ch_var.get(), self._raw.shape[0] if self._raw.ndim > 1 else 1)
        spacing = self._spacing_var.get()
        mode = self._mode_var.get()

        self._ax.clear()
        self._ax.set_facecolor("#1e1e1e")

        n_samples = self._raw.shape[-1]
        t = np.arange(n_samples) / self._fs

        raw_2d = self._raw if self._raw.ndim > 1 else self._raw[np.newaxis, :]
        filt_2d = (self._filtered if self._filtered is not None else raw_2d)
        if filt_2d.ndim == 1:
            filt_2d = filt_2d[np.newaxis, :]

        colors_raw = "#44aaff"
        colors_filt = "#ff8844"

        for i in range(min(n_show, raw_2d.shape[0])):
            offset = i * spacing
            if mode in ("Raw", "Both"):
                sig = raw_2d[i]
                sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)
                self._ax.plot(t, sig_norm + offset, color=colors_raw, lw=0.6, alpha=0.85, label="Raw" if i == 0 else "")
            if mode in ("Filtered", "Both") and filt_2d is not None:
                sig = filt_2d[i] if i < filt_2d.shape[0] else filt_2d[0]
                sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)
                self._ax.plot(t, sig_norm + offset + (0.3 if mode == "Both" else 0),
                              color=colors_filt, lw=0.6, alpha=0.85, label="Filtered" if i == 0 else "")

        self._ax.set_xlabel("Time (s)", color="#cccccc")
        self._ax.set_ylabel("Channel", color="#cccccc")
        self._ax.set_title("EEG Signals", color="#eeeeee")
        self._ax.tick_params(colors="#aaaaaa")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#555555")

        if mode == "Both":
            handles, labels = self._ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            self._ax.legend(unique.values(), unique.keys(), facecolor="#2a2a2a", labelcolor="white", fontsize=8)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def highlight_seizure(self, start_s: float, end_s: float):
        self._ax.axvspan(start_s, end_s, color="#ff4444", alpha=0.25, label="Seizure")
        self._canvas.draw_idle()
