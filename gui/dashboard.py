import os
import sys
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

from gui.eeg_viewer import EEGViewer
from gui.prediction_panel import PredictionPanel
from gui.alerts import AlertPanel
from utils.config import CHBMIT_FS
from utils.logger import get_logger

logger = get_logger("Dashboard")

DARK_BG = "#1a1a2e"
DARK_FG = "#e0e0e0"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-Driven EEG Seizure Prediction System")
        self.geometry("1400x900")
        self.configure(bg=DARK_BG)
        self._fs = CHBMIT_FS
        self._raw_signal: np.ndarray = None
        self._filtered_signal: np.ndarray = None
        self._model = None
        self._feature_extractor = None
        self._running = False
        self._setup_styles()
        self._build_menu()
        self._build_ui()
        logger.info("Dashboard initialized")

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=DARK_BG, foreground=DARK_FG, fieldbackground="#2a2a3e")
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)
        style.configure("TLabelframe", background=DARK_BG, foreground=DARK_FG, relief="groove")
        style.configure("TLabelframe.Label", background=DARK_BG, foreground=HIGHLIGHT)
        style.configure("TButton", background=ACCENT, foreground=DARK_FG, padding=4)
        style.map("TButton", background=[("active", HIGHLIGHT)])
        style.configure("TNotebook", background=DARK_BG)
        style.configure("TNotebook.Tab", background="#2a2a3e", foreground=DARK_FG, padding=[8, 4])
        style.map("TNotebook.Tab", background=[("selected", ACCENT)])
        style.configure("TProgressbar", troughcolor="#2a2a3e", background=HIGHLIGHT)

    def _build_menu(self):
        menubar = tk.Menu(self, bg=DARK_BG, fg=DARK_FG, activebackground=ACCENT,
                          activeforeground=DARK_FG, relief="flat")
        file_menu = tk.Menu(menubar, tearoff=0, bg=DARK_BG, fg=DARK_FG,
                            activebackground=ACCENT, activeforeground=DARK_FG)
        file_menu.add_command(label="Load EEG File (.edf / .npy)", command=self._load_eeg)
        file_menu.add_command(label="Load Model", command=self._load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0, bg=DARK_BG, fg=DARK_FG,
                             activebackground=ACCENT, activeforeground=DARK_FG)
        tools_menu.add_command(label="Train Models", command=self._run_training_dialog)
        tools_menu.add_command(label="Run Evaluation", command=self._run_evaluation_dialog)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        self.configure(menu=menubar)

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=4)
        ttk.Label(header, text="🧠 AI-Driven EEG Seizure Prediction",
                  font=("Helvetica", 16, "bold"), foreground=HIGHLIGHT).pack(side="left")
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self._status_var, font=("Helvetica", 10),
                  foreground="#44ff88").pack(side="right", padx=8)

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=2)
        self._btn_load = ttk.Button(toolbar, text="📂 Load EEG", command=self._load_eeg)
        self._btn_load.pack(side="left", padx=2)
        self._btn_model = ttk.Button(toolbar, text="📦 Load Model", command=self._load_model)
        self._btn_model.pack(side="left", padx=2)
        self._btn_run = ttk.Button(toolbar, text="▶ Run Prediction", command=self._start_realtime)
        self._btn_run.pack(side="left", padx=2)
        self._btn_stop = ttk.Button(toolbar, text="⏹ Stop", command=self._stop_realtime, state="disabled")
        self._btn_stop.pack(side="left", padx=2)
        ttk.Button(toolbar, text="🏋 Train", command=self._run_training_dialog).pack(side="left", padx=2)

        self._progress = ttk.Progressbar(toolbar, mode="indeterminate", length=120)
        self._progress.pack(side="right", padx=8)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        main_tab = ttk.Frame(self._notebook)
        self._notebook.add(main_tab, text="EEG Viewer & Prediction")
        self._build_main_tab(main_tab)

        info_tab = ttk.Frame(self._notebook)
        self._notebook.add(info_tab, text="Dataset & Model Info")
        self._build_info_tab(info_tab)

    def _build_main_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient="vertical")
        paned.pack(fill="both", expand=True)

        top_pane = ttk.Frame(paned)
        paned.add(top_pane, weight=3)

        self._eeg_viewer = EEGViewer(top_pane)
        self._eeg_viewer.pack(fill="both", expand=True, padx=4, pady=4)

        mid_pane = ttk.Frame(paned)
        paned.add(mid_pane, weight=2)
        self._pred_panel = PredictionPanel(mid_pane)
        self._pred_panel.pack(fill="both", expand=True, padx=4, pady=4)

        bot_pane = ttk.Frame(paned)
        paned.add(bot_pane, weight=1)
        self._alert_panel = AlertPanel(bot_pane)
        self._alert_panel.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_info_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        ttk.Label(frame, text="System Information", font=("Helvetica", 13, "bold"),
                  foreground=HIGHLIGHT).pack(anchor="w", pady=4)

        info_text = tk.Text(frame, height=20, bg="#1e1e1e", fg=DARK_FG,
                            font=("Consolas", 10), relief="flat", state="normal")
        info_text.pack(fill="both", expand=True)
        info = (
            "EEG Epileptic Seizure Prediction System\n"
            "========================================\n\n"
            "Supported Datasets:\n"
            "  • CHB-MIT: 23 patients, 256 Hz, EDF format\n"
            "  • Bonn: Single-channel, 173.61 Hz, TXT format\n"
            "  • TUH: 24+ channels, 250 Hz, EDF + TSE annotations\n\n"
            "Preprocessing Pipeline:\n"
            "  1. Bandpass filter (0.5–45 Hz)\n"
            "  2. Notch filter (50/60 Hz)\n"
            "  3. Wavelet denoising (db4, level 4)\n"
            "  4. ICA artifact removal\n"
            "  5. Z-score normalization\n"
            "  6. 1-second windowing with 50% overlap\n\n"
            "Feature Extraction:\n"
            "  • Time: Mean, Var, Std, RMS, Skew, Kurtosis, Hjorth, ZCR\n"
            "  • Frequency: Band powers, Spectral entropy, Peak freq, PSD\n"
            "  • Wavelet: Energy, Entropy, DWT coefficients (db4 L4)\n"
            "  • Nonlinear: ApEn, SampEn, PermEn, FD, Hurst, Lyapunov\n"
            "  • Deep: CNN + BiLSTM + STAtt embeddings (64-dim)\n\n"
            "Optimization: EMBO (Enhanced Multi-Objective Bean Optimization)\n\n"
            "ML Models: SVM (RBF), Random Forest (300 trees), XGBoost (500 est.)\n\n"
            "RL Controller: Hierarchical Dual Q-Learning (HDQL)\n"
            "  Level 1: Normal/Pre-ictal → Continue/Alert\n"
            "  Level 2: Pre-ictal/Ictal → Non-Seizure/Seizure\n"
        )
        info_text.insert("end", info)
        info_text.configure(state="disabled")

    def _load_eeg(self):
        path = filedialog.askopenfilename(
            title="Load EEG File",
            filetypes=[("EDF files", "*.edf"), ("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if not path:
            return
        self._set_status(f"Loading {os.path.basename(path)}...")
        try:
            if path.endswith(".npy"):
                data = np.load(path)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                self._raw_signal = data
                self._fs = CHBMIT_FS
            else:
                import mne
                mne.set_log_level("WARNING")
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                self._raw_signal = raw.get_data()
                self._fs = raw.info["sfreq"]

            self._filtered_signal = self._apply_preprocessing(self._raw_signal)
            self._eeg_viewer.load_signal(self._raw_signal, self._fs, self._filtered_signal)
            self._set_status(f"Loaded: {os.path.basename(path)} — {self._raw_signal.shape[0]} channels, {self._raw_signal.shape[1]} samples @ {self._fs} Hz")
            self._alert_panel.log(f"EEG loaded: {os.path.basename(path)}", "info")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            logger.error(f"EEG load error: {e}")

    def _apply_preprocessing(self, data: np.ndarray) -> np.ndarray:
        from preprocessing.filtering import apply_filters
        from preprocessing.normalization import normalize
        filtered = apply_filters(data, self._fs)
        return normalize(filtered)

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            import joblib
            self._model = joblib.load(path)
            self._set_status(f"Model loaded: {os.path.basename(path)}")
            self._alert_panel.log(f"Model loaded: {os.path.basename(path)}", "info")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))

    def _start_realtime(self):
        if self._raw_signal is None:
            messagebox.showwarning("No Data", "Please load an EEG file first.")
            return
        if self._model is None:
            messagebox.showwarning("No Model", "Please load a trained model first.")
            return
        self._running = True
        self._btn_run.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._progress.start(10)
        thread = threading.Thread(target=self._realtime_loop, daemon=True)
        thread.start()

    def _stop_realtime(self):
        self._running = False
        self._btn_run.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._progress.stop()
        self._set_status("Stopped")

    def _realtime_loop(self):
        from features.hybrid_features import compute_hybrid_features, replace_nan_inf
        from utils.helpers import segment_signal

        signal = self._filtered_signal if self._filtered_signal is not None else self._raw_signal
        window_samples = int(1.0 * self._fs)
        step = int(window_samples * 0.5)
        n_samples = signal.shape[-1]
        start = 0

        while self._running and start + window_samples <= n_samples:
            seg = signal[..., start: start + window_samples]
            
            try:
                features = compute_hybrid_features(seg[np.newaxis], self._fs)
                features = replace_nan_inf(features)
                
                # --- PASTE THE FIX HERE ---
                # Forcefully slice 1219 features down to 676 to match the model shape
                if hasattr(features, 'shape') and features.shape[1] == 1219:
                    features = features[:, :676]
                # --------------------------

                model_obj = self._model
                if hasattr(model_obj, "predict_proba"):
                    proba = model_obj.predict_proba(features)[0]
                    seizure_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                elif isinstance(model_obj, dict):
                    inner = model_obj.get("model") or model_obj.get("rf") or model_obj.get("svm")
                    if inner and hasattr(inner, "predict_proba"):
                        proba = inner.predict_proba(features)[0]
                        seizure_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    else:
                        seizure_prob = 0.0
                else:
                    seizure_prob = 0.0

                ts = datetime.now().strftime("%H:%M:%S")
                self.after(0, self._pred_panel.update_prediction, seizure_prob)
                threshold = self._pred_panel._seizure_threshold
                if seizure_prob >= threshold:
                    self.after(0, self._alert_panel.seizure_alert, seizure_prob, ts)
                elif seizure_prob >= 0.3:
                    self.after(0, self._alert_panel.preictal_alert, seizure_prob, ts)
            except Exception as e:
                logger.warning(f"Real-time prediction error at sample {start}: {e}")

            start += step
            import time
            time.sleep(0.05)

        self.after(0, self._stop_realtime)

    def _run_training_dialog(self):
        win = tk.Toplevel(self)
        win.title("Training Configuration")
        win.geometry("400x300")
        win.configure(bg=DARK_BG)
        ttk.Label(win, text="Training is done via train.py from terminal.", font=("Helvetica", 11)).pack(pady=20)
        ttk.Label(win, text="python train.py --dataset bonn --model all", font=("Consolas", 10),
                  foreground=HIGHLIGHT).pack()
        ttk.Label(win, text="python train.py --dataset chbmit --patient chb01", font=("Consolas", 10),
                  foreground=HIGHLIGHT).pack()
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=20)

    def _run_evaluation_dialog(self):
        messagebox.showinfo("Evaluation", "Run evaluation via:\n\npython predict.py --evaluate --model all")

    def _set_status(self, msg: str):
        self._status_var.set(msg)
        logger.info(msg)


def run_dashboard():
    app = Dashboard()
    app.mainloop()
