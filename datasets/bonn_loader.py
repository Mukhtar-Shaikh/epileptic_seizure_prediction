import os
import numpy as np
from typing import Tuple, Dict, List
from utils.config import BONN_DIR, BONN_FS
from utils.logger import get_logger

logger = get_logger("Bonn")

BONN_SETS = {
    "Z": 0,
    "O": 0,
    "N": 0,
    "F": 0,
    "S": 1,
}


class BonnLoader:
    """
    Loader for the Bonn University EEG dataset.
    5 sets (Z, O, N, F, S), each with 100 single-channel .txt files of 4097 samples at 173.61 Hz.
    Sets S = seizure (label 1); others = non-seizure (label 0).
    Directory structure: data/Bonn/Z/*.txt, data/Bonn/S/*.txt, ...
    """

    def __init__(self, data_dir: str = BONN_DIR, fs: float = BONN_FS):
        self.data_dir = data_dir
        self.fs = fs

    def load_set(self, set_name: str) -> Tuple[np.ndarray, np.ndarray]:
        set_dir = os.path.join(self.data_dir, set_name)
        if not os.path.isdir(set_dir):
            raise FileNotFoundError(f"Bonn set directory not found: {set_dir}")

        label = BONN_SETS.get(set_name.upper(), 0)
        signals, labels = [], []

        for fname in sorted(os.listdir(set_dir)):
            fpath = os.path.join(set_dir, fname)
            if not (fname.endswith(".txt") or fname.endswith(".dat")):
                continue
            try:
                data = np.loadtxt(fpath)
                signals.append(data)
                labels.append(label)
                logger.debug(f"Loaded {fname}: shape={data.shape}")
            except Exception as e:
                logger.warning(f"Could not load {fpath}: {e}")

        return np.array(signals), np.array(labels)

    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        X_all, y_all = [], []
        for set_name in BONN_SETS:
            set_dir = os.path.join(self.data_dir, set_name)
            if not os.path.isdir(set_dir):
                logger.warning(f"Bonn set {set_name} not found at {set_dir}")
                continue
            X, y = self.load_set(set_name)
            X_all.append(X)
            y_all.append(y)
            logger.info(f"Set {set_name}: {len(X)} samples, label={BONN_SETS[set_name]}")

        if not X_all:
            return np.empty((0,)), np.empty((0,))
        return np.vstack(X_all), np.concatenate(y_all)

    def build_windowed_dataset(
        self, window_sec: float = 1.0, overlap: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        from utils.helpers import segment_signal

        X_all, y_all = self.load_all()
        if X_all.ndim == 1:
            return np.empty((0,)), np.empty((0,))

        X_win, y_win = [], []
        for signal, label in zip(X_all, y_all):
            signal_2d = signal[np.newaxis, :]
            segs = segment_signal(signal_2d, self.fs, window_sec, overlap)
            for seg in segs:
                X_win.append(seg)
                y_win.append(label)

        return np.array(X_win), np.array(y_win)
