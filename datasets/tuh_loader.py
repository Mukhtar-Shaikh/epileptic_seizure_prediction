import os
import numpy as np
from typing import List, Tuple, Dict
from utils.config import TUH_DIR, TUH_FS
from utils.logger import get_logger

logger = get_logger("TUH")


class TUHLoader:
    """
    Loader for the TUH EEG Corpus.
    Expects .edf files with paired .tse or .lbl annotation files.
    Directory: data/TUH/<patient_id>/*.edf
    """

    def __init__(self, data_dir: str = TUH_DIR, fs: float = TUH_FS):
        self.data_dir = data_dir
        self.fs = fs

    def _parse_tse(self, tse_path: str) -> List[Tuple[float, float, str]]:
        segments = []
        if not os.path.exists(tse_path):
            return segments
        with open(tse_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        label = parts[2]
                        segments.append((start, end, label))
                    except ValueError:
                        pass
        return segments

    def load_file(self, edf_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        try:
            import mne
            mne.set_log_level("WARNING")
        except ImportError:
            raise ImportError("mne is required: pip install mne")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data = raw.get_data()
        ch_names = raw.ch_names
        n_samples = data.shape[1]
        labels = np.zeros(n_samples, dtype=np.int32)

        tse_path = edf_path.replace(".edf", ".tse")
        segments = self._parse_tse(tse_path)
        for start_s, end_s, lbl in segments:
            if lbl.lower() in ("seiz", "seizure", "bckg_seiz"):
                start_idx = int(start_s * self.fs)
                end_idx = int(end_s * self.fs)
                labels[start_idx : min(end_idx, n_samples)] = 1

        return data, labels, ch_names

    def collect_edf_files(self) -> List[str]:
        edf_files = []
        if not os.path.isdir(self.data_dir):
            logger.warning(f"TUH directory not found: {self.data_dir}")
            return edf_files
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".edf"):
                    edf_files.append(os.path.join(root, f))
        return sorted(edf_files)

    def build_dataset(
        self, window_sec: float = 1.0, overlap: float = 0.5, max_files: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        from utils.helpers import segment_signal

        edf_files = self.collect_edf_files()
        if max_files:
            edf_files = edf_files[:max_files]

        X_all, y_all = [], []
        for edf_path in edf_files:
            try:
                data, labels, _ = self.load_file(edf_path)
                segs = segment_signal(data, self.fs, window_sec, overlap)
                window_samples = int(window_sec * self.fs)
                step = int(window_samples * (1 - overlap))
                for i, seg in enumerate(segs):
                    start = i * step
                    end = start + window_samples
                    seg_labels = labels[start:end]
                    y_val = 1 if seg_labels.mean() >= 0.5 else 0
                    X_all.append(seg)
                    y_all.append(y_val)
                logger.info(f"Processed {os.path.basename(edf_path)}: {len(segs)} segments")
            except Exception as e:
                logger.warning(f"Skipping {edf_path}: {e}")

        if not X_all:
            return np.empty((0,)), np.empty((0,))
        return np.array(X_all), np.array(y_all)
