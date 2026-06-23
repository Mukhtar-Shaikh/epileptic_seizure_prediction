import os
import re
import numpy as np
from typing import List, Tuple, Dict
from utils.config import CHBMIT_DIR, CHBMIT_FS
from utils.logger import get_logger

logger = get_logger("CHBMIT")


class CHBMITLoader:
    """
    Loader for the CHB-MIT Scalp EEG dataset.
    Expects .edf files organized per patient: data/CHBMIT/chb01/chb01_01.edf
    and summary files: data/CHBMIT/chb01/chb01-summary.txt
    """

    def __init__(self, data_dir: str = CHBMIT_DIR, fs: float = CHBMIT_FS):
        self.data_dir = data_dir
        self.fs = fs

    def _parse_summary(self, summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
        seizure_times: Dict[str, List[Tuple[float, float]]] = {}
        current_file = None
        with open(summary_path, "r") as f:
            for line in f:
                line = line.strip()
                m = re.match(r"File Name:\s+(.+\.edf)", line, re.IGNORECASE)
                if m:
                    current_file = m.group(1).strip()
                    seizure_times.setdefault(current_file, [])
                if current_file:
                    ms = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)", line, re.IGNORECASE)
                    me = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)", line, re.IGNORECASE)
                    if ms:
                        seizure_times[current_file].append((float(ms.group(1)), None))
                    if me and seizure_times.get(current_file):
                        last = seizure_times[current_file][-1]
                        seizure_times[current_file][-1] = (last[0], float(me.group(1)))
        return seizure_times

    def load_patient(
        self, patient_id: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Returns (signals_list, labels_list, channel_names) for all EDF files of a patient.
        Each signal is shape (n_channels, n_samples).
        Labels are per-sample: 1 = seizure, 0 = normal.
        """
        try:
            import mne
            mne.set_log_level("WARNING")
        except ImportError:
            raise ImportError("mne is required: pip install mne")

        patient_dir = os.path.join(self.data_dir, patient_id)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        summary_files = [f for f in os.listdir(patient_dir) if "summary" in f.lower()]
        seizure_map: Dict[str, List[Tuple[float, float]]] = {}
        if summary_files:
            seizure_map = self._parse_summary(os.path.join(patient_dir, summary_files[0]))

        edf_files = sorted(f for f in os.listdir(patient_dir) if f.endswith(".edf"))
        signals_list, labels_list, channel_names = [], [], []

        for edf_name in edf_files:
            edf_path = os.path.join(patient_dir, edf_name)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                data = raw.get_data()
                ch_names = raw.ch_names
                n_samples = data.shape[1]
                labels = np.zeros(n_samples, dtype=np.int32)
                for start_s, end_s in seizure_map.get(edf_name, []):
                    if end_s is None:
                        continue
                    start_idx = int(start_s * self.fs)
                    end_idx = int(end_s * self.fs)
                    labels[start_idx : min(end_idx, n_samples)] = 1
                signals_list.append(data)
                labels_list.append(labels)
                if not channel_names:
                    channel_names = ch_names
                logger.info(f"Loaded {edf_name}: shape={data.shape}, seizure_samples={labels.sum()}")
            except Exception as e:
                logger.warning(f"Could not load {edf_name}: {e}")

        return signals_list, labels_list, channel_names

    def load_all_patients(self) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[str]]]:
        result = {}
        if not os.path.isdir(self.data_dir):
            logger.warning(f"CHB-MIT directory not found: {self.data_dir}")
            return result
        for patient_id in sorted(os.listdir(self.data_dir)):
            patient_path = os.path.join(self.data_dir, patient_id)
            if os.path.isdir(patient_path) and patient_id.startswith("chb"):
                try:
                    result[patient_id] = self.load_patient(patient_id)
                except Exception as e:
                    logger.error(f"Failed to load patient {patient_id}: {e}")
        return result

    def iter_segments(
        self,
        patient_ids: List[str] = None,
        window_sec: float = 1.0,
        overlap: float = 0.5,
        max_files: int = None,
    ):
        """
        Generator: yields (segments, labels) one EDF file at a time.
        segments shape: (n_windows, n_channels, window_samples)  dtype=float32
        labels shape:   (n_windows,)  dtype=int32
        Memory stays bounded to one file at a time.
        """
        import gc
        from utils.helpers import segment_signal

        patients = patient_ids or (
            sorted(
                d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith("chb")
            )
            if os.path.isdir(self.data_dir)
            else []
        )

        files_processed = 0
        for pid in patients:
            patient_dir = os.path.join(self.data_dir, pid)
            summary_files = [f for f in os.listdir(patient_dir) if "summary" in f.lower()]
            seizure_map: Dict[str, List[Tuple[float, float]]] = {}
            if summary_files:
                seizure_map = self._parse_summary(os.path.join(patient_dir, summary_files[0]))

            edf_files = sorted(f for f in os.listdir(patient_dir) if f.endswith(".edf"))
            for edf_name in edf_files:
                if max_files is not None and files_processed >= max_files:
                    return
                edf_path = os.path.join(patient_dir, edf_name)
                try:
                    import mne
                    mne.set_log_level("WARNING")
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    data = raw.get_data().astype(np.float32)
                    del raw
                    n_samples = data.shape[1]
                    labels = np.zeros(n_samples, dtype=np.int32)
                    for start_s, end_s in seizure_map.get(edf_name, []):
                        if end_s is None:
                            continue
                        s_idx = int(start_s * self.fs)
                        e_idx = int(end_s * self.fs)
                        labels[s_idx : min(e_idx, n_samples)] = 1

                    segs = segment_signal(data, self.fs, window_sec, overlap)
                    segs = segs.astype(np.float32)
                    window_samples = int(window_sec * self.fs)
                    step = int(window_samples * (1 - overlap))
                    seg_labels = np.array([
                        1 if labels[i * step: i * step + window_samples].mean() >= 0.5 else 0
                        for i in range(len(segs))
                    ], dtype=np.int32)

                    logger.info(f"Streaming {edf_name}: {len(segs)} windows, seizure={seg_labels.sum()}")
                    files_processed += 1
                    yield segs, seg_labels

                    del data, segs, seg_labels, labels
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Skipping {edf_name}: {e}")

    def build_dataset(
        self, patient_ids: List[str] = None, window_sec: float = 1.0, overlap: float = 0.5,
        max_files: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load entire dataset into RAM. Use iter_segments() for large datasets."""
        X_all, y_all = [], []
        for segs, seg_labels in self.iter_segments(patient_ids, window_sec, overlap, max_files):
            X_all.append(segs)
            y_all.append(seg_labels)
        if not X_all:
            return np.empty((0,)), np.empty((0,))
        return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)
