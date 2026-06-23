import numpy as np
from scipy import signal as sp_signal
from utils.config import BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ
from utils.logger import get_logger

logger = get_logger("Filter")


def bandpass_filter(data: np.ndarray, fs: float, low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = min(high / nyq, 0.999)
    b, a = sp_signal.butter(order, [low_n, high_n], btype="band")
    if data.ndim == 1:
        return sp_signal.filtfilt(b, a, data)
    return np.array([sp_signal.filtfilt(b, a, ch) for ch in data])


def notch_filter(data: np.ndarray, fs: float, freq: float = NOTCH_FREQ, Q: float = 30.0) -> np.ndarray:
    nyq = fs / 2.0
    w0 = freq / nyq
    if w0 >= 1.0:
        logger.warning(f"Notch frequency {freq} Hz exceeds Nyquist {nyq} Hz, skipping.")
        return data
    b, a = sp_signal.iirnotch(w0, Q)
    if data.ndim == 1:
        return sp_signal.filtfilt(b, a, data)
    return np.array([sp_signal.filtfilt(b, a, ch) for ch in data])


def apply_filters(data: np.ndarray, fs: float, notch_freqs=(50.0, 60.0)) -> np.ndarray:
    data = bandpass_filter(data, fs)
    for nf in notch_freqs:
        data = notch_filter(data, fs, freq=nf)
    logger.debug(f"Filtering complete: shape={data.shape}")
    return data
