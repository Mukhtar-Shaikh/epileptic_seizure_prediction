import numpy as np
import pywt
from utils.config import WAVELET, WAVELET_LEVEL
from utils.logger import get_logger

logger = get_logger("ArtifactRemoval")


def wavelet_denoise(data: np.ndarray, wavelet: str = WAVELET, level: int = WAVELET_LEVEL, threshold_mode: str = "soft") -> np.ndarray:
    def _denoise_channel(ch: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(ch)))
        denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]]
        return pywt.waverec(denoised, wavelet)[:len(ch)]

    if data.ndim == 1:
        return _denoise_channel(data)
    return np.array([_denoise_channel(ch) for ch in data])


def remove_ica_artifacts(data: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Lightweight ICA-based artifact removal using sklearn FastICA.
    Removes components with extreme kurtosis (eye blinks, muscle artifacts).
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        logger.warning("sklearn not available, skipping ICA.")
        return data

    if data.ndim == 1:
        return data

    n_ch = data.shape[0]
    n_comp = n_components or min(n_ch, 20)
    ica = FastICA(n_components=n_comp, max_iter=200, tol=0.01, random_state=42)

    try:
        sources = ica.fit_transform(data.T)
        kurts = np.array([_kurtosis(sources[:, i]) for i in range(n_comp)])
        bad = np.abs(kurts) > 5.0
        sources[:, bad] = 0.0
        reconstructed = sources @ ica.mixing_.T + ica.mean_
        logger.debug(f"ICA: removed {bad.sum()} components out of {n_comp}")
        return reconstructed.T
    except Exception as e:
        logger.warning(f"ICA failed: {e}, returning original data.")
        return data


def _kurtosis(x: np.ndarray) -> float:
    mu = x.mean()
    sigma = x.std() + 1e-10
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)


def remove_artifacts(data: np.ndarray, fs: float, use_ica: bool = True, use_wavelet: bool = True) -> np.ndarray:
    if use_wavelet:
        data = wavelet_denoise(data)
    if use_ica and data.ndim == 2:
        data = remove_ica_artifacts(data)
    logger.debug(f"Artifact removal complete: shape={data.shape}")
    return data
