import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CHBMIT_DIR = os.path.join(DATA_DIR, "CHBMIT")
BONN_DIR = os.path.join(DATA_DIR, "Bonn")
TUH_DIR = os.path.join(DATA_DIR, "TUH")

CHBMIT_FS = 256
BONN_FS = 173.61
TUH_FS = 250

WINDOW_SIZE_SEC = 1.0
OVERLAP = 0.5

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 50.0

WAVELET = "db4"
WAVELET_LEVEL = 4

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10,
    "gamma": "scale",
    "probability": True,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": 42,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

CNN_EMBEDDING_DIM = 64
LSTM_EMBEDDING_DIM = 64
CNN_EPOCHS = 20
LSTM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

HDQL_ALPHA = 0.1
HDQL_GAMMA = 0.9
HDQL_EPSILON = 0.1
HDQL_EPISODES = 500

EMBO_ALPHA = 0.4
EMBO_BETA = 0.3
EMBO_GAMMA = 0.2
EMBO_DELTA = 0.1
EMBO_POPULATION = 30
EMBO_ITERATIONS = 50

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
