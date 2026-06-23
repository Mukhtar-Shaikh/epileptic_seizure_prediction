import numpy as np
from typing import Tuple
from utils.config import EMBO_ALPHA, EMBO_BETA, EMBO_GAMMA, EMBO_DELTA, EMBO_POPULATION, EMBO_ITERATIONS
from utils.logger import get_logger

logger = get_logger("EMBO")


def _mutual_information(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Feature relevance via mutual information with class labels."""
    from sklearn.feature_selection import mutual_info_classif
    return mutual_info_classif(X, y, random_state=42)


def _redundancy(X: np.ndarray, mask: np.ndarray) -> float:
    """Mean pairwise correlation among selected features."""
    selected = X[:, mask > 0.5]
    if selected.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(selected.T)
    n = corr.shape[0]
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.mean(np.abs(upper)))


def _class_separability(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Fisher criterion: inter-class / intra-class variance ratio."""
    selected = X[:, mask > 0.5]
    if selected.shape[1] == 0:
        return 0.0
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    overall_mean = selected.mean(axis=0)
    sb = sum(np.sum(y == c) * np.outer(selected[y == c].mean(axis=0) - overall_mean,
                                        selected[y == c].mean(axis=0) - overall_mean)
             for c in classes)
    sw = sum(np.cov(selected[y == c].T, bias=True) if selected[y == c].shape[0] > 1 else np.zeros((selected.shape[1], selected.shape[1]))
             for c in classes)
    sw_trace = np.trace(sw) + 1e-10
    return float(np.trace(sb) / sw_trace)


def _computational_cost(mask: np.ndarray) -> float:
    return float(mask.sum() / len(mask))


def fitness(
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    relevance: np.ndarray,
    alpha: float = EMBO_ALPHA,
    beta: float = EMBO_BETA,
    gamma: float = EMBO_GAMMA,
    delta: float = EMBO_DELTA,
) -> float:
    rel = float(np.dot(relevance, mask) / (relevance.sum() + 1e-10))
    red = _redundancy(X, mask)
    sep = _class_separability(X, y, mask)
    sep_norm = min(sep / (sep + 1.0), 1.0)
    cost = _computational_cost(mask)
    score = alpha * rel - beta * red + gamma * sep_norm - delta * cost
    return float(score)


class EMBO:
    """
    Enhanced Multi-Objective Bean Optimization for feature selection.

    Fitness = α·Relevance - β·Redundancy + γ·Separability - δ·Cost
    """

    def __init__(
        self,
        n_population: int = EMBO_POPULATION,
        n_iterations: int = EMBO_ITERATIONS,
        alpha: float = EMBO_ALPHA,
        beta: float = EMBO_BETA,
        gamma: float = EMBO_GAMMA,
        delta: float = EMBO_DELTA,
        threshold: float = 0.5,
    ):
        self.n_population = n_population
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.threshold = threshold
        self.best_mask_: np.ndarray = None
        self.best_fitness_: float = -np.inf
        self.selected_indices_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EMBO":
        n_features = X.shape[1]
        logger.info(f"EMBO: optimizing {n_features} features, pop={self.n_population}, iters={self.n_iterations}")

        relevance = _mutual_information(X, y)

        population = np.random.uniform(0, 1, (self.n_population, n_features))
        fit_scores = np.array([
            fitness(p > self.threshold, X, y, relevance, self.alpha, self.beta, self.gamma, self.delta)
            for p in population
        ])

        for iteration in range(self.n_iterations):
            best_idx = np.argmax(fit_scores)
            best_bean = population[best_idx].copy()

            for i in range(self.n_population):
                direction = best_bean - population[i]
                step = np.random.uniform(0.3, 0.9) * direction
                noise = np.random.normal(0, 0.1, n_features)
                candidate = np.clip(population[i] + step + noise, 0, 1)
                c_score = fitness(candidate > self.threshold, X, y, relevance, self.alpha, self.beta, self.gamma, self.delta)
                if c_score > fit_scores[i]:
                    population[i] = candidate
                    fit_scores[i] = c_score

            current_best_idx = np.argmax(fit_scores)
            if fit_scores[current_best_idx] > self.best_fitness_:
                self.best_fitness_ = fit_scores[current_best_idx]
                self.best_mask_ = population[current_best_idx] > self.threshold

            if (iteration + 1) % 10 == 0:
                logger.info(f"EMBO iter {iteration+1}/{self.n_iterations}: best_fitness={self.best_fitness_:.4f}, selected={self.best_mask_.sum()}")

        if self.best_mask_ is None or self.best_mask_.sum() == 0:
            self.best_mask_ = np.ones(n_features, dtype=bool)

        self.selected_indices_ = np.where(self.best_mask_)[0]
        logger.info(f"EMBO done: selected {len(self.selected_indices_)}/{n_features} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("EMBO not fitted yet. Call fit() first.")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
