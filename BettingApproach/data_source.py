# data_source.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple

__all__ = [
    "DataSource",
    "OneHotSource",
    "GaussianMixtureSource",
    "MNISTSource",
    "CIFAR10Source",
]

class DataSource(ABC):
    """
    Abstract base class for data streams.
    """
    @abstractmethod
    def reset(self):
        """Rewind to the start of the stream."""
        pass

    @abstractmethod
    def next(self):
        """Return the next sample from the stream. Raises StopIteration when exhausted."""
        pass

class ArraySource(DataSource):
    """
    Wrapper for a simple array of data. 
    """
    def __init__(self, X):
        self.X = X
        self.reset()
    def reset(self):
        self._i = 0
    def next(self):
        if self._i >= len(self.X):
            raise StopIteration
        x = self.X[self._i]
        self._i += 1
        return x


class OneHotSource(DataSource):
    """
    Emits one-hot vectors of length L+1 according to a fixed probability distribution p.
    Also tracks the optimal index argmax(p).
    """
    def __init__(self, p: np.ndarray, length: int):
        assert p.ndim == 1 and np.isclose(p.sum(), 1.0), "p must be a probability vector"
        self.p = p
        self.length = length
        self.optimal_index = int(np.argmax(p))  # most likely class under p
        self.reset()

    def reset(self):
        self._i = 0

    def next(self) -> np.ndarray:
        if self._i >= self.length:
            raise StopIteration
        self._i += 1
        j = np.random.choice(len(self.p), p=self.p)
        vec = np.zeros_like(self.p, dtype=int)
        vec[j] = 1
        return vec

class GaussianMixtureSource(DataSource):
    """
    Emits continuous samples from a mixture of Gaussians in R^d.
    Each component k has weight w_k, mean mu_k, and covariance Sigma_k.
    """
    def __init__(self,
                 weights: np.ndarray,
                 mus: np.ndarray,
                 covs: np.ndarray,
                 length: int):
        assert np.isclose(weights.sum(), 1.0), "weights must sum to 1"
        self.weights = weights
        self.mus = mus
        self.covs = covs
        self.n_components = len(weights)
        self.length = length
        self.reset()

    def reset(self):
        self._i = 0

    def next(self) -> np.ndarray:
        if self._i >= self.length:
            raise StopIteration
        self._i += 1
        k = np.random.choice(self.n_components, p=self.weights)
        return np.random.multivariate_normal(self.mus[k], self.covs[k])

