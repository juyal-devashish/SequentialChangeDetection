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

    def __iter__(self):
        """Allow `for z in OneHotSource(...)` loops."""
        self.reset()
        while True:
            try:
                yield self.next()
            except StopIteration:
                break

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


def create_gap_pmf(delta: float, n_classes: int, *, alt: bool, rng=None) -> np.ndarray:
    """
    Generate a probability mass function that meets the user-specified
    “gap-b” construction.

    Construction rules
    ------------------
    1. Pick b ∼ Unif(0, (1-delta)/2).
    2. Choose j uniformly from {1,…, n_classes−1}.
    3. If alt is True :  p[0] = b,         p[j] = b + delta     (j is the max).
       If alt is False:  p[0] = b + delta, p[j] = b             (0 is the max).
    4. For every remaining index i ∉ {0, j}, draw r_i ∼ Unif(0,1).
       Scale {r_i} so their total probability mass equals 1 − (2b + delta).
    5. Return the resulting length-`n_classes` vector (sums exactly to 1).

    Parameters
    ----------
    delta : float
        The desired gap (0 < delta < 1).
    n_classes : int
        Number of classes (≥ 2; if =2 the result is deterministic).
    alt : bool
        • True  → the largest probability sits at some index j ≠ 0  
        • False → index 0 has the largest probability.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility.  A new one is created
        if None is supplied.

    Returns
    -------
    np.ndarray
        A probability vector of length `n_classes` satisfying the gap rule.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie strictly between 0 and 1.")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")

    rng = rng or np.random.default_rng()

    # 1. Draw b
    b = rng.uniform(0.0, (1.0 - delta) / 2.0)

    # 2. Pick j (cannot be 0)
    j = int(rng.integers(1, n_classes))

    pmf = np.zeros(n_classes)

    # 3–4. Set the two special coordinates
    if alt:
        pmf[0] = b
        pmf[j] = b + delta
    else:
        pmf[0] = b + delta
        pmf[j] = b

    # Remaining probability mass
    remaining_mass = 1.0 - (2.0 * b + delta)

    if remaining_mass < 0:      # should never happen given the range of b
        raise RuntimeError("Internal error: negative remaining mass")

    # 5. Fill the other coordinates (if any)
    other_idx = [i for i in range(n_classes) if i not in (0, j)]
    if other_idx:
        r = rng.random(len(other_idx))
        r_sum = r.sum()
        # In the extremely unlikely event r_sum is 0, make them uniform
        if r_sum == 0.0:
            r[:] = 1.0
            r_sum = len(r)
        pmf[other_idx] = remaining_mass * r / r_sum
    else:
        # This happens only when n_classes == 2; remaining_mass must be zero
        pmf /= pmf.sum()  # safety normalisation

    return pmf

class ChangePointSource(DataSource):
    """
    A streaming DataSource that switches from a pre-change generator to a post-change generator at a specified changepoint.

    Attributes:
        pre_source    : DataSource yielding samples under the null distribution
        post_source   : DataSource yielding samples under the alternative distribution
        T_c           : changepoint index (1-based). Observations 1..T_c from pre_source, then from post_source.
        length        : total number of samples to generate (if None, streams until sources exhaust)
        randomize_Tc  : if True, draws a new T_c uniformly in [1, length] on reset()
    """

    def __init__(
        self,
        pre_source: DataSource,
        post_source: DataSource,
        T_c: int,
        length: Optional[int] = None,
        randomize_Tc: bool = False,
    ):
        self.pre_source = pre_source
        self.post_source = post_source
        self.T_c = T_c
        self.length = length
        self.randomize_Tc = randomize_Tc
        self._t = 0

    def reset(self) -> None:
        """Reset both underlying sources and optionally randomize the changepoint."""
        self.pre_source.reset()
        self.post_source.reset()
        self._t = 0
        if self.randomize_Tc:
            if self.length is None:
                raise ValueError("Must specify 'length' to randomize changepoint.")
            self.T_c = np.random.randint(1, self.length + 1)

    def next(self):
        """Return the next sample, advancing time and switching at T_c."""
        self._t += 1
        if self.length is not None and self._t > self.length:
            raise StopIteration
        if self._t <= self.T_c:
            return self.pre_source.next()
        else:
            return self.post_source.next()

    def __iter__(self):
        """Support iteration by returning samples until exhaustion (no error on no-change)."""
        self.reset()
        while True:
            try:
                yield self.next()
            except StopIteration:
                break
