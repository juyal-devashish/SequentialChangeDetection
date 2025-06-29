import numpy as np
from data_source import OneHotSource

class IdentityClassifier:
    """Classifier that treats its input as a one-hot or probability vector
    and returns the argmax index."""
    def predict(self, x):
        return int(np.argmax(x))


class SequentialHypothesisTest:
    """
    Implements the e-process–based sequential hypothesis test:
      H0: class 0 is most likely
      H1: some other class is most likely

    Parameters
    ----------
    classifier : object
        Any object with a .predict(x) -> int in {0,…,L} interface.
    n_classes : int
        Total number of classes L+1.
    alpha : float
        Target Type-I error rate. Threshold = 1/alpha.
    M : int, optional
        Number of grid points to discretize λ over (-1,0], by default 200.
    smoothing : float, optional
        Pseudocount added to counts, by default 0.5.
    grid : np.ndarray, optional
        Custom grid of λ values (length M), by default None.
    N_max : int or None, optional
        Maximum number of steps to run before giving up (None = no cap).
    """
    def __init__(self,
                 classifier,
                 n_classes: int,
                 alpha: float,
                 M: int = 200,
                 smoothing: float = 0.5,
                 grid: np.ndarray = None,
                 N_max: int = None, 
                 eta: float = 1.0,        # ONS learning rate, Unused
                 reg: float = 1e-2        # initial regularization for A, Unused
                 ):
        self.clf = classifier
        self.L = n_classes - 1
        self.alpha = alpha
        self.threshold = 1.0 / alpha
        self.smooth = smoothing
        self.N_max = N_max if N_max is not None else float('inf')

        # build λ-grid
        if grid is not None:
            assert grid.ndim == 1 and len(grid) == M
            self.lambdas = grid
        else:
            self.lambdas = np.linspace(-1.0, 0.0, num=M, endpoint=True)
        self.M = len(self.lambdas)

        self.reset()

    def reset(self):
        """Re-initialize all internal state."""
        self.t = 0
        # counts of observed predicted labels
        self.N = np.zeros(self.L + 1, dtype=float)
        # initial smoothed estimate p̂₀ = [1/(L+1), …]
        base = (self.L + 1) * self.smooth
        self.p_hat = np.full(self.L + 1, fill_value=self.smooth / base)
        # initial chosen index ĵ₁
        self.hat_j = 0
        # e-process wealth vector W₀(i) = 1
        self.W = np.ones(self.M, dtype=float)

    def update(self, x_t):
        """
        Process one observation x_t, update the e-process.

        Parameters
        ----------
        x_t : input to classifier or one-hot vector

        Returns
        -------
        stopped : bool
            True if W̄_t ≥ 1/α (reject H0).
        tau : int or None
            The stopping time if stopped; otherwise None.
        """
        self.t += 1

        # ensure x_t is 2D if coming from sklearn
        xt = np.atleast_2d(x_t)

        # 1) get classifier label
        pred = self.clf.predict(xt)
        # if sklearn, pred is an array; if IdentityClassifier, pred is int
        ell_t = int(pred[0]) if hasattr(pred, "__len__") else int(pred)

        # 2) compute δ_t = 1{ell_t=0} - 1{ell_t=hat_j}
        delta = (1 if ell_t == 0 else 0) - (1 if ell_t == self.hat_j else 0)

        # 3) update wealth vector
        self.W *= (1.0 + self.lambdas * delta)

        # 4) check threshold
        W_bar = self.W.mean()
        if W_bar >= self.threshold:
            return True, self.t

        # 5) update counts based on the observed label ell_t
        self.N[ell_t] += 1
        denom = self.t + (self.L + 1) * self.smooth
        self.p_hat = (self.N + self.smooth) / denom

        # 6) update hat_j to argmax p̂
        self.hat_j = int(np.argmax(self.p_hat))

        # 7) if reached N_max without stopping, treat as no-reject
        if self.t >= self.N_max:
            return False, int(self.N_max)

        return False, None

    def run(self, X_seq):
        """
        Run on an iterable of observations.

        Returns
        -------
        (flag, time)
        flag : bool
            True if H0 rejected; False otherwise.
        time : int
            Stopping time if rejected, or N_max if not.
        """
        self.reset()
        for x in X_seq:
            stopped, tau = self.update(x)
            if stopped or tau == self.N_max:
                return stopped, tau if stopped else int(self.N_max)
        # exhausted sequence without reject
        return False, min(self.t, int(self.N_max))

    def run_on_source(self, src):
        """
        Drive the test using a DataSource with .reset() and .next().

        Parameters
        ----------
        src : DataSource
            Any object implementing reset() and next().

        Returns
        -------
        (flag, time)
        """
        src.reset()
        self.reset()
        while self.t < self.N_max:
            try:
                x = src.next()
            except StopIteration:
                break
            stopped, tau = self.update(x)
            if stopped or tau == self.N_max:
                return stopped, tau if stopped else int(self.N_max)
        return False, int(min(self.t, self.N_max))

class SequentialHypothesisTestONS:
    """
    E-process–based sequential test with betting fraction chosen by Online Newton Step (ONS).
    Everything else matches SequentialHypothesisTest, except no λ-grid.
    """
    def __init__(
        self,
        classifier,
        n_classes: int,
        alpha: float,
        M: int = 200,            # kept for interface compatibility, but unused
        smoothing: float = 0.5,
        grid: np.ndarray = None, # unused
        N_max: int = None,
        eta: float = 1.0,        # ONS learning rate
        reg: float = 1e-2        # initial regularization for A
    ):
        # --- same initialization as SequentialHypothesisTest ---
        self.clf = classifier
        self.L = n_classes - 1
        self.alpha = alpha
        self.threshold = 1.0 / alpha
        self.smooth = smoothing
        self.N_max = N_max if N_max is not None else float('inf')
        # counts for smoothing
        base = (self.L + 1) * self.smooth
        self.p_hat = np.full(self.L + 1, fill_value=self.smooth / base)
        self.hat_j = 0
        # ONS-specific state
        self.eta = eta
        self.A = reg        # scalar in 1-D
        self.lambda_ = 0.0  # initial betting fraction
        # wealth process
        self.W = 1.0
        # time
        self.t = 0

    def reset(self) -> None:
        """Reset time, classifier, counts, and ONS state."""
        self.t = 0
        base = (self.L + 1) * self.smooth
        self.p_hat = np.full(self.L + 1, fill_value=self.smooth / base)
        self.hat_j = 0
        self.A = self.A  # keep initial reg
        self.lambda_ = 0.0
        self.W = 1.0

    def update(self, x_t):
        """
        Process one observation, update ONS bet and wealth.
        Returns (stopped: bool, tau: Optional[int]).
        """
        self.t += 1
        # 1) classify
        ell = self.clf.predict(x_t)
        # 2) compute delta: 1{ell==0} - 1{ell==hat_j}
        delta = (1 if ell == 0 else 0) - (1 if ell == self.hat_j else 0)
        # 3) compute gradient of loss = -log(1+lambda*delta)
        #    grad = -delta / (1 + lambda * delta)
        grad = -delta / max(1e-12, (1.0 + self.lambda_ * delta))
        # 4) ONS update: A += grad^2, lambda ← lambda - eta * grad / A
        self.A += grad * grad
        self.lambda_ -= self.eta * grad / self.A
        # project lambda to [-1,0]
        self.lambda_ = min(0.0, max(-1.0, self.lambda_))
        # 5) update wealth: W ← W * (1 + lambda * delta)
        self.W *= (1.0 + self.lambda_ * delta)
        # 6) check threshold
        if self.W >= self.threshold:
            return True, self.t
        # 7) update counts, p_hat, hat_j (same as original)
        self.p_hat[ell] += 1
        normalized = self.p_hat / self.p_hat.sum()
        # hat_j = argmax over 1..L of normalized[j]
        self.hat_j = int(np.argmax(normalized[1:]) + 1)
        # 8) enforce N_max if desired
        if self.t >= self.N_max:
            return False, int(self.N_max)
        return False, None

    def run_on_source(self, src: OneHotSource):
        """
        Drive test on a DataSource; return (flag, stop_time).
        """
        src.reset()
        self.reset()
        for x in src:
            stopped, tau = self.update(x)
            if stopped or tau == self.N_max:
                return stopped, tau
        return False, int(min(self.t, self.N_max))



if __name__ == '__main__':

    # Experiment: Type I error and expected stopping time under alternative
    n_classes = 3
    alpha = 0.05
    M = 100
    smoothing = 0.5
    N_max = 2000
    runs = 200
    p0 = np.array([0.6, 0.2, 0.2])  # null distribution
    p1 = np.array([0.2, 0.6, 0.2])  # alternative distribution

    # Type I error
    errors = 0
    for _ in range(runs):
        src = OneHotSource(p0, N_max)
        test = SequentialHypothesisTest(IdentityClassifier(), n_classes, alpha, M, smoothing, N_max=N_max)
        flag, t = test.run_on_source(src)
        if flag:
            errors += 1
    print(f"Type I error (alpha={alpha}): {errors/runs:.3f}")

    # Expected stopping time under alternative
    times = []
    for _ in range(runs):
        src = OneHotSource(p1, N_max)
        test = SequentialHypothesisTest(IdentityClassifier(), n_classes, alpha, M, smoothing, N_max=N_max)
        flag, t = test.run_on_source(src)
        times.append(t)
    print(f"Expected stopping time (alpha={alpha}): {np.mean(times):.2f}")


