import numpy as np
from sklearn.svm import SVR
from scipy.stats import norm

class SVMBasedCUSUM:
    """
    SVM-based CuSum detector for sequential change detection.

    Trains an SVR model to estimate the log-likelihood ratio between two distributions
    and uses the estimated scores in a CuSum loop.
    """
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.model = SVR()

    def fit(self, X_p, X_q):
        """
        Fit the SVR model on samples from two distributions P and Q.

        Parameters:
        - X_p: np.ndarray, samples from distribution P (post-change)
        - X_q: np.ndarray, samples from distribution Q (pre-change)
        """
        X = np.concatenate([X_p, X_q])[:, None]
        # Use approximate log-likelihood ratio for training targets
        y = np.log(norm.pdf(X.ravel(), 1.0, 1.0) / norm.pdf(X.ravel(), 0.0, 1.0))
        self.model.fit(X, y)

    def run(self, stream):
        """
        Run the CUSUM procedure using predicted LLRs from the trained SVM model.

        Parameters:
        - stream: np.ndarray, sequence of observations

        Returns:
        - W: np.ndarray, CuSum statistics
        - tau: int or None, stopping time if threshold crossed
        """
        llrs = self.model.predict(stream[:, None])
        W = np.zeros(len(stream))
        tau = None
        for n in range(1, len(stream)):
            W[n] = max(0, W[n - 1] + llrs[n])
            if tau is None and W[n] >= self.threshold:
                tau = n
        return W, tau
