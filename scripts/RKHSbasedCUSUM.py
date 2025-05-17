
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize

class RKHSLikelihoodRatioEstimator:
    def __init__(self, sigma=1.0, lambda_reg=1e-2):
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.alpha = None
        self.X_p = None

    def _compute_kernel_matrices(self, X_p, X_q):
        gamma = 1.0 / (2 * self.sigma ** 2)
        K_PP = rbf_kernel(X_p[:, np.newaxis], X_p[:, np.newaxis], gamma=gamma)
        K_QP = rbf_kernel(X_q[:, np.newaxis], X_p[:, np.newaxis], gamma=gamma)
        return K_PP, K_QP

    def _loss(self, alpha, K_PP, K_QP):
        n_p = K_PP.shape[0]
        n_q = K_QP.shape[0]
        g_q = K_QP @ alpha
        g_p = K_PP @ alpha
        g_p = np.clip(g_p, 1e-8, None)
        term_q = np.mean(g_q)
        term_p = -np.mean(np.log(g_p))
        reg = 0.5 * self.lambda_reg * (alpha @ K_PP @ alpha)
        return term_q + term_p + reg

    def _grad(self, alpha, K_PP, K_QP):
        n_p = K_PP.shape[0]
        g_p = K_PP @ alpha
        g_p = np.clip(g_p, 1e-8, None)
        grad_q = np.mean(K_QP, axis=0)
        grad_p = -(K_PP @ (1.0 / g_p)) / n_p
        grad_reg = self.lambda_reg * (K_PP @ alpha)
        return grad_q + grad_p + grad_reg

    def fit(self, X_p, X_q):
        self.X_p = X_p
        K_PP, K_QP = self._compute_kernel_matrices(X_p, X_q)
        n_p = X_p.shape[0]
        alpha0 = np.ones(n_p) / n_p
        bounds = [(0, None)] * n_p
        result = minimize(
            fun=lambda a: self._loss(a, K_PP, K_QP),
            x0=alpha0,
            jac=lambda a: self._grad(a, K_PP, K_QP),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        self.alpha = result.x

    def predict_log(self, X):
        if self.alpha is None or self.X_p is None:
            raise ValueError("Estimator not fitted.")
        gamma = 1.0 / (2 * self.sigma ** 2)
        K = rbf_kernel(X[:, np.newaxis], self.X_p[:, np.newaxis], gamma=gamma)
        return np.log(np.clip(K @ self.alpha, 1e-8, None))


class RKHSBasedCUSUM:
    def __init__(self, threshold=5.0, sigma=1.0, lambda_reg=1e-2):
        self.threshold = threshold
        self.estimator = RKHSLikelihoodRatioEstimator(sigma=sigma, lambda_reg=lambda_reg)

    def fit(self, X_p, X_q):
        self.estimator.fit(X_p, X_q)

    def run(self, X_stream):
        log_llrs = self.estimator.predict_log(X_stream)
        W = np.zeros(len(X_stream))
        tau = None
        for n in range(1, len(X_stream)):
            W[n] = max(0, W[n-1] + log_llrs[n])
            if tau is None and W[n] >= self.threshold:
                tau = n
        return W, tau