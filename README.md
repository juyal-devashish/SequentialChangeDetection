# ML based Sequential Change Detection Schemes

At the moment, this repository implements and evaluates a sequential change detection framework using the Cumulative Sum (CuSum) algorithm from [Xie et. al.](https://arxiv.org/abs/2104.04186), enhanced with estimated likelihood ratios (LLRs) via convex risk minimization, based on methods outlined by [Nguyen et al.](https://arxiv.org/abs/0809.0853), inshort, the project is exploring change detection in real-time data streams.

## Problem Statement

The core task addressed by this project is to detect an unknown changepoint in a stream of observations drawn from two distinct distributions.

### Stream of Independent \(X\)-valued Observations

Let \(X_1, X_2, \dots\) denote a stream of independent \(X\)-valued observations. For some (unknown)  
\[
T \in \mathbb{N} \cup \{\infty\},
\]  
we have
\[
X_t \sim
\begin{cases}
P_0, & t \le T,\\
P_1 \neq P_0, & t > T,
\end{cases}
\]
where:
- Both \(P_0\) and \(P_1\) are unknown.
- \(P_0, P_1 \in \mathcal{P}\) for a known class of distributions \(\mathcal{P}\).

### Hypothesis Testing

We wish to decide between the two hypotheses
\[
\begin{aligned}
H_0 &: T = \infty \quad (\text{no changepoint}),\\
H_1 &: T < \infty \quad (\text{changepoint exists}).
\end{aligned}
\]

### Objective

Define a stopping time \(\tau\) at which we declare “change detected” so as to
1. **Minimize false alarms** under \(H_0\), and  
2. **Minimize detection delay**  
   \[
     (\tau - T)^+ \;=\; \max\{\tau - T,\,0\}
   \]  
   under \(H_1\).

## Performance Measures

### 1. No Changepoint (\(T = \infty\))

For a desired false‑alarm level \(\alpha \in (0, 1)\), one may control either
- **Average Run Length (ARL):**  
  \[
    \mathbb{E}_\infty[\tau] \;\ge\; \frac{1}{\alpha},
  \]
- **Probability of False Alarm (PFA):**  
  \[
    P_\infty(\tau < \infty) \;\le\; \alpha.
  \]

### 2. Changepoint Exists (\(T < \infty\))

Under \(H_1\), we require that the detection delay be small in a worst‑case sense:
\[
\text{(i.e.)}\quad \sup_{T<\infty}\mathbb{E}_T\bigl[(\tau - T)^+\bigr]
\quad\text{or}\quad
\sup_{T<\infty}P_T\bigl((\tau - T)^+ > d\bigr)
\]
for small \(d\).


## Tentative Objectives (Copied from Slack)
### 1. Implementation and Theoretical Validation of CuSum
- Implement the CuSum (Cumulative Sum) scheme in Python.
- Test the implementation with a set of examples.
- Plot the detection delay vs. effect size curve and compare the results with the theoretical predictions.
- Plot the histogram of stopping times when there is no change, and verify if the Average Run Length (ARL) constraint is satisfied.

### 2. Empirical Study of Likelihood Ratio Misspecification
- CuSum requires the exact knowledge of the likelihood ratio for optimal performance.
- Conduct an empirical study on the effect of misspecification of the likelihood ratio.
- Investigate the consequences of a mismatch between the true likelihood ratio and the one used by CuSum.
- Identify benign mismatches and understand the types of mismatches that are more costly.

### 3. Learning Likelihood Ratio Model from Training Data
- Assume the availability of training data from pre-change and post-change distributions.
- Use this data to learn a likelihood ratio model based on the variational definition from Nguyen, Wainwright, and Jordan.
- Integrate the learned model into the CuSum scheme.
- Empirically study the impact of different factors on performance:
  1. Size of the training datasets.
  2. Complexity of machine learning models used for likelihood ratio approximation (e.g., linear models, SVMs, neural networks, tree models).
  3. Effect of data imbalance (e.g., having a large number of pre-change data compared to post-change data).
  4. Impact of using different f-divergences other than the default relative entropy in the variational definition of the likelihood ratio.

## Methodology

### CUSUM Algorithm from Xie et. al.

CUSUM monitors the cumulative sum of log-likelihood ratios (LLRs) to signal distribution changes. The algorithm stops and flags a change when the statistic surpasses a defined threshold, optimized based on false alarm rates.

### Likelihood Ratio Estimation from Nguyen et. al.

Due to unknown distributions \$P\$ and \$Q\$, we employ convex risk minimization methods as introduced by Nguyen et al. to estimate the likelihood ratio containing samples from both distributions.

## References

* Granjon, P. (2013). *The CUSUM algorithm - a small review*.
* Xie, L., et al. (2021). *Sequential (Quickest) Change Detection: Classical Results and New Directions*.
* Nguyen, X., Wainwright, M., Jordan, M. (2007). *Estimating divergence functionals and the likelihood ratio by convex risk minimization*.
