# ML based Sequential Change Detection Schemes

At the moment, this repository implements and evaluates a sequential change detection framework using the Cumulative Sum (CuSum) algorithm from [Xie et. al.](https://arxiv.org/abs/2104.04186), enhanced with estimated likelihood ratios (LLRs) via convex risk minimization, based on methods outlined by [Nguyen et al.](https://arxiv.org/abs/0809.0853), inshort, the project is exploring change detection in real-time data streams.

## Problem Statement

The core task addressed by this project is to detect an unknown changepoint in a sequence of observations drawn from two distinct distributions:

### Stream of Independent \( X \)-valued Observations

Let \( X_1, X_2, \dots \) denote a stream of independent \( X \)-valued observations:
- For some \( T \in \mathbb{N} \cup \{\infty\} \):
  - \( X_t \sim P_0 \) for \( t \leq T \),
  - \( X_t \sim P_1 \neq P_0 \) for \( t > T \).
  
### Mild Requirements on the Distributions:
- Both \( P_0 \) and \( P_1 \) are unknown.
- \( P_0, P_1 \in \mathcal{P} \) for some known class of distributions \( \mathcal{P} \).

### Hypothesis Testing:
Decide between:
- \( H_0 : T = \infty \) (no changepoint), versus
- \( H_1 : T < \infty \) (changepoint exists).

### Objective:
Define a stopping time \( \tau \) to declare a detection, that:
- Minimizes false alarms under \( H_0 \), and
- Has a small detection delay, \( (\tau - T)^+ \), under \( H_1 \).

## Performance Measures

### When \( T = \infty \) (No Changepoint):
For an \( \alpha \in (0, 1) \), control:
- **Average Run Length (ARL)**: \( \mathbb{E}_\infty[\tau] \geq \frac{1}{\alpha} \), or
- **Probability of False Alarm (PFA)**: \( P_\infty(\tau < \infty) \leq \alpha \).

### When \( T < \infty \):
Ensure that:
- \( (\tau - T)^+ \) is small, either in expectation or with high probability.
- The guarantee should hold for the worst-case choice of \( T \).


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
