# Machine Learning based Sequential Change Detection Schemes

At the moment, this repository implements and evaluates a sequential change detection framework using the Cumulative Sum (CuSum) algorithm from [Xie et. al.](https://arxiv.org/abs/2104.04186), enhanced with estimated likelihood ratios (LLRs) via convex risk minimization, based on methods outlined by [Nguyen et al.](https://arxiv.org/abs/0809.0853), inshort, the project is exploring change detection in real-time data streams.

## Problem Statement

The core task addressed by this project is to detect an unknown changepoint in a sequence of observations drawn from two distinct distributions:

* Observations \$X\_i \sim P\$ (pre-change) for \$i \leq T\$.
* Observations \$X\_i \sim Q\$ (post-change) for \$i > T\$.

Here, \$P\$ and \$Q\$ are unknown probability distributions, and obviously, \$i\$ is the changepoint. Our goal is to detect the change from \$P\$ to \$Q\$ as soon as possible while controlling the false alarm rate.

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

Due to unknown distributions \$P\$ and \$Q\$, we employ convex risk minimization methods as introduced by Nguyen et al. to estimate the likelihood ratio directly from a reference dataset containing samples from both distributions.

Specifically, we solve the convex optimization:

$\min_{g \in \mathcal{G}} E_{Q}[g(X)] - E_{P}[\log(g(X))] + \frac{\lambda}{2} I^2(g),$

where \$\mathcal{G}\$ is a suitable function class and \$I(g)\$ is a complexity regularizer.

## References

* Granjon, P. (2013). *The CUSUM algorithm - a small review*.
* Xie, L., et al. (2021). *Sequential (Quickest) Change Detection: Classical Results and New Directions*.
* Nguyen, X., Wainwright, M., Jordan, M. (2007). *Estimating divergence functionals and the likelihood ratio by convex risk minimization*.
