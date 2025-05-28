# ML based Sequential Change Detection Schemes

At the moment, this repository implements and evaluates a sequential change detection framework using the Cumulative Sum (CuSum) algorithm from [Xie et. al.](https://arxiv.org/abs/2104.04186), enhanced with estimated likelihood ratios (LLRs) via convex risk minimization, based on methods outlined by [Nguyen et al.](https://arxiv.org/abs/0809.0853), inshort, the project is exploring change detection in real-time data streams.

## Stream of Independent X-valued Observations

Let X₁, X₂, ... denote a stream of independent X-valued observations:
- For some T ∈ ℕ ∪ {∞}:
  - Xₜ ∼ P₀ for t ≤ T,
  - Xₜ ∼ P₁ ≠ P₀ for t > T.

### Mild Requirements on the Distributions:
- Both P₀ and P₁ are unknown.
- P₀, P₁ ∈ P for some known class of distributions P.

### Hypothesis Testing:
Decide between:
- H₀ : T = ∞ (no changepoint), versus
- H₁ : T < ∞ (changepoint exists).

### Objective:
Define a stopping time τ to declare a detection, that:
- Minimizes false alarms under H₀, and
- Has a small detection delay, (τ - T)⁺, under H₁.

## Performance Measures

### When T = ∞ (No Changepoint):
For an α ∈ (0, 1), control:
- **Average Run Length (ARL)**: E₋₀[τ] ≥ 1/α, or
- **Probability of False Alarm (PFA)**: P₋₀(τ < ∞) ≤ α.

### When T < ∞:
Ensure that:
- (τ - T)⁺ is small, either in expectation or with high probability.
- The guarantee should hold for the worst-case choice of T.



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


# 🔬 RKHS-based CuSum Experiments: Full Task Checklist

This repository contains experiments based on the variational formulation of likelihood ratio estimation (Nguyen, Wainwright, Jordan 2010), plugged into a CuSum change-point detection scheme. Below is the complete task breakdown.

---

## ✅ Core Setup

- [x] Implement RKHS-based LLR estimator (Nguyen et al.)
- [x] Implement CuSum scheme accepting `log g(x)`
- [x] Generate synthetic data from pre- and post-change distributions
- [x] Compute ground truth log-likelihood ratio `log(p/q)`
- [x] Run RKHS + CuSum baseline with balanced training data

---

## 📈 Training Size Variation

- [ ] Loop over training sizes `n ∈ {50, 100, 200, 400, 800, 1600}`
- [ ] For each `n`, generate `X_p, X_q`
- [ ] Train RKHS estimator on each pair
- [ ] Run CuSum on each, record 100 trials
- [ ] Plot average detection delay vs training size

---

## ⚙️ Model Complexity Comparison

- [ ] Train Linear Regression on labeled data
- [ ] Train SVM (RBF) with probability output
- [ ] Train Neural Network (MLP)
- [ ] Train XGBoost or Decision Tree
- [ ] Plot estimated `log g(x)` vs RKHS vs True
- [ ] Run CuSum for each model and measure detection delay
- [ ] Plot detection delay vs model type (bar plot)

---

## ⚖️ Data Imbalance Analysis

- [ ] Fix total size (e.g. 1000), vary `(n_P, n_Q)` like:
  - `(800,200)`
  - `(500,500)`
  - `(200,800)`
- [ ] Train RKHS estimator for each imbalance
- [ ] Run CuSum and record average detection delay
- [ ] Plot detection delay vs imbalance setting

---

## 🔁 Divergence Comparison

- [ ] Modify RKHS objective to support alternative `f`-divergences:
  - [ ] KL (already implemented)
  - [ ] Pearson χ²
  - [ ] Hellinger
- [ ] Train estimator with each divergence
- [ ] Run CuSum and measure delay
- [ ] Plot detection delay vs divergence

---

## 📊 Benchmark & Summary

- [ ] Compute MSE between estimated and true log-likelihood ratios
- [ ] Create summary table:
  - Rows = model / setting
  - Columns = MSE, average delay
- [ ] Final plots: summary figures for presentation
 ----

## 📅 Change Detection Paper Timeline

| **Week** | **Goals**                             | **Tasks**                                                                                                                                          | **Deliverables**                                                                 |
|----------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Week 1** | Finalize Plan and Setup               | - Confirm structure and estimator types  <br> - Reorganize existing code for empirical CUSUM  <br> - Set up template on Overleaf                   | ✅ Final project outline  <br> ✅ Overleaf template                              |
| **Week 2** | Baseline LLR Experiments              | - Implement experiments with oracle, RKHS, NN, Linear models  <br> - Evaluate ARL and delay under *ideal LLR approximation*  <br> - Sweep training sizes | 📊 Plots: Detection delay vs training size  <br> 📄 Notes: Baseline performance |
| **Week 3** | Failure Cases and Model Mismatch      | - Design failure scenarios (linear model on nonlinear LLRs, RKHS with high variance)  <br> - Show ARL/delay breakdown  <br> - Visualize LLR error | 📊 Plots: failure cases  <br> 📄 Section 3 draft                                 |
| **Week 4** | Classifier-based CUSUM                | - Implement betting-based scheme (Section 2.2)  <br> - Train classifier (logistic/NN)  <br> - Simulate \(W_n\) process  <br> - Compare ARL/delay   | 📈 Betting-based detector plots  <br> 📄 Section 4 draft                         |
| **Week 5** | Multiclass Extension                  | - Create 3-class post-change scenario  <br> - Implement multiclass betting (e.g., one-vs-rest or softmax-based scheme)  <br> - Validate detection   | 📊 Plots: multiclass detection  <br> 📄 Section 5 draft                          |
| **Week 6** | Unified Experiments + Summary Tables  | - Run full comparison: Oracle vs RKHS vs NN vs Classifier  <br> - Collect metrics  <br> - Build ARL/delay summary table                           | 📊 Final comparative table  <br> 📄 Complete Section 6                          |
| **Week 7** | Full Paper Drafting                   | - Write Sections 1, 2, and 7 (Intro, LLR, Conclusion)  <br> - Polish experimental write-ups  <br> - Format references, figures                     | 📝 Full Overleaf draft  <br> 📄 Clean bibliography                              |
| **Week 8** | Revision + Feedback                   | - Get advisor feedback  <br> - Run requested experiments  <br> - Final proofreading                                                                | ✅ Final PDF  <br> 📬 Submission-ready manuscript                                |

---

