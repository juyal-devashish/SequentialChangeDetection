# Classifier Based Sequential Testing and Change Detection

This project provides a modular framework for implementing and experimenting with sequential hypothesis tests (e‑process based) and corresponding change-point detection methods. It includes:

* \`\`: Abstract data source interface plus concrete implementations

  * `DataSource` (abstract base)
  * `OneHotSource`: emits one-hot vectors from a categorical distribution
  * `GaussianMixtureSource`: draws from a Gaussian mixture in $\mathbb{R}^d$
  * `MNISTSource`: streams MNIST images with optional single-digit or mixture modes
  * `CIFAR10Source`: analogous to `MNISTSource` for CIFAR-10 (TODO). 

* \`\`: Sequential hypothesis testing class

  * `IdentityClassifier`: wrapper for one-hot or probability vectors
  * `SequentialHypothesisTest`: implements the e‑process test, with:

    * configurable grid size $M$, smoothing, and false-alarm rate $\alpha$
    * methods `update()`, `run()`, and `run_on_source()`
    * built-in support for maximum sample length $N_{\max}$
  * Demo under `if __name__ == '__main__'` for type-I error and stopping time experiments

## Sub-Task Structure

```
BettingApproach/
├── data_source.py
├── sequential_test.py
├── README.md       # this document
└── experiments/    # optional folder for Jupyter notebooks or scripts
```

## Installation

1. Ensure Python 3.8+ is installed.
2. Install dependencies:

```bash
pip install numpy scipy matplotlib
```

## Usage

### 1. Define a data source

```python
from data_source import OneHotSource
# Categorical one-hot source under p
p = np.array([0.6, 0.2, 0.2])
src = OneHotSource(p, length=1000)
```

### 2. Initialize the hypothesis test

```python
from sequential_test import SequentialHypothesisTest, IdentityClassifier
# Configure: 3 classes, alpha=0.05, grid M=200, smoothing=0.5, N_max=1000
test = SequentialHypothesisTest(
    classifier=IdentityClassifier(),
    n_classes=3,
    alpha=0.05,
    M=200,
    smoothing=0.5,
    N_max=1000
)
```

### 3. Run on the data source

```python
flag, tau = test.run_on_source(src)
if flag:
    print(f"Null rejected at time {tau}")
else:
    print(f"Null not rejected within {tau} samples")
```

### 4. Experimental scripts

* Use the built-in demo in `sequential_test.py` for quick Monte Carlo of type-I error and stopping time.
* For custom experiments, create scripts or notebooks under `experiments/`.

## TODO

* **Add new data sources** by subclassing `DataSource` and implementing `reset()` and `next()`.
* **Alternate grids or smoothing**: pass custom `grid` or `smoothing` parameters to `SequentialHypothesisTest`.
* **Change-point detection**: build on `SequentialHypothesisTest` to detect distribution changes at unknown times.

##
