"""
Sanity Check Experiment that plots the histogram of the stopping time of the Sequential test for a fixed alternative. 
"""
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))


import numpy as np
import matplotlib.pyplot as plt
from data_source import OneHotSource
from sequential_test import SequentialHypothesisTest, IdentityClassifier
from tqdm import tqdm 

delta=0.1
assert delta <0.2
p_alt = np.array([0.2, 0.2 + delta, 0.2, 0.2-delta, 0.2])     # alternative: class 0 is *not* most likely
n_classes = len(p_alt)
N = 4000
src = OneHotSource(p_alt, length=N)

alpha = 0.05
num_trials=500

# initialize the test 
test = SequentialHypothesisTest(
    classifier=IdentityClassifier(),
    n_classes=n_classes,
    alpha=alpha,
    M=200,
    smoothing=0.5,
    N_max=N
)

StoppingTimes = np.zeros((num_trials,))

for trial in tqdm(range(num_trials)):
    test.reset()
    src.reset()
    stopped = False
    for t in range(1, N+1):
        x = src.next()
        stopped, tau = test.update(x)
        if stopped:
            break
    if stopped:
        StoppingTimes[trial] = t
    else:
        StoppingTimes[trial]= np.inf

#  Plot the Histogram
plt.figure()
plt.hist(StoppingTimes[np.isfinite(StoppingTimes)], density=True, alpha=0.3)
plt.axvline(x=StoppingTimes[np.isfinite(StoppingTimes)].mean(), linestyle='--')
plt.xlabel("Stopping Times", fontsize=10)
plt.ylabel("Density")
plt.yticks([])
plt.show()