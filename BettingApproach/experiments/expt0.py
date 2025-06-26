"""
Sanity Check Experiment that plots the variation of the stopping time vs. effect size of the Sequential test
"""
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import numpy as np
import matplotlib.pyplot as plt
from data_source import OneHotSource, create_gap_pmf
from sequential_test import SequentialHypothesisTest, IdentityClassifier
from tqdm import tqdm 


def one_trial(p_alt,  N=4000, test=None, alpha = 0.05,  num_trials=500, M=200, progress_bar=False, plot_fig=False):
    n_classes = len(p_alt)
    src = OneHotSource(p_alt, length=N)

    # initialize the test 
    if test is None:
        test = SequentialHypothesisTest(
            classifier=IdentityClassifier(),
            n_classes=n_classes,
            alpha=alpha,
            M=M,
            smoothing=0.5,
            N_max=N
        )

    StoppingTimes = np.zeros((num_trials,))

    range_ = tqdm(range(num_trials)) if progress_bar else range(num_trials)
    for trial in range_:
        test.reset()
        src.reset()
        stopped = False
        for t in range(1, N+1):
            x = src.next()
            stopped, _ = test.update(x)
            if stopped:
                break
        if stopped:
            StoppingTimes[trial] = t
        else:
            StoppingTimes[trial]= np.inf

    #  Plot the Histogram
    if plot_fig:
        plt.figure()
        plt.hist(StoppingTimes[np.isfinite(StoppingTimes)], density=True, alpha=0.3)
        plt.axvline(x=StoppingTimes[np.isfinite(StoppingTimes)].mean(), linestyle='--')
        plt.xlabel("Stopping Times", fontsize=10)
        plt.ylabel("Density")
        plt.yticks([])
        plt.show()
    # return the collected stopping time values 
    print(f"Number of rejections:{(np.isfinite(StoppingTimes).sum())}")
    return StoppingTimes

def run_expt0(Delta, NN, alpha=0.05, num_trials=500, n_classes=5, progress_bar=True, plot_fig=True):
    StoppingTimesMean = np.zeros(Delta.shape)
    StoppingTimesSTD = np.zeros(Delta.shape)

    num_expts = len(Delta)     
    range_ = tqdm(range(num_expts)) if progress_bar else range(num_expts)

    for i in range_:
        delta, N = Delta[i], NN[i]
        # generate the source 
        p_alt = create_gap_pmf(delta, n_classes, alt=True)
        StoppingTimes_i = one_trial(p_alt=p_alt, N=N, alpha=alpha,
            num_trials=num_trials, progress_bar=True, plot_fig=False)
        # collect the statistics 
        mask = np.isfinite(StoppingTimes_i) 
        StoppingTimesMean[i] = StoppingTimes_i[mask].mean()
        StoppingTimesSTD[i] = StoppingTimes_i[mask].std()

    # plot the results if needed 
    if plot_fig: 
        plt.figure() 
        plt.plot(Delta, StoppingTimesMean, marker="o", linewidth=1.5, label="Observed")
        y1 =  StoppingTimesMean - StoppingTimesSTD
        y2 =  StoppingTimesMean + StoppingTimesSTD
        plt.fill_between(Delta, y1 = y1, y2=y2, alpha=0.25) 

        # quadratic fit 
        D_ = np.geomspace(start = Delta.min(), stop=Delta.max(), num=1000)
        best_fit = (StoppingTimesMean *(Delta**2)).mean()
        S_ = best_fit * (1/D_**2)
        plt.plot(D_, S_, linestyle='--', label='Quadratic Fit')

        plt.xlabel(r"$\delta$", fontsize=12)
        plt.ylabel("Stopping Time")
        plt.title(r"$\mathbb{E}_{H_1}[N]$ versus $\delta$")
        plt.legend()
        plt.show()
    
    return StoppingTimesMean, StoppingTimesSTD


if __name__=='__main__':
    Delta = np.geomspace(0.01, 0.25, num=10)
    NN = np.clip(100*(1/Delta**2), a_min=1000, a_max=75000).astype(int)
    ST, STstd= run_expt0(Delta, NN, alpha=0.05, num_trials=500, n_classes=5, progress_bar=False, plot_fig=True)