import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import argparse
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from data_source import OneHotSource, create_gap_pmf
from sequential_test import SequentialHypothesisTest, IdentityClassifier, SequentialHypothesisTestONS
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 0: Stopping time vs effect size for sequential test"
    )
    parser.add_argument(
        '--delta_min', type=float, default=0.09,
        help='minimum Δ for the sweep'
    )
    parser.add_argument(
        '--delta_max', type=float, default=0.25,
        help='maximum Δ for the sweep'
    )
    parser.add_argument(
        '--num_delta', type=int, default=5,
        help='number of Δ values'
    )
    parser.add_argument(
        '--alpha0', type=float, default=0.01,
        help='α level for the sequential test'
    )
    parser.add_argument(
        '--num_trials', type=int, default=50,
        help='number of Monte Carlo trials per Δ'
    )
    parser.add_argument(
        '--n_classes', type=int, default=5,
        help='number of classes (L+1)'
    )
    parser.add_argument(
        '--progress_bar', action='store_true',
        help='show tqdm progress bars'
    )
    parser.add_argument(
        '--save_figs', action='store_true',
        help='flag to save figures to results/figures'
    )
    args = parser.parse_args()
    if args.delta_max <= args.delta_min:
        parser.error('--delta_max must be greater than --delta_min')
    return args


def setup_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(root, 'results', 'figures')
    log_dir = os.path.join(root, 'results', 'logs')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return fig_dir, log_dir


def one_trial(p_alt, N=4000, test=None, alpha=0.05,
              num_trials=500, M=200,
              progress_bar=False, plot_fig=False):
    n_classes = len(p_alt)
    src = OneHotSource(p_alt, length=N)

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
    trials = tqdm(range(num_trials)) if progress_bar else range(num_trials)
    for trial in trials:
        test.reset()
        src.reset()
        stopped = False
        for t in range(1, N+1):
            x = src.next()
            stopped, _ = test.update(x)
            if stopped:
                break
        StoppingTimes[trial] = t if stopped else np.inf

    if plot_fig:
        plt.figure()
        fin = StoppingTimes[np.isfinite(StoppingTimes)]
        plt.hist(fin, density=True, alpha=0.3)
        plt.axvline(x=fin.mean(), linestyle='--')
        plt.xlabel('Stopping Times')
        plt.ylabel('Density')
        plt.title(f'Histogram of Stopping Times')
        plt.show()
    print(f"Number of rejections: {np.isfinite(StoppingTimes).sum()}")
    return StoppingTimes


def run_expt0(Delta, NN,
              alpha=0.05,
              num_trials=500,
              n_classes=5,
              progress_bar=True,
              plot_fig=True,
              save_fig=False,
              fig_dir=None,
              log_dir=None):
    StoppingTimesMean = np.zeros_like(Delta)
    StoppingTimesSTD = np.zeros_like(Delta)
    num_expts = len(Delta)
    exps = tqdm(range(num_expts)) if progress_bar else range(num_expts)

    for i in exps:
        delta, N = Delta[i], NN[i]
        p_alt = create_gap_pmf(delta, n_classes, alt=True)
        ST_i = one_trial(p_alt=p_alt,
                         N=N,
                         alpha=alpha,
                         num_trials=num_trials,
                         progress_bar=False,
                         plot_fig=False)
        mask = np.isfinite(ST_i)
        StoppingTimesMean[i] = ST_i[mask].mean()
        StoppingTimesSTD[i]  = ST_i[mask].std()

    if plot_fig or save_fig:
        plt.figure()
        plt.plot(Delta, StoppingTimesMean, marker='o', linewidth=1.5, label='Observed')
        y1 = StoppingTimesMean - StoppingTimesSTD
        y2 = StoppingTimesMean + StoppingTimesSTD
        plt.fill_between(Delta, y1, y2, alpha=0.25)
        D_ = np.geomspace(Delta.min(), Delta.max(), num=1000)
        best_fit = (StoppingTimesMean * (Delta**2)).mean()
        S_ = best_fit * (1/D_**2)
        plt.plot(D_, S_, linestyle='--', label='Quadratic Fit')
        plt.xlabel(r'$\delta$')
        plt.ylabel('Stopping Time')
        plt.title(r'$\mathbb{E}_{H_1}[N]$ vs $\delta$')
        plt.legend()
        if save_fig and fig_dir:
            ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            delta_min_str = f"{Delta.min():.3f}".replace('.', '_')
            delta_max_str = f"{Delta.max():.3f}".replace('.', '_')
            alpha_str     = f"{alpha:.4f}".replace('.', '_')
            fname = (
                f"expt0_delta0_{delta_min_str}-{delta_max_str}"
                f"_alpha{alpha_str}_{ts}.png"
            )
            # fname = f'expt0_delta_{Delta.min():.3f}-{Delta.max():.3f}_alpha{alpha:.4f}_{ts}.png'
            path = os.path.join(fig_dir, fname)
            plt.savefig(path, dpi=300)
            print(f'Saved figure to {path}')
        if plot_fig:
            plt.show()
        plt.close()

    if log_dir:
        log_path = os.path.join(log_dir, 'expt0_log.json')
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'Delta': Delta.tolist(),
            'means': StoppingTimesMean.tolist(),
            'stds': StoppingTimesSTD.tolist(),
            'alpha0': alpha,
            'num_trials': num_trials,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f'Logged results to {log_path}')

    return StoppingTimesMean, StoppingTimesSTD

if __name__ == '__main__':
    args = parse_args()
    fig_dir, log_dir = setup_paths()

    Delta = np.geomspace(args.delta_min, args.delta_max, num=args.num_delta)
    NN = np.clip(100*(1/Delta**2), a_min=1000, a_max=25000).astype(int)

    run_expt0(
        Delta, NN,
        alpha=args.alpha0,
        num_trials=args.num_trials,
        n_classes=args.n_classes,
        progress_bar=args.progress_bar,
        plot_fig=not args.save_figs,
        save_fig=args.save_figs,
        fig_dir=fig_dir,
        log_dir=log_dir
    )
