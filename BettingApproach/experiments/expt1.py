import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
import argparse
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_source import OneHotSource, create_gap_pmf, ChangePointSource
from change_detector import ChangeDetector
from sequential_test import SequentialHypothesisTest, IdentityClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 1: ARL and detection-delay studies"
    )
    parser.add_argument(
        "--n_classes", type=int, default=5,
        help="number of classes (L+1)"
    )
    parser.add_argument(
        "--delta0", type=float, default=0.1,
        help="Effect size Δ for the ARL experiment"
    )
    parser.add_argument(
        "--alpha0", type=float, default=0.001,
        help="α for the ARL experiment"
    )
    parser.add_argument(
        "--alpha1", type=float, default=0.05,
        help="α for the delay experiment"
    )
    parser.add_argument(
        "--num_trials_arl", type=int, default=500,
        help="Number of Monte Carlo trials for ARL estimation"
    )
    parser.add_argument(
        "--num_trials_delay", type=int, default=200,
        help="Number of Monte Carlo trials for delay estimation"
    )
    parser.add_argument(
        "--Tc", type=int, default=10,
        help="Change-point time T_c (should be ≪ 1/α1)"
    )
    parser.add_argument(
        "--N_max", type=int, default=5000,
        help="Maximum stream length (N_pre + N_post)"
    )
    parser.add_argument(
        "--delta1_min", type=float, default=0.1,
        help="Minimum Δ for delay sweep"
    )
    parser.add_argument(
        "--delta1_max", type=float, default=0.3,
        help="Maximum Δ for delay sweep"
    )
    parser.add_argument(
        "--num_delta1", type=int, default=5,
        help="Number of Δ values to sweep"
    )
    parser.add_argument(
        "--W", type=int, default=500,
        help="Maximum Window Size = maximum number of 'active tests'"
    )
    parser.add_argument(
        "--M", type=int, default=200,
        help="grid size for mixture method"
    )
    parser.add_argument(
        "--save_figs", action='store_true',
        help="Flag to save figures to results/figures"
    )
    args = parser.parse_args()
    if args.delta1_max <= args.delta1_min:
        parser.error("--delta1_max must be strictly greater than --delta1_min")
    return args


def setup_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(root, "results", "figures")
    log_dir = os.path.join(root, "results", "logs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return fig_dir, log_dir


def one_trial_arl(p_null, detector_ctor, N, alpha=0.001):
    src = OneHotSource(p_null, length=N)
    detector = detector_ctor()
    alarm, t_hat = detector.run_on_source(src)
    return t_hat if alarm else np.inf


def run_arl_expt(p_null, N, detector_ctor,
                  alpha=0.001, num_trials=500,
                  plot_fig=True, save_fig=False,
                  fig_dir=None, log_dir=None,
                  delta0=None):
    stops = np.zeros(num_trials)
    for i in tqdm(range(num_trials)):
        stops[i] = one_trial_arl(p_null, detector_ctor, N, alpha)
    finite = np.isfinite(stops)
    arl = stops[finite].mean()
    print(f"Estimated ARL (mean stopping time): {arl:.1f}")

    # Plot
    if plot_fig or save_fig:
        plt.figure()
        plt.hist(stops[finite], bins=50, density=True, alpha=0.4)
        plt.axvline(arl, linestyle='--', color='r', label=f'Mean={arl:.1f}')
        plt.title(f"Histogram under H0 (Δ={delta0}, α={alpha})")
        plt.xlabel("Stopping time")
        plt.ylabel("Density")
        plt.legend()
        if save_fig and fig_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"expt1_arl_delta{delta0:.3f}_alpha{alpha:.4f}_{ts}.png"
            path = os.path.join(fig_dir, fname)
            plt.savefig(path, dpi=300)
            print(f"Saved ARL figure to {path}")
        if plot_fig:
            plt.show()
        plt.close()

    # Logging
    if log_dir:
        log_path = os.path.join(log_dir, "expt1_arl_log.json")
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "delta0": delta0,
            "alpha0": alpha,
            "num_trials": num_trials,
            "arl": float(arl)
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Logged ARL results to {log_path}")

    return stops


def run_delay_expt(Delta, NN, detector_ctor,
                   n_classes, Tc=10,
                   alpha=0.05,
                   num_trials=200,
                   plot_fig=True,
                   save_fig=False,
                   fig_dir=None,
                   log_dir=None):
    means = np.zeros_like(Delta)
    stds = np.zeros_like(Delta)
    for idx, (delta, N) in enumerate(zip(Delta, NN)):
        delays = []
        for _ in range(num_trials):
            alarm, tau = one_trial_delay(
                delta, n_classes, detector_ctor,
                N_pre=Tc, N_post=N, Tc=Tc)
            if tau >= Tc:
                delays.append(tau - Tc)
        delays = np.array(delays)
        means[idx] = delays.mean()
        stds[idx] = delays.std()
        print(f"Δ={delta:.3f}: mean delay={means[idx]:.1f}")

    # Plot
    if plot_fig or save_fig:
        plt.figure()
        plt.errorbar(Delta, means, yerr=stds, fmt='o-', label='Observed')
        D_plot = np.geomspace(Delta.min(), Delta.max(), 200)
        c_hat = means[0] * (Delta[0]**2)
        plt.plot(D_plot, c_hat/(D_plot**2), '--', label=r'$c/\Delta^2$')
        plt.xlabel(r'$\Delta$')
        plt.ylabel(r'$\mathbb{E}[(	au-T_c)^+]$')
        plt.title(f"Delay vs Δ (T_c={Tc}, α={alpha})")
        plt.legend()
        if save_fig and fig_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"expt1_delay_Tc{Tc}_alpha{alpha:.4f}_{ts}.png"
            path = os.path.join(fig_dir, fname)
            plt.savefig(path, dpi=300)
            print(f"Saved delay figure to {path}")
        if plot_fig:
            plt.show()
        plt.close()

    # Logging
    if log_dir:
        log_path = os.path.join(log_dir, "expt1_delay_log.json")
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "Delta": Delta.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "Tc": Tc,
            "alpha1": alpha,
            "num_trials": num_trials
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Logged delay results to {log_path}")

    return means, stds


def one_trial_delay(delta, n_classes, detector_ctor, N_pre, N_post, Tc=None):
    p_null = create_gap_pmf(delta, n_classes, alt=False)
    p_alt  = create_gap_pmf(delta, n_classes, alt=True)
    if Tc is not None:
        N_pre = Tc
    if Tc is None:
        Tc = N_pre
    pre_src = OneHotSource(p_null, length=N_pre)
    post_src = OneHotSource(p_alt, length=N_post)
    cp = ChangePointSource(pre_src, post_src, T_c=Tc, length=N_pre+N_post)
    detector = detector_ctor()
    return detector.run_on_source(cp)

if __name__ == '__main__':
    args = parse_args()
    fig_dir, log_dir = setup_paths()

    # ARL experiment
    n_classes = args.n_classes
    N_max = args.N_max
    p_null = create_gap_pmf(args.delta0, n_classes, alt=False)
    def ctor0():
        return ChangeDetector(
            test_ctor=lambda: SequentialHypothesisTest(
                classifier=IdentityClassifier(),
                n_classes=n_classes,
                alpha=args.alpha0,
                M=args.M,
                smoothing=0.5,
                N_max=N_max
            ), N_max=N_max, W=args.W
        )
    run_arl_expt(
        p_null, N_max, ctor0,
        alpha=args.alpha0,
        num_trials=args.num_trials_arl,
        plot_fig=not args.save_figs,
        save_fig=args.save_figs,
        fig_dir=fig_dir,
        log_dir=log_dir,
        delta0=args.delta0
    )

    # Delay experiment
    Delta = np.geomspace(args.delta1_min, args.delta1_max, args.num_delta1)
    NN = np.full_like(Delta, args.N_max)
    def ctor1():
        return ChangeDetector(
            test_ctor=lambda: SequentialHypothesisTest(
                classifier=IdentityClassifier(),
                n_classes=n_classes,
                alpha=args.alpha1,
                M=args.M,
                smoothing=0.5,
                N_max=args.N_max
            ), N_max=args.N_max, W=args.W
        )
    run_delay_expt(
        Delta, NN, ctor1,
        n_classes=n_classes,
        Tc=args.Tc,
        alpha=args.alpha1,
        num_trials=args.num_trials_delay,
        plot_fig=not args.save_figs,
        save_fig=args.save_figs,
        fig_dir=fig_dir,
        log_dir=log_dir
    )
