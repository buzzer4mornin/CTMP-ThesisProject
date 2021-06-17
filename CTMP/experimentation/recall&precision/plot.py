import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='nflx', type=str, help="['nflx', 'original']")
parser.add_argument("--TOP_M_start", default=10, type=int, help="Start of Top-M recommendation")
parser.add_argument("--TOP_M_end", default=100, type=int, help="End of Top-M recommendation")
parser.add_argument("--pred_type", default='out-of-matrix', type=str, help="['in-matrix', 'out-of-matrix']")
parser.add_argument("--k", default=100, type=int, help="K-fold Cross Validation which was used")
parser.add_argument("--folder", default=7, type=int, help="Which fold of K-fold Cross Validation to test")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


def plot(args):
    if args.pred_type == "in-matrix":
        pass
    elif args.pred_type == "out-of-matrix":
        r_mean = np.empty(shape=(90,))
        p_mean = np.empty(shape=(90,))
        for i in range(1, 6):
            with open(f"./NFLX/k={args.k};iter=50/{args.folder}/Recall-out-of-matrix-50000sample-_{i}fold-50th.pkl", "rb") as f:
                r_TEST = pickle.load(f)
            with open(f"./NFLX/k={args.k};iter=50/{args.folder}/Precision-out-of-matrix-50000sample-_{i}fold-50th.pkl", "rb") as f:
                p_TEST = pickle.load(f)
            r_mean += np.array(r_TEST)
            p_mean += np.array(p_TEST)
        r_mean /= 5
        p_mean /= 5

    # PLOT recall graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(range(args.TOP_M_start, args.TOP_M_end), r_mean, label="test", linewidth=0.7, color="m", markeredgewidth=0.01, markerfacecolor='m', marker='o', markevery=10)
    ax1.set_xlabel('Top-M', fontsize=11)
    ax1.set_ylabel('Recall', fontsize=11)
    # ax1.set_title(f"IMPORT SOME NAME HERE")
    ax1.legend()

    # PLOT precision graph
    ax2.plot(range(args.TOP_M_start, args.TOP_M_end), p_mean, label="test", linewidth=0.7, color="m", markeredgewidth=0.01, markerfacecolor='m', marker='o', markevery=10)
    ax2.set_xlabel('Top-M', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    # ax2.set_title(f"IMPORT SOME NAME HERE")
    ax2.legend()

    # plot configs
    ax1.set_xlim([10, 100])
    ax2.set_xlim([10, 100])
    ax1.set_xticks(np.arange(10, 101, 10))
    ax2.set_xticks(np.arange(10, 101, 10))
    ax1.set_yticks(np.arange(0.05, 0.46, 0.05))
    ax2.set_yticks(np.arange(0.10, 0.24, 0.02))
    ax1.grid()
    ax2.grid()
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, bottom=0.15)
    fig.suptitle(f'{args.pred_type} predictions', fontsize=14)

    # Save results
    plt.savefig(f"./NFLX/k={args.k};iter=50/{args.folder}/mean.png")
    plt.show()

if __name__ == '__main__':
    import pickle
    import random
    import time
    import matplotlib.pyplot as plt
    import os

    NUM_THREADS = "1"
    os.environ["OMP_NUM_THREADS"] = NUM_THREADS
    os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = NUM_THREADS
    os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
    os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
    import numpy as np

    args = parser.parse_args([] if "__file__" not in globals() else None)
    assert args.pred_type in ['in-matrix', 'out-of-matrix']

    plot(args)
