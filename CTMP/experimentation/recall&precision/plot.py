import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='NFLX', type=str, help="['nflx', 'original']")
parser.add_argument("--TOP_M_start", default=10, type=int, help="Start of Top-M recommendation")
parser.add_argument("--TOP_M_end", default=100, type=int, help="End of Top-M recommendation")
parser.add_argument("--pred_type", default='in-matrix', type=str, help="['in-matrix', 'out-of-matrix']")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--p", default=0.7, type=float, help="K-fold Cross Validation which was used")
parser.add_argument("--k", default=50, type=int, help="K-fold Cross Validation which was used")
parser.add_argument("--folder", default=7, type=int, help="Which fold of K-fold Cross Validation to test")
parser.add_argument("--fold", default=4, type=int, help="Which fold of K-fold Cross Validation to test")


def plot(args):
    r_mean = np.empty(shape=(90,))
    p_mean = np.empty(shape=(90,))
    r_mean_out = np.empty(shape=(90,))
    p_mean_out = np.empty(shape=(90,))
    for i in range(1, 6):
        try:
            with open(f"./{args.dataset}/p={args.p}/k={args.k}/{args.folder}/Recall-in-matrix-50000sample-p={args.p}-k={args.k}-folder={args.folder}-fold={args.fold}.pkl", "rb") as f:
                r_TEST = pickle.load(f)
            with open(f"./{args.dataset}/p={args.p}/k={args.k}/{args.folder}/Precision-in-matrix-50000sample-p={args.p}-k={args.k}-folder={args.folder}-fold={args.fold}.pkl", "rb") as f:
                p_TEST = pickle.load(f)
            with open(f"./{args.dataset}/p={args.p}/k={args.k}/{args.folder}/Recall-out-of-matrix-50000sample-p={args.p}-k={args.k}-folder={args.folder}-fold={args.fold}.pkl", "rb") as f:
                r_TEST_out = pickle.load(f)
            with open(f"./{args.dataset}/p={args.p}/k={args.k}/{args.folder}/Precision-out-of-matrix-50000sample-p={args.p}-k={args.k}-folder={args.folder}-fold={args.fold}.pkl", "rb") as f:
                p_TEST_out = pickle.load(f)
        except:
            continue
        r_mean += np.array(r_TEST)
        p_mean += np.array(p_TEST)
        r_mean_out += np.array(r_TEST_out)
        p_mean_out += np.array(p_TEST_out)
    r_mean /= 5
    p_mean /= 5
    r_mean_out /= 5
    p_mean_out /= 5

    # PLOT recall graph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5))
    ax1.plot(range(args.TOP_M_start, args.TOP_M_end), r_mean*100, label="in-matrix", linewidth=0.7, color="m", markeredgewidth=0.01, markerfacecolor='m', marker='o', markevery=10)
    ax1.plot(range(args.TOP_M_start, args.TOP_M_end), r_mean_out*100, label="out-of-matrix", linewidth=0.7, color="y", markeredgewidth=0.01, markerfacecolor='y', marker='o', markevery=10)
    ax1.set_xlabel('Top-M', fontsize=11)
    ax1.set_ylabel('Recall (%)', fontsize=11)
    # ax1.set_title(f"IMPORT SOME NAME HERE")
    ax1.legend()

    # PLOT precision graph
    ax2.plot(range(args.TOP_M_start, args.TOP_M_end), p_mean*100, label="in-matrix", linewidth=0.7, color="m", markeredgewidth=0.01, markerfacecolor='m', marker='o', markevery=10)
    ax2.plot(range(args.TOP_M_start, args.TOP_M_end), p_mean_out*100, label="out-of-matrix", linewidth=0.7, color="y", markeredgewidth=0.01, markerfacecolor='y', marker='o', markevery=10)
    ax2.set_xlabel('Top-M', fontsize=11)
    ax2.set_ylabel('Precision (%)', fontsize=11)
    # ax2.set_title(f"IMPORT SOME NAME HERE")
    ax2.legend()

    # plot configs
    ax1.set_xlim([10, 100])
    ax2.set_xlim([10, 100])
    ax1.set_xticks(np.arange(10, 101, 10))
    ax2.set_xticks(np.arange(10, 101, 10))
    ax1.set_yticks(np.arange(5, 46, 5))
    ax2.set_yticks(np.arange(10, 24, 2))
    ax1.grid()
    ax2.grid()
    plt.subplots_adjust(hspace=0.3, left=0.18, right=0.94, bottom=0.1, top=0.9)

    fig.suptitle(f'{{K=100, lamb=1, alpha=1, p=0.9}}', fontsize=12)

    # Save results
    # plt.savefig(f"./{args.dataset}/p={args.p}/k={args.k}/{args.folder}/{args.pred_type}-mean.png")
    # plt.savefig(f"./{args.dataset}/p={args.p}/k={args.k}/{args.folder}/result.png")
    plt.savefig(f"./result.png")
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
