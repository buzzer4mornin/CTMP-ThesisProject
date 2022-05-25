import pickle
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor


# Run Evaluation
# python ./model/Evaluation.py nflx 5 1 NFX_50_50_100_1.0_1.0_0.9 1000 10 100 out-of-matrix


class MyEvaluation:
    def __init__(self, user_train, user_test, movie_train, movie_test, output_dir,
                 sample_test=1000, TOP_M_start=10, TOP_M_end=100, pred_type='out-of-matrix', seed=42):

        assert pred_type in ['in-matrix', 'out-of-matrix', 'both']
        self.folder = f"output-data/{output_dir}"
        self.sample_test = sample_test
        self.TOP_M_start = TOP_M_start
        self.TOP_M_end = TOP_M_end
        self.pred_type = pred_type
        self.seed = seed

        # Set seed
        np.random.seed(self.seed)

        self.rating_GroupForUser_TRAIN = user_train
        self.rating_GroupForUser_TEST = user_test
        self.rating_GroupForMovie_TRAIN = movie_train
        self.rating_GroupForMovie_TEST = movie_test

        self.mu = np.load(f"./{self.folder}/mu.npy")
        self.shp = np.load(f"./{self.folder}/shp.npy")
        self.rte = np.load(f"./{self.folder}/rte.npy")

        # TODO: maybe float64 -> float32? Compare results if changed!
        # print(self.mu.dtype)
        # print(self.shp.dtype)
        # print(self.rte.dtype)

        # Group items separately
        self.cold_items_TRAIN, self.cold_items_TEST, self.noncold_items_TRAIN, self.noncold_items_TEST = self.group_items()

        # Generate test set
        self.test_set = self.generate_test_set()

        # Average Recalls and Precisions over all users of test set across the Top-M
        # TEST
        self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST = [], []
        self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST = [], []

        # Update them accordingly
        self.avg_recall_precision()

    def group_items(self) -> list:
        """Number of cold items - 5,577/25,900 || Number of noncold items - 20,323/25,900"""
        cold_items_TRAIN, cold_items_TEST = [], []
        noncold_items_TRAIN, noncold_items_TEST = [], []

        for movie_id in self.rating_GroupForMovie_TEST:
            if len(self.rating_GroupForMovie_TEST[movie_id]) != 0:
                noncold_items_TEST.append(movie_id)
            else:
                cold_items_TEST.append(movie_id)

        for movie_id in self.rating_GroupForMovie_TRAIN:
            if len(self.rating_GroupForMovie_TRAIN[movie_id]) != 0:
                noncold_items_TRAIN.append(movie_id)
            else:
                cold_items_TRAIN.append(movie_id)

        print(f"Training set: cold-{len(cold_items_TRAIN)}, noncold-{len(noncold_items_TRAIN)}")
        print(f"Testing set: cold-{len(cold_items_TEST)}, noncold-{len(noncold_items_TEST)}")
        return cold_items_TRAIN, cold_items_TEST, noncold_items_TRAIN, noncold_items_TEST

    def generate_test_set(self) -> list:
        sample = random.sample(list(self.rating_GroupForUser_TEST.keys()), self.sample_test)
        test_set = []
        for u in sample:
            if len(self.rating_GroupForUser_TEST[u]) > 0 and len(self.rating_GroupForUser_TRAIN[u]) > 0:
                test_set.append(u)
        # avg = 0
        # for i in self.rating_GroupForUser_TRAIN:
        #       avg += len(self.rating_GroupForUser_TRAIN[i])
        # print(avg/len(self.rating_GroupForUser_TRAIN))
        # exit()
        return test_set

    # TODO old one
    def predict_in_matrix(self, user_id, top_m) -> None:
        """Compute in-matrix recall and precision for a given user, then add them to the sum"""
        ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
        actual_TEST = self.rating_GroupForUser_TEST[user_id]
        sorted_ratings = np.argsort(-ratings)
        # s = np.setdiff1d(sorted_ratings, self.rating_GroupForUser_TRAIN[user_id], assume_unique=True)
        predicted_top_M_TEST = np.setdiff1d(sorted_ratings, self.cold_items_TEST, assume_unique=True)[:top_m]
        top_m_correct_TEST = np.sum(np.in1d(predicted_top_M_TEST, actual_TEST) * 1)
        self.recalls_in_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
        self.precisions_in_matrix_TEST += (top_m_correct_TEST / top_m)

    def predict_out_of_matrix(self, user_id, top_m) -> None:
        # Compute out-of-matrix recall and precision for a given user, then add them to the sum
        ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
        actual_TEST = self.rating_GroupForUser_TEST[user_id]
        predicted_top_M_TEST = np.argsort(-ratings)[:top_m]
        # predicted_top_M_TEST = np.setdiff1d(sorted_ratings, self.rating_GroupForUser_TRAIN[user_id], assume_unique=True)[:top_m]
        top_m_correct_TEST = np.sum(np.in1d(predicted_top_M_TEST, actual_TEST) * 1)
        self.recalls_out_of_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
        self.precisions_out_of_matrix_TEST += (top_m_correct_TEST / top_m)

    def avg_recall_precision(self) -> None:
        for top in range(self.TOP_M_start, self.TOP_M_end):
            # make all metrics zero for new iteration
            print(f"Top-M: {top}")
            self.recalls_in_matrix_TEST, self.precisions_in_matrix_TEST = 0, 0
            self.recalls_out_of_matrix_TEST, self.precisions_out_of_matrix_TEST = 0, 0

            for usr in self.test_set:
                self.predict_in_matrix(usr, top)
                self.predict_out_of_matrix(usr, top)

            self.avg_recalls_in_matrix_TEST.append(self.recalls_in_matrix_TEST / len(self.test_set))
            self.avg_precisions_in_matrix_TEST.append(self.precisions_in_matrix_TEST / len(self.test_set))
            self.avg_recalls_out_of_matrix_TEST.append(self.recalls_out_of_matrix_TEST / len(self.test_set))
            self.avg_precisions_out_of_matrix_TEST.append(self.precisions_out_of_matrix_TEST / len(self.test_set))

    # def save(self) -> None:
    #     self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST
    #     self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST

    def plot(self) -> None:
        r_TEST, p_TEST = self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST
        r_TEST_out, p_TEST_out = self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST

        r_mean = np.array(r_TEST)
        p_mean = np.array(p_TEST)
        r_mean_out = np.array(r_TEST_out)
        p_mean_out = np.array(p_TEST_out)

        # PLOT recall graph
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5))
        ax1.plot(range(TOP_M_start, TOP_M_end), r_mean * 100, label="in-matrix", linewidth=0.7, color="m",
                 markeredgewidth=0.15, markerfacecolor='m', marker='o', markevery=10)
        ax1.plot(range(TOP_M_start, TOP_M_end), r_mean_out * 100, label="out-of-matrix", linewidth=0.7,
                 color="y", markeredgewidth=0.01, markerfacecolor='y', marker='o', markevery=10)
        ax1.set_xlabel('Top-M', fontsize=11)
        ax1.set_ylabel('Recall (%)', fontsize=11)
        # ax1.set_title(f"IMPORT SOME NAME HERE")
        ax1.legend()

        # PLOT precision graph
        ax2.plot(range(TOP_M_start, TOP_M_end), p_mean * 100, label="in-matrix", linewidth=0.7, color="m",
                 markeredgewidth=0.01, markerfacecolor='m', marker='o', markevery=10)
        ax2.plot(range(TOP_M_start, TOP_M_end), p_mean_out * 100, label="out-of-matrix", linewidth=0.7,
                 color="y", markeredgewidth=0.01, markerfacecolor='y', marker='o', markevery=10)
        ax2.set_xlabel('Top-M', fontsize=11)
        ax2.set_ylabel('Precision (%)', fontsize=11)
        # ax2.set_title(f"IMPORT SOME NAME HERE")
        ax2.legend()

        # Plot configs
        ax1.set_xlim([10, 100])
        ax2.set_xlim([10, 100])
        ax1.set_xticks(np.arange(10, 101, 10))
        ax2.set_xticks(np.arange(10, 101, 10))
        ax1.set_yticks(np.arange(5, 56, 5))
        ax2.set_yticks(np.arange(15, 41, 5))
        ax1.grid()
        ax2.grid()
        plt.subplots_adjust(hspace=0.3, left=0.18, right=0.94, bottom=0.1, top=0.9)

        fig.suptitle(f'{{K=50, lamb=1, alpha=1, p=0.7}}', fontsize=12)

        # Save results
        plt.savefig(f'./{self.folder}/FIGURE.png')
        plt.show()


if __name__ == '__main__':
    import sys

    which_dataset = sys.argv[1]
    k_cross_val = int(sys.argv[2])
    which_k_cross_fold = int(sys.argv[3])
    iteration = sys.argv[4]
    sample_test = int(sys.argv[5])
    TOP_M_start = int(sys.argv[6])
    TOP_M_end = int(sys.argv[7])
    pred_type = sys.argv[8]

    train_folds = pickle.load(
        open(f"./input-data/train_NFLX_{k_cross_val}_folds.pkl", "rb")) if which_dataset == "nflx" else pickle.load(
        open(f"./input-data/train_{k_cross_val}_folds.pkl", "rb"))
    test_folds = pickle.load(
        open(f"./input-data/test_NFLX_{k_cross_val}_folds.pkl", "rb")) if which_dataset == "nflx" else pickle.load(
        open(f"./input-data/train_{k_cross_val}_folds.pkl", "rb"))

    rating_GroupForUser_train = train_folds[which_k_cross_fold - 1][0]
    rating_GroupForUser_test = test_folds[which_k_cross_fold - 1][0]
    rating_GroupForMovie_train = train_folds[which_k_cross_fold - 1][1]
    rating_GroupForMovie_test = test_folds[which_k_cross_fold - 1][1]

    eval = MyEvaluation(rating_GroupForUser_train, rating_GroupForUser_test,
                        rating_GroupForMovie_train, rating_GroupForMovie_test,
                        iteration, sample_test, TOP_M_start, TOP_M_end, pred_type)

    eval.plot()
