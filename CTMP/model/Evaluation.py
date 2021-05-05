import pickle
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor


class MyEvaluation:
    def __init__(self, user_train, user_test, movie_train, movie_test, iteration,
                 sample_test=50, TOP_M_start=10, TOP_M_end=100, pred_type='out-of-matrix', seed=42):

        assert pred_type in ['in-matrix', 'out-of-matrix', 'both']
        self.folder = f"output-data/{iteration}"
        self.sample_test = sample_test
        self.TOP_M_start = TOP_M_start
        self.TOP_M_end = TOP_M_end
        self.pred_type = pred_type
        self.seed = seed

        # Set seed
        np.random.seed(self.seed)

        self.rating_GroupForUser_TRAIN, self.rating_GroupForUser_TEST = user_train, user_test,
        self.rating_GroupForMovie_TRAIN, self.rating_GroupForMovie_TEST = movie_train, movie_test

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
        # TRAIN
        self.avg_recalls_in_matrix_TRAIN, self.avg_precisions_in_matrix_TRAIN = [], []
        self.avg_recalls_out_of_matrix_TRAIN, self.avg_precisions_out_of_matrix_TRAIN = [], []
        # TEST
        self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST = [], []
        self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST = [], []

        # Update them accordingly
        self.avg_recall_precision()

    def group_items(self) -> list:
        """Number of cold items - 5,577/25,900 || Number of noncold items - 20,323/25,900"""
        cold_items_TRAIN, cold_items_TEST = [], []
        noncold_items_TRAIN, noncold_items_TEST = [], []

        for movie_id in self.rating_GroupForMovie_TRAIN:
            if len(self.rating_GroupForMovie_TRAIN[movie_id]) != 0:
                noncold_items_TRAIN.append(movie_id)
            else:
                cold_items_TRAIN.append(movie_id)

        for movie_id in self.rating_GroupForMovie_TEST:
            if len(self.rating_GroupForMovie_TEST[movie_id]) != 0:
                noncold_items_TEST.append(movie_id)
            else:
                cold_items_TEST.append(movie_id)

        print(f"Training set: cold-{len(cold_items_TRAIN)}, noncold-{len(noncold_items_TRAIN)}")
        print(f"Testing set: cold-{len(cold_items_TEST)}, noncold-{len(noncold_items_TEST)}")
        return cold_items_TRAIN, cold_items_TEST, noncold_items_TRAIN, noncold_items_TEST

    def generate_test_set(self) -> list:
        # TODO: consider both TRAIN and TEST ??
        sample = random.sample(list(self.rating_GroupForUser_TEST.keys()), self.sample_test)
        test_set = []
        for u in sample:
            if len(self.rating_GroupForUser_TEST[u]) > 0:
                test_set.append(u)
        # avg = 0
        # for i in self.rating_GroupForUser_TRAIN:
        #       avg += len(self.rating_GroupForUser_TRAIN[i])
        # print(avg/len(self.rating_GroupForUser_TRAIN))
        # exit()
        return test_set

    # TODO add new one
    # here ...
    #TODO old one
    def predict_in_matrix(self, user_id, top_m) -> None:
        """Compute in-matrix recall and precision for a given user, then add them to the sum"""
        ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
        actual_TRAIN = self.rating_GroupForUser_TRAIN[user_id]
        actual_TEST = self.rating_GroupForUser_TEST[user_id]
        sorted_ratings = np.argsort(-ratings)
        predicted_top_M_TRAIN = np.setdiff1d(sorted_ratings, self.cold_items_TRAIN, assume_unique=True)[:top_m]
        predicted_top_M_TEST = np.setdiff1d(sorted_ratings, self.cold_items_TEST, assume_unique=True)[:top_m]
        top_m_correct_TRAIN = np.sum(np.in1d(predicted_top_M_TRAIN, actual_TRAIN) * 1)
        top_m_correct_TEST = np.sum(np.in1d(predicted_top_M_TEST, actual_TEST) * 1)
        self.recalls_in_matrix_TRAIN += (top_m_correct_TRAIN / len(self.rating_GroupForUser_TRAIN[user_id]))
        self.precisions_in_matrix_TRAIN += (top_m_correct_TRAIN / top_m)
        self.recalls_in_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
        self.precisions_in_matrix_TEST += (top_m_correct_TEST / top_m)

    # TODO old one
    # def predict_out_of_matrix(self, user_id, top_m) -> None:
    #     """Compute out-of-matrix recall and precision for a given user, then add them to the sum"""
    #     ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
    #     actual_TRAIN = self.rating_GroupForUser_TRAIN[user_id]
    #     actual_TEST = self.rating_GroupForUser_TEST[user_id]
    #     predicted_top_M = np.argsort(-ratings)[:top_m]
    #     top_m_correct_TRAIN = np.sum(np.in1d(predicted_top_M, actual_TRAIN) * 1)
    #     top_m_correct_TEST = np.sum(np.in1d(predicted_top_M, actual_TEST) * 1)
    #     self.recalls_out_of_matrix_TRAIN += (top_m_correct_TRAIN / len(self.rating_GroupForUser_TRAIN[user_id]))
    #     self.precisions_out_of_matrix_TRAIN += (top_m_correct_TRAIN / top_m)
    #     self.recalls_out_of_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
    #     self.precisions_out_of_matrix_TEST += (top_m_correct_TEST / top_m)
    # TODO new one
    def predict_out_of_matrix(self, user_id, top_m) -> None:
        """Compute out-of-matrix recall and precision for a given user, then add them to the sum"""
        ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
        actual_TRAIN = self.rating_GroupForUser_TRAIN[user_id]
        actual_TEST = self.rating_GroupForUser_TEST[user_id]
        sorted_ratings = np.argsort(-ratings)
        predicted_top_M_TEST = np.setdiff1d(sorted_ratings, self.rating_GroupForUser_TRAIN[user_id], assume_unique=True)[:top_m]
        predicted_top_M_TRAIN = np.argsort(-ratings)[:top_m]
        top_m_correct_TRAIN = np.sum(np.in1d(predicted_top_M_TRAIN, actual_TRAIN) * 1)
        top_m_correct_TEST = np.sum(np.in1d(predicted_top_M_TEST, actual_TEST) * 1)
        self.recalls_out_of_matrix_TRAIN += (top_m_correct_TRAIN / len(self.rating_GroupForUser_TRAIN[user_id]))
        self.precisions_out_of_matrix_TRAIN += (top_m_correct_TRAIN / top_m)
        self.recalls_out_of_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
        self.precisions_out_of_matrix_TEST += (top_m_correct_TEST / top_m)

    def avg_recall_precision(self) -> None:
        for top in range(self.TOP_M_start, self.TOP_M_end):
            # make all metrics zero for new iteration
            print(f"Top-M: {top}")
            self.recalls_in_matrix_TRAIN, self.precisions_in_matrix_TRAIN = 0, 0
            self.recalls_out_of_matrix_TRAIN, self.precisions_out_of_matrix_TRAIN = 0, 0

            self.recalls_in_matrix_TEST, self.precisions_in_matrix_TEST = 0, 0
            self.recalls_out_of_matrix_TEST, self.precisions_out_of_matrix_TEST = 0, 0

            if self.pred_type == "both":
                for usr in self.test_set:
                    self.predict_in_matrix(usr, top)
                    self.predict_out_of_matrix(usr, top)
                self.avg_recalls_in_matrix_TRAIN.append(self.recalls_in_matrix_TRAIN / self.sample_test)
                self.avg_precisions_in_matrix_TRAIN.append(self.precisions_in_matrix_TRAIN / self.sample_test)
                self.avg_recalls_out_of_matrix_TRAIN.append(self.recalls_out_of_matrix_TRAIN / self.sample_test)
                self.avg_precisions_out_of_matrix_TRAIN.append(self.precisions_out_of_matrix_TRAIN / self.sample_test)
            elif self.pred_type == "in-matrix":
                for usr in self.test_set:
                    self.predict_in_matrix(usr, top)
                self.avg_recalls_in_matrix_TRAIN.append(self.recalls_in_matrix_TRAIN / self.sample_test)
                self.avg_precisions_in_matrix_TRAIN.append(self.precisions_in_matrix_TRAIN / self.sample_test)
                self.avg_recalls_in_matrix_TEST.append(self.recalls_in_matrix_TEST / self.sample_test)
                self.avg_precisions_in_matrix_TEST.append(self.precisions_in_matrix_TEST / self.sample_test)

            elif self.pred_type == "out-of-matrix":
                for usr in self.test_set:
                    self.predict_out_of_matrix(usr, top)
                self.avg_recalls_out_of_matrix_TRAIN.append(self.recalls_out_of_matrix_TRAIN / self.sample_test)
                self.avg_precisions_out_of_matrix_TRAIN.append(self.precisions_out_of_matrix_TRAIN / self.sample_test)
                self.avg_recalls_out_of_matrix_TEST.append(self.recalls_out_of_matrix_TEST / self.sample_test)
                self.avg_precisions_out_of_matrix_TEST.append(self.precisions_out_of_matrix_TEST / self.sample_test)

    def plot(self) -> None:
        if self.pred_type == "both":
            r_i_TRAIN, p_i_TRAIN = self.avg_recalls_in_matrix_TRAIN, self.avg_precisions_in_matrix_TRAIN
            r_o_TRAIN, p_o_TRAIN = self.avg_recalls_out_of_matrix_TRAIN, self.avg_precisions_out_of_matrix_TRAIN
            r_i_TEST, p_i_TEST = self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST
            r_o_TEST, p_o_TEST = self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST
        elif self.pred_type == "in-matrix":
            r_TRAIN, p_TRAIN = self.avg_recalls_in_matrix_TRAIN, self.avg_precisions_in_matrix_TRAIN
            r_TEST, p_TEST = self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST
        elif self.pred_type == "out-of-matrix":
            r_TRAIN, p_TRAIN = self.avg_recalls_out_of_matrix_TRAIN, self.avg_precisions_out_of_matrix_TRAIN
            r_TEST, p_TEST = self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST

        # PLOT recall graph
        plt.ioff()  # Turn interactive plotting off
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        if self.pred_type == "both":
            ax1.plot(range(self.TOP_M_start, self.TOP_M_end), r_i_TRAIN, label="in-matrix-train")
            ax1.plot(range(self.TOP_M_start, self.TOP_M_end), r_o_TRAIN, label="out-of-matrix-train")
            ax1.plot(range(self.TOP_M_start, self.TOP_M_end), r_i_TEST, label="in-matrix-test")
            ax1.plot(range(self.TOP_M_start, self.TOP_M_end), r_o_TEST, label="out-of-matrix-test")
        else:
            ax1.plot(range(self.TOP_M_start, self.TOP_M_end), r_TRAIN, label="train")
            ax1.plot(range(self.TOP_M_start, self.TOP_M_end), r_TEST, label="test")
        ax1.set_xlabel('Top-M', fontsize=11)
        ax1.set_ylabel('Recall', fontsize=11)
        ax1.set_title(f"Sample size: {self.sample_test}")
        ax1.legend()

        # PLOT precision graph
        if self.pred_type == "both":
            ax2.plot(range(self.TOP_M_start, self.TOP_M_end), p_i_TRAIN, label="in-matrix-train")
            ax2.plot(range(self.TOP_M_start, self.TOP_M_end), p_o_TRAIN, label="out-of-matrix-train")
            ax2.plot(range(self.TOP_M_start, self.TOP_M_end), p_i_TEST, label="in-matrix-test")
            ax2.plot(range(self.TOP_M_start, self.TOP_M_end), p_o_TEST, label="out-of-matrix-test")
        else:
            ax2.plot(range(self.TOP_M_start, self.TOP_M_end), p_TRAIN, label="train")
            ax2.plot(range(self.TOP_M_start, self.TOP_M_end), p_TEST, label="test")
        ax2.set_xlabel('Top-M', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title(f"Sample size: {self.sample_test}")
        ax2.legend()

        # plot configs
        ax1.grid()
        ax2.grid()
        plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, bottom=0.15)
        fig.suptitle(f'{self.pred_type} predictions', fontsize=14)
        plt.savefig(f'./{self.folder}/FIGURE.png')
        # plt.show()
