import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='nflx', type=str, help="['nflx', 'original']")
parser.add_argument("--sample_test", default=50, type=int, help="Size of test set")
parser.add_argument("--TOP_M_start", default=10, type=int, help="Start of Top-M recommendation")
parser.add_argument("--TOP_M_end", default=11, type=int, help="End of Top-M recommendation")
parser.add_argument("--pred_type", default='out-of-matrix', type=str, help="['in-matrix', 'out-of-matrix']")
parser.add_argument("--k_cross_val", default=5, type=int, help="K-fold Cross Validation which was used")
parser.add_argument("--fold", default=1, type=int, help="Which fold of K-fold Cross Validation to test")
parser.add_argument("--iter", default=50, type=int, help="Which iteration result to test")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


# parser.add_argument("--folder", default=".test", type=str, help="Folder of saved outputs")


class Evaluation:
    def __init__(self, args):

        # Set seed
        np.random.seed(args.seed)

        train_folds = pickle.load(open(f"../input-data/train_NFLX_{args.k_cross_val}_folds.pkl", "rb"))
        test_folds = pickle.load(open(f"../input-data/test_NFLX_{args.k_cross_val}_folds.pkl", "rb"))

        self.rating_GroupForUser_TRAIN = train_folds[args.fold - 1][0]
        self.rating_GroupForUser_TEST = test_folds[args.fold - 1][0]

        self.rating_GroupForMovie_TRAIN = train_folds[args.fold - 1][1]
        self.rating_GroupForMovie_TEST = test_folds[args.fold - 1][1]

        # self.mu = np.load(f"../input-data/eval/{args.fold}-fold/{args.iter}th-iter/mu.npy")
        # self.shp = np.load(f"../input-data/eval/{args.fold}-fold/{args.iter}th-iter/shp.npy")
        # self.rte = np.load(f"../input-data/eval/{args.fold}-fold/{args.iter}th-iter/rte.npy")

        self.mu = np.load(f"../output-data/{args.fold}-fold/{args.iter}th-iter/mu.npy")
        self.shp = np.load(f"../output-data/{args.fold}-fold/{args.iter}th-iter/shp.npy")
        self.rte = np.load(f"../output-data/{args.fold}-fold/{args.iter}th-iter/rte.npy")

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

    def distributions_on_ratings(self):
        # ORIGINAL SET
        data = []
        for key in self.rating_GroupForUser_TEST:
            data.append(len(self.rating_GroupForUser_TEST[key]))

        # SAMPLE SET
        # data = []
        # sample = random.sample(list(self.rating_GroupForUser_TEST.keys()), args.sample_test)
        # for usr in sample:
        #     data.append(len(self.rating_GroupForUser_TEST[usr]))

        bins = np.arange(0, 700, 5)
        plt.xlim([min(data) - 5, max(data) + 5])
        plt.hist(data, bins=bins, alpha=0.5)
        plt.title('Random Gaussian data (fixed bin size)')
        plt.xlabel('variable X (bin size = 5)')
        plt.ylabel('count')
        plt.show()

    def generate_test_set(self) -> list:
        sample = random.sample(list(self.rating_GroupForUser_TEST.keys()), args.sample_test)
        test_set = []
        for usr in sample:
            if len(self.rating_GroupForUser_TEST[usr]) > 0 and len(self.rating_GroupForUser_TRAIN[usr]) > 0:
                test_set.append(usr)
        return test_set

    def _reset_vars(self):
        self.recalls_in_matrix_TEST = self.precisions_in_matrix_TEST = \
            self.recalls_out_of_matrix_TEST = self.precisions_out_of_matrix_TEST = \
            self.zero_place = 0

    def predict_in_matrix(self, user_id, top_m, ratings) -> None:
        actual_TEST = self.rating_GroupForUser_TEST[user_id]
        sorted_ratings = np.argsort(-ratings)
        predicted_top_M_TEST = np.setdiff1d(sorted_ratings, self.cold_items_TRAIN, assume_unique=True)[:top_m]
        predicted_top_M_TEST = np.setdiff1d(predicted_top_M_TEST, self.rating_GroupForUser_TRAIN[user_id],
                                            assume_unique=True)
        if len(predicted_top_M_TEST) == 0:
            self.zero_place += 1
        else:
            top_m_correct_TEST = np.sum(np.in1d(predicted_top_M_TEST, actual_TEST) * 1)
            self.recalls_in_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
            self.precisions_in_matrix_TEST += (top_m_correct_TEST / len(predicted_top_M_TEST))

    def predict_out_of_matrix(self, user_id, top_m, ratings) -> None:
        actual_TEST = self.rating_GroupForUser_TEST[user_id]
        sorted_ratings = np.argsort(-ratings)
        predicted_top_M_TEST = sorted_ratings[:top_m]
        predicted_top_M_TEST = np.setdiff1d(predicted_top_M_TEST, self.rating_GroupForUser_TRAIN[user_id],
                                            assume_unique=True)
        if len(predicted_top_M_TEST) == 0:
            self.zero_place += 1
        else:
            top_m_correct_TEST = np.sum(np.in1d(predicted_top_M_TEST, actual_TEST) * 1)
            self.recalls_out_of_matrix_TEST += (top_m_correct_TEST / len(self.rating_GroupForUser_TEST[user_id]))
            self.precisions_out_of_matrix_TEST += (top_m_correct_TEST / len(predicted_top_M_TEST))

    def avg_recall_precision(self) -> None:
        self.test_set = sorted(self.test_set)
        whole_rating = np.dot(self.shp[self.test_set] / self.rte[self.test_set], self.mu.T)
        for top in range(args.TOP_M_start, args.TOP_M_end):
            self._reset_vars()
            print(f"Top-M: {top}")
            if args.pred_type == "in-matrix":
                for i in range(len(self.test_set)):
                    self.predict_in_matrix(self.test_set[i], top, whole_rating[i])
                self.avg_recalls_in_matrix_TEST.append(
                    self.recalls_in_matrix_TEST / (len(self.test_set) - self.zero_place))
                self.avg_precisions_in_matrix_TEST.append(
                    self.precisions_in_matrix_TEST / (len(self.test_set) - self.zero_place))

            elif args.pred_type == "out-of-matrix":
                for i in range(len(self.test_set)):
                    self.predict_out_of_matrix(self.test_set[i], top, whole_rating[i])
                self.avg_recalls_out_of_matrix_TEST.append(
                    self.recalls_out_of_matrix_TEST / (len(self.test_set) - self.zero_place))
                self.avg_precisions_out_of_matrix_TEST.append(
                    self.precisions_out_of_matrix_TEST / (len(self.test_set) - self.zero_place))

    def plot(self) -> None:
        if args.pred_type == "in-matrix":
            r_TEST, p_TEST = self.avg_recalls_in_matrix_TEST, self.avg_precisions_in_matrix_TEST
        elif args.pred_type == "out-of-matrix":
            r_TEST, p_TEST = self.avg_recalls_out_of_matrix_TEST, self.avg_precisions_out_of_matrix_TEST

        # PLOT recall graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax1.plot(range(args.TOP_M_start, args.TOP_M_end), r_TEST, label="test")
        ax1.set_xlabel('Top-M', fontsize=11)
        ax1.set_ylabel('Recall', fontsize=11)
        # ax1.set_title(f"IMPORT SOME NAME HERE")
        ax1.legend()

        # PLOT precision graph
        ax2.plot(range(args.TOP_M_start, args.TOP_M_end), p_TEST, label="test")

        ax2.set_xlabel('Top-M', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        # ax2.set_title(f"IMPORT SOME NAME HERE")
        ax2.legend()

        # plot configs
        ax1.grid()
        ax2.grid()
        plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, bottom=0.15)
        fig.suptitle(f'{args.pred_type} predictions', fontsize=14)

        # save results
        plt.savefig(f'../input-data/eval/Image-{args.pred_type}-{args.sample_test}sample->>{args.fold}fold-{args.iter}th.png')
        with open(f"../input-data/eval/Recall-{args.pred_type}-{args.sample_test}sample->>{args.fold}fold-{args.iter}th.pkl", "wb") as f:
            pickle.dump(r_TEST, f)
        with open(f"../input-data/eval/Precision-{args.pred_type}-{args.sample_test}sample->>{args.fold}fold-{args.iter}th.pkl", "wb") as f:
            pickle.dump(p_TEST, f)

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
    assert args.fold in range(1, 6)
    # assert args.iter in [30, 50, 100]
    s = time.time()
    eval = Evaluation(args)
    print("SECONDS:", time.time() - s)
    eval.plot()

