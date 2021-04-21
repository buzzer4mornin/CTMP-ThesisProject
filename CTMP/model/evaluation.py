import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", default=200, type=int, help="Size of test set")
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--TOP_M_start", default=10, type=int, help="Start of Top-M recommendation")
parser.add_argument("--TOP_M_end", default=100, type=int, help="End of Top-M recommendation")
parser.add_argument("--pred_type", default='in-matrix', type=str, help="Type of prediction - ['in-matrix', 'out-of-matrix', 'both']")
parser.add_argument("--seed", default=11, type=int, help="Random seed.")


class Evaluation:
    def __init__(self, args):
        # Set seed
        np.random.seed(args.seed)

        # Read data
        with open("../saved-outputs/rating_GroupForUser.pkl", "rb") as f:
            self.rating_GroupForUser = pickle.load(f)

        with open("../saved-outputs/rating_GroupForMovie.pkl", "rb") as f:
            self.rating_GroupForMovie = pickle.load(f)

        self.mu = np.load("../saved-outputs/mu.npy")
        self.shp = np.load("../saved-outputs/shp.npy")
        self.rte = np.load("../saved-outputs/rte.npy")

        # Group items separately
        self.cold_items, self.noncold_items = self.group_items()

        # Generate test set
        self.test_set = self.generate_test_set()

        # Average Recalls and Precisions over all users of test set across the Top-M
        # For both in_matrix and out-of-matrix prediction
        self.avg_recalls_in_matrix, self.avg_precisions_in_matrix = [], []
        self.avg_recalls_out_of_matrix, self.avg_precisions_out_of_matrix = [], []
        # Update them accordingly
        self.avg_recall_precision()

    def group_items(self) -> list:
        """Number of cold items - 5,577/25,900 || Number of noncold items - 20,323/25,900"""
        cold_items = []
        noncold_items = []
        for movie_id in range(len(self.rating_GroupForMovie)):
            if len(self.rating_GroupForMovie[movie_id]) != 0:
                noncold_items.append(movie_id)
            else:
                cold_items.append(movie_id)
        return cold_items, noncold_items

    def generate_test_set(self) -> list:
        generator = np.random.RandomState(args.seed)
        permutation = generator.permutation(len(self.rating_GroupForUser))
        c = 0
        test_set = []
        for i in permutation:
            if len(self.rating_GroupForUser[i]) > 0:
                test_set.append(i)
                c += 1
            if c == args.test_size:
                break
        return test_set

    def predict_in_matrix(self, user_id, top_m) -> None:
        """Compute in-matrix recall and precision for a given user, then add them to the sum"""
        ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
        actual = self.rating_GroupForUser[user_id]
        sorted_ratings = np.argsort(-ratings)
        predicted_top_M = np.setdiff1d(sorted_ratings, self.cold_items, assume_unique=True)[:top_m]
        top_m_correct = np.sum(np.in1d(predicted_top_M, actual) * 1)
        self.recalls_in_matrix += (top_m_correct / len(self.rating_GroupForUser[user_id]))
        self.precisions_in_matrix += (top_m_correct / top_m)

    def predict_out_of_matrix(self, user_id, top_m) -> None:
        """Compute out-of-matrix recall and precision for a given user, then add them to the sum"""
        ratings = np.dot((self.shp[user_id] / self.rte[user_id]), self.mu.T)
        actual = self.rating_GroupForUser[user_id]
        predicted_top_M = np.argsort(-ratings)[:top_m]
        top_m_correct = np.sum(np.in1d(predicted_top_M, actual) * 1)
        self.recalls_out_of_matrix += (top_m_correct / len(self.rating_GroupForUser[user_id]))
        self.precisions_out_of_matrix += (top_m_correct / top_m)

    def avg_recall_precision(self) -> None:
        for top in range(args.TOP_M_start, args.TOP_M_end):
            # make all metrics zero for new iteration
            print(f"Top iteration: {top}")
            self.recalls_in_matrix, self.precisions_in_matrix = 0, 0
            self.recalls_out_of_matrix, self.precisions_out_of_matrix = 0, 0

            if args.pred_type == "both":
                for usr in self.test_set:
                    self.predict_in_matrix(usr, top)
                    self.predict_out_of_matrix(usr, top)
                self.avg_recalls_in_matrix.append(self.recalls_in_matrix / args.test_size)
                self.avg_precisions_in_matrix.append(self.precisions_in_matrix / args.test_size)
                self.avg_recalls_out_of_matrix.append(self.recalls_out_of_matrix / args.test_size)
                self.avg_precisions_out_of_matrix.append(self.precisions_out_of_matrix / args.test_size)
            elif args.pred_type == "in-matrix":
                for usr in self.test_set:
                    self.predict_in_matrix(usr, top)
                self.avg_recalls_in_matrix.append(self.recalls_in_matrix / args.test_size)
                self.avg_precisions_in_matrix.append(self.precisions_in_matrix / args.test_size)
            elif args.pred_type == "out-of-matrix":
                for usr in self.test_set:
                    self.predict_out_of_matrix(usr, top)
                self.avg_recalls_out_of_matrix.append(self.recalls_out_of_matrix / args.test_size)
                self.avg_precisions_out_of_matrix.append(self.precisions_out_of_matrix / args.test_size)

    def plot(self) -> None:
        if args.pred_type == "both":
            r_i, p_i = self.avg_recalls_in_matrix, self.avg_precisions_in_matrix
            r_o, p_o = self.avg_recalls_out_of_matrix, self.avg_precisions_out_of_matrix
        elif args.pred_type == "in-matrix":
            r, p = self.avg_recalls_in_matrix, self.avg_precisions_in_matrix
        elif args.pred_type == "out-of-matrix":
            r, p = self.avg_recalls_out_of_matrix, self.avg_precisions_out_of_matrix

        # PLOT recall graph
        fig, ax = plt.subplots()
        if args.pred_type == "both":
            ax.plot(range(args.TOP_M_start, args.TOP_M_end), r_i, label="in-matrix")
            ax.plot(range(args.TOP_M_start, args.TOP_M_end), r_o, label="out-of-matrix")
        else:
            ax.plot(range(args.TOP_M_start, args.TOP_M_end), r, label=f"{args.pred_type}")
        ax.set_xlabel('Top-M')
        ax.set_ylabel('Recall')
        ax.set_title(f"Test size of {args.test_size}")
        ax.legend()
        plt.grid()
        plt.show()

        # PLOT precision graph
        fig, ax = plt.subplots()
        if args.pred_type == "both":
            ax.plot(range(args.TOP_M_start, args.TOP_M_end), p_i, label="in-matrix")
            ax.plot(range(args.TOP_M_start, args.TOP_M_end), p_o, label="out-of-matrix")
        else:
            ax.plot(range(args.TOP_M_start, args.TOP_M_end), p, label=f"{args.pred_type}")
        ax.set_xlabel('Top-M')
        ax.set_ylabel('Precision')
        ax.set_title(f"Test size of {args.test_size}")
        ax.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    assert args.pred_type in ['in-matrix', 'out-of-matrix', 'both']
    e = Evaluation(args)
    e.plot()

# =========== Saved Results ==============
# --- Set size = 2,000, in-matrix ---
# --- Set size = 2,000, out-of-matrix ---