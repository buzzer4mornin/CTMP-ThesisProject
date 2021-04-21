import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

# ======================================================================================================================
# =============================================== Read Data ============================================================

# with open("../saved-outputs/phi.pkl", "rb") as f:
#    phi = pickle.load(f)

with open("../saved-outputs/rating_GroupForUser.pkl", "rb") as f:
    rating_GroupForUser = pickle.load(f)

with open("../saved-outputs/rating_GroupForMovie.pkl", "rb") as f:
    rating_GroupForMovie = pickle.load(f)

mu = np.load("../saved-outputs/mu.npy")
shp = np.load("../saved-outputs/shp.npy")
rte = np.load("../saved-outputs/rte.npy")


def main():
    # Get environment variables
    test_size = int(sys.argv[1])
    TOP_M_start = int(sys.argv[2])
    TOP_M_end = int(sys.argv[3])

    # ==================================================================================================================
    # ============================================ Generate Test Set ===================================================
    generator = np.random.RandomState(11)
    permutation = generator.permutation(len(rating_GroupForUser))
    c = 0
    test_user_ids = []
    for i in permutation:
        if len(rating_GroupForUser[i]) > 0:
            test_user_ids.append(i)
            c += 1
        if c == test_size:
            break

    # ==================================================================================================================
    # ============================ Compute Average Recalls and Precisions for Top-M ====================================

    # size of noncold items - 20,323 (out of 25,900)
    noncold_items = []
    cold_items = []
    for movie_id in range(len(rating_GroupForMovie)):
        if len(rating_GroupForMovie[movie_id]) != 0:
            noncold_items.append(movie_id)
        else:
            cold_items.append(movie_id)

    def per_user(user_id, TOP_M, in_matrix):
        ratings = np.dot((shp[user_id] / rte[user_id]), mu.T)
        actual = rating_GroupForUser[user_id]

        if in_matrix:
            sorted_ratings = np.argsort(-ratings)
            '''predicted_top_M_ = []
            t = 0
            for i in sorted_ratings:
                if i not in cold_items:
                    predicted_top_M_.append(i)
                    t += 1
                # else:
                #    print("yess")
                if t == TOP_M:
                    break'''
            predicted_top_M = np.setdiff1d(sorted_ratings, cold_items, assume_unique=True)[:TOP_M]
            top_m_correct = np.sum(np.in1d(predicted_top_M, actual) * 1)
        else:
            predicted_top_M = np.argsort(-ratings)[:TOP_M]
            top_m_correct = np.sum(np.in1d(predicted_top_M, actual) * 1)  # add this??? --> + np.sum(np.in1d(predicted_top_M, cold_items) * 1)

        recall = top_m_correct / len(rating_GroupForUser[user_id])
        precision = top_m_correct / TOP_M
        return recall, precision

    def average_recalls_precisions(top_start, top_end, in_matrix=None):
        r, p = [], []
        for top in range(top_start, top_end):
            print("iteration:", top)
            recall_sum, precision_sum = 0, 0
            for usr in test_user_ids:
                i, j = per_user(usr, top, in_matrix)
                recall_sum += i
                precision_sum += j
            avg_recall = recall_sum / test_size
            avg_precision = precision_sum / test_size
            r.append(avg_recall)
            p.append(avg_precision)
        return r, p

    avg_r_in, avg_p_in = average_recalls_precisions(TOP_M_start, TOP_M_end, in_matrix=True)
    avg_r_out, avg_p_out = average_recalls_precisions(TOP_M_start, TOP_M_end, in_matrix=False)

    """PLOT RECALL GRAPH"""
    fig, ax = plt.subplots()
    ax.plot(range(TOP_M_start, TOP_M_end), avg_r_in, label="in-matrix")
    ax.plot(range(TOP_M_start, TOP_M_end), avg_r_out, label="out-of-matrix")
    ax.set_xlabel('Top-M')
    ax.set_ylabel('Recall')
    ax.set_title(f"Test size of {test_size}")
    ax.legend()
    plt.grid()
    plt.show()

    """PLOT PRECISION GRAPH"""
    fig, ax = plt.subplots()
    ax.plot(range(TOP_M_start, TOP_M_end), avg_p_in, label="in-matrix")
    ax.plot(range(TOP_M_start, TOP_M_end), avg_p_out, label="out-of-matrix")
    ax.set_xlabel('Top-M')
    ax.set_ylabel('Precision')
    ax.set_title(f"Test size of {test_size}")
    ax.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

# =========== Saved Results ==============

# --- Set size = 2,000, in-matrix ---


# --- Set size = 2,000, out-of-matrix ---
