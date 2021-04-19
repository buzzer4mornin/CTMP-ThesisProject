import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

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

# ======================================================================================================================
# ============================================ Generate Test Set =======================================================
generator = np.random.RandomState(42)
permutation = generator.permutation(len(rating_GroupForUser))
test_size = 20000
c = 0
test_user_ids = []
for i in permutation:
    if len(rating_GroupForUser[i]) > 0:
        test_user_ids.append(i)
        c += 1
    if c == test_size:
        break

# ======================================================================================================================
# ============================ Compute Average Recalls and Precisions for Top-M ========================================

noncold_items = []
for movie_id in range(len(rating_GroupForMovie)):
    if len(rating_GroupForMovie[movie_id]) != 0:
        noncold_items.append(movie_id)


def per_user(user_id, TOP_M, in_matrix):
    if in_matrix:
        _mu = mu[noncold_items, :]
    else:
        _mu = mu

    ratings = np.dot((shp[user_id] / rte[user_id]), _mu.T)
    predicted_top_M = np.argsort(-ratings)[:TOP_M]
    actual = rating_GroupForUser[user_id]

    top_m_correct = np.sum(np.in1d(predicted_top_M, actual) * 1)
    recall = top_m_correct / len(rating_GroupForUser[user_id])
    precision = top_m_correct / TOP_M
    return recall, precision


def average_recalls_precisions(top_start, top_end):
    r, p = [], []
    for top in range(top_start, top_end):
        print("iteration:", top)
        recall_sum, precision_sum = 0, 0
        for usr in test_user_ids:
            i, j = per_user(usr, top, in_matrix=False)
            recall_sum += i
            precision_sum += j
        avg_recall = recall_sum / test_size
        avg_precision = precision_sum / test_size
        r.append(avg_recall)
        p.append(avg_precision)
    return r, p


avg_r, avg_p = average_recalls_precisions(1, 100)
print(avg_r)
print(avg_p)

"""PLOT RECALL GRAPH"""
fig, ax = plt.subplots()
ax.plot(range(1, 100), avg_r, label="CTMP")
ax.set_xlabel('Top-M')
ax.set_ylabel('Recall')
ax.set_title(f"Test size of {test_size} users")
ax.legend()
plt.grid()
plt.show()

"""PLOT PRECISION GRAPH"""
fig, ax = plt.subplots()
ax.plot(range(1, 100), avg_p, label="CTMP")
ax.set_xlabel('Top-M')
ax.set_ylabel('Precision')
ax.set_title(f"Test size of {test_size} users")
ax.legend()
plt.grid()
plt.show()
