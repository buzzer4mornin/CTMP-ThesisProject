"""num_docs: ---
num_terms: ---
num_topics: 100 for MVLNS, 50 for NFLX
user_size: ---
tops: 10
lamb: 1.0
e: 0.3
f: 0.3
alpha: 1.0
iter_infer: 100
iter_train: 50
BOPE-p: 0.7

WE CONSIDER 50th iteration results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

movie_df_M = pd.read_pickle("./processed-files/df_movie_CLEANED")
movie_df_F = pd.read_pickle("./processed-files/df_movie_NFLX_CLEANED")
movie_df_M["MOVIEID_MVLNS"] = np.arange(len(movie_df_M))
movie_df_F["MOVIEID_NFLX"] = np.arange(len(movie_df_F))

joint = pd.merge(movie_df_M, movie_df_F, how='inner', on='TT')
joint = joint.drop(["MOVIEPLOT_y", "TT", "MOVIEID_x", "MOVIEID_y"], axis=1)
#print("-- Total # of Joint Movies from MovieLens and NFLX:", joint.shape[0])

theta_mv = np.load("./transfer-learning/theta_MVLNS.npy")
theta_nf = np.load("./transfer-learning/theta_NFLX.npy")

# Compute Sparsity
"""def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype=np.float)
    for d in range(batch_size):
        sparsity[d] = len(np.where(doc_tp[d] > 1e-20)[0])
    sparsity /= num_topics
    return np.mean(sparsity)
s = compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
print("-- Sparse dimensions: {:.1f}%".format(s*100))"""


# BESTS: 573, 3754, 6318, 2188, 340, 1060, 3788, 4125
i = np.random.randint(6756)
# i=573
row = np.array(joint)[i]
plot, mv, nf = row
y_mv = theta_mv[mv]
y_nf = theta_nf[nf]
tops_mv = open("./transfer-learning/topn_output_MVLNS.txt", "r")
topics_mv = tops_mv.readlines()
tops_nf = open("./transfer-learning/topn_output_NFLX_k=50.txt", "r")
topics_nf = tops_nf.readlines()

print(i)
print("-------")
print(plot)
print("-------")
print("MVLNS proportions:", np.where(y_mv > 0.05)[0])
print("NFLX proportions:", np.where(y_nf > 0.05)[0])
print("-------")


for i in np.where(y_mv > 0.05)[0]:
    print(topics_mv[i])

for i in np.where(y_nf > 0.05)[0]:
    print(topics_nf[i])

# fig, axs = plt.subplots(2, 2, figsize=(9, 4))
# for i, ax in enumerate(axs.reshape(-1)):
#     if i in range(0, 2):
#         y = theta_mv[mv[i]]
#         x = np.arange(100)
#         ax.set_xticks(np.arange(0, 101, 1))
#     else:
#         y = theta_nf[nf[i-2]]
#         x = np.arange(50)
#         ax.set_xticks(np.arange(0, 51, 1))
#
#     ax.bar(x, y, color="black")
#     ax.set_ylim([0, 1])
#     ax.set_xlabel('Topics')
# plt.show()