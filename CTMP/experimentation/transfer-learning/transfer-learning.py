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


movie_df_M = pd.read_pickle("../../../db-files/processed-files/df_movie_CLEANED")
movie_df_F = pd.read_pickle("../../../db-files/processed-files/df_movie_NFLX_CLEANED")
movie_df_M["MOVIEID_MVLNS"] = np.arange(len(movie_df_M))
movie_df_F["MOVIEID_NFLX"] = np.arange(len(movie_df_F))

joint = pd.merge(movie_df_M, movie_df_F, how='inner', on='TT')
joint = joint.drop(["MOVIEPLOT_y", "TT", "MOVIEID_x", "MOVIEID_y"], axis=1)
#print("-- Total # of Joint Movies from MovieLens and NFLX:", joint.shape[0])

theta_mv = np.load("theta_MVLNS.npy")
theta_nf = np.load("theta_NFLX.npy")

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
i = 573
row = np.array(joint)[i]
plot, mv, nf = row
y_mv = theta_mv[mv]
y_nf = theta_nf[nf]
tops_mv = open("topn_output_MVLNS.txt", "r")
topics_mv = tops_mv.readlines()
tops_nf = open("topn_output_NFLX_k=50.txt", "r")
topics_nf = tops_nf.readlines()

print(i)
print("-------")
print(plot)
print("-------")
print("MVLNS proportions:", np.where(y_mv > 0.01)[0]+1)
print("NFLX proportions:", np.where(y_nf > 0.01)[0]+1)
print("-------")


for j in np.where(y_mv > 0.01)[0]:
    print(topics_mv[j])

for j in np.where(y_nf > 0.01)[0]:
    print(topics_nf[j])

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4))

y = y_mv*100
x = np.arange(100)
ax1.set_xticks(np.arange(0, 101, 1))
ax1.set_xlim([-1, 101])
ax1.bar(x, y, color="black")
ax1.set_ylim([0, 100])
ax1.set_ylabel('topic proportion %', fontsize=12)
ax1.set_xlabel('100 Topics \n(CTMP on MovieLens 20M)', fontsize=12)
ax1.set_xticklabels([])

y = y_nf*100
x = np.arange(50)
ax2.set_xticks(np.arange(0, 51, 1))
ax2.set_xlim([-1, 51])
ax2.bar(x, y, color="black")
ax2.set_ylim([0, 100])
ax2.set_ylabel('topic proportion %', fontsize=12)
ax2.set_xlabel('50 Topics \n(CTMP on NETFLIX)', fontsize=12)
ax2.set_xticklabels([])


plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, bottom=0.15)
fig.suptitle(f'The Naked City (1948)', fontsize=14)
plt.show()