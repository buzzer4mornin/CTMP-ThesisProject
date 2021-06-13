import matplotlib.pyplot as plt
import numpy as np

theta = np.load("./theta.npy")

def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype=np.float)
    for d in range(batch_size):
        sparsity[d] = len(np.where(doc_tp[d] > 1e-20)[0])
    sparsity /= num_topics
    return np.mean(sparsity)

s = compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
print("Sparse dimensions - {:.1f}%".format(s*100))

fig, axs = plt.subplots(2, 3, figsize=(9, 4))
ys = [11, 2, 17, 20, 26, 22]

for i, ax in enumerate(axs.reshape(-1)):
    y = theta[ys[i]]
    x = np.arange(100)
    ax.bar(x, y, color="black")
    ax.set_ylim([0, 1])
    ax.set_xlabel('Topic')
    ax.set_xticks([])
    ax.set_xticklabels([])

plt.show()

# """fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 4))
#
# y = theta[11]
# x = np.arange(100)
# ax1.bar(x, y, color="black")
# ax1.set_ylim([0, 1])
# ax1.set_xlabel('Topic')
# ax1.set_xticks([])
# ax1.set_xticklabels([])
#
#
# y = theta[2]
# x = np.arange(100)
# ax2.bar(x, y, color="black")
# ax2.set_ylim([0, 1])
# ax2.set_xlabel('Topic')
# ax2.set_xticks([])
# ax2.set_xticklabels([])
#
#
# y = theta[17]
# x = np.arange(100)
# ax3.bar(x, y, color="black")
# ax3.set_ylim([0, 1])
# ax3.set_xlabel('Topic')
# ax3.set_xticks([])
# ax3.set_xticklabels([])
#
#
# y = theta[20]
# x = np.arange(100)
# ax4.bar(x, y, color="black")
# ax4.set_ylim([0, 1])
# ax4.set_xlabel('Topic')
# ax4.set_xticks([])
# ax4.set_xticklabels([])
#
#
# y = theta[26]
# x = np.arange(100)
# ax5.bar(x, y, color="black")
# ax5.set_ylim([0, 1])
# ax5.set_xlabel('Topic')
# ax5.set_xticks([])
# ax5.set_xticklabels([])
#
#
# y = theta[22]
# x = np.arange(100)
# ax6.bar(x, y, color="black")
# ax6.set_ylim([0, 1])
# ax6.set_xlabel('Topic')
# ax6.set_xticks([])
# ax6.set_xticklabels([])
#
# plt.show()"""



