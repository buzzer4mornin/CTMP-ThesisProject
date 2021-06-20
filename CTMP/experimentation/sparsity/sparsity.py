import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

nflx_p07_k100 = np.array([[98.3988,  95.8885, 80.2701],
                         [98.9981,  98.9496, 95.9888],
                         [98.9975,  98.9480, 96.1034]])

fig = plt.figure(figsize=(9, 4))


ax = fig.add_subplot(111)
ax.set_title('p=0.7; k=100')
plt.imshow(nflx_p07_k100)

cax = fig.add_axes([0.5, 0.16, 0.235, 0.7])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
t = [0, 1, 2]
x = ["lamb=1", "lamb=10", "lamb=100"]
y = ["alpha=1", "alpha=0.1", "alpha=0.01"]
ax.xaxis.set_ticks(t)
ax.yaxis.set_ticks(t)
ax.set_xticklabels(x)
ax.set_yticklabels(y)
# ax.set_xlabel("something")

plt.colorbar(orientation='vertical')
plt.subplots_adjust(wspace=1, left=0.1, right=0.95, bottom=0.15)
plt.show()


exit()
theta = np.load("./theta.npy")

def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype=np.float)
    for d in range(batch_size):
        sparsity[d] = len(np.where(doc_tp[d] < 1e-20)[0])
    sparsity /= num_topics
    return np.mean(sparsity)


s = compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
print("Sparse dimensions - {:.4f}%".format(s * 100))

exit()
# TODO
# PUT one example for above 95%
#    another example for 80% (lamb=100)
# TODO: Examples of “sparse” estimates.
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
