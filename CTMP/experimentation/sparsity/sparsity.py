import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ======================================== Joint Matrix Heat MAP =======================================================

"""nflx_p07_k50 = np.array([[96.3344, 91.3271, 61.0266],
                         [97.9888, 97.8130, 91.3925],
                         [97.9878, 97.8059, 91.3563]])


nflx_p07_k100 = np.array([[98.3988,  95.8885, 80.2701],
                         [98.9981,  98.9496, 95.9888],
                         [98.9975,  98.9480, 96.1034]])

nflx_p09_k50 = np.array([[95.8686,  89.9838, 56.2083],
                         [97.9947, 97.8990,  91.8691],
                         [97.9952, 97.9076,  92.3017]])


nflx_p09_k100 = np.array([[98.2607, 95.1714, 77.6233],
                         [98.9992, 98.9796, 96.4929],
                         [98.9995, 98.9480, 96.1034]])


fig = plt.figure(figsize=(9, 4))


ax = fig.add_subplot(111)
ax.set_title('k=100', fontsize=16)#, color="blue")
plt.imshow(nflx_p09_k100, cmap='viridis', interpolation='antialiased')

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

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

plt.colorbar(orientation='vertical', cax=cax)
plt.subplots_adjust(wspace=0.5, left=0.1, right=0.95, bottom=0.18, top=0.82)

for i in t:
    for j in t:
        if i==2 and j==0:
            ax.text(i, j, f"{round(nflx_p09_k100[j][i], 2)}%", color='white', ha='center', va='center')
        else:
            ax.text(i, j, f"{round(nflx_p09_k100[j][i], 2)}%", color='black', ha='center', va='center')

plt.show()
exit()"""

# ================================================ MERGE PHOTOS ========================================================
"""import sys
from PIL import Image

images = [Image.open(x) for x in ['07s.jpg', '09s.jpg']]
widths, heights = zip(*(i.size for i in images))

horiz = False

if horiz:
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), color="white")
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
else:
    total_width = sum(widths)
    total_width = int(total_width / 2)
    max_height = 2 * max(heights) + 100
    new_im = Image.new('RGB', (total_width, max_height), color="white")
    x_offset = 0
    for im in images:
        new_im.paste(im, (0, x_offset))
        x_offset += im.size[1]

new_im.save('xx.jpg')

exit()
"""

# ================================================== Sparsity ==========================================================
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

# ================================================ Sparsity Examples ===================================================
"""# TODO
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

# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 4))
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
# plt.show()
"""
